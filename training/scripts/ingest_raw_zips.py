"""data/raw/data/*.zip (CVAT에서 내보낸 "YOLO Pose 1.0" export)을 압축 해제
없이 바로 읽어서 data/raw/img/, data/raw/label/ 에 평평하게(flatten) 모아준다.

CVAT task export zip 구조 (실제 관찰된 구조):
    <task>/img/*.png|jpg
    <task>/label/data.yaml                       # kpt_shape, names 등
    <task>/label/labels/<Train|train>/*.txt       # 이미지와 동일한 파일명(.txt)
    <task>/label/<Train|train>.txt                # CVAT 참고용 목록 (사용 안 함)

라벨 폴더 이름은 "label"뿐 아니라 "autolabel"(모델이 자동으로 라벨링한
배치)도 나온다 — 폴더 이름이 "label"로 끝나기만 하면(정확히 "label/"이든
"autolabel/"이든) 같은 구조로 취급해서 인식한다.

zip 하나에 task 폴더가 여러 개 들어있는 경우(예: 프로젝트 전체를 한 번에
내보낸 번들 zip)도 있고, 서로 다른 zip에 같은 task_id가 중복으로 들어있는
경우도 있다(예: 001.zip 의 "001" 과 다른 번들 zip 안의 "001" 이 이미지 장수가
살짝 다름 — 어느 쪽이 최신인지 이 스크립트는 알 수 없다).

이런 충돌을 조용히 덮어쓰지 않기 위해 아래 우선순위로 소스를 고른다:
  1. zip 파일명이 task_id와 정확히 같은 "단독 task zip" (예: 001.zip -> "001")
     을 최우선으로 사용한다.
  2. 단독 zip이 없는 task_id(번들에만 존재)는 번들 zip에서 그대로 가져온다.
  3. 같은 task_id가 여러 zip에 존재하면(=충돌) 어떤 소스를 선택했고 어떤
     소스를 버렸는지, 각각 이미지/라벨 매칭 장수가 몇 장이었는지를
     data/raw/_ingest_conflicts.log 에 기록하고 콘솔에도 경고를 출력한다.
     -> 확실하지 않으면 반드시 이 로그를 열어서 직접 확인하세요.

data/raw/img/, data/raw/label/ 은 사람이 직접 채우는 폴더가 아니라 이
스크립트가 채우는 산출물입니다 — data/raw/data/ 에 zip만 올려두고 이 스크립트를
실행하세요. 재실행해도 안전합니다(같은 결정이면 파일을 그대로 덮어쓸 뿐,
중복 생성되지 않습니다).

사용법:
    python scripts/ingest_raw_zips.py
"""
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from utils.paths import RAW_DIR, RAW_IMAGES_DIR, RAW_LABELS_DIR

ZIP_DIR = RAW_DIR / "data"
CONFLICT_LOG = RAW_DIR / "_ingest_conflicts.log"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_KPT_SHAPE = [18, 3]


@dataclass
class TaskSource:
    zip_path: Path
    task: str
    img_entries: dict = field(default_factory=dict)   # stem -> zip-internal path
    label_entries: dict = field(default_factory=dict)  # stem -> zip-internal path

    @property
    def is_primary(self):
        return self.zip_path.stem == self.task

    @property
    def n_matched(self):
        return len(set(self.img_entries) & set(self.label_entries))


def list_tasks_in_zip(zip_path: Path, zf: zipfile.ZipFile):
    names = zf.namelist()
    return sorted({n.split("/")[0] for n in names if "/" in n})


def warn_if_kpt_shape_mismatch(zf, zip_path, task, label_prefix):
    yaml_name = f"{label_prefix}data.yaml"
    if yaml_name not in zf.namelist():
        return
    spec = yaml.safe_load(zf.read(yaml_name))
    kpt_shape = spec.get("kpt_shape")
    if kpt_shape != EXPECTED_KPT_SHAPE:
        print(
            f"[ingest][경고] {zip_path.name}:{task}: data.yaml의 kpt_shape={kpt_shape} 가 "
            f"configs/pig_pose.yaml 기대값 {EXPECTED_KPT_SHAPE} 과 다릅니다."
        )


def find_label_root(names, task):
    """<task>/ 바로 아래에서 이름이 "label"로 끝나는 폴더(label, autolabel 등)를 찾는다."""
    candidates = sorted({
        n.split("/")[1] for n in names
        if n.startswith(f"{task}/") and len(n.split("/")) > 2 and n.split("/")[1].endswith("label")
    })
    return f"{task}/{candidates[0]}/" if candidates else None


def scan_zip(zip_path: Path):
    """이 zip 안의 각 task 폴더에 대해 이미지/라벨 목록을 모은 TaskSource 리스트 반환."""
    sources = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for task in list_tasks_in_zip(zip_path, zf):
            img_prefix = f"{task}/img/"
            label_prefix = find_label_root(names, task)
            labels_dir_prefix = None
            if label_prefix:
                for n in names:
                    if n.startswith(label_prefix + "labels/") and n.lower().endswith(".txt"):
                        labels_dir_prefix = "/".join(n.split("/")[:4]) + "/"
                        break

            if label_prefix:
                warn_if_kpt_shape_mismatch(zf, zip_path, task, label_prefix)

            img_entries = {
                Path(n).stem: n
                for n in names
                if n.startswith(img_prefix) and Path(n).suffix.lower() in IMAGE_EXTS
            }
            label_entries = {}
            if labels_dir_prefix:
                label_entries = {
                    Path(n).stem: n
                    for n in names
                    if n.startswith(labels_dir_prefix) and n.lower().endswith(".txt")
                }
            sources.append(TaskSource(zip_path, task, img_entries, label_entries))
    return sources


def choose_winners(all_sources):
    """task_id -> 선택된 TaskSource. 충돌은 (task_id, winner, losers) 리스트로 함께 반환."""
    by_task = {}
    for src in all_sources:
        by_task.setdefault(src.task, []).append(src)

    winners, conflicts = {}, []
    for task, sources in by_task.items():
        primaries = [s for s in sources if s.is_primary]
        if primaries:
            winner = primaries[0]
        else:
            winner = max(sources, key=lambda s: s.n_matched)

        winners[task] = winner
        if len(sources) > 1:
            losers = [s for s in sources if s is not winner]
            conflicts.append((task, winner, losers))

    return winners, conflicts


def write_conflict_log(conflicts):
    if not conflicts:
        if CONFLICT_LOG.exists():
            CONFLICT_LOG.unlink()
        return

    lines = [
        "이 파일은 scripts/ingest_raw_zips.py 가 자동 생성했습니다.",
        "같은 task_id가 여러 zip에서 발견되어, 아래와 같이 하나를 선택했습니다.",
        "확실하지 않다면 직접 두 zip의 내용을 비교해보고 필요시 재선택하세요.",
        "",
    ]
    for task, winner, losers in conflicts:
        lines.append(f"[task={task}] 선택: {winner.zip_path.name} (매칭 {winner.n_matched}장)")
        for loser in losers:
            lines.append(f"           제외: {loser.zip_path.name} (매칭 {loser.n_matched}장)")
        lines.append("")

    CONFLICT_LOG.write_text("\n".join(lines), encoding="utf-8")


def ingest_task(src: TaskSource):
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    matched, unlabeled = 0, 0
    with zipfile.ZipFile(src.zip_path) as zf:
        for stem, img_name in src.img_entries.items():
            label_name = src.label_entries.get(stem)
            if label_name is None:
                unlabeled += 1
                continue
            ext = Path(img_name).suffix
            out_img = RAW_IMAGES_DIR / f"{src.task}__{stem}{ext}"
            out_lbl = RAW_LABELS_DIR / f"{src.task}__{stem}.txt"
            out_img.write_bytes(zf.read(img_name))
            out_lbl.write_bytes(zf.read(label_name))
            matched += 1
    return matched, unlabeled


def main():
    zips = sorted(ZIP_DIR.glob("*.zip"))
    if not zips:
        raise SystemExit(f"{ZIP_DIR} 에 zip 파일이 없습니다. CVAT task export zip을 여기에 올려주세요.")

    all_sources = []
    for zip_path in zips:
        all_sources.extend(scan_zip(zip_path))

    winners, conflicts = choose_winners(all_sources)
    write_conflict_log(conflicts)

    if conflicts:
        print(f"[ingest][경고] {len(conflicts)}개 task에서 zip 간 데이터 충돌 발견 -> {CONFLICT_LOG} 확인 필요")
        for task, winner, losers in conflicts:
            loser_desc = ", ".join(f"{l.zip_path.name}({l.n_matched}장)" for l in losers)
            print(f"  - {task}: {winner.zip_path.name}({winner.n_matched}장) 선택, {loser_desc} 제외")

    total_matched, total_unlabeled = 0, 0
    for task in sorted(winners):
        src = winners[task]
        m, u = ingest_task(src)
        total_matched += m
        total_unlabeled += u
        print(f"[ingest] task={task} <- {src.zip_path.name}: 매칭 {m}장, 라벨 없어서 제외 {u}장")

    print(
        f"\n[ingest] 완료: 총 {total_matched}장을 data/raw/img, data/raw/label 로 병합"
        f" (라벨 없어 제외 {total_unlabeled}장). 다음 단계: python scripts/split_dataset.py"
    )


if __name__ == "__main__":
    main()
