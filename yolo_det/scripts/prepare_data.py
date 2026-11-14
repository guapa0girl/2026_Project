"""CVAT "for images 1.1" export zip → YOLO detect 데이터셋 (2클래스: 0 other / 1 visiting).

- 원본: counting/data/labels/source/<task>_sorted.zip  (사진 + annotations.xml, 돼지마다 skeleton + State)
- 박스: pose 라벨과 완전히 같은 규칙
    * 박스 = skeleton 점(화면 밖 제외)의 최소·최대 범위   (training/scripts/cvat_xml_to_raw_zip.py)
    * 0~1로 정규화한 뒤 이미지 경계로 clip, 면적 0이면 제외  (training/scripts/utils/label_io.py)
- 클래스: State 속성 → STATE_MAP (visiting=1, other=0). State가 없거나 Unlabeling인 돼지가 있으면 중단
  (그 돼지가 배경으로 학습되는 것을 막기 위해).
- 분할: pose 모델과 같은 사진으로 나뉘도록 training/scripts/split_dataset.py --test-zip 방식을 그대로 재현
    * 이미지 이름 = "<task>__<원래 이름>"  (ingest_raw_zips.py 규칙)
    * test task(000)는 통째로 test
    * 나머지를 이름순 정렬 → random.Random(seed).shuffle → 앞에서 round(N × valid) 장이 valid
- 사람 정답 마릿수(clip 전, XML 기준)를 data/gt_counts.csv 로 따로 저장 → evaluate.py가 사용.

사용법 (레포 루트에서):
    python yolo_det/scripts/prepare_data.py
    python yolo_det/scripts/prepare_data.py --train-tasks 001 002 --test-task 000 --valid 0.15 --seed 42
"""
import argparse
import csv
import random
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASS_NAMES, CLS_VISITING, CONFIG_YAML, DATA_DIR, DEFAULT_SOURCE_DIR,  # noqa: E402
                    GT_COUNTS_CSV, IMAGE_EXTS, RESOLVED_YAML, SPLITS, STATE_MAP)


def _clip01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def read_task(zip_path: Path, task: str):
    """zip 하나 → [{name, data(bytes), boxes[(cls, cx, cy, w, h)], n_pigs, n_visiting, n_dropped}]"""
    zf = zipfile.ZipFile(zip_path)
    names = [n for n in zf.namelist() if not n.startswith("__MACOSX") and not n.endswith("/")]
    xml_names = [n for n in names if n.endswith("annotations.xml")]
    if len(xml_names) != 1:
        raise SystemExit(f"{zip_path}: annotations.xml 이 {len(xml_names)}개 — 1개여야 함")
    by_base = {Path(n).name: n for n in names if Path(n).suffix.lower() in IMAGE_EXTS}
    root = ET.fromstring(zf.read(xml_names[0]))

    items, problems = [], []
    for im in root.findall("image"):
        base = Path(im.get("name")).name
        if base not in by_base:
            raise SystemExit(f"{zip_path}: XML의 이미지 {base} 가 zip에 없음")
        W, H = float(im.get("width")), float(im.get("height"))
        boxes, n_pigs, n_vis, n_drop = [], 0, 0, 0
        for sk in im.findall("skeleton"):
            raw = None
            for a in sk.findall("attribute"):
                if a.get("name", "").lower() == "state":
                    raw = (a.text or "").strip()
            key = (raw or "").lower()
            if key not in STATE_MAP or STATE_MAP[key] is None:
                problems.append(f"{base}: State={raw!r}")
                continue
            cls = STATE_MAP[key]

            xs, ys = [], []
            for p in sk.findall("points"):
                if p.get("outside") == "1":
                    continue
                x, y = map(float, p.get("points").split(";")[0].split(","))
                xs.append(x)
                ys.append(y)
            if not xs:          # pose 변환도 점이 하나도 없는 skeleton은 건너뜀
                continue
            n_pigs += 1
            n_vis += cls == CLS_VISITING

            # 정규화 → clip (pose 라벨과 같은 순서·같은 계산)
            cx = (min(xs) + max(xs)) / 2 / W
            cy = (min(ys) + max(ys)) / 2 / H
            w, h = (max(xs) - min(xs)) / W, (max(ys) - min(ys)) / H
            # pose 변환은 소수 6자리 문자열로 한 번 저장한 뒤 clip → 같은 반올림을 거쳐 박스를 완전히 일치시킴
            cx, cy, w, h = (round(v, 6) for v in (cx, cy, w, h))
            x1, y1 = _clip01(cx - w / 2), _clip01(cy - h / 2)
            x2, y2 = _clip01(cx + w / 2), _clip01(cy + h / 2)
            if x2 <= x1 or y2 <= y1:
                n_drop += 1
                continue
            boxes.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))

        items.append({"name": f"{task}__{base}", "orig": base, "task": task, "data": zf.read(by_base[base]),
                      "boxes": boxes, "n_pigs": n_pigs, "n_visiting": n_vis, "n_dropped": n_drop})
    if problems:
        raise SystemExit(f"{zip_path}: State 라벨이 없거나 알 수 없는 돼지 {len(problems)}마리 "
                         f"(예: {problems[:5]}) → 라벨을 고치거나 common.STATE_MAP에 추가하세요")
    return items


def split_like_pose(items, test_task: str, valid_ratio: float, seed: int):
    """training/scripts/split_dataset.py (--test-zip) 와 같은 결과가 나오도록 재현."""
    pairs = sorted(items, key=lambda it: Path(it["name"]))   # sorted(raw/img.iterdir())와 같은 정렬(Windows는 대소문자 무시)
    test = [it for it in pairs if it["task"] == test_task]
    rest = [it for it in pairs if it["task"] != test_task]
    rng = random.Random(seed)
    rng.shuffle(rest)
    n_valid = round(len(rest) * valid_ratio)
    return {"train": rest[n_valid:], "valid": rest[:n_valid], "test": test}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="<task>_sorted.zip 이 있는 폴더")
    ap.add_argument("--train-tasks", nargs="+", default=["001", "002"])
    ap.add_argument("--test-task", default="000")
    ap.add_argument("--valid", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src = Path(a.source_dir)
    items = []
    for task in [*a.train_tasks, a.test_task]:
        zp = src / f"{task}_sorted.zip"
        if not zp.exists():
            raise SystemExit(f"{zp} 가 없음 — CVAT export zip을 {src} 에 두세요")
        task_items = read_task(zp, task)
        c = Counter(cls for it in task_items for cls, *_ in it["boxes"])
        print(f"[prepare] {task}: 사진 {len(task_items)}장, 돼지 {sum(it['n_pigs'] for it in task_items)}마리 "
              f"(visiting {c[1]} / other {c[0]}, 경계 밖이라 제외 {sum(it['n_dropped'] for it in task_items)})")
        items += task_items

    splits = split_like_pose(items, a.test_task, a.valid, a.seed)

    # data/ 를 새로 채움 (이전 결과 삭제)
    for s in SPLITS:
        for sub in ("images", "labels"):
            d = DATA_DIR / s / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        for f in DATA_DIR.glob(f"{s}/labels.cache"):
            f.unlink()

    rows = []
    for s, its in splits.items():
        for it in its:
            stem = Path(it["name"]).stem
            (DATA_DIR / s / "images" / it["name"]).write_bytes(it["data"])
            lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in it["boxes"]]
            (DATA_DIR / s / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            rows.append({"image": stem, "orig_image": Path(it["orig"]).stem, "task": it["task"], "split": s,
                         "n_pigs": it["n_pigs"], "n_visiting": it["n_visiting"]})
        c = Counter(cls for it in its for cls, *_ in it["boxes"])
        by_task = Counter(it["task"] for it in its)
        print(f"[split] {s:5s}: 사진 {len(its):3d} ({', '.join(f'{t} {n}' for t, n in sorted(by_task.items()))}) "
              f"| visiting {c[1]} / other {c[0]}")

    with open(GT_COUNTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # 학습용 yaml: configs/pig_det.yaml 의 클래스 이름 + 이 머신의 절대경로
    names = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))["names"]
    if {int(k): v for k, v in names.items()} != CLASS_NAMES:
        raise SystemExit(f"{CONFIG_YAML} 의 names {names} 가 common.CLASS_NAMES {CLASS_NAMES} 와 다름")
    RESOLVED_YAML.write_text(yaml.safe_dump({
        "path": DATA_DIR.as_posix(), "train": "train/images", "val": "valid/images", "test": "test/images",
        "names": names}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[prepare] 완료 → {DATA_DIR}  (학습 yaml: {RESOLVED_YAML.name}, 사람 정답 마릿수: {GT_COUNTS_CSV.name})")


if __name__ == "__main__":
    main()
