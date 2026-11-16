"""CVAT "for images 1.1" export zip → YOLO detect 데이터셋 (2클래스: 0 other / 1 visiting).

- 원본: counting/data/labels/source/<task>_sorted.zip  (사진 + annotations.xml, 돼지마다 skeleton + State)
- 박스: pose 라벨과 완전히 같은 규칙
    * 박스 = skeleton 점(화면 밖 제외)의 최소·최대 범위   (training/scripts/cvat_xml_to_raw_zip.py)
    * 0~1로 정규화한 뒤 이미지 경계로 clip, 면적 0이면 제외  (training/scripts/utils/label_io.py)
- 클래스: State 속성 → STATE_MAP (visiting=1, other=0). State가 없거나 Unlabeling인 돼지가 있으면 중단
  (그 돼지가 배경으로 학습되는 것을 막기 위해).
- 분할 (--split, 두 방식 모두 test task(000)는 통째로 test)
    * random (기본): pose 모델과 같은 사진으로 나뉘도록 training/scripts/split_dataset.py --test-zip 방식을 그대로 재현
        - 이미지 이름 = "<task>__<원래 이름>"  (ingest_raw_zips.py 규칙)
        - 나머지를 이름순 정렬 → random.Random(seed).shuffle → 앞에서 round(N × valid) 장이 valid
    * sorted: 시간대 분할. train task마다 시간순(영상 시작 시각 → 영상 안 위치)으로 정렬하고
      마지막 영상 클립들을 통째로 valid로 뗌. 떼는 클립 수는 valid 장수가 round(N × valid)에 가장 가깝게.
      클립 경계에서 자르므로 같은 클립의 비슷한 사진이 train과 valid에 갈리지 않음.
      ※ 돼지 포즈 모델 가중치(--init)는 random 분할로 학습됐으므로 sorted와 같이 쓰면 valid 누수 → COCO 시작만 사용
    * kfold: 시간대 K-fold. train task마다 시간순으로 정렬해 K개 연속 구간으로 나누고, --fold 번째 구간이 valid.
      구간 경계는 근처(±n/(4K)장)에 클립 경계가 있으면 거기에 맞춤. 클립 중간에서 잘리면 valid와 붙어 있는
      같은 클립 사진 --gap 장을 train에서 뺌(바로 옆 사진이 거의 같아서). test(000)는 모든 fold에서 같음.
      ※ pose 가중치(--init)와는 같이 못 씀(누수)
- 분할 기록: data/split.json (방식, valid 사진 목록). train.py가 models/<run>/<size>/에 복사하고
  evaluate.py가 지금 data/와 같은 분할인지 확인함.
- 사람 정답 마릿수(clip 전, XML 기준)를 data/gt_counts.csv 로 따로 저장 → evaluate.py가 사용.

사용법 (레포 루트에서):
    python yolo_det/scripts/prepare_data.py
    python yolo_det/scripts/prepare_data.py --train-tasks 001 002 --test-task 000 --valid 0.15 --seed 42
    python yolo_det/scripts/prepare_data.py --split sorted        # 시간대 분할
    YOLO_DET_DATA=data_k5f0 python yolo_det/scripts/prepare_data.py --split kfold --k 5 --fold 0   # 시간대 5-fold의 0번
"""
import argparse
import csv
import json
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASS_NAMES, CLS_VISITING, CONFIG_YAML, DATA_DIR, DEFAULT_SOURCE_DIR,  # noqa: E402
                    GT_COUNTS_CSV, IMAGE_EXTS, RESOLVED_YAML, SPLIT_JSON, SPLITS, STATE_MAP)


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


# 원래 이름 예: "0088_frame_0088__[CH005] 20260707222421-20260707224411_min018"
#              "0098_0098_frame_0045__frame_0043__[CH010] 20260716091255-20260716093200_f0111200_novinf"
# → (영상 시작 시각, 영상 안 위치: min 번호 또는 프레임 번호)
# ※ 이름의 시각은 실제 촬영 시각보다 9시간 느림(실제 KST = 이름 시각 + 9시간, 예: 002 "20260716012647" → 10:26:47).
#   여기서는 정렬에만 쓰고 모든 사진에 같은 9시간이 더해지므로 순서·분할 결과는 그대로.
#   실제 시각으로 본 분포는 training/scripts/time_distribution.py (training/reports/data/time_distribution.md)
_TIME_RE = re.compile(r"\] (\d{14})-\d{14}_(?:min(\d+)|f(\d+))")


def time_key(orig: str):
    m = _TIME_RE.search(orig)
    if not m:
        raise SystemExit(f"사진 이름에서 시간을 읽을 수 없음: {orig} → prepare_data._TIME_RE 확인")
    return m.group(1), int(m.group(2) or m.group(3))


def split_sorted(items, test_task: str, valid_ratio: float):
    """train task마다 시간순 정렬 → 마지막 클립들을 통째로 valid."""
    test = sorted([it for it in items if it["task"] == test_task], key=lambda it: Path(it["name"]))
    train, valid = [], []
    for task in sorted({it["task"] for it in items} - {test_task}):
        its = sorted([it for it in items if it["task"] == task], key=lambda it: time_key(it["orig"]))
        clips = []                                   # 시간순 클립 목록 [[item, ...], ...]
        for it in its:
            c = time_key(it["orig"])[0]
            if not clips or time_key(clips[-1][0]["orig"])[0] != c:
                clips.append([])
            clips[-1].append(it)
        target = round(len(its) * valid_ratio)
        # 뒤에서 k개 클립(1 ≤ k < 전체 클립 수)을 뗐을 때 장수가 target에 가장 가까운 k
        best_k = min(range(1, len(clips)), key=lambda k: (abs(sum(map(len, clips[-k:])) - target), k))
        v = [it for c in clips[-best_k:] for it in c]
        t = [it for c in clips[:-best_k] for it in c]
        print(f"[split] {task}: 클립 {len(clips)}개 중 마지막 {best_k}개 → valid {len(v)}장 (목표 {target}장), "
              f"train {len(t)}장 | valid 시작: {v[0]['orig']}")
        train += t
        valid += v
    return {"train": train, "valid": valid, "test": test}


def split_kfold(items, test_task: str, k: int, fold: int, gap: int):
    """train task마다 시간순 정렬 → K개 연속 구간 → fold번째 구간이 valid."""
    test = sorted([it for it in items if it["task"] == test_task], key=lambda it: Path(it["name"]))
    train, valid, notes = [], [], []
    for task in sorted({it["task"] for it in items} - {test_task}):
        its = sorted([it for it in items if it["task"] == task], key=lambda it: time_key(it["orig"]))
        n = len(its)
        clip = [time_key(it["orig"])[0] for it in its]
        clip_starts = [i for i in range(1, n) if clip[i] != clip[i - 1]]
        tol = max(1, n // (4 * k))
        bounds = [0]
        for j in range(1, k):
            b = round(j * n / k)
            near = [s for s in clip_starts if abs(s - b) <= tol and s > bounds[-1]]
            bounds.append(min(near, key=lambda s: (abs(s - b), s)) if near else b)
        bounds.append(n)
        lo, hi = bounds[fold], bounds[fold + 1]
        drop = set()
        if lo > 0 and clip[lo - 1] == clip[lo]:                      # 앞 경계가 클립 중간
            drop |= {i for i in range(max(0, lo - gap), lo) if clip[i] == clip[lo]}
        if hi < n and clip[hi] == clip[hi - 1]:                      # 뒤 경계가 클립 중간
            drop |= {i for i in range(hi, min(n, hi + gap)) if clip[i] == clip[hi - 1]}
        v = its[lo:hi]
        t = [it for i, it in enumerate(its) if not (lo <= i < hi) and i not in drop]
        cut_mid = [b for b in bounds[1:-1] if clip[b] == clip[b - 1]]
        print(f"[split] {task}: 구간 경계 {bounds} (클립 중간 경계 {cut_mid or '없음'}) → fold {fold}: "
              f"valid {len(v)}장 [{lo}:{hi}], train {len(t)}장, 경계라 뺀 사진 {len(drop)}장")
        notes.append({"task": task, "bounds": bounds, "mid_clip_bounds": cut_mid, "dropped": len(drop)})
        train += t
        valid += v
    return {"train": train, "valid": valid, "test": test}, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="<task>_sorted.zip 이 있는 폴더")
    ap.add_argument("--train-tasks", nargs="+", default=["001", "002"])
    ap.add_argument("--test-task", default="000")
    ap.add_argument("--valid", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42, help="random 분할에만 쓰임")
    ap.add_argument("--split", choices=["random", "sorted", "kfold"], default="random",
                    help="random: pose 실험과 같은 seed 분할 / sorted: 시간대(클립 단위) 분할 / kfold: 시간대 K-fold")
    ap.add_argument("--k", type=int, default=5, help="kfold: 구간 수")
    ap.add_argument("--fold", type=int, default=None, help="kfold: valid로 쓸 구간 번호 (0 ~ K-1)")
    ap.add_argument("--gap", type=int, default=1, help="kfold: 클립 중간 경계에서 train에서 뺄 장수")
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

    notes = None
    if a.split == "kfold" and (a.fold is None or not 0 <= a.fold < a.k):
        raise SystemExit(f"--split kfold 에는 --fold 0~{a.k - 1} 이 필요")
    if a.split == "random":
        splits = split_like_pose(items, a.test_task, a.valid, a.seed)
    elif a.split == "sorted":
        splits = split_sorted(items, a.test_task, a.valid)
    else:
        splits, notes = split_kfold(items, a.test_task, a.k, a.fold, a.gap)
    split_name = f"kfold_k{a.k}_f{a.fold}" if a.split == "kfold" else a.split

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

    SPLIT_JSON.write_text(json.dumps({
        "split": split_name, "valid_ratio": a.valid if a.split != "kfold" else None,
        "seed": a.seed if a.split == "random" else None, "kfold": notes and {"k": a.k, "fold": a.fold, "gap": a.gap,
                                                                          "tasks": notes},
        "train_tasks": a.train_tasks, "test_task": a.test_task,
        "valid_images": sorted(Path(it["name"]).stem for it in splits["valid"])},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # 학습용 yaml: configs/pig_det.yaml 의 클래스 이름 + 이 머신의 절대경로
    names = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))["names"]
    if {int(k): v for k, v in names.items()} != CLASS_NAMES:
        raise SystemExit(f"{CONFIG_YAML} 의 names {names} 가 common.CLASS_NAMES {CLASS_NAMES} 와 다름")
    RESOLVED_YAML.write_text(yaml.safe_dump({
        "path": DATA_DIR.as_posix(), "train": "train/images", "val": "valid/images", "test": "test/images",
        "names": names}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[prepare] 분할 방식 = {split_name}")
    print(f"[prepare] 완료 → {DATA_DIR}  (학습 yaml: {RESOLVED_YAML.name}, 사람 정답 마릿수: {GT_COUNTS_CSV.name})")


if __name__ == "__main__":
    main()
