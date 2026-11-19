"""학습 전 1회 대조: yolo_det/<data폴더>/split.json  ↔  training/data/splits/<name>.json

- valid: split.json의 valid_images ↔ json의 "valid"
- train: <data폴더>/train/images 사진 이름 ↔ json의 "train"
- test : <data폴더>/test/images 사진 이름 ↔ json의 "test"
항목은 모두 "<task>__<원래 이름 stem>". 하나라도 다르면 종료 코드 1 (run_kfold.sh가 학습 전에 멈춤).

규칙(11/17): 먼저 만든 사람의 training/data/splits/ 가 기준, 나중에 돌리는 사람이 맞춤.

사용법 (레포 루트에서):
    YOLO_DET_DATA=data_rk5f0 python yolo_det/scripts/compare_split.py --ref training/data/splits/rk5_f0.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, IMAGE_EXTS, SPLIT_JSON  # noqa: E402


def stems(d: Path):
    return sorted(p.stem for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="기준 분할 json (training/data/splits/<name>.json)")
    a = ap.parse_args()

    ref_path = Path(a.ref)
    if not ref_path.exists():
        raise SystemExit(f"[compare] 기준 파일 {ref_path} 가 없음 → 먼저 만든 사람의 분할 json을 받거나 "
                         f"training/scripts/make_study_splits.py 로 생성")
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    mine = {"valid": sorted(json.loads(SPLIT_JSON.read_text(encoding="utf-8"))["valid_images"]),
            "train": stems(DATA_DIR / "train" / "images"),
            "test": stems(DATA_DIR / "test" / "images")}

    ok = True
    for s in ("train", "valid", "test"):
        r, m = set(ref.get(s, [])), set(mine[s])
        if r == m:
            print(f"[compare] {s:5s} 일치 ({len(m)}장)")
            continue
        ok = False
        print(f"[compare] {s:5s} 다름: 기준 {len(r)}장 / 지금 {len(m)}장, "
              f"기준에만 {len(r - m)}장 (예: {sorted(r - m)[:3]}), 지금에만 {len(m - r)}장 (예: {sorted(m - r)[:3]})")
    print(f"[compare] {DATA_DIR.name}/split.json ↔ {ref_path} : {'일치 → 학습 진행' if ok else '불일치 → 학습 중단'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
