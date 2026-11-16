"""YOLO 직접 판정(yolo_det)과 같은 분할 목록을 만들어 포즈 모델·RF 학습에서 같이 쓰게 함.

분할 3가지 (test는 모두 000 86장):
    random : 001+002를 seed 42로 섞어 15% valid (162 / 28) — 지금 포즈 모델(ep200)과 같은 분할
    sorted : 001·002 각각 마지막 클립을 통째로 valid (159 / 31)
    k5_f0 ~ k5_f4 : 001·002를 시간순 5구간, fold마다 한 구간이 valid (148~155 / 34~42)

분할 규칙은 yolo_det/scripts/prepare_data.py의 함수를 그대로 불러 씀 → YOLO와 사진 목록이 완전히 같음.
결과: data/splits/<name>.json  {"train": [...], "valid": [...], "test": [...]}  (이름 = "<task>__<원래 이름 stem>")

사용법 (training/ 에서):
    python scripts/make_study_splits.py
    python scripts/split_dataset.py --split-file data/splits/sorted.json     # 포즈 학습용 data/processed 채우기
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # training/
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "yolo_det" / "scripts"))
from prepare_data import read_task, split_kfold, split_like_pose, split_sorted  # noqa: E402

OUT_DIR = ROOT / "data" / "splits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default=str(REPO / "counting" / "data" / "labels" / "source"))
    ap.add_argument("--train-tasks", nargs="+", default=["001", "002"])
    ap.add_argument("--test-task", default="000")
    ap.add_argument("--valid", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gap", type=int, default=1)
    a = ap.parse_args()

    items = []
    for task in [*a.train_tasks, a.test_task]:
        items += read_task(Path(a.source_dir) / f"{task}_sorted.zip", task)
    for it in items:
        it.pop("data")                                   # 사진 바이트는 필요 없음

    splits = {"random": split_like_pose(items, a.test_task, a.valid, a.seed),
              "sorted": split_sorted(items, a.test_task, a.valid)}
    for f in range(a.k):
        splits[f"k{a.k}_f{f}"], _ = split_kfold(items, a.test_task, a.k, f, a.gap)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, sp in splits.items():
        d = {s: sorted(Path(it["name"]).stem for it in its) for s, its in sp.items()}
        (OUT_DIR / f"{name}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[splits] {name:7s} train {len(d['train'])} / valid {len(d['valid'])} / test {len(d['test'])}")
    print(f"[splits] 저장 → {OUT_DIR}")


if __name__ == "__main__":
    main()
