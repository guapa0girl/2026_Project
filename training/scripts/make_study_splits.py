"""YOLO 직접 판정(yolo_det)과 같은 분할 목록을 만들어 포즈 모델·RF 학습에서 같이 쓰게 함.

분할 (test는 모두 000 86장):
    random : 001+002를 seed 42로 섞어 15% valid (162 / 28) — 지금 포즈 모델(ep200)과 같은 분할
    sorted : 001·002 각각 마지막 클립을 통째로 valid (159 / 31)
    k5_f0 ~ k5_f4 : 시간대 5-fold. 001·002를 시간순 5구간, fold마다 한 구간이 valid (148~155 / 34~42)
    rk5_f0 ~ rk5_f4 : 랜덤 5-fold. task마다 seed 42로 섞고, fold마다 train·valid 장수를 k5_f<i>와 task별로 똑같이 맞춤

분할 규칙은 yolo_det/scripts/prepare_data.py의 함수를 그대로 불러 씀 → YOLO와 사진 목록이 완전히 같음.
결과: data/splits/<name>.json  {"train": [...], "valid": [...], "test": [...]}  (이름 = "<task>__<원래 이름 stem>")

사용법 (training/ 에서):
    python scripts/make_study_splits.py
    python scripts/make_study_splits.py --export-images ../counting/data/raw/train_img   # + 001·002 사진(원래 이름) → valid 평가용
    python scripts/split_dataset.py --split-file data/splits/sorted.json     # 포즈 학습용 data/processed 채우기
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # training/
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "yolo_det" / "scripts"))
from prepare_data import read_task, split_kfold, split_like_pose, split_random_kfold, split_sorted  # noqa: E402

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
    ap.add_argument("--export-images", default=None,
                    help="001+002 사진을 원래 이름으로 이 폴더에 꺼냄 (counting 평가에서 valid 사진 추론용)")
    a = ap.parse_args()

    items = []
    for task in [*a.train_tasks, a.test_task]:
        items += read_task(Path(a.source_dir) / f"{task}_sorted.zip", task)
    if a.export_images:
        exp = Path(a.export_images)
        exp.mkdir(parents=True, exist_ok=True)
        n = 0
        for it in items:
            if it["task"] != a.test_task:
                (exp / it["orig"]).write_bytes(it["data"])
                n += 1
        print(f"[splits] 사진 {n}장 → {exp}")
    for it in items:
        it.pop("data")                                   # 사진 바이트는 필요 없음

    splits = {"random": split_like_pose(items, a.test_task, a.valid, a.seed),
              "sorted": split_sorted(items, a.test_task, a.valid)}
    for f in range(a.k):
        splits[f"k{a.k}_f{f}"], _ = split_kfold(items, a.test_task, a.k, f, a.gap)
    for f in range(a.k):
        splits[f"rk{a.k}_f{f}"], _ = split_random_kfold(items, a.test_task, a.k, f, a.gap, a.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, sp in splits.items():
        d = {s: sorted(Path(it["name"]).stem for it in its) for s, its in sp.items()}
        (OUT_DIR / f"{name}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[splits] {name:7s} train {len(d['train'])} / valid {len(d['valid'])} / test {len(d['test'])}")
        # 검증: test task가 train/valid에 없음, train·valid 겹치지 않음
        if any(s.startswith(f"{a.test_task}__") for s in d["train"] + d["valid"]) or set(d["train"]) & set(d["valid"]):
            raise SystemExit(f"[splits] {name}: test 사진이 train/valid에 있거나 train·valid가 겹침")
    # 검증: 랜덤 5-fold 장수 = 시간대 5-fold 장수 (task별), 랜덤 fold valid가 겹치지 않고 전체를 덮음
    seen = []
    for f in range(a.k):
        r, t = splits[f"rk{a.k}_f{f}"], splits[f"k{a.k}_f{f}"]
        for s in ("train", "valid"):
            if sorted(it["task"] for it in r[s]) != sorted(it["task"] for it in t[s]):
                raise SystemExit(f"[splits] rk{a.k}_f{f} {s} 장수가 k{a.k}_f{f}와 다름")
        seen += [it["name"] for it in r["valid"]]
    n_trainval = sum(it["task"] != a.test_task for it in items)
    if len(seen) != len(set(seen)) or len(seen) != n_trainval:
        raise SystemExit("[splits] 랜덤 5-fold valid가 겹치거나 전체를 덮지 않음")
    print(f"[splits] 검증 통과: test 미포함, rk{a.k} 장수 = k{a.k} 장수(task별), rk{a.k} valid가 {n_trainval}장을 한 번씩 덮음")
    print(f"[splits] 저장 → {OUT_DIR}")


if __name__ == "__main__":
    main()
