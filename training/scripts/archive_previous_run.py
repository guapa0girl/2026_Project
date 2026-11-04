"""새로운 데이터로 재학습하기 전에, 이전에 학습된 모델/지표 결과를
past_train/ 아래로 옮겨서 보관한다.

이 프로젝트는 파인튜닝을 하지 않고 매번 사전학습 가중치(coco)에서 새로
전체 재학습을 하기 때문에, models/results/ 에는 항상 "가장 최근에 학습된
모델"만 남아있어야 한다. 이전 결과가 남아있으면 이 스크립트가 자동으로
past_train/run_<timestamp>/ 로 이동시킨다.

train.py 가 각 사이즈(n/s/m) 학습 직전에 자동으로 호출하므로, 보통 사용자가
직접 실행할 필요는 없다. 수동으로 정리하고 싶을 때만 CLI로 실행하면 된다.
"""
import argparse
import shutil
from datetime import datetime
from typing import Optional

from utils.paths import METRICS_DIR, PAST_TRAIN_DIR, RESULTS_DIR, SIZES


def archive_size(size: str, run_id: str) -> bool:
    """size(n/s/m)에 해당하는 기존 results/metrics 를 past_train/<run_id>/ 로 이동.
    옮길 것이 있었으면 True, 없었으면 False."""
    moved = False
    dest_root = PAST_TRAIN_DIR / run_id / size

    results_src = RESULTS_DIR / size
    if results_src.exists() and any(results_src.iterdir()):
        dest = dest_root / "model"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(results_src), str(dest))
        moved = True

    metrics_src = METRICS_DIR / size
    if metrics_src.exists() and any(metrics_src.iterdir()):
        dest = dest_root / "metrics"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(metrics_src), str(dest))
        moved = True

    return moved


def archive_all(sizes=SIZES, run_id: Optional[str] = None) -> str:
    run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    any_moved = False
    for size in sizes:
        if archive_size(size, run_id):
            any_moved = True
    if any_moved:
        print(f"[archive] 이전 학습 결과를 past_train/{run_id}/ 로 이동했습니다.")
    else:
        print("[archive] 이동할 이전 결과가 없습니다 (첫 학습).")
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이전 학습 결과를 past_train/ 으로 아카이브")
    parser.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    args = parser.parse_args()
    archive_all(args.sizes)
