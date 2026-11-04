"""n/s/m/l/x 학습 진행 상황을 막대그래프(%)로 보여준다.
백그라운드로 오래 도는 학습(특히 l/x)의 진행률을 아무 때나 빠르게
확인하고 싶을 때 사용한다.

사용법:
    python scripts/training_status.py
"""
import csv
import time
from pathlib import Path

import yaml

from utils.paths import CONFIGS_DIR, RESULTS_DIR, SIZE_LABELS, SIZES

SIZE_CONFIG_FILE = {
    "n": CONFIGS_DIR / "train_nano.yaml",
    "s": CONFIGS_DIR / "train_small.yaml",
    "m": CONFIGS_DIR / "train_medium.yaml",
    "l": CONFIGS_DIR / "train_large.yaml",
    "x": CONFIGS_DIR / "train_xlarge.yaml",
}

BAR_WIDTH = 40
STALE_SECONDS = 180  # 이보다 오래 로그 갱신이 없으면 "중단 의심"으로 표시


def target_epochs(size):
    with open(SIZE_CONFIG_FILE[size], encoding="utf-8") as f:
        return yaml.safe_load(f)["epochs"]


def current_epoch_and_time(size):
    csv_path = RESULTS_DIR / size / "results.csv"
    if not csv_path.exists():
        return 0, None
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return 0, None
    last = rows[-1]
    return int(float(last[0])), csv_path.stat().st_mtime


def has_final_weights(size):
    return (RESULTS_DIR / size / "weights" / "best.pt").exists() and \
        (RESULTS_DIR / size / "weights" / "last.pt").exists() and \
        (RESULTS_DIR / size).joinpath("val_test").exists()


def render_bar(pct):
    # 일부 Windows 콘솔(cp949)에서 유니코드 블록 문자가 깨져서 ASCII로 표현
    filled = int(BAR_WIDTH * pct / 100)
    return "#" * filled + "-" * (BAR_WIDTH - filled)


def main():
    now = time.time()
    rows = []
    for size in SIZES:
        total = target_epochs(size)
        epoch, mtime = current_epoch_and_time(size)
        pct = min(100, round(epoch / total * 100))

        if has_final_weights(size) and epoch >= total:
            status = "완료"
        elif epoch == 0 and mtime is None:
            status = "대기"
        elif mtime is not None and (now - mtime) > STALE_SECONDS:
            status = f"중단 의심 (마지막 갱신 {int((now - mtime) / 60)}분 전)"
        else:
            status = "진행 중"

        rows.append((size, epoch, total, pct, status))

    label_w = max(len(f"{s} ({SIZE_LABELS[s]})") for s in SIZES)
    for size, epoch, total, pct, status in rows:
        label = f"{size} ({SIZE_LABELS[size]})".ljust(label_w)
        print(f"{label}  [{render_bar(pct)}] {epoch:>3}/{total}  {pct:>3}%  {status}")

    overall = sum(min(e, t) for _, e, t, _, _ in rows) / sum(t for _, _, t, _, _ in rows) * 100
    print(f"\n전체  [{render_bar(overall)}] {overall:.1f}%")


if __name__ == "__main__":
    main()
