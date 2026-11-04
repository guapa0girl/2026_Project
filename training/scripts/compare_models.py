"""nano / small / medium 세 모델의 성능을 한 번에 비교하는 리포트를 만든다.

reports/metrics/<size>/summary.json (scripts/evaluate.py가 생성) 을 모두 읽어서
  - reports/comparison/latest_comparison.csv  : train/valid/test x n/s/m 전체 표
  - reports/comparison/latest_comparison.png  : test split 기준 pose 성능 막대그래프
를 만든다. bbox보다 pose 성능(mAP, PCK)을 우선적으로 비교한다.

train.py가 매 학습 후 자동으로 호출하지만, 단독으로도 실행 가능:
    python scripts/compare_models.py
"""
import csv
import json

from utils.paths import COMPARISON_DIR, METRICS_DIR, SIZE_LABELS, SIZES, SPLITS

ROWS_HEADER = [
    "size", "split",
    "pose_mAP50", "pose_mAP50-95",
    "box_mAP50", "box_mAP50-95",
    "PCK@0.05", "PCK@0.1", "PCK@0.2",
    "matched_instances", "missed_instances",
]


def load_rows():
    rows = []
    for size in SIZES:
        summary_path = METRICS_DIR / size / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        for split in SPLITS:
            if split not in summary:
                continue
            r = summary[split]
            official, pck = r["official"], r["pck"]
            rows.append({
                "size": SIZE_LABELS[size],
                "split": split,
                "pose_mAP50": official["pose"]["mAP50"],
                "pose_mAP50-95": official["pose"]["mAP50-95"],
                "box_mAP50": official["box"]["mAP50"],
                "box_mAP50-95": official["box"]["mAP50-95"],
                "PCK@0.05": pck["by_threshold"]["PCK@0.05"]["overall"],
                "PCK@0.1": pck["by_threshold"]["PCK@0.1"]["overall"],
                "PCK@0.2": pck["by_threshold"]["PCK@0.2"]["overall"],
                "matched_instances": pck["matched_instances"],
                "missed_instances": pck["missed_instances"],
            })
    return rows


def write_csv(rows):
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMPARISON_DIR / "latest_comparison.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROWS_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[compare] 표 저장: {out_path}")
    return out_path


def write_chart(rows):
    test_rows = [r for r in rows if r["split"] == "test"]
    if not test_rows:
        print("[compare] test split 결과가 없어 그래프는 생략합니다.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[compare] matplotlib이 설치되어 있지 않아 그래프는 생략합니다 (pip install matplotlib).")
        return

    metrics = ["pose_mAP50", "pose_mAP50-95", "PCK@0.1", "PCK@0.2"]
    sizes = [r["size"] for r in test_rows]

    x = np.arange(len(metrics))
    width = 0.8 / max(len(sizes), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, row in enumerate(test_rows):
        values = [row[m] or 0 for m in metrics]
        ax.bar(x + i * width, values, width, label=row["size"])

    ax.set_xticks(x + width * (len(sizes) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score")
    ax.set_title(f"Pig Pose - test split comparison ({' vs '.join(sizes)})")
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    out_path = COMPARISON_DIR / "latest_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] 그래프 저장: {out_path}")


def main():
    rows = load_rows()
    if not rows:
        print("[compare] reports/metrics/ 에 결과가 없습니다. 먼저 evaluate.py / train.py를 실행하세요.")
        return
    write_csv(rows)
    write_chart(rows)

    test_rows = sorted(
        (r for r in rows if r["split"] == "test"),
        key=lambda r: (r["pose_mAP50-95"] or 0),
        reverse=True,
    )
    if test_rows:
        best = test_rows[0]
        print(
            f"[compare] test split 기준 pose mAP50-95 최고 모델: {best['size']} "
            f"(mAP50-95={best['pose_mAP50-95']:.4f}, PCK@0.2={best['PCK@0.2']})"
        )


if __name__ == "__main__":
    main()
