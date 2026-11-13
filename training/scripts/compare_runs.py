"""Compare training runs that differ only in a setting (e.g. epochs 100 vs 200), size by size.

Reads reports/metrics/<run>/<size>/summary.json and models/results/<run>/<size>/results.csv
(written when train.py ran with POSE_RUN=<run>).

Usage (from training/):
    python scripts/compare_runs.py --runs ep100 ep200 ep300
-> reports/comparison/runs_<a>_vs_<b>[_vs_<c>...].csv / .png
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIZES = [("n", "nano"), ("s", "small"), ("m", "medium"), ("l", "large")]
METRICS = [("pose_mAP50", "pose mAP50"), ("pose_mAP50-95", "pose mAP50-95"),
           ("PCK@0.1", "PCK@0.1"), ("PCK@0.2", "PCK@0.2")]


def flat(split_summary):
    o, p = split_summary["official"], split_summary["pck"]["by_threshold"]
    return {"pose_mAP50": o["pose"]["mAP50"], "pose_mAP50-95": o["pose"]["mAP50-95"],
            "box_mAP50-95": o["box"]["mAP50-95"],
            "PCK@0.1": p["PCK@0.1"]["overall"], "PCK@0.2": p["PCK@0.2"]["overall"],
            "missed": split_summary["pck"]["missed_instances"],
            "matched": split_summary["pck"]["matched_instances"]}


def run_info(run, size):
    s = ROOT / "reports" / "metrics" / run / size / "summary.json"
    r = ROOT / "models" / "results" / run / size / "results.csv"
    if not s.exists():
        return None
    summ = json.loads(s.read_text(encoding="utf-8"))
    info = {f"test_{k}": v for k, v in flat(summ["test"]).items()}
    info.update({f"valid_{k}": v for k, v in flat(summ["valid"]).items()})
    if r.exists():
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(open(r, encoding="utf-8"))]
        key = "metrics/mAP50-95(P)"
        best = max(rows, key=lambda x: float(x[key]))
        info.update({"epochs_run": len(rows), "best_epoch": int(float(best["epoch"])),
                     "train_minutes": round(float(rows[-1]["time"]) / 60, 1)})
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    out_dir = ROOT / "reports" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / ("runs_" + "_vs_".join(a.runs))

    rows = []
    for size, label in SIZES:
        for run in a.runs:
            info = run_info(run, size)
            if info:
                rows.append({"size": label, "run": run, **info})
    if not rows:
        raise SystemExit("summary.json 이 없습니다 (학습이 끝났는지 확인)")
    cols = list(rows[0].keys())
    with open(str(stem) + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"{'size':7s} {'run':6s} {'epochs':>6s} {'best':>5s} {'min':>6s} | test " +
          " ".join(f"{m:>13s}" for m, _ in METRICS) + " | valid pose mAP50-95")
    for r in rows:
        print(f"{r['size']:7s} {r['run']:6s} {r.get('epochs_run', ''):>6} {r.get('best_epoch', ''):>5} "
              f"{r.get('train_minutes', ''):>6} | " + " ".join(f"{r['test_' + m]:13.3f}" for m, _ in METRICS) +
              f" | {r['valid_pose_mAP50-95']:.3f}")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.4))
    labels = [l for _, l in SIZES]
    x = np.arange(len(labels))
    palette = ["#b9d3f2", "#8fb8ea", "#2a78d6", "#174a8c"]
    colors = palette[-len(a.runs):] if len(a.runs) <= len(palette) else None
    w = 0.8 / len(a.runs)
    for ax, (m, title) in zip(axes, METRICS):
        for i, run in enumerate(a.runs):
            vals = [next((r["test_" + m] for r in rows if r["size"] == l and r["run"] == run), np.nan) for l in labels]
            b = ax.bar(x + (i - (len(a.runs) - 1) / 2) * w, vals, w, label=run, color=colors[i] if colors else None)
            ax.bar_label(b, fmt="%.2f", padding=2, fontsize=6)
        ax.set_xticks(x, labels)
        ax.set_title(f"test {title}", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#e1e0d9")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"YOLOv8-pose {' vs '.join(a.runs)} — test = 000 (1-min sample, 86 imgs, not in training)")
    fig.tight_layout()
    fig.savefig(str(stem) + ".png", dpi=130)
    print("saved", str(stem) + ".csv", str(stem) + ".png")


if __name__ == "__main__":
    main()
