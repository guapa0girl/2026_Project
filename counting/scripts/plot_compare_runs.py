"""Compare several evaluate_methods.py runs (e.g. 90 vs 226 Stage 1 labels) per method.

Usage:
    python scripts/plot_compare_runs.py --run "90 imgs (CH005)=outputs/eval_001" \
        --run "226 imgs=outputs/eval_all" --run "226 imgs, strict=outputs/eval_all_strict" \
        --out ../docs/figures/compare_runs.png
Panels: Stage 1 F1 | Stage 2 F1 | Stage 2 count error per frame. Grouped bars = runs.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = [("geo", "Geo"), ("rf", "RF"), ("rf_geo", "RF+Geo"), ("rf_geo_leg", "RF+Geo+Leg")]
COLORS = ["#b9b8b0", "#2a78d6", "#8fb8ea", "#e07b39"]

p = argparse.ArgumentParser()
p.add_argument("--run", action="append", required=True, help='"label=folder"')
p.add_argument("--out", required=True)
a = p.parse_args()


def load(folder, name):
    f = Path(folder) / name
    return {r["method"]: r for r in csv.DictReader(open(f, encoding="utf-8-sig"))} if f.exists() else None


runs = []
for spec in a.run:
    label, folder = spec.split("=", 1)
    runs.append((label, load(folder, "stage1_metrics.csv"), load(folder, "stage2_metrics.csv")))

panels = [("Stage 1: visiting F1 (higher = better)", 1, "visiting_F1"),
          ("Stage 2: visiting F1 (higher = better)", 2, "visiting_F1"),
          ("Stage 2: count error per frame, pigs (lower = better)", 2, "count_MAE")]
x = np.arange(len(METHODS))
w = 0.8 / len(runs)
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (title, st, col) in zip(axes, panels):
    for i, (label, s1, s2) in enumerate(runs):
        d = s1 if st == 1 else s2
        if d is None:
            continue
        vals = [float(d[m][col]) if m in d else np.nan for m, _ in METHODS]
        b = ax.bar(x + (i - (len(runs) - 1) / 2) * w, vals, w, label=label, color=COLORS[i % len(COLORS)])
        ax.bar_label(b, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x, [n for _, n in METHODS])
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="#e1e0d9")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if col == "visiting_F1":
        ax.set_ylim(0, 1.1)
axes[0].legend(loc="lower right", fontsize=9)
fig.suptitle("Effect of more Stage 1 labels on each method")
fig.tight_layout()
fig.savefig(a.out, dpi=130)
print("saved", a.out)
