"""Bar chart of evaluate_methods.py results: Geo / RF / RF+Geo / RF+Geo+Leg.

Usage:
    python scripts/plot_methods.py outputs/eval/final          # -> outputs/eval/final/methods.png
    python scripts/plot_methods.py outputs/eval/final --out <png> --title "<subtitle>"
Panels: per-frame visiting count error (MAE, bias in label) for Stage 1 (human keypoints,
clip-level CV) and Stage 2 (model keypoints, 000) | Stage 2 total visiting count error (%).
Missing stages are skipped.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"geo": "Geo\n(rule)", "rf": "RF", "rf_geo": "RF+Geo", "rf_geo_leg": "RF+Geo\n+Leg"}

p = argparse.ArgumentParser()
p.add_argument("folder")
p.add_argument("--out", default=None)
p.add_argument("--title", default="")
a = p.parse_args()
folder = Path(a.folder)


def load(name):
    f = folder / name
    if not f.exists():
        return None
    return {r["method"]: r for r in csv.DictReader(open(f, encoding="utf-8-sig"))}


s1, s2 = load("stage1_metrics.csv"), load("stage2_metrics.csv")
panels = []
if s1:
    panels.append(("Stage 1: count error per frame (pigs)\nhuman keypoints, clip-level CV - lower = better",
                   s1, "count_MAE"))
if s2:
    panels.append(("Stage 2: count error per frame (pigs)\nmodel keypoints, 000 - lower = better",
                   s2, "count_MAE"))
    panels.append(("Stage 2: total visiting count error (%)\nmodel vs human - closer to 0 = better",
                   s2, "total_error_pct"))

blue, grey = "#2a78d6", "#b9b8b0"
fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8))
axes = [axes] if len(panels) == 1 else axes
for ax, (title, d, col) in zip(axes, panels):
    methods = [m for m in LABELS if m in d]
    vals = [float(d[m][col]) for m in methods]
    best = min(range(len(vals)), key=lambda i: abs(vals[i]))
    bars = ax.bar([LABELS[m] for m in methods], vals,
                  color=[blue if i == best else grey for i in range(len(vals))])
    if col == "count_MAE":
        txt = [f"{v:.2f}\n(bias {float(d[m]['count_bias']):+.2f})" for m, v in zip(methods, vals)]
        ax.set_ylim(0, max(vals) * 1.35 or 1)
    else:
        txt = [f"{v:+.1f}%" for v in vals]
        lim = max(abs(v) for v in vals) * 1.35 or 1
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="#888780", lw=0.8)
    ax.bar_label(bars, labels=txt, padding=3)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="#e1e0d9")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Visiting count: Geo vs RF vs RF+Geo vs RF+Geo+Leg" + (f"  -  {a.title}" if a.title else ""))
fig.tight_layout()
out = Path(a.out) if a.out else folder / "methods.png"
fig.savefig(out, dpi=130)
print("saved", out)
