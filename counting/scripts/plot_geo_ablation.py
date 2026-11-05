"""Bar chart of the Geo ablation (run_geo_ablation.ps1) Stage 2 results.

Usage:
    python scripts/plot_geo_ablation.py                       # outputs/eval/geo_ablation
    python scripts/plot_geo_ablation.py --root <folder> --out <png>
Left: visiting F1 (higher = better). Right: per-frame count error (lower = better).
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = [("dir60", "dir 60°\n(current)"), ("dir90", "dir 90°"), ("nodir", "dir off"),
            ("dir60_posture", "dir 60°\n+ lying"), ("nodir_posture", "dir off\n+ lying")]

p = argparse.ArgumentParser()
p.add_argument("--root", default="outputs/eval/geo_ablation")
p.add_argument("--out", default=None)
a = p.parse_args()
root = Path(a.root)

rows = []
for key, label in VARIANTS:
    f = root / key / "stage2_metrics.csv"
    if not f.exists():
        continue
    r = next(x for x in csv.DictReader(open(f, encoding="utf-8-sig")) if x["method"] == "geo")
    rows.append((label, float(r["visiting_F1"]), float(r["count_MAE"]), float(r["count_bias"])))

labels = [r[0] for r in rows]
blue, grey = "#2a78d6", "#b9b8b0"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))

f1 = [r[1] for r in rows]
best = max(range(len(f1)), key=f1.__getitem__)
b = a1.bar(labels, f1, color=[blue if i == best else grey for i in range(len(f1))])
a1.bar_label(b, fmt="%.3f", padding=3)
a1.set_ylim(0, 1)
a1.set_title("visiting F1 (higher = better)")

mae = [r[2] for r in rows]
best = min(range(len(mae)), key=mae.__getitem__)
b = a2.bar(labels, mae, color=[blue if i == best else grey for i in range(len(mae))])
a2.bar_label(b, labels=[f"{m:.2f}\n(bias {bi:+.2f})" for _, _, m, bi in rows], padding=3)
a2.set_ylim(0, max(mae) * 1.35)
a2.set_title("count error per frame, pigs (lower = better)")

for ax in (a1, a2):
    ax.grid(axis="y", color="#e1e0d9")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Geo rule conditions - eval 50 images (CH010), model keypoints")
fig.tight_layout()
out = Path(a.out) if a.out else root / "geo_ablation.png"
fig.savefig(out, dpi=130)
print("saved", out)
