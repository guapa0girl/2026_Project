"""Join every <clip>_1s.csv in a count_day.py output folder into one timeline and
re-bin it on clock boundaries (e.g. 10:20, 10:30, ...). No YOLO re-run.

Usage:
    python scripts/plot_day.py outputs/ch010_20260716_day            # 10-min bins
    python scripts/plot_day.py outputs/ch010_20260716_day --bin-min 5
-> <folder>/day_<N>min.csv, <folder>/day_<N>min.png

A bin that only partly overlaps the recording (first/last) is kept; check
n_samples_ok (600 = full 10-min bin at 1 sample/s).
"""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

p = argparse.ArgumentParser()
p.add_argument("folder")
p.add_argument("--bin-min", type=int, default=10)
p.add_argument("--column", default="pigs_geometric")
a = p.parse_args()

folder = Path(a.folder)
files = sorted(folder.glob("*_1s.csv"))
if not files:
    raise SystemExit(f"{folder}: *_1s.csv 없음")

samples = []  # (wall_time, value)
for f in files:
    for r in csv.DictReader(open(f, encoding="utf-8")):
        v = r.get(a.column, "nan")
        if v in ("", "nan"):
            continue
        samples.append((datetime.fromisoformat(r["wall_time"]), float(v)))
samples.sort()

BIN = timedelta(minutes=a.bin_min)
day0 = samples[0][0].replace(hour=0, minute=0, second=0)
bins = {}
for t, v in samples:
    k = day0 + ((t - day0) // BIN) * BIN
    bins.setdefault(k, []).append(v)

stem = folder / f"day_{a.bin_min}min"
with open(str(stem) + ".csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["bin_start", "bin_end", "n_samples_ok", f"avg_{a.column}", f"max_{a.column}"])
    for k in sorted(bins):
        v = bins[k]
        w.writerow([k.isoformat(), (k + BIN).isoformat(), len(v), round(sum(v) / len(v), 3), int(max(v))])

# 1-min mean as a faint line under the binned bars, for context
minute = {}
for t, v in samples:
    minute.setdefault(t.replace(second=0), []).append(v)
mt = sorted(minute)

keys = sorted(bins)
full = a.bin_min * 60
fig, ax = plt.subplots(figsize=(14, 4.8))
ax.plot([t + timedelta(seconds=30) for t in mt], [sum(minute[t]) / len(minute[t]) for t in mt],
        color="#9db8d9", linewidth=1, label="1-min mean")
ax.bar([k + BIN / 2 for k in keys], [sum(bins[k]) / len(bins[k]) for k in keys], width=BIN * 0.9,
       color=["#2a78d6" if len(bins[k]) >= 0.9 * full else "#a9c6ec" for k in keys], alpha=0.85)
handles = ax.get_legend_handles_labels()[0] + [
    Patch(color="#2a78d6", alpha=0.85, label=f"{a.bin_min}-min mean"),
    Patch(color="#a9c6ec", alpha=0.85, label=f"{a.bin_min}-min mean (partial bin, < 90% covered)")]
ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, a.bin_min)))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.grid(which="major", axis="y", color="#e1e0d9")
ax.set_axisbelow(True)
ax.set_ylim(bottom=0)
ax.set_xlabel("time of day (KST)")
ax.set_ylabel("pigs feeding")
ax.set_title(f"Feeding count {samples[0][0]:%Y-%m-%d %H:%M}~{samples[-1][0]:%H:%M} KST "
             f"({len(files)} clips, {a.column})")
ax.legend(handles=handles, loc="upper left")
fig.tight_layout()
fig.savefig(str(stem) + ".png", dpi=130)
print("saved", str(stem) + ".csv", str(stem) + ".png", f"{len(keys)} bins, {len(samples)} samples")
