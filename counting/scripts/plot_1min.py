"""Re-bin a count_full_video.py <prefix>_1s.csv into 1-minute mean/max (no YOLO re-run).

Usage:
    python scripts/plot_1min.py outputs/ch010_20260716/CH010_..._1s.csv
-> <prefix>_1min.csv, <prefix>_1min.png
"""
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

src = Path(sys.argv[1])  # <prefix>_1s.csv
prefix = Path(str(src)[: -len("_1s.csv")])
BIN = 60

rows = list(csv.DictReader(open(src, encoding="utf-8")))
t0 = datetime.fromisoformat(rows[0]["wall_time"]) - timedelta(seconds=int(rows[0]["elapsed_sec"]))
bins = {}
for r in rows:
    bins.setdefault(int(r["elapsed_sec"]) // BIN, []).append(int(r["pigs_geometric"]))

out = []
for b in sorted(bins):
    v = bins[b]
    out.append((b * BIN, t0 + timedelta(seconds=b * BIN), len(v), sum(v) / len(v), max(v)))

with open(str(prefix) + "_1min.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["bin_start_sec", "wall_time", "n_samples_ok", "avg_pigs_geometric", "max_pigs_geometric"])
    for s, t, n, m, mx in out:
        w.writerow([s, t.isoformat(), n, round(m, 3), mx])

# x = bin centre so each point sits in the middle of its minute
times = [t + timedelta(seconds=BIN / 2) for _, t, *_ in out]
plt.figure(figsize=(14, 4.8))
plt.plot(times, [o[3] for o in out], marker="o", markersize=4, color="#2a78d6", label="geometric (1-min mean)")
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
ax.grid(which="major", axis="x", color="#c7c6bd", linewidth=0.9)
ax.grid(which="minor", axis="x", color="#e8e7e0", linewidth=0.6)
ax.grid(which="major", axis="y", color="#e1e0d9")
ax.set_ylim(bottom=0)
plt.xlabel("time of day (KST)")
plt.ylabel("pigs feeding (mean per 60s)")
plt.title(f"Feeding count {t0:%Y-%m-%d %H:%M}~{out[-1][1] + timedelta(seconds=BIN):%H:%M} KST -- farm_ch010 (60s bins, mean)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(str(prefix) + "_1min.png", dpi=130)
print("saved", str(prefix) + "_1min.csv", str(prefix) + "_1min.png", len(out), "bins")
