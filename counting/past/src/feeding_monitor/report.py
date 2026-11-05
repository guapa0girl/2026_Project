"""Save/plot the aggregator's BinResult series -- the "시간대(X축) x 섭취 마리
수(Y축)" graph that's the research's stated final output.

If the config gives a `video_start_time`, bins are labeled with wall-clock
time (e.g. "09:05") instead of raw elapsed seconds, matching how the target
graph in the research plan is read (by time of day, not by video offset).
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .aggregator import BinResult


def save_csv(bins: list[BinResult], path: str | Path, timestamps: Optional[list[datetime]] = None) -> None:
    """`timestamps`, if given (one per bin, e.g. true wall-clock times for a
    live run -- see run_realtime.py), adds an absolute `timestamp` column.
    Without it the CSV only has elapsed-time-since-run-start, which can't be
    compared across runs recorded on different days (see compare_stages.py)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["bin_start_sec", "avg_pigs", "n_samples"]
        if timestamps is not None:
            header.insert(0, "timestamp")
        w.writerow(header)
        for i, b in enumerate(bins):
            row = [b.bin_start_sec, round(b.avg_pigs, 3), b.n_samples]
            if timestamps is not None:
                row.insert(0, timestamps[i].strftime("%Y-%m-%d %H:%M:%S"))
            w.writerow(row)


def plot_timeseries(
    bins: list[BinResult],
    path: str | Path,
    title: str = "Feeding count over time",
    video_start_time: Optional[str] = None,
    timestamps: Optional[list[datetime]] = None,
) -> None:
    """x-axis priority: explicit `timestamps` (true per-bin wall-clock, e.g.
    a live run) > `video_start_time` + bin_start_sec (only accurate when
    bin_start_sec comes from a real fixed fps, i.e. a recorded video file)
    > plain elapsed seconds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not bins:
        raise ValueError("no bins to plot")

    avg = [b.avg_pigs for b in bins]
    peak = max(avg) if avg else 0.0

    plt.figure(figsize=(10, 4))
    if timestamps is not None:
        x = timestamps
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gcf().autofmt_xdate()
        xlabel = "time of day"
    elif video_start_time:
        t0 = datetime.fromisoformat(video_start_time)
        x = [t0 + timedelta(seconds=b.bin_start_sec) for b in bins]
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.gcf().autofmt_xdate()
        xlabel = "time of day"
    else:
        x = [b.bin_start_sec for b in bins]
        xlabel = "elapsed time (s)"

    plt.plot(x, avg, marker="o", color="#2a78d6", markersize=3)
    plt.fill_between(x, avg, color="#2a78d6", alpha=0.12)
    plt.ylim(0, max(3, round(peak) + 1))
    plt.xlabel(xlabel)
    plt.ylabel("avg pigs feeding")
    plt.title(title)
    plt.grid(axis="y", color="#e1e0d9")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def summarize(bins: list[BinResult], fps: float) -> dict:
    if not bins:
        return {"peak_avg_pigs": 0.0, "pig_seconds": 0.0, "n_bins": 0}
    return {
        "peak_avg_pigs": round(max(b.avg_pigs for b in bins), 2),
        "pig_seconds": round(sum(b.avg_pigs * b.n_samples for b in bins) / fps, 1),
        "n_bins": len(bins),
    }
