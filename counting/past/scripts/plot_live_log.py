"""Turn a live feeding-count .txt log (written by run_realtime.py /
run_analysis.py --live-log, one "YYYY-MM-DD HH:MM:SS  Count : N  (n=samples)"
line per bin) into the graph the research is actually after: which hour of
the day pigs feed most. Safe to run against a log that's still being
written -- it just reads whatever lines exist so far, so you can regenerate
the graph periodically during a long monitoring session.

Two modes:
  hourly (default)  fold every day in the log together by hour-of-day (0-23),
                     n_samples-weighted average -- the daily feeding-activity
                     pattern. Hours with zero logged bins are left as a gap
                     (not drawn as 0), with a coverage panel underneath
                     showing how many bins actually fed each hour's average --
                     a silent camera/network outage must not read as
                     "pigs don't feed at night".
  timeline           plot the raw count chronologically, no folding --
                     useful for a single short session.

Usage:
    python scripts/plot_live_log.py --log outputs/realtime/pen01__..._live_log.txt
    python scripts/plot_live_log.py --log outputs/realtime/pen01__..._live_log.txt --mode timeline
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_monitor.daily_pattern import hourly_pattern
from feeding_monitor.live_log import parse_log


def parse_args():
    p = argparse.ArgumentParser(description="Plot a feeding-by-time-of-day graph from a live_log.txt")
    p.add_argument("--log", required=True, help="path to a *_live_log.txt written by run_realtime.py / run_analysis.py --live-log")
    p.add_argument("--mode", choices=["hourly", "timeline"], default="hourly")
    p.add_argument("--out", default=None, help="default: <log>_hourly.png / <log>_timeline.png next to the log")
    p.add_argument("--title", default=None)
    return p.parse_args()


def plot_hourly(rows, out_path, title):
    pattern = hourly_pattern(rows)
    n_days = len({ts.date() for ts, _, _ in rows}) if rows else 0
    missing_hours = [h for h, a in zip(pattern.hours, pattern.avg_pigs) if math.isnan(a)]

    fig, (ax_top, ax_cov) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.08},
    )

    present_hours = [h for h, a in zip(pattern.hours, pattern.avg_pigs) if not math.isnan(a)]
    present_avgs = [a for a in pattern.avg_pigs if not math.isnan(a)]
    ax_top.bar(present_hours, present_avgs, color="#2a78d6", width=0.8, label="avg pigs feeding")
    for h in missing_hours:
        ax_top.axvspan(h - 0.4, h + 0.4, color="#cccccc", alpha=0.35, hatch="//", zorder=0)
    if missing_hours:
        ax_top.bar([], [], color="#cccccc", alpha=0.35, hatch="//", label="no data")
        ax_top.legend(loc="upper right", fontsize=8)

    ax_top.set_ylabel("avg pigs feeding")
    ax_top.set_title(title or f"Feeding activity by hour of day (n={n_days} day(s) of log)")
    ax_top.grid(axis="y", color="#e1e0d9")

    ax_cov.bar(pattern.hours, pattern.n_obs, color="#8fa6bf", width=0.8)
    ax_cov.set_xlabel("hour of day")
    ax_cov.set_ylabel("bins logged\n(coverage)")
    ax_cov.set_xticks(range(0, 24, 2))
    ax_cov.grid(axis="y", color="#e1e0d9")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)

    if present_hours:
        peak_h = present_hours[max(range(len(present_avgs)), key=lambda i: present_avgs[i])]
        peak_msg = f"peak hour = {peak_h:02d}:00"
    else:
        peak_msg = "no data at all"
    gap_msg = f", missing hours: {missing_hours}" if missing_hours else ""
    print(f"[saved] {out_path}  ({n_days} distinct day(s) in the log, {peak_msg}{gap_msg})")


def plot_timeline(rows, out_path, title):
    ts_list = [r[0] for r in rows]
    counts = [r[1] for r in rows]

    plt.figure(figsize=(10, 4))
    plt.plot(ts_list, counts, color="#2a78d6", linewidth=1)
    plt.fill_between(ts_list, counts, color="#2a78d6", alpha=0.12)
    plt.xlabel("time")
    plt.ylabel("pigs feeding")
    plt.title(title or "Feeding count over time (live log)")
    plt.gcf().autofmt_xdate()
    plt.grid(axis="y", color="#e1e0d9")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"[saved] {out_path}")


def main():
    args = parse_args()
    log_path = Path(args.log)
    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"{log_path}에서 파싱된 줄이 없습니다 (아직 기록이 없거나 형식이 다릅니다).")
    print(f"[info] {len(rows)}개 줄 로드됨: {rows[0][0]} ~ {rows[-1][0]}")

    out_path = Path(args.out) if args.out else log_path.with_name(f"{log_path.stem}_{args.mode}.png")
    if args.mode == "hourly":
        plot_hourly(rows, out_path, args.title)
    else:
        plot_timeline(rows, out_path, args.title)


if __name__ == "__main__":
    main()
