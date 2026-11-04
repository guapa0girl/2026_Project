"""Overlay several live_log.txt logs (different growth stages and/or pens)
on one hour-of-day chart -- the multi-stage comparison that's the long-term
goal in the research plan (이유자돈/육성돈/비육돈 섭취 패턴 비교).

Reads *_live_log.txt (from run_realtime.py / run_analysis.py --live-log),
NOT feeding_binned.csv: a binned CSV's bin_start_sec is elapsed-time-since-
that-run-started, which has no relationship between two pens recorded on
different days at different times -- only a log's real timestamps can be
folded onto a shared 0-23h axis (see src/feeding_monitor/daily_pattern.py).

Three metrics (--metric):
  shape (default)  each pen's daily profile normalized to sum to 1 before
                    averaging across days -- cancels out pig count / feeder
                    size / appetite scale, leaving only the comparable
                    thing: *when* each stage feeds relative to its own day.
                    This is the metric that actually answers "do stages
                    feed at different times," not "which pen has more pigs."
  per-pig           avg_pigs / --capacity (a rough concurrent-feeding rate,
                    %) -- needs --capacity LABEL=N for every --run.
  raw               unnormalized avg pigs feeding per hour. Only meaningful
                    if every pen has the same headcount and feeder -- kept
                    for within-pen or same-pen-different-day comparisons,
                    NOT for cross-stage comparison.

Usage:
    python scripts/compare_stages.py \
        --run "이유자돈=outputs/realtime/pen01_live_log.txt" \
        --run "육성돈=outputs/realtime/pen02_live_log.txt" \
        --run "비육돈=outputs/realtime/pen03_live_log.txt"

    python scripts/compare_stages.py --metric per-pig \
        --run "이유자돈=outputs/realtime/pen01_live_log.txt" --capacity "이유자돈=10" \
        --run "육성돈=outputs/realtime/pen02_live_log.txt" --capacity "육성돈=8"
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_monitor.daily_pattern import daily_shape, hourly_pattern
from feeding_monitor.live_log import parse_log

COLORS = ["#2a78d6", "#2fae5b", "#d68a2a", "#a12ad6", "#d62a4d"]


def parse_args():
    p = argparse.ArgumentParser(description="Overlay multiple live_log.txt runs on a shared hour-of-day axis")
    p.add_argument("--run", action="append", required=True, metavar="LABEL=path/to/live_log.txt",
                    help="repeatable; label is shown in the legend (e.g. growth stage or pen id)")
    p.add_argument("--capacity", action="append", default=[], metavar="LABEL=N",
                    help="feeder capacity or expected pig count per label, required for --metric per-pig")
    p.add_argument("--metric", choices=["shape", "per-pig", "raw"], default="shape")
    p.add_argument("--out", default=str(ROOT / "outputs" / "stage_comparison.png"))
    p.add_argument("--title", default=None)
    return p.parse_args()


def parse_label_value_pairs(specs, cast=str):
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"'{spec}'는 LABEL=value 형식이어야 합니다.")
        label, value = spec.split("=", 1)
        out[label] = cast(value)
    return out


def main():
    args = parse_args()
    runs = parse_label_value_pairs(args.run)
    capacities = parse_label_value_pairs(args.capacity, cast=int)

    if args.metric == "per-pig":
        missing = [label for label in runs if label not in capacities]
        if missing:
            raise SystemExit(f"--metric per-pig 이려면 모든 --run에 대응하는 --capacity가 필요합니다. 누락: {missing}")

    plt.figure(figsize=(10, 4.5))
    n_days_by_label = {}
    for i, (label, path) in enumerate(runs.items()):
        rows = parse_log(path)
        if not rows:
            print(f"[warn] {label} ({path}): 파싱된 줄이 없어 건너뜁니다.")
            continue
        color = COLORS[i % len(COLORS)]

        if args.metric == "shape":
            result = daily_shape(rows)
            ys = result.shape
            n_days_by_label[label] = result.n_days
        else:
            pattern = hourly_pattern(rows)
            ys = pattern.avg_pigs
            if args.metric == "per-pig":
                ys = [y / capacities[label] * 100 if y == y else y for y in ys]  # y==y filters NaN
            n_days_by_label[label] = len({ts.date() for ts, _, _ in rows})

        xs = list(range(24))
        # matplotlib silently skips NaN points in a line plot, leaving a visible gap
        # for hours with no data instead of a misleading interpolated/zero value.
        plt.plot(xs, ys, marker="o", markersize=3, label=label, color=color)

    ylabel = {
        "shape": "share of daily total feeding activity",
        "per-pig": "concurrent feeding rate (% of capacity)",
        "raw": "avg pigs feeding",
    }[args.metric]
    plt.xlabel("hour of day")
    plt.ylabel(ylabel)
    plt.xticks(range(0, 24, 2))
    title = args.title or f"Feeding pattern by hour of day ({args.metric})"
    coverage_note = ", ".join(f"{label}: {n}d" for label, n in n_days_by_label.items())
    plt.title(f"{title}\n({coverage_note})", fontsize=11)
    plt.legend()
    plt.grid(axis="y", color="#e1e0d9")
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    print(f"[saved] {out_path}")
    if args.metric == "raw":
        print("[note] --metric raw is not scale-comparable across pens with different pig counts / "
              "feeder capacity -- use the default --metric shape for actual stage comparison.")


if __name__ == "__main__":
    main()
