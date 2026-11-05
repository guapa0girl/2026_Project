"""Fold a feeding-count time series onto a 24-hour clock: the "which hour of
the day do pigs feed most" computation shared by scripts/plot_live_log.py
(one pen) and scripts/compare_stages.py (several pens/growth stages
overlaid, where only the *shape* of each pen's daily curve -- not the raw
head-count -- is actually comparable across pens of different size/capacity).

Rows are (timestamp, count, n_samples) triples, one per aggregation bin, as
produced by live_log.parse_log(). n_samples (how many raw per-frame counts
were averaged into that bin -- see FeedingAggregator) is used as a weight:
a bin built from more samples is a more reliable reading of that instant,
and a bin/hour with zero samples is genuinely missing data, not a measured
zero (e.g. a camera/network outage overnight must not silently read as
"pigs don't feed at night").
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import NamedTuple


class HourlyPattern(NamedTuple):
    hours: list[int]        # 0..23
    avg_pigs: list[float]   # n_samples-weighted mean per hour; NaN where n_obs == 0
    n_obs: list[int]        # how many logged bins fell in that hour (coverage)


def hourly_pattern(rows: list[tuple[datetime, int, int]]) -> HourlyPattern:
    weighted_sum: dict[int, float] = defaultdict(float)
    weight_total: dict[int, float] = defaultdict(float)
    obs_count: dict[int, int] = defaultdict(int)

    for ts, count, n_samples in rows:
        h = ts.hour
        w = max(n_samples, 1)
        weighted_sum[h] += count * w
        weight_total[h] += w
        obs_count[h] += 1

    hours = list(range(24))
    n_obs = [obs_count[h] for h in hours]
    avg_pigs = [weighted_sum[h] / weight_total[h] if obs_count[h] > 0 else math.nan for h in hours]
    return HourlyPattern(hours=hours, avg_pigs=avg_pigs, n_obs=n_obs)


class DailyShape(NamedTuple):
    hours: list[int]
    shape: list[float]   # proportion of that day's total feeding activity per hour; NaN if no days had data
    n_days: int


def daily_shape(rows: list[tuple[datetime, int, int]]) -> DailyShape:
    """Normalize each calendar day's 24-hour profile to sum to 1 before
    averaging across days -- removes pen size / feeder capacity / appetite
    scale so different growth stages become comparable on timing structure
    alone (peak hour, night-feeding share, peak width), independent of how
    many pigs or feeding spaces that pen has.
    """
    by_day: dict = defaultdict(lambda: defaultdict(list))
    for ts, count, n_samples in rows:
        by_day[ts.date()][ts.hour].append((count, max(n_samples, 1)))

    day_shapes = []
    for day, hour_buckets in by_day.items():
        hour_avgs = []
        for h in range(24):
            entries = hour_buckets.get(h)
            if not entries:
                hour_avgs.append(0.0)  # no reading this hour on this day -> treated as 0 for that day's shape
                continue
            wsum = sum(c * w for c, w in entries)
            wtot = sum(w for _, w in entries)
            hour_avgs.append(wsum / wtot)
        total = sum(hour_avgs)
        if total <= 0:
            continue  # a day with literally no feeding detected can't be normalized (0/0) -- skip it
        day_shapes.append([v / total for v in hour_avgs])

    hours = list(range(24))
    if not day_shapes:
        return DailyShape(hours=hours, shape=[math.nan] * 24, n_days=0)

    n_days = len(day_shapes)
    shape = [sum(day[h] for day in day_shapes) / n_days for h in range(24)]
    return DailyShape(hours=hours, shape=shape, n_days=n_days)
