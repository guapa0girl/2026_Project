"""Real-time feeding-count text log: mirrors each closed aggregation bin to
the terminal AND a continuously-appended .txt file, in the same
`YYYY-MM-DD HH:MM:SS  Count : N  (n=samples)` line. With
`aggregation.bin_size: 1s` in the pen config, that's one line per second
while a live source is monitored.

Timestamp source matters: for a live stream, FeedingAggregator's
`bin_start_sec` is `frame_count / assumed_fps`, and the assumed fps (a CLI
default or guess) will not match the real, variable delivery/processing
rate of an RTSP stream -- over a day, even a small fps mismatch drifts the
implied clock by hours, silently moving the peak-hour result of this whole
study. So for a genuinely live source, always pass an explicit `ts` (the
real `datetime.now()` captured the instant the bin closed) to `write()`;
the `start_wall_time + bin_start_sec` fallback is only correct when
bin_start_sec is derived from a real, fixed fps (a recorded video file's
own container metadata), not a live source's guessed one.

This is the artifact scripts/plot_live_log.py later reads to find which
hour of the day pigs feed most -- the log keeps growing across a live
session (or across many sessions, if appended to) independent of any single
run's in-memory bin list.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .aggregator import BinResult

LOG_LINE_FMT = "{ts:%Y-%m-%d %H:%M:%S}  Count : {count}  (n={n_samples})"
LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+Count\s*:\s*(\d+)\s*(?:\(n=(\d+)\))?\s*$"
)


class LiveLogWriter:
    def __init__(self, path: str | Path, start_wall_time: Optional[datetime] = None, echo: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")
        self.start_wall_time = start_wall_time or datetime.now()
        self.echo = echo

    def write(self, bin_result: BinResult, ts: Optional[datetime] = None) -> str:
        """Append one log line. Pass `ts` explicitly (the real wall clock at
        the moment this bin closed) for a live source -- see module
        docstring. Without it, the timestamp is derived from
        `start_wall_time + bin_result.bin_start_sec`, which is only accurate
        when the aggregator's fps is a real, known-fixed value (a recorded
        video file), not guessed (a live stream).
        """
        if ts is None:
            ts = self.start_wall_time + timedelta(seconds=bin_result.bin_start_sec)
        # ceil, not round: avg_pigs is a fractional average over the bin (or
        # smoothing window), but "0.5 pigs" isn't a real state -- any
        # nonzero activity in the window should surface as >=1, not get
        # rounded down to 0 (2026-08-25, user preference: no such thing as
        # half a pig).
        line = LOG_LINE_FMT.format(ts=ts, count=math.ceil(bin_result.avg_pigs), n_samples=bin_result.n_samples)
        self._file.write(line + "\n")
        self._file.flush()  # so `tail -f` / a concurrent reader sees it immediately
        if self.echo:
            print(line)
        return line

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def parse_log(path: str | Path) -> list[tuple[datetime, int, int]]:
    """Returns (timestamp, count, n_samples) triples. n_samples is 0 for
    lines written before the (n=...) suffix existed."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LOG_LINE_RE.match(line.rstrip("\n"))
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                n_samples = int(m.group(3)) if m.group(3) is not None else 0
                rows.append((ts, int(m.group(2)), n_samples))
    return rows
