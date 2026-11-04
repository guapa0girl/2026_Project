"""Turn per-frame "which track IDs are feeding right now" into the research's
actual deliverable: avg pigs feeding per time bin, at whatever resolution the
config asks for (5s for a 1-minute test clip, 1min/1h for a full barn day).

Single incremental design so the exact same class backs both:
  - offline (scripts/run_analysis.py): feed every frame, then flush the tail
  - live (scripts/run_realtime.py): feed frames as they arrive, get a
    BinResult back the instant each bin closes, for a live dashboard/printout

Also implements the "지속시간 조건(bout)" item from the project's roadmap:
a track only counts as feeding once it's been classified as feeding for
`min_bout_frames` consecutive frames, which filters out a nose that grazes
the ring for one flickering frame while walking past.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .config import AggregationConfig


@dataclass
class BinResult:
    bin_start_sec: float
    avg_pigs: float
    n_samples: int


class FeedingAggregator:
    def __init__(self, fps: float, aggregation_cfg: AggregationConfig, min_bout_sec: float = 0.0):
        self.fps = fps
        self.bin_sec = aggregation_cfg.bin_size_sec
        smooth_frames = max(1, round(aggregation_cfg.smooth_sec * fps)) if aggregation_cfg.smooth_sec > 0 else 1
        self.min_bout_frames = max(0, round(min_bout_sec * fps))

        self._smooth_buf: deque[float] = deque(maxlen=smooth_frames)
        self._track_streak: dict[int, int] = {}
        self._bin_idx: Optional[int] = None
        self._bin_values: list[float] = []

    def _bouted_count(self, feeding_track_ids: set[int]) -> int:
        for tid in list(self._track_streak):
            if tid not in feeding_track_ids:
                del self._track_streak[tid]
        if self.min_bout_frames <= 0:
            for tid in feeding_track_ids:
                self._track_streak[tid] = self._track_streak.get(tid, 0) + 1
            return len(feeding_track_ids)

        count = 0
        for tid in feeding_track_ids:
            streak = self._track_streak.get(tid, 0) + 1
            self._track_streak[tid] = streak
            if streak >= self.min_bout_frames:
                count += 1
        return count

    def ingest(self, frame_idx: int, feeding_track_ids: set[int]) -> Optional[BinResult]:
        """Feed one frame's feeding track IDs. Returns a BinResult iff this
        frame closed out the previous time bin."""
        raw_count = self._bouted_count(feeding_track_ids)
        self._smooth_buf.append(raw_count)
        smoothed = sum(self._smooth_buf) / len(self._smooth_buf)

        bin_idx = int((frame_idx / self.fps) // self.bin_sec)
        if self._bin_idx is None:
            self._bin_idx = bin_idx

        if bin_idx != self._bin_idx:
            result = self._close_bin()
            self._bin_idx = bin_idx
            self._bin_values = [smoothed]
            return result

        self._bin_values.append(smoothed)
        return None

    def flush(self) -> Optional[BinResult]:
        """Emit whatever partial bin remains (call once at end of stream)."""
        return self._close_bin()

    def _close_bin(self) -> Optional[BinResult]:
        if not self._bin_values or self._bin_idx is None:
            return None
        return BinResult(
            bin_start_sec=self._bin_idx * self.bin_sec,
            avg_pigs=sum(self._bin_values) / len(self._bin_values),
            n_samples=len(self._bin_values),
        )
