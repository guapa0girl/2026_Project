"""Unit tests for the live per-bin .txt log: format, explicit-ts override
(the A1 fix -- live runs must stamp real wall-clock, not derived elapsed
time), and round-trip parsing. Run with:
    python -m unittest discover -s tests -p "test_*.py"
"""

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_monitor.aggregator import BinResult
from feeding_monitor.live_log import LiveLogWriter, parse_log


class TestLiveLog(unittest.TestCase):
    def test_write_then_parse_round_trip(self):
        start = datetime(2026, 8, 20, 9, 0, 0)
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "live_log.txt"

            with LiveLogWriter(log_path, start_wall_time=start, echo=False) as writer:
                writer.write(BinResult(bin_start_sec=0, avg_pigs=2.4, n_samples=15))
                writer.write(BinResult(bin_start_sec=1, avg_pigs=3.0, n_samples=15))
                writer.write(BinResult(bin_start_sec=2, avg_pigs=1.6, n_samples=15))

            rows = parse_log(log_path)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0], (datetime(2026, 8, 20, 9, 0, 0), 3, 15))   # ceil(2.4) = 3, not round's 2 -- no such thing as half a pig
            self.assertEqual(rows[1], (datetime(2026, 8, 20, 9, 0, 1), 3, 15))
            self.assertEqual(rows[2], (datetime(2026, 8, 20, 9, 0, 2), 2, 15))   # ceil(1.6) = 2

    def test_explicit_ts_overrides_bin_start_sec_derived_time(self):
        # A1: a live source must use the real wall clock at bin-close time,
        # not start_wall_time + bin_start_sec (which drifts when the
        # aggregator's assumed fps doesn't match the actual delivery rate).
        start = datetime(2026, 8, 20, 9, 0, 0)
        real_close_time = datetime(2026, 8, 20, 9, 4, 12)  # e.g. actual fps ran slower than assumed
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "live_log.txt"
            with LiveLogWriter(log_path, start_wall_time=start, echo=False) as writer:
                # bin_start_sec says "+1s", but the true close time is +4m12s later
                line = writer.write(BinResult(bin_start_sec=1, avg_pigs=1.0, n_samples=1), ts=real_close_time)
            self.assertIn("2026-08-20 09:04:12", line)
            self.assertNotIn("09:00:01", line)

    def test_append_mode_preserves_earlier_lines(self):
        start = datetime(2026, 8, 20, 9, 0, 0)
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "live_log.txt"

            with LiveLogWriter(log_path, start_wall_time=start, echo=False) as w1:
                w1.write(BinResult(bin_start_sec=0, avg_pigs=1.0, n_samples=1))

            with LiveLogWriter(log_path, start_wall_time=start, echo=False) as w2:
                w2.write(BinResult(bin_start_sec=1, avg_pigs=2.0, n_samples=1))

            rows = parse_log(log_path)
            self.assertEqual(len(rows), 2)

    def test_line_format_matches_spec(self):
        start = datetime(2026, 8, 20, 9, 0, 0)
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "live_log.txt"
            with LiveLogWriter(log_path, start_wall_time=start, echo=False) as writer:
                line = writer.write(BinResult(bin_start_sec=0, avg_pigs=3.0, n_samples=12))
            self.assertEqual(line, "2026-08-20 09:00:00  Count : 3  (n=12)")


if __name__ == "__main__":
    unittest.main()
