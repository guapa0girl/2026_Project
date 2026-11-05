"""Unit tests for hour-of-day folding: NaN for genuinely missing hours (A2),
n_samples weighting, and shape normalization across days (A4).
Run with: python -m unittest discover -s tests -p "test_*.py"
"""

import math
import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_monitor.daily_pattern import daily_shape, hourly_pattern


def ts(h, m=0, d=20):
    return datetime(2026, 8, d, h, m, 0)


class TestHourlyPattern(unittest.TestCase):
    def test_missing_hour_is_nan_not_zero(self):
        # only hour 9 has any data; hour 3 (e.g. an overnight outage) must
        # not silently read as "measured zero pigs feeding"
        rows = [(ts(9), 2, 1), (ts(9, 30), 4, 1)]
        pattern = hourly_pattern(rows)
        self.assertTrue(math.isnan(pattern.avg_pigs[3]))
        self.assertEqual(pattern.n_obs[3], 0)
        self.assertAlmostEqual(pattern.avg_pigs[9], 3.0)
        self.assertEqual(pattern.n_obs[9], 2)

    def test_n_samples_weighting(self):
        # a bin built from 100 raw samples should outweigh one from 1 sample
        rows = [(ts(9), 0, 100), (ts(9, 30), 10, 1)]
        pattern = hourly_pattern(rows)
        self.assertLess(pattern.avg_pigs[9], 1.0)  # heavily pulled toward the 0-count, high-weight bin

    def test_folds_across_multiple_days(self):
        rows = [(ts(9, d=19), 2, 1), (ts(9, d=20), 4, 1)]
        pattern = hourly_pattern(rows)
        self.assertAlmostEqual(pattern.avg_pigs[9], 3.0)
        self.assertEqual(pattern.n_obs[9], 2)


class TestDailyShape(unittest.TestCase):
    def test_shape_sums_to_one(self):
        rows = [(ts(9), 2, 1), (ts(15), 2, 1)]
        result = daily_shape(rows)
        total = sum(v for v in result.shape if not math.isnan(v))
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertAlmostEqual(result.shape[9], 0.5)
        self.assertAlmostEqual(result.shape[15], 0.5)

    def test_scale_invariant_across_pens(self):
        # same timing structure, 10x the headcount -- shape must match exactly
        small_pen = [(ts(9), 1, 1), (ts(15), 3, 1)]
        big_pen = [(ts(9), 10, 1), (ts(15), 30, 1)]
        s_small = daily_shape(small_pen)
        s_big = daily_shape(big_pen)
        for a, b in zip(s_small.shape, s_big.shape):
            if math.isnan(a):
                self.assertTrue(math.isnan(b))
            else:
                self.assertAlmostEqual(a, b, places=6)

    def test_day_with_zero_total_is_excluded_not_divide_by_zero(self):
        rows = [(ts(9), 0, 1), (ts(15), 0, 1)]  # a day with literally no feeding detected
        result = daily_shape(rows)
        self.assertEqual(result.n_days, 0)
        self.assertTrue(all(math.isnan(v) for v in result.shape))


if __name__ == "__main__":
    unittest.main()
