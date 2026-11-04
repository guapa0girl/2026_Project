"""Unit tests for the 3-condition feeding rule, using synthetic keypoints so
this runs without a model checkpoint or video. Run with:

    python -m pytest tests/
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_monitor.config import FeederROI, FeedingRule, load_config
from feeding_monitor.feeding_classifier import classify
from feeding_monitor.pose_tracker import KEYPOINT_NAMES, Detection

FEEDER = FeederROI(cx=500.0, cy=500.0, r=100.0, ring_inner_ratio=0.20, ring_outer_ratio=1.10)
RULE = FeedingRule(nose_index=1, neck_index=0, direction_angle_max_deg=60.0, keypoint_conf_min=0.5)


def make_detection(neck, nose, neck_conf=0.9, nose_conf=0.9):
    kpts = [(0.0, 0.0, 0.0)] * 2
    kpts[RULE.neck_index] = (*neck, neck_conf)
    kpts[RULE.nose_index] = (*nose, nose_conf)
    return Detection(frame_idx=0, track_id=1, det_conf=0.9, box=(0, 0, 10, 10), kpts=kpts)


class TestFeedingClassifier(unittest.TestCase):
    def test_nose_at_rim_facing_center_is_feeding(self):
        # neck further from center than nose, both roughly on the same radial line
        det = make_detection(neck=(560.0, 500.0), nose=(550.0, 500.0))  # nose ~50px from center, inside ring
        verdict = classify(det, FEEDER, RULE)
        self.assertTrue(verdict.is_feeding, verdict.reason)

    def test_sleeping_beside_feeder_wrong_direction_rejected(self):
        # nose is within ring distance, but head points away from the feeder center
        det = make_detection(neck=(550.0, 500.0), nose=(560.0, 500.0))  # head points outward, away from center
        verdict = classify(det, FEEDER, RULE)
        self.assertFalse(verdict.is_feeding)
        self.assertEqual(verdict.reason, "wrong_direction")

    def test_nose_over_lid_center_too_close_rejected(self):
        det = make_detection(neck=(520.0, 500.0), nose=(505.0, 500.0))  # 5px from center, inside ring_inner
        verdict = classify(det, FEEDER, RULE)
        self.assertFalse(verdict.is_feeding)
        self.assertEqual(verdict.reason, "out_of_ring")

    def test_far_from_feeder_rejected(self):
        det = make_detection(neck=(0.0, 0.0), nose=(10.0, 0.0))
        verdict = classify(det, FEEDER, RULE)
        self.assertFalse(verdict.is_feeding)
        self.assertEqual(verdict.reason, "out_of_ring")

    def test_low_confidence_rejected(self):
        det = make_detection(neck=(560.0, 500.0), nose=(550.0, 500.0), nose_conf=0.2)
        verdict = classify(det, FEEDER, RULE)
        self.assertFalse(verdict.is_feeding)
        self.assertEqual(verdict.reason, "low_confidence")


class TestKeypointIndicesMatchSkeleton(unittest.TestCase):
    """Regression guard for the nose_index=12 bug (that was actually l_ear,
    not nose, in the project's real 18-point CVAT skeleton -- see
    pose_tracker.KEYPOINT_NAMES). A pen config's nose_index/neck_index must
    resolve to the correspondingly-named keypoint, or every feeding
    classification silently uses the wrong point on the pig."""

    def test_default_feeding_rule_indices_are_correct(self):
        rule = FeedingRule()
        self.assertEqual(KEYPOINT_NAMES[rule.nose_index], "nose")
        self.assertEqual(KEYPOINT_NAMES[rule.neck_index], "neck")

    def test_example_configs_indices_are_correct(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            config = load_config(path)
            self.assertEqual(
                KEYPOINT_NAMES[config.feeding_rule.nose_index], "nose",
                f"{path.name}: nose_index={config.feeding_rule.nose_index} is not 'nose'",
            )
            self.assertEqual(
                KEYPOINT_NAMES[config.feeding_rule.neck_index], "neck",
                f"{path.name}: neck_index={config.feeding_rule.neck_index} is not 'neck'",
            )


if __name__ == "__main__":
    unittest.main()
