"""Typed config loaded from a YAML file (configs/*.yaml).

One config = one pen/camera setup: which model checkpoint to run, where the
circular feeder is in pixel space, which keypoints count as nose/neck, and
what time resolution to aggregate feeding counts into. Keeping all of this in
YAML (instead of constants edited in a script, as in the original
ObjectDetectionTracking_GPU/aggregate_feeding.py) is what lets the same
pipeline run unmodified across pens, growth stages, and video lengths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_BIN_UNIT_SECONDS = {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hr": 3600}


def parse_duration(value) -> float:
    """Parse '5s', '1min', '30min', '1h', or a bare number of seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    m = re.fullmatch(r"\s*([\d.]+)\s*([a-zA-Z]+)\s*", str(value))
    if not m:
        raise ValueError(f"Cannot parse duration: {value!r}")
    n, unit = m.groups()
    unit = unit.lower()
    if unit not in _BIN_UNIT_SECONDS:
        raise ValueError(f"Unknown duration unit '{unit}' in {value!r}")
    return float(n) * _BIN_UNIT_SECONDS[unit]


@dataclass
class FeederROI:
    cx: float
    cy: float
    r: float
    ring_inner_ratio: float = 0.20   # fraction of r; excludes the lid center a nose can't reach
    ring_outer_ratio: float = 1.10   # fraction of r; margin for a pig with its nose at the rim

    @property
    def ring_inner(self) -> float:
        return self.r * self.ring_inner_ratio

    @property
    def ring_outer(self) -> float:
        return self.r * self.ring_outer_ratio


@dataclass
class FeedingRule:
    """The 3-condition feeding判定 from the research: distance + direction + confidence."""

    # 0-based indices into the 18-point CVAT skeleton, ordered by the label
    # spec's svg data-node-id (creation order): tail_tip, tail, l_hip, r_hip,
    # l_foot, r_foot, r_flank, l_flank, l_shoulder, r_shoulder, l_hand, neck,
    # nose, l_ear, l_ear_tip, r_ear, r_ear_tip, r_hand. NOT the label config
    # JSON's "sublabels" array display order (which can be freely reordered
    # in the CVAT UI independent of actual export order -- an earlier version
    # of this file used that order and got nose_index wrong as a result). See
    # pose_tracker.py's KEYPOINT_NAMES for the same list -- keep both in sync.
    nose_index: int = 12
    neck_index: int = 11
    direction_angle_max_deg: float = 60.0
    keypoint_conf_min: float = 0.5
    min_bout_sec: float = 0.0   # future: require this many consecutive seconds before counting (hysteresis)

    # --- condition 4: posture (see posture.py) -----------------------------
    # Off by default -- these ramp/weight/threshold values are first-guess
    # placeholders, not calibrated against labeled data (no scripts/
    # calibrate_posture.py exists yet, unlike the RF path's ~940 labeled
    # frames). Flip posture_enabled on per-config only after checking it
    # against real footage for that pen/growth-stage -- the direction-angle
    # 60->62->60 episode (2026-08-25: a 2 degree relaxation alone produced
    # a real lying-pig false positive at night) is the cautionary example
    # for shipping an unvalidated threshold change here.
    posture_enabled: bool = False
    posture_limb_low: float = 0.3
    posture_limb_high: float = 0.7
    posture_aspect_low: float = 0.45
    posture_aspect_high: float = 0.85
    posture_asym_low: float = 0.15
    posture_asym_high: float = 0.5
    posture_weights: tuple[float, float, float] = (0.3, 0.4, 0.3)   # (limb, aspect, asymmetry)
    posture_score_max: float = 0.5     # score above this -> lying (excluded)
    posture_min_evidence: float = 0.5  # below this -> insufficient evidence, fail open (pass)


@dataclass
class AggregationConfig:
    smooth_sec: float = 1.0
    bin_size: str = "5s"
    video_start_time: Optional[str] = None  # ISO datetime; enables wall-clock x-axis instead of elapsed time

    @property
    def bin_size_sec(self) -> float:
        return parse_duration(self.bin_size)


@dataclass
class ModelConfig:
    weights: str
    imgsz: int = 960
    conf: float = 0.4
    iou: float = 0.5
    tracker: str = "bytetrack.yaml"
    device: Optional[str] = None
    n_keypoints: int = 18


@dataclass
class PenConfig:
    pen_id: str
    growth_stage: str  # "이유자돈" | "육성돈" | "비육돈" (free text, used only for grouping/plots)
    model: ModelConfig
    feeder: FeederROI
    feeding_rule: FeedingRule = field(default_factory=FeedingRule)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    expected_pig_count: Optional[int] = None   # total pigs in the pen -- sanity check only
    feeder_capacity: Optional[int] = None      # physical simultaneous feeding spaces at the feeder;
                                                # observed count above this flags likely overlap/occlusion double-counting
    notes: str = ""


def load_config(path: str | Path) -> PenConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    model = ModelConfig(**raw["model"])
    feeder = FeederROI(**raw["feeder"])
    feeding_rule = FeedingRule(**raw.get("feeding_rule", {}))
    aggregation = AggregationConfig(**raw.get("aggregation", {}))

    return PenConfig(
        pen_id=raw["pen_id"],
        growth_stage=raw["growth_stage"],
        model=model,
        feeder=feeder,
        feeding_rule=feeding_rule,
        aggregation=aggregation,
        expected_pig_count=raw.get("expected_pig_count"),
        feeder_capacity=raw.get("feeder_capacity"),
        notes=raw.get("notes", ""),
    )
