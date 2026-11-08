"""Second feeding-classification approach (RandomForest features), compared
against the geometric rule (feeding_classifier.classify).

Where feeding_classifier.classify() only asks "is the nose in the ring,
pointed at the center, with good confidence", this extracts 6 features
(zone containment/distance + how far the head and spine are dipped down +
neck height + keypoint confidence). The RandomForest that uses them is trained
in scripts/evaluate_methods.py / scripts/apply_methods.py (method_features.py).

Uses polygon feeder zones (configs/zones/feeder_zones_ch010.yaml, a 48-gon of
the same circle as the geo ROI -- scripts/make_circle_zone.py), so geo and RF
look at the same feeder area.

Keypoint indices match pose_tracker.KEYPOINT_NAMES (nose=12, neck=11,
l_hip=2, r_hip=3, l_shoulder=8, r_shoulder=9) -- both projects' CVAT
skeletons independently converged on the same node-id-ordered scheme.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import yaml

from .pose_tracker import Detection

NOSE, NECK = 12, 11
L_HIP, R_HIP = 2, 3
L_SHOULDER, R_SHOULDER = 8, 9

FEATURE_NAMES = [
    "nose_in_zone",
    "nose_zone_dist_norm",
    "head_dip_deg",
    "spine_dip_deg",
    "neck_rel_height",
    "key_kpt_conf",
]

Point = tuple[float, float]
Zone = list[Point]


def load_zones(zones_yaml: str | Path) -> list[Zone]:
    with open(zones_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [[tuple(pt) for pt in zone] for zone in data["zones"]]


def point_in_polygon(point: Point, polygon: Zone) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def polygon_center(polygon: Zone) -> Point:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def angle_from_vertical(vx: float, vy: float) -> float:
    """Angle (degrees) between the vector and straight down (+y in image
    coordinates). 0 = pointing straight down (head fully dipped), 180 = up."""
    norm = math.hypot(vx, vy)
    if norm < 1e-6:
        return 90.0
    cos_a = max(-1.0, min(1.0, vy / norm))
    return math.degrees(math.acos(cos_a))


class PostureFeatures(NamedTuple):
    nose_in_zone: float
    nose_zone_dist_norm: float
    head_dip_deg: float
    spine_dip_deg: float
    neck_rel_height: float
    key_kpt_conf: float

    def as_row(self) -> list[float]:
        return [
            self.nose_in_zone, self.nose_zone_dist_norm, self.head_dip_deg,
            self.spine_dip_deg, self.neck_rel_height, self.key_kpt_conf,
        ]


def extract_features(det: Detection, zones: list[Zone]) -> PostureFeatures:
    kpts = det.kpts
    if len(kpts) <= max(NOSE, NECK, L_HIP, R_HIP, L_SHOULDER, R_SHOULDER):
        raise ValueError(f"expected at least {max(NOSE, NECK, L_HIP, R_HIP, L_SHOULDER, R_SHOULDER) + 1} keypoints")

    nose = kpts[NOSE][:2]
    neck = kpts[NECK][:2]
    hips_center = ((kpts[L_HIP][0] + kpts[R_HIP][0]) / 2, (kpts[L_HIP][1] + kpts[R_HIP][1]) / 2)
    shoulders_center = ((kpts[L_SHOULDER][0] + kpts[R_SHOULDER][0]) / 2, (kpts[L_SHOULDER][1] + kpts[R_SHOULDER][1]) / 2)

    x1, y1, x2, y2 = det.box
    bbox_diag = math.hypot(x2 - x1, y2 - y1) + 1e-6

    nose_in_zone = 0.0
    min_dist = None
    for zone in zones:
        if point_in_polygon(nose, zone):
            nose_in_zone = 1.0
        zcx, zcy = polygon_center(zone)
        d = math.hypot(nose[0] - zcx, nose[1] - zcy) / bbox_diag
        if min_dist is None or d < min_dist:
            min_dist = d
    if min_dist is None:
        min_dist = 10.0  # no zones configured -> treat as very far

    head_dip_deg = angle_from_vertical(nose[0] - neck[0], nose[1] - neck[1])
    spine_dip_deg = angle_from_vertical(neck[0] - hips_center[0], neck[1] - hips_center[1])
    neck_rel_height = (neck[1] - shoulders_center[1]) / bbox_diag

    key_idx = [NOSE, NECK, L_HIP, R_HIP, L_SHOULDER, R_SHOULDER]
    key_kpt_conf = sum(1 for i in key_idx if kpts[i][2] > 0) / len(key_idx)

    return PostureFeatures(
        nose_in_zone=nose_in_zone,
        nose_zone_dist_norm=min_dist,
        head_dip_deg=head_dip_deg,
        spine_dip_deg=spine_dip_deg,
        neck_rel_height=neck_rel_height,
        key_kpt_conf=key_kpt_conf,
    )


def rule_based_classify(features: PostureFeatures, dip_thresh_deg: float = 55.0, dist_thresh: float = 0.35) -> bool:
    near_zone = features.nose_in_zone > 0.5 or features.nose_zone_dist_norm < dist_thresh
    head_down = features.head_dip_deg < dip_thresh_deg
    return bool(near_zone and head_down)
