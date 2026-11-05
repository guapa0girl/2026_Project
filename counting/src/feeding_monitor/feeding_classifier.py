"""The feeding core: 4 keypoint-based conditions, all must hold.

    1) Confidence -- nose/neck keypoint confidence both clear the threshold
                    (rejects keypoints hallucinated onto an occluded pig in
                    a crowded pile)
    2) Distance  -- nose falls within the feeder's ring (excludes the lid
                    center a nose can't physically reach, and anything too
                    far away)
    3) Direction -- the neck->nose vector points at the feeder center within
                    `direction_angle_max_deg` (this is what rejects a pig
                    whose rear overlaps the ROI but whose head points away --
                    the exact failure mode that made the original bbox-only
                    approach overcount)
    4) Posture   -- the pig is not lying on its side (see posture.py)

Why condition 4 exists: conditions 1-3 all pass for a pig asleep against the
feeder rim with its nose resting inside the ring and pointed at the center.
Direction was originally described as rejecting "a pig sleeping next to the
feeder", but that only holds when the sleeping pig's head happens to point
away. A pig lying with its snout on the rim is geometrically
indistinguishable from a feeding one on position and heading alone -- and
this was observed being counted in real geo footage. Posture is the only
remaining discriminator, and it is precisely the information a rotated
bounding box cannot represent.

Runs per detection and is driven entirely by the pen config (configs/*.yaml).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import FeederROI, FeedingRule
from .pose_tracker import Detection
from .posture import PostureScore, evaluate_posture


@dataclass
class FeedingVerdict:
    is_feeding: bool
    distance: float | None
    angle_deg: float | None
    nose_conf: float | None
    neck_conf: float | None
    reason: str  # which condition failed first, or "ok"
    # --- condition 4 diagnostics (appended with defaults: existing positional
    #     constructions and any CSV writers keyed on the first six fields stay valid)
    posture_score: float | None = None
    posture_evidence: float | None = None
    limb_visibility: float | None = None
    aspect_ratio: float | None = None
    lateral_asymmetry: float | None = None


def _verdict(is_feeding, distance, angle, nose_conf, neck_conf, reason,
             posture: PostureScore | None = None) -> FeedingVerdict:
    """Fold an optional PostureScore into the verdict's diagnostic fields."""
    return FeedingVerdict(
        is_feeding=is_feeding,
        distance=distance,
        angle_deg=angle,
        nose_conf=nose_conf,
        neck_conf=neck_conf,
        reason=reason,
        posture_score=posture.score if posture else None,
        posture_evidence=posture.evidence if posture else None,
        limb_visibility=posture.limb_visibility if posture else None,
        aspect_ratio=posture.aspect_ratio if posture else None,
        lateral_asymmetry=posture.lateral_asymmetry if posture else None,
    )


def classify(det: Detection, feeder: FeederROI, rule: FeedingRule) -> FeedingVerdict:
    if rule.nose_index >= len(det.kpts) or rule.neck_index >= len(det.kpts):
        return _verdict(False, None, None, None, None, "missing_keypoints")

    nx, ny, nose_conf = det.kpts[rule.nose_index]
    kx, ky, neck_conf = det.kpts[rule.neck_index]

    # --- 1) confidence ----------------------------------------------------
    if nose_conf < rule.keypoint_conf_min or neck_conf < rule.keypoint_conf_min:
        return _verdict(False, None, None, nose_conf, neck_conf, "low_confidence")

    # --- 2) distance ------------------------------------------------------
    distance = math.hypot(nx - feeder.cx, ny - feeder.cy)
    if not (feeder.ring_inner <= distance <= feeder.ring_outer):
        return _verdict(False, distance, None, nose_conf, neck_conf, "out_of_ring")

    # --- 3) direction -----------------------------------------------------
    head_vec = (nx - kx, ny - ky)                   # neck -> nose: which way the head points
    feeder_vec = (feeder.cx - nx, feeder.cy - ny)   # nose -> feeder center
    head_norm = math.hypot(*head_vec)
    feeder_norm = math.hypot(*feeder_vec)
    if head_norm < 1e-3 or feeder_norm < 1e-3:
        return _verdict(False, distance, None, nose_conf, neck_conf, "degenerate_vector")

    cos_angle = (head_vec[0] * feeder_vec[0] + head_vec[1] * feeder_vec[1]) / (head_norm * feeder_norm)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.degrees(math.acos(cos_angle))

    if angle > rule.direction_angle_max_deg:
        return _verdict(False, distance, angle, nose_conf, neck_conf, "wrong_direction")

    # --- 4) posture -------------------------------------------------------
    # Evaluated last so the diagnostic columns are only populated for
    # detections that already cleared 1-3. That keeps the threshold-calibration
    # sample (scripts/calibrate_posture.py) restricted to exactly the ambiguous
    # population this condition is meant to split: pigs at the feeder, head
    # pointed in -- some feeding, some asleep.
    if not rule.posture_enabled:
        return _verdict(True, distance, angle, nose_conf, neck_conf, "ok")

    posture = evaluate_posture(det.kpts, rule)
    if posture.lying:
        return _verdict(False, distance, angle, nose_conf, neck_conf, "lying_posture", posture)

    return _verdict(True, distance, angle, nose_conf, neck_conf, "ok", posture)