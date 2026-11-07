"""One place that turns a Detection into every feature the methods use, so the
evaluation (scripts/evaluate_methods.py) and the day graph
(scripts/apply_methods.py) compute exactly the same numbers.

Methods
  geo         feeding_classifier.classify()  -- rule, no training
  rf          RandomForest on the 6 posture_features            (README "RF 단독")
  rf_geo      rf + Geo rule values (distance/r, head angle, 3 posture signals)
  rf_geo_leg  rf_geo + leg-geometry lying features (leg_features.py)
"""

from __future__ import annotations

import math

from .feeding_classifier import classify
from .leg_features import LEG_FEATURE_NAMES, leg_features
from .posture import evaluate_posture
from .posture_features import FEATURE_NAMES, extract_features

RF_FEATURES = list(FEATURE_NAMES)
GEO_FEATURES = ["geo_nose_dist_over_r", "geo_head_angle_deg",
                "geo_limb_visibility", "geo_aspect_ratio", "geo_lateral_asymmetry"]
LEG_FEATURES = list(LEG_FEATURE_NAMES)

METHOD_FEATURES = {
    "rf": RF_FEATURES,
    "rf_geo": RF_FEATURES + GEO_FEATURES,
    "rf_geo_leg": RF_FEATURES + GEO_FEATURES + LEG_FEATURES,
}
METHODS = ["geo"] + list(METHOD_FEATURES)

SENTINEL = -1.0          # "not computable" -> RandomForest can't take NaN
LEG_CONF_MIN = 0.3       # model keypoints below this are ignored (human points are 1.0)


def _fill(v) -> float:
    return SENTINEL if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def geo_values(det, feeder, rule) -> list[float]:
    nx, ny, _ = det.kpts[rule.nose_index]
    kx, ky, _ = det.kpts[rule.neck_index]
    dist = math.hypot(nx - feeder.cx, ny - feeder.cy) / max(feeder.r, 1e-6)
    hv, fv = (nx - kx, ny - ky), (feeder.cx - nx, feeder.cy - ny)
    hn, fn = math.hypot(*hv), math.hypot(*fv)
    if hn < 1e-3 or fn < 1e-3:
        angle = 0.0 if fn < 1e-3 else 180.0
    else:
        c = max(-1.0, min(1.0, (hv[0] * fv[0] + hv[1] * fv[1]) / (hn * fn)))
        angle = math.degrees(math.acos(c))
    p = evaluate_posture(det.kpts, rule)
    return [dist, angle, _fill(p.limb_visibility), _fill(p.aspect_ratio), _fill(p.lateral_asymmetry)]


def featurize_det(det, cfg, zones) -> dict:
    """All features + the geo rule's verdict for one detection."""
    rec = dict(zip(RF_FEATURES, (_fill(v) for v in extract_features(det, zones).as_row())))
    rec.update(dict(zip(GEO_FEATURES, geo_values(det, cfg.feeder, cfg.feeding_rule))))
    rec.update({k: _fill(v) for k, v in leg_features(det.kpts, LEG_CONF_MIN).items()})
    v = classify(det, cfg.feeder, cfg.feeding_rule)
    rec["pred_geo"] = int(v.is_feeding)
    rec["geo_reason"] = v.reason
    return rec
