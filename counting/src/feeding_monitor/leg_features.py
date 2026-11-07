"""Leg-geometry features for "lying on its side" -- uses keypoint POSITIONS only.

Why a second lying signal: posture.py (condition 4) decides lying from keypoint
CONFIDENCE (limb visibility, left/right confidence asymmetry). That breaks in
two ways for this study:
  * human-corrected keypoints (Stage 1) have no real confidence, so the signal
    is meaningless there;
  * on CH010 model keypoints it removed every visiting pig (Stage 2, F1 0).
A pig sleeping on its side stretches all four legs out to ONE side of the
body, roughly perpendicular to the spine (a "ㄷ" shape seen from above). A
standing pig keeps its legs under the body: short, split to both sides. Those
are 2D geometry facts, so they mean the same thing for human and model points.

All lengths are divided by body length (neck->tail), all angles are relative
to the spine -> independent of camera height, image size and pig size.

Features (NaN when the needed points are missing; callers fill a sentinel):
  leg_extension     mean |joint->paw| / body length          (lying: larger)
  leg_angle_deg     mean angle between leg and spine, 0-90   (lying: ~90)
  paw_offset        mean |paw distance from spine line| / L   (lying: larger)
  leg_side_bias     |mean side sign of the 4 paws|, 0-1       (lying "ㄷ": ~1, standing: ~0)
  body_bend_deg     angle neck-midbody-tail, 180 = straight  (curled: smaller)
"""

from __future__ import annotations

import math

# 18-keypoint order (same as pose_tracker / posture.py)
TAIL, L_HIP, R_HIP, L_FOOT, R_FOOT = 1, 2, 3, 4, 5
R_FLANK, L_FLANK, L_SHOULDER, R_SHOULDER, L_HAND = 6, 7, 8, 9, 10
NECK, R_HAND = 11, 17

LEGS = ((L_SHOULDER, L_HAND), (R_SHOULDER, R_HAND), (L_HIP, L_FOOT), (R_HIP, R_FOOT))
LEG_FEATURE_NAMES = ["leg_extension", "leg_angle_deg", "paw_offset", "leg_side_bias", "body_bend_deg"]
NAN = float("nan")


def _ok(kpts, i, conf_min):
    return i < len(kpts) and kpts[i][2] > conf_min


def leg_features(kpts, conf_min: float = 0.0) -> dict[str, float]:
    out = dict.fromkeys(LEG_FEATURE_NAMES, NAN)
    if not (_ok(kpts, NECK, conf_min) and _ok(kpts, TAIL, conf_min)):
        return out
    nx, ny = kpts[NECK][:2]
    tx, ty = kpts[TAIL][:2]
    sx, sy = tx - nx, ty - ny                     # spine vector neck -> tail
    L = math.hypot(sx, sy)
    if L < 1e-3:
        return out
    ux, uy = sx / L, sy / L

    ext, ang, off, side = [], [], [], []
    for joint, paw in LEGS:
        if not _ok(kpts, paw, conf_min):
            continue
        px, py = kpts[paw][:2]
        # signed distance of the paw from the spine line (cross product)
        d = ux * (py - ny) - uy * (px - nx)
        off.append(abs(d) / L)
        side.append(1.0 if d > 0 else -1.0)
        if _ok(kpts, joint, conf_min):
            jx, jy = kpts[joint][:2]
            vx, vy = px - jx, py - jy
            n = math.hypot(vx, vy)
            ext.append(n / L)
            if n > 1e-3:
                c = abs(vx * ux + vy * uy) / n     # |cos| -> angle folded to 0..90
                ang.append(math.degrees(math.acos(min(1.0, c))))

    if ext:
        out["leg_extension"] = sum(ext) / len(ext)
    if ang:
        out["leg_angle_deg"] = sum(ang) / len(ang)
    if off:
        out["paw_offset"] = sum(off) / len(off)
    if len(side) >= 2:                            # one paw alone says nothing about "one side"
        out["leg_side_bias"] = abs(sum(side)) / len(side)

    mids = [i for i in (L_FLANK, R_FLANK) if _ok(kpts, i, conf_min)]
    if mids:
        mx = sum(kpts[i][0] for i in mids) / len(mids)
        my = sum(kpts[i][1] for i in mids) / len(mids)
        a = (nx - mx, ny - my)
        b = (tx - mx, ty - my)
        na, nb = math.hypot(*a), math.hypot(*b)
        if na > 1e-3 and nb > 1e-3:
            c = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
            out["body_bend_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    return out
