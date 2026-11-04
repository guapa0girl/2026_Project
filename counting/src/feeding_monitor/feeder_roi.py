"""Circular feeder ROI: auto-detection from a frame, and a visual calibration
overlay for tuning configs/*.yaml by eye (see scripts/calibrate_feeder.py).

Assumes a top-down camera and a single circular feeder with a distinctly
colored lid per pen (matches the research setup: red lid, top-down view,
no wide-angle distortion to correct).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import FeederROI


def detect_feeder_circle(
    frame: np.ndarray,
    hsv_lower1=(0, 100, 80), hsv_upper1=(10, 255, 255),
    hsv_lower2=(170, 100, 80), hsv_upper2=(180, 255, 255),
) -> Optional[FeederROI]:
    """Find a red circular lid via HSV thresholding + minEnclosingCircle.

    Red wraps around hue 0/180 in OpenCV's HSV, hence two ranges merged with OR.
    Returns None if no plausible blob is found -- fall back to manual
    calibration (scripts/calibrate_feeder.py) in that case.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array(hsv_lower1), np.array(hsv_upper1))
    mask2 = cv2.inRange(hsv, np.array(hsv_lower2), np.array(hsv_upper2))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:  # too small to be a feeder lid
        return None

    (cx, cy), r = cv2.minEnclosingCircle(largest)
    return FeederROI(cx=float(cx), cy=float(cy), r=float(r))


def draw_calibration(frame: np.ndarray, feeder: FeederROI) -> np.ndarray:
    img = frame.copy()
    cv2.circle(img, (int(feeder.cx), int(feeder.cy)), int(feeder.r), (0, 0, 255), 2)
    cv2.circle(img, (int(feeder.cx), int(feeder.cy)), int(feeder.ring_inner), (0, 255, 255), 2)
    cv2.circle(img, (int(feeder.cx), int(feeder.cy)), int(feeder.ring_outer), (0, 255, 255), 2)
    cv2.drawMarker(img, (int(feeder.cx), int(feeder.cy)), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(
        img,
        f"cx={feeder.cx:.0f} cy={feeder.cy:.0f} r={feeder.r:.0f} "
        f"(ring {feeder.ring_inner:.0f}-{feeder.ring_outer:.0f})",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    return img
