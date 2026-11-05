"""YOLO-Pose + ByteTrack wrapper: detect + track pigs and yield per-frame,
per-pig keypoint records.

scripts/count_full_video.py uses `PoseTracker.model.predict()` on one seeked
frame per second; `track()` is kept for continuous tracking over a clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import ModelConfig

# The project's actual CVAT skeleton label spec, in real YOLO-pose export
# order == kpt0..kpt17. This IS ordered by the SVG's data-node-id ascending
# (creation order), NOT the label config JSON's "sublabels" array display
# order -- those two orders differ (sublabels can be freely reordered/
# drag-and-dropped in the CVAT label editor without affecting the already-
# exported point order), and this file previously got that wrong once
# already: assuming sublabels order put "nose" last (kpt17) instead of right
# after neck (kpt12), which silently used l_ear as "nose" throughout the
# feeding classifier and drew garbled skeleton lines. This ordering matches
# training/configs/pig_pose.yaml -- keep both in sync if the CVAT skeleton is
# ever edited. Also keep in sync with
# FeedingRule.nose_index/neck_index in config.py.
KEYPOINT_NAMES: list[str] = [
    "tail_tip",     # 0
    "tail",         # 1
    "l_hip",        # 2
    "r_hip",        # 3
    "l_foot",       # 4
    "r_foot",       # 5
    "r_flank",      # 6
    "l_flank",      # 7
    "l_shoulder",   # 8
    "r_shoulder",   # 9
    "l_hand",       # 10
    "neck",         # 11
    "nose",         # 12
    "l_ear",        # 13
    "l_ear_tip",    # 14
    "r_ear",        # 15
    "r_ear_tip",    # 16
    "r_hand",       # 17
]

# Per-keypoint colors, straight from the CVAT label config (hex RGB as CVAT
# stores them; converted to BGR below for cv2), reordered to match
# KEYPOINT_NAMES above (i.e. this is by kpt index, not by sublabels order).
KEYPOINT_COLORS_HEX: list[str] = [
    "#3D3DF5", "#0000ff", "#24b353", "#f59331", "#014217", "#8d6d0d",
    "#ee8d2c", "#26bd58", "#23c056", "#f6912c", "#043614", "#ff0000",
    "#fa7dbb", "#C2B7FF", "#C2B7FF", "#fcfc1a", "#fafa2d", "#a36323",
]


def _hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


KEYPOINT_COLORS: list[tuple[int, int, int]] = [_hex_to_bgr(h) for h in KEYPOINT_COLORS_HEX]

# 1-indexed edges (kpt index + 1), matching draw_detection's `pts[a-1]` lookup.
# Parsed from the CVAT label spec's SVG <line data-node-from/to> connectivity,
# translated through the corrected (node-id) index order above.
PIG_SKELETON: list[tuple[int, int]] = [
    (7, 10), (18, 10), (2, 4), (16, 17), (12, 16), (13, 12), (14, 15),
    (12, 14), (9, 11), (3, 5), (4, 6), (7, 4), (12, 10), (9, 12),
    (8, 9), (3, 8), (2, 3), (1, 2),
]

SKELETON_COLOR = (255, 160, 0)
BOX_COLOR = (60, 220, 60)
TEXT_COLOR = (255, 255, 255)


@dataclass
class Detection:
    frame_idx: int
    track_id: Optional[int]
    det_conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    kpts: list[tuple[float, float, float]]  # (x, y, conf) per keypoint, in config order


def resolve_device(requested: Optional[str]):
    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return 0
    print("[warn] no CUDA GPU detected, falling back to CPU")
    return "cpu"


def draw_detection(frame, det: Detection, kpt_conf_thresh: float = 0.5, is_feeding: bool = False):
    x1, y1, x2, y2 = map(int, det.box)
    color = (0, 140, 255) if is_feeding else BOX_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"pig {det.track_id}" + (" FEEDING" if is_feeding else "")
    cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    pts = {}
    for idx, (x, y, c) in enumerate(det.kpts):
        if c < kpt_conf_thresh:
            continue
        pts[idx] = (int(x), int(y))
        color = KEYPOINT_COLORS[idx] if idx < len(KEYPOINT_COLORS) else (0, 220, 255)
        cv2.circle(frame, pts[idx], 3, color, -1, cv2.LINE_AA)
    for a, b in PIG_SKELETON:
        if (a - 1) in pts and (b - 1) in pts:
            cv2.line(frame, pts[a - 1], pts[b - 1], SKELETON_COLOR, 2, cv2.LINE_AA)
    return frame


def _parse_detections(result, frame_idx: int) -> list[Detection]:
    boxes = result.boxes
    kpts_all = result.keypoints
    detections: list[Detection] = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)
        kxy = kpts_all.xy.cpu().numpy() if kpts_all is not None else None
        kconf = kpts_all.conf.cpu().numpy() if kpts_all is not None and kpts_all.conf is not None else None

        for i in range(len(boxes)):
            track_id = int(ids[i]) if ids[i] is not None else None
            kpts_i = []
            if kxy is not None:
                for j in range(kxy.shape[1]):
                    x, y = kxy[i, j]
                    c = float(kconf[i, j]) if kconf is not None else 1.0
                    kpts_i.append((float(x), float(y), c))
            detections.append(
                Detection(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    det_conf=float(confs[i]),
                    box=tuple(xyxy[i].tolist()),
                    kpts=kpts_i,
                )
            )
    return detections


class PoseTracker:
    def __init__(self, model_cfg: ModelConfig):
        self.cfg = model_cfg
        self.device = resolve_device(model_cfg.device)
        self.model = YOLO(model_cfg.weights)

    def track(self, source, persist: bool = True, start_frame: int = 0) -> Iterator[tuple[int, np.ndarray, list[Detection]]]:
        """Yield (frame_idx, raw_bgr_frame, detections) for every frame of `source`.

        `source` is anything Ultralytics accepts: a video file path, an RTSP
        URL, or a webcam index (int). `start_frame` > 0 seeks a video file to
        that frame first (e.g. to render a clip from the middle of a long
        recording without paying for full-inference on the skipped prefix)
        -- this switches to feeding the model one manually-read frame at a time
        (still with persist=True for continuous ByteTrack IDs) instead of
        Ultralytics' own stream=True file reader, which always starts at 0.
        """
        if start_frame > 0:
            cap = cv2.VideoCapture(source)
            # seeking isn't 100% reliable on these files' corrupted AVI
            # headers (occasional isolated failures elsewhere too, e.g.
            # count_full_video.py's ~4/630 seek misses) -- nudge forward a
            # few frames and retry rather than silently yielding 0 frames.
            frame_idx = start_frame
            ok = False
            for nudge in range(31):
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + nudge)
                ok, first_frame = cap.read()
                if ok:
                    frame_idx = start_frame + nudge
                    break
            if not ok:
                cap.release()
                raise RuntimeError(f"seek to frame {start_frame} (+0..30 nudge) failed on {source!r}")
            try:
                frame = first_frame
                while frame is not None:
                    result = self.model.track(
                        source=frame,
                        tracker=self.cfg.tracker,
                        conf=self.cfg.conf,
                        iou=self.cfg.iou,
                        imgsz=self.cfg.imgsz,
                        device=self.device,
                        persist=persist,
                        verbose=False,
                    )[0]
                    yield frame_idx, frame, _parse_detections(result, frame_idx)
                    frame_idx += 1
                    ok, frame = cap.read()
                    if not ok:
                        # a mid-stream read failure isn't necessarily the end
                        # of the file -- this project's AVI files have
                        # isolated bad spots (corrupted headers); re-seek a
                        # little further forward and retry a few times
                        # before actually giving up, so one bad frame
                        # doesn't truncate an otherwise-long render.
                        frame = None
                        for nudge in range(1, 31):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + nudge)
                            ok, retry_frame = cap.read()
                            if ok:
                                frame_idx += nudge
                                frame = retry_frame
                                break
            finally:
                cap.release()
            return

        results_gen = self.model.track(
            source=source,
            tracker=self.cfg.tracker,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.device,
            persist=persist,
            stream=True,
            verbose=False,
        )

        frame_idx = 0
        for result in results_gen:
            yield frame_idx, result.orig_img, _parse_detections(result, frame_idx)
            frame_idx += 1
