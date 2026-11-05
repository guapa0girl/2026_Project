"""Wires the pieces together: PoseTracker -> feeding_classifier -> FeedingAggregator
-> report. One function, reused by both scripts/run_analysis.py (a recorded
video, drained to completion) and scripts/run_realtime.py (a live source,
where `on_bin` is how the caller gets results as they close instead of
waiting for the stream to end).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import cv2

from .aggregator import BinResult, FeedingAggregator
from .config import PenConfig
from .feeding_classifier import classify
from .pose_tracker import PoseTracker, draw_detection

N_KPTS_DEFAULT = 18


COUNT_OVERLAY_COLOR = (0, 255, 255)


def run_pipeline(
    config: PenConfig,
    source,
    fps: float,
    out_dir: str | Path,
    run_name: str,
    save_video: bool = True,
    save_frame_csv: bool = True,
    show: bool = False,
    window_name: Optional[str] = None,
    frame_size: Optional[tuple[int, int]] = None,
    on_bin: Optional[Callable[[BinResult], None]] = None,
    on_frame: Optional[Callable[[int, "cv2.typing.MatLike"], None]] = None,
    max_frames: Optional[int] = None,
) -> list[BinResult]:
    """`on_frame(frame_idx, frame)`, if given, is called once per processed
    frame (after the Count/keypoint overlay is drawn) -- e.g. to periodically
    save a calibration-check snapshot (see run_realtime.py's
    --calib-snapshot-every) so ROI drift from a bumped/cleaned camera can be
    spotted after the fact instead of silently corrupting the count."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = PoseTracker(config.model)
    aggregator = FeedingAggregator(fps, config.aggregation, config.feeding_rule.min_bout_sec)
    draw_overlay = save_video or show
    window_name = window_name or f"pig feeding monitor -- {config.pen_id}"
    overflow_frames = 0
    overflow_warned = False

    video_writer = None
    frame_csv = None
    frame_csv_writer = None
    if save_frame_csv:
        frame_csv = open(out_dir / f"{run_name}_tracks.csv", "w", newline="", encoding="utf-8")
        frame_csv_writer = csv.writer(frame_csv)
        kpt_cols = [f"kpt{i}_{axis}" for i in range(config.model.n_keypoints) for axis in ("x", "y", "conf")]
        frame_csv_writer.writerow(
            ["frame", "track_id", "det_conf", "x1", "y1", "x2", "y2", "feeding"] + kpt_cols
        )

    all_bins: list[BinResult] = []
    try:
        for frame_idx, frame, detections in tracker.track(source):
            if max_frames is not None and frame_idx >= max_frames:
                break
            if save_video and video_writer is None:
                h, w = frame_size or frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(out_dir / f"{run_name}_tracked.mp4"), fourcc, fps, (w, h))

            feeding_track_ids = set()
            for det in detections:
                verdict = classify(det, config.feeder, config.feeding_rule)
                if verdict.is_feeding and det.track_id is not None:
                    feeding_track_ids.add(det.track_id)

                if frame_csv_writer is not None:
                    row = [frame_idx, det.track_id, det.det_conf, *det.box, verdict.is_feeding]
                    for x, y, c in det.kpts:
                        row += [x, y, c]
                    frame_csv_writer.writerow(row)

                if draw_overlay:
                    draw_detection(frame, det, config.feeding_rule.keypoint_conf_min, is_feeding=verdict.is_feeding)

            over_capacity = config.feeder_capacity is not None and len(feeding_track_ids) > config.feeder_capacity
            if over_capacity:
                overflow_frames += 1
                if not overflow_warned:
                    print(
                        f"[warn] frame {frame_idx}: count {len(feeding_track_ids)} exceeds "
                        f"feeder_capacity={config.feeder_capacity} -- likely overlap/occlusion double-count "
                        f"(further occurrences counted, not printed)"
                    )
                    overflow_warned = True

            if draw_overlay:
                label = f"Count : {len(feeding_track_ids)}"
                if config.feeder_capacity is not None:
                    label += f" / {config.feeder_capacity}"
                color = (0, 0, 255) if over_capacity else COUNT_OVERLAY_COLOR
                cv2.putText(
                    frame, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA,
                )
            if video_writer is not None:
                video_writer.write(frame)
            if show:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if on_frame is not None:
                on_frame(frame_idx, frame)

            closed_bin = aggregator.ingest(frame_idx, feeding_track_ids)
            if closed_bin is not None:
                all_bins.append(closed_bin)
                if on_bin is not None:
                    on_bin(closed_bin)

        tail = aggregator.flush()
        if tail is not None:
            all_bins.append(tail)
            if on_bin is not None:
                on_bin(tail)
    finally:
        if video_writer is not None:
            video_writer.release()
        if frame_csv is not None:
            frame_csv.close()
        if show:
            cv2.destroyAllWindows()
        if overflow_frames:
            print(f"[warn] {overflow_frames} frame(s) exceeded feeder_capacity={config.feeder_capacity} this run")

    return all_bins


def probe_video(source: str) -> tuple[float, int, int]:
    """Read fps/width/height from a video file without decoding all of it."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height
