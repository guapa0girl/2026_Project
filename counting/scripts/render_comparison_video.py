"""Render a single side-by-side video: left half annotated with this
project's geometric classifier's verdict, right half with the ported
RandomForest posture classifier's verdict -- running pose inference once
per frame and drawing both, so the two panels are pixel-for-pixel the same
frame/detections and only the feeding verdict (box color + Count) differs.
That makes it easy to see exactly which pigs the two methods disagree on.

Usage:
    python scripts/render_comparison_video.py --config configs/farm_ch005_20260610.yaml \
        --source "data/raw/[CH005] 202606100923_44.avi" --fps 200 --max-frames 2000 \
        --zones configs/zones/feeder_zones_ch005.yaml --model models/posture_random_forest.joblib
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from feeding_monitor.config import load_config
from feeding_monitor.feeding_classifier import classify
from feeding_monitor.pose_tracker import PoseTracker, draw_detection
from feeding_monitor.posture_classifier import MLPostureModel
from feeding_monitor.posture_features import load_zones

LABEL_COLOR = (255, 255, 255)


def parse_args():
    p = argparse.ArgumentParser(description="Side-by-side geometric vs RandomForest annotated video")
    p.add_argument("--config", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--fps", type=float, default=None, help="override container fps (see run_analysis.py --fps)")
    p.add_argument("--start-frame", type=int, default=0, help="seek here before starting (e.g. to render a clip from the middle of a long recording)")
    p.add_argument("--max-frames", type=int, default=None, help="frames to process AFTER --start-frame (not an absolute frame index)")
    p.add_argument("--zones", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--rf-threshold", type=float, default=0.5, help="RandomForest 'eating' probability cutoff -- lower is more lenient (e.g. 0.4)")
    p.add_argument("--out", default=None, help="default: outputs/<source_stem>_geo_vs_rf.mp4")
    p.add_argument("--panel-width", type=int, default=960,
                    help="resize each panel to this width before combining -- mp4v (no openh264 DLL here, "
                         "so avc1/H.264 isn't available) fails to write very large combined frames")
    p.add_argument("--playback-fps", type=float, default=None,
                    help="fps to WRITE the output video at (default: same as --fps/source). Writing a 200fps "
                         "source at 200fps plays correctly-paced but looks unnaturally fast to watch/scrub in "
                         "most players; pass a standard rate (25/30) here -- combine with --frame-stride to "
                         "keep real-time pacing instead of slow-motion.")
    p.add_argument("--frame-stride", type=int, default=1,
                    help="only draw+write every Nth decoded frame (inference/tracking still runs on all of "
                         "them, for ByteTrack continuity) -- lets one render span a longer chunk of source "
                         "video without the output file/duration growing by the same factor. Pick N so "
                         "playback_fps * N ~= source fps for natural-speed (not slow-motion) playback.")
    return p.parse_args()


def label_panel(frame, title, count, label_h):
    h, w = frame.shape[:2]
    bar = cv2.copyMakeBorder(frame, label_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(bar, f"{title}  Count : {count}", (10, label_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_COLOR, 2, cv2.LINE_AA)
    return bar


def main():
    args = parse_args()
    config = load_config(args.config)
    zones = load_zones(args.zones)
    ml_model = MLPostureModel.load(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.source}")
    probed_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    fps = args.fps if args.fps is not None else probed_fps
    playback_fps = args.playback_fps if args.playback_fps is not None else fps

    panel_w = args.panel_width
    panel_h = round(height * panel_w / width)
    label_h = 40
    out_w, out_h = panel_w * 2, panel_h + label_h

    out_path = Path(args.out) if args.out else ROOT / "outputs" / f"{Path(args.source).stem}_geo_vs_rf.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # mp4v silently fails to write very large combined frames (e.g. doubling a
    # 1920x1080 source gives 3840x1126) on this machine's ffmpeg build -- and
    # avc1/H.264 isn't usable without the openh264 DLL -- so downscale each
    # panel first (--panel-width) instead of writing at full source resolution.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, playback_fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"VideoWriter를 열지 못했습니다 (codec/크기 문제일 수 있음): {out_path}")

    print(f"[info] source~{fps:.0f}fps, stride={args.frame_stride} -> writing at {playback_fps:.0f}fps "
          f"({'natural speed' if abs(playback_fps * args.frame_stride - fps) < 1 else 'NOT 1x real-time -- adjust --frame-stride'})")

    tracker = PoseTracker(config.model)
    n_frames = 0
    n_written = 0
    try:
        for frame_idx, frame, detections in tracker.track(args.source, start_frame=args.start_frame):
            if args.max_frames is not None and (frame_idx - args.start_frame) >= args.max_frames:
                break
            n_frames += 1
            if frame_idx % args.frame_stride != 0:
                continue

            geo_frame = frame.copy()
            rf_frame = frame.copy()
            ml_preds = ml_model.predict_batch(detections, zones, threshold=args.rf_threshold) if detections else []

            geo_count = 0
            rf_count = 0
            for det, rf_verdict in zip(detections, ml_preds):
                geo_verdict = classify(det, config.feeder, config.feeding_rule).is_feeding
                geo_count += geo_verdict
                rf_count += rf_verdict
                draw_detection(geo_frame, det, config.feeding_rule.keypoint_conf_min, is_feeding=geo_verdict)
                draw_detection(rf_frame, det, config.feeding_rule.keypoint_conf_min, is_feeding=rf_verdict)

            geo_frame = cv2.resize(geo_frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            rf_frame = cv2.resize(rf_frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            geo_frame = label_panel(geo_frame, "GEOMETRIC (this project)", geo_count, label_h)
            rf_frame = label_panel(rf_frame, "RANDOMFOREST (ported)", rf_count, label_h)
            combined = cv2.hconcat([geo_frame, rf_frame])
            writer.write(combined)
            n_written += 1
    finally:
        writer.release()

    print(f"[saved] {out_path}  ({n_written} frames written / {n_frames} decoded, "
          f"{n_written / playback_fps:.1f}s playback)")


if __name__ == "__main__":
    main()
