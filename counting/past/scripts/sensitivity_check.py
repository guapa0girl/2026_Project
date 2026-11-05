"""Threshold sensitivity sweep: re-run the feeding classifier over an
already-produced per-frame tracks CSV (from run_analysis.py, frame-csv
enabled -- i.e. WITHOUT --no-frame-csv) across a grid of
direction_angle_max_deg / ring_outer_ratio values, without re-running
detection.

Why this matters for a 3-stage comparison specifically: growth stages
differ in body size, so a fixed angle/ring threshold means something
different for a small weaned pig than a large finisher. If the ranking or
shape of results flips when thresholds move a bit, an observed "stage
difference" might just be a threshold artifact, not a real behavioral one.
Run this per stage and check whether peak-count / total pig-seconds stay in
the same relative order across the grid ("robust to thresholds") before
trusting a cross-stage comparison.

This does NOT re-run YOLO -- it's pure post-hoc reclassification from the
saved keypoints, so it's fast enough to grid-search.

Usage:
    python scripts/run_analysis.py --config configs/pen01.yaml --source data/raw/clip.mp4 --no-video
    python scripts/sensitivity_check.py --tracks-csv outputs/pen01__clip_tracks.csv --config configs/pen01.yaml
"""

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_monitor.config import FeederROI, FeedingRule, load_config
from feeding_monitor.feeding_classifier import classify
from feeding_monitor.pose_tracker import Detection


def parse_args():
    p = argparse.ArgumentParser(description="Sweep feeding-rule thresholds over an existing tracks CSV")
    p.add_argument("--tracks-csv", required=True, help="*_tracks.csv from run_analysis.py (needs --live-log NOT required, but frame-csv must be enabled)")
    p.add_argument("--config", required=True, help="base pen config -- feeder ROI, nose/neck index, fps come from here")
    p.add_argument("--angles", default="45,60,75", help="comma-separated direction_angle_max_deg values")
    p.add_argument("--ring-outer-ratios", default="1.0,1.1,1.2", help="comma-separated ring_outer_ratio values")
    p.add_argument("--out-csv", default=None, help="default: <tracks-csv>_sensitivity.csv")
    return p.parse_args()


def load_rows(path, n_keypoints):
    kpt_cols = [f"kpt{i}_{axis}" for i in range(n_keypoints) for axis in ("x", "y", "conf")]
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("track_id", "") == "":
                continue
            kpts = []
            for i in range(n_keypoints):
                x = float(row[f"kpt{i}_x"])
                y = float(row[f"kpt{i}_y"])
                c = float(row[f"kpt{i}_conf"])
                kpts.append((x, y, c))
            rows.append((int(row["frame"]), int(float(row["track_id"])), kpts))
    return rows


def evaluate(rows, feeder, rule):
    per_frame_counts = {}
    for frame_idx, track_id, kpts in rows:
        det = Detection(frame_idx=frame_idx, track_id=track_id, det_conf=1.0, box=(0, 0, 0, 0), kpts=kpts)
        verdict = classify(det, feeder, rule)
        if verdict.is_feeding:
            per_frame_counts.setdefault(frame_idx, set()).add(track_id)
    counts = [len(s) for s in per_frame_counts.values()]
    return {
        "peak_count": max(counts) if counts else 0,
        "total_feeding_frames": sum(counts),
        "n_frames_with_any_feeding": len(counts),
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    rows = load_rows(args.tracks_csv, config.model.n_keypoints)
    if not rows:
        raise SystemExit(f"{args.tracks_csv}에서 유효한 track row를 찾지 못했습니다.")
    print(f"[info] {len(rows)} keypoint row 로드됨")

    angles = [float(a) for a in args.angles.split(",")]
    ratios = [float(r) for r in args.ring_outer_ratios.split(",")]

    results = []
    for angle, ratio in product(angles, ratios):
        feeder = FeederROI(
            cx=config.feeder.cx, cy=config.feeder.cy, r=config.feeder.r,
            ring_inner_ratio=config.feeder.ring_inner_ratio, ring_outer_ratio=ratio,
        )
        rule = FeedingRule(
            nose_index=config.feeding_rule.nose_index, neck_index=config.feeding_rule.neck_index,
            direction_angle_max_deg=angle, keypoint_conf_min=config.feeding_rule.keypoint_conf_min,
            min_bout_sec=config.feeding_rule.min_bout_sec,
        )
        metrics = evaluate(rows, feeder, rule)
        results.append({"angle_max_deg": angle, "ring_outer_ratio": ratio, **metrics})
        print(f"  angle<={angle:5.1f}  ring_outer={ratio:.2f}  -> peak={metrics['peak_count']:2d}  "
              f"total_feeding_frames={metrics['total_feeding_frames']}")

    out_csv = Path(args.out_csv) if args.out_csv else Path(args.tracks_csv).with_name(
        f"{Path(args.tracks_csv).stem}_sensitivity.csv"
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    peaks = {r["peak_count"] for r in results}
    print(f"\n[saved] {out_csv}")
    if len(peaks) == 1:
        print(f"[info] peak_count은 이 grid 전체에서 {peaks.pop()}으로 불변 -- 임계값에 강건함")
    else:
        print(f"[info] peak_count이 grid 안에서 {sorted(peaks)}로 변함 -- 결과가 임계값에 민감할 수 있음, "
              "단계 간 비교 전에 확인 필요")


if __name__ == "__main__":
    main()
