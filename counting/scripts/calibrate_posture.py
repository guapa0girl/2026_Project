"""Small human-labeled calibration for feeding_classifier's condition 3
(direction) and condition 4 (posture) -- the same "sample, human labels,
fit" shape as validate_counts.py, but at the per-detection posture level
instead of per-frame count level.

Why this exists: condition 4's ramp thresholds (posture_limb_low/high,
posture_aspect_low/high, posture_asym_low/high, posture_score_max) were
shipped as first-guess placeholders with no labeled data behind them (see
models/README.md's RF section for the contrast -- that one has ~940
hand-labeled frames). A 2026-08-25 review of real CH005 night footage found
condition 4 numerically inert (score architecturally capped below its own
threshold) and condition 3's 60 degree cutoff loose enough to admit a pig
resting nose-at-rim near the outer ring. Guessing a second/third set of
numbers blind repeats the same mistake (see the direction_angle_max_deg
60->62->60 episode) -- this tool instead pulls real candidates spanning the
full range (near-center/aligned to far/misaligned), a human labels each
"eating" / "lying" / "unsure", and `fit` derives thresholds from that.

Two-phase workflow:

  sample  Pull N detections that pass condition 1 (keypoint confidence) and
          sit within a generous distance band of the feeder (not filtered by
          the current angle/ring cutoffs -- that would bias the sample away
          from exactly the boundary cases calibration needs). Saves an
          annotated frame per detection plus labels.csv with every
          diagnostic feature already filled in and a blank `label` column.

  fit     Once labels.csv's `label` column is filled in (eating / lying /
          unsure -- leave unsure rows blank-ish, they're excluded), reports
          each feature's distribution split by class and a suggested
          threshold per feature (the cut point that best separates the two
          labeled classes on that single feature).

Usage:
    python scripts/calibrate_posture.py sample --config configs/farm_ch005_v2m_night.yaml \
        --source "data/raw/[CH005] 20260610015924-20260610025848_02.avi" --n 40 \
        --out-dir outputs/calibration/night_posture
    # ... open outputs/calibration/night_posture/frames/*.jpg, fill in labels.csv's label column ...
    python scripts/calibrate_posture.py fit --labels outputs/calibration/night_posture/labels.csv

    # to add more labels later: `sample` again into a NEW --out-dir (a second
    # round, or a different video/growth-stage), label that one too, then
    # `fit` with both paths -- rows from every file passed are pooled first:
    python scripts/calibrate_posture.py sample --config configs/farm_ch005_v2m_night.yaml \
        --source "data/raw/[CH005] 20260610015924-20260610025848_02.avi" --n 40 \
        --out-dir outputs/calibration/night_posture_round2
    python scripts/calibrate_posture.py fit \
        --labels outputs/calibration/night_posture/labels.csv \
                 outputs/calibration/night_posture_round2/labels.csv
"""
import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from feeding_monitor.config import load_config
from feeding_monitor.pose_tracker import Detection, PoseTracker, draw_detection
from feeding_monitor.posture import evaluate_posture

FEATURES = ["distance", "angle", "limb_visibility", "aspect_ratio", "lateral_asymmetry"]


def parse_args():
    p = argparse.ArgumentParser(description="Human-labeled calibration for direction/posture thresholds")
    sub = p.add_subparsers(dest="mode", required=True)

    ps = sub.add_parser("sample")
    ps.add_argument("--config", required=True)
    ps.add_argument("--source", required=True)
    ps.add_argument("--n", type=int, default=40)
    ps.add_argument("--out-dir", required=True)
    ps.add_argument("--min-gap-sec", type=float, default=45.0,
                     help="minimum time between samples of the *same spot* (dedupe a stationary pig getting resampled)")
    ps.add_argument("--dist-band", type=float, default=1.4,
                     help="sample detections up to this many x feeder.r from center (beyond the current ring) so boundary cases aren't excluded")

    pf = sub.add_parser("fit")
    pf.add_argument("--labels", required=True, nargs="+",
                     help="one or more labels.csv paths (e.g. from separate sample rounds, or day+night) -- rows from all of them are pooled before fitting")

    return p.parse_args()


def cmd_sample(args):
    config = load_config(args.config)
    rule = config.feeding_rule
    feeder = config.feeder

    cap = cv2.VideoCapture(args.source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 200.0
    cap.release()

    tracker = PoseTracker(config.model)
    cap = cv2.VideoCapture(args.source)

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # scan generously, keep the ones that pass condition 1 + the wide distance
    # band, then spread the final sample evenly across time so one stationary
    # pig can't dominate (see the 2026-08-25 finding: it did, at N=800 dense).
    SCAN_N = max(400, args.n * 15)
    last_pos_time = {}  # rounded (x,y) -> last frame_idx kept, for de-dup
    candidates = []

    for k in range(SCAN_N):
        t = int(k * total_frames / SCAN_N)
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame = cap.read()
        if not ok:
            continue
        result = tracker.model.predict(frame, conf=config.model.conf, imgsz=config.model.imgsz,
                                        device=tracker.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            continue
        kxy = result.keypoints.xy.cpu().numpy()
        kconf = result.keypoints.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        for i in range(len(result.boxes)):
            nx, ny = kxy[i, rule.nose_index]
            nc = kconf[i, rule.nose_index]
            kx, ky = kxy[i, rule.neck_index]
            ncf = kconf[i, rule.neck_index]
            if nc < rule.keypoint_conf_min or ncf < rule.keypoint_conf_min:
                continue
            distance = math.hypot(nx - feeder.cx, ny - feeder.cy)
            if distance > feeder.r * args.dist_band:
                continue

            pos_key = (round(nx / 40), round(ny / 40))
            if pos_key in last_pos_time and (t - last_pos_time[pos_key]) / fps < args.min_gap_sec:
                continue
            last_pos_time[pos_key] = t

            head_vec = (nx - kx, ny - ky)
            feeder_vec = (feeder.cx - nx, feeder.cy - ny)
            hn, fn = math.hypot(*head_vec), math.hypot(*feeder_vec)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, (head_vec[0]*feeder_vec[0]+head_vec[1]*feeder_vec[1])/(hn*fn))))) if hn > 1e-3 and fn > 1e-3 else None

            kpts = [(float(kxy[i, j, 0]), float(kxy[i, j, 1]), float(kconf[i, j])) for j in range(kxy.shape[1])]
            det = Detection(frame_idx=t, track_id=i, det_conf=1.0, box=tuple(boxes[i].tolist()), kpts=kpts)
            posture = evaluate_posture(kpts, rule)

            candidates.append({
                "t": t, "det": det, "frame": frame.copy(),
                "distance": distance, "angle": angle,
                "limb_visibility": posture.limb_visibility,
                "aspect_ratio": posture.aspect_ratio,
                "lateral_asymmetry": posture.lateral_asymmetry,
            })
    cap.release()

    print(f"[info] {len(candidates)} candidates after dedup, selecting {min(args.n, len(candidates))} spread across time")
    if len(candidates) > args.n:
        step = len(candidates) / args.n
        candidates = [candidates[int(i * step)] for i in range(args.n)]

    rows = []
    for idx, c in enumerate(candidates):
        annotated = c["frame"]
        draw_detection(annotated, c["det"], rule.keypoint_conf_min, is_feeding=True)
        img_name = f"sample_{idx:03d}_t{c['t']}.jpg"
        cv2.imwrite(str(frames_dir / img_name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append({
            "id": idx, "image": img_name, "frame": c["t"], "elapsed_sec": round(c["t"] / fps, 1),
            "distance": round(c["distance"], 1),
            "angle": round(c["angle"], 1) if c["angle"] is not None else "",
            "limb_visibility": round(c["limb_visibility"], 3) if c["limb_visibility"] is not None else "",
            "aspect_ratio": round(c["aspect_ratio"], 3) if c["aspect_ratio"] is not None else "",
            "lateral_asymmetry": round(c["lateral_asymmetry"], 4) if c["lateral_asymmetry"] is not None else "",
            "label": "",
        })

    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "image", "frame", "elapsed_sec"] + FEATURES + ["label"])
        w.writeheader()
        w.writerows(rows)

    print(f"[saved] {len(rows)} frames -> {frames_dir}")
    print(f"[saved] {labels_path}")
    print("다음: frames/*.jpg 를 보면서 labels.csv의 label 열을 채워주세요 (eating / lying / unsure)")


def cmd_fit(args):
    rows = []
    for path in args.labels:
        file_rows = list(csv.DictReader(open(path, encoding="utf-8")))
        print(f"[info] {path}: {len(file_rows)} rows")
        rows.extend(file_rows)
    eating = [r for r in rows if r["label"].strip().lower() == "eating"]
    lying = [r for r in rows if r["label"].strip().lower() == "lying"]
    unsure = [r for r in rows if r["label"].strip().lower() not in ("eating", "lying")]
    print(f"pooled across {len(args.labels)} file(s): eating={len(eating)}  lying={len(lying)}  unlabeled/unsure={len(unsure)}")
    if not eating or not lying:
        raise SystemExit("both 'eating' and 'lying' need at least one labeled row -- label more frames first")

    for feat in FEATURES:
        ev = sorted(float(r[feat]) for r in eating if r[feat] not in ("", None))
        lv = sorted(float(r[feat]) for r in lying if r[feat] not in ("", None))
        if not ev or not lv:
            print(f"{feat}: insufficient data (eating n={len(ev)}, lying n={len(lv)})")
            continue

        def stats(name, vals):
            n = len(vals)
            return f"{name} n={n} min={vals[0]:.3f} median={vals[n//2]:.3f} max={vals[-1]:.3f}"

        print(f"\n--- {feat} ---")
        print("  " + stats("eating", ev))
        print("  " + stats("lying", lv))

        # best single-feature split point: try every midpoint between
        # adjacent sorted values from both classes, keep the one with the
        # highest classification accuracy on this labeled set.
        all_vals = sorted(set(ev + lv))
        best_thresh, best_acc, best_dir = None, -1, None
        for a, b in zip(all_vals, all_vals[1:]):
            mid = (a + b) / 2
            for direction in ("lying_above", "lying_below"):
                if direction == "lying_above":
                    correct = sum(1 for v in ev if v <= mid) + sum(1 for v in lv if v > mid)
                else:
                    correct = sum(1 for v in ev if v > mid) + sum(1 for v in lv if v <= mid)
                acc = correct / (len(ev) + len(lv))
                if acc > best_acc:
                    best_acc, best_thresh, best_dir = acc, mid, direction
        if best_thresh is not None:
            rule = "lying if value > thresh" if best_dir == "lying_above" else "lying if value <= thresh"
            print(f"  suggested split: thresh={best_thresh:.3f} ({rule}), accuracy on labeled set={best_acc:.0%}")


def main():
    args = parse_args()
    if args.mode == "sample":
        cmd_sample(args)
    else:
        cmd_fit(args)


if __name__ == "__main__":
    main()
