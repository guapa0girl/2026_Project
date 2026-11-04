"""Ground-truth validation for the feeding COUNT (not keypoint accuracy --
training/evaluate_pose.py already covers that, and it's a different thing:
pose can be accurate while a classifier's thresholds still get the count
wrong). There's no RFID ground truth here, so this samples frames for a
human to label "how many pigs are feeding right now", then scores each
classifier's prediction against that -- the actual apples-to-apples
comparison scripts/compare_classifiers.py can't give on its own, since it
only measures how much the two methods agree with *each other*.

Two-phase workflow:

  sample  Pull N frames from a video at spread-out timestamps (stratified
          across the video, so both busy and quiet moments are covered),
          save them as images plus a CSV with each configured method's
          predicted count already filled in -- a human fills in
          `human_count` next to them by eye. Pass --ml-model/--zones to
          also predict with the ported RandomForest classifier alongside
          this project's geometric one; omit them to score the geometric
          method alone.

  score   Once the CSV's human_count column is filled in, compute MAE and
          basic precision/recall (treating "count > 0" as the binary
          feeding/not-feeding event) for every predicted_count_* column
          found -- so with both methods sampled, this directly answers
          "which one is more accurate," side by side.

Run this per growth stage/pen separately -- a detector that scores well in
one stage's lighting/pig-size/feeder shape is not guaranteed to score well
in another, and averaging across stages would hide that.

Usage:
    python scripts/validate_counts.py sample --config configs/pen01.yaml --source data/raw/clip.mp4 --n 20 --out-dir outputs/validation/pen01
    python scripts/validate_counts.py sample --config configs/farm_ch005_20260610.yaml --source "data/raw/[CH005] 202606100923_44.avi" \
        --fps 200 --n 20 --out-dir outputs/validation/farm_ch005 \
        --ml-model models/posture_random_forest.joblib --zones configs/zones/feeder_zones_ch005.yaml
    # ... open outputs/validation/<pen>/frames/*.jpg, fill in the human_count column of labels.csv ...
    python scripts/validate_counts.py score --labels outputs/validation/farm_ch005/labels.csv
"""

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from feeding_monitor.config import load_config
from feeding_monitor.feeding_classifier import classify
from feeding_monitor.pipeline import probe_video
from feeding_monitor.pose_tracker import Detection, PoseTracker

METHOD_PREFIX = "predicted_count_"


def parse_args():
    p = argparse.ArgumentParser(description="Sample frames for human ground-truth labeling, then score predicted vs human count")
    sub = p.add_subparsers(dest="mode", required=True)

    ps = sub.add_parser("sample", help="extract frames + predicted counts for a human to label")
    ps.add_argument("--config", required=True)
    ps.add_argument("--source", required=True)
    ps.add_argument("--n", type=int, default=20, help="how many frames to sample")
    ps.add_argument("--out-dir", required=True)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--fps", type=float, default=None, help="override container fps (see run_analysis.py --fps -- some DVR AVI headers are wrong)")
    ps.add_argument("--ml-model", default=None, help="also predict with the ported RandomForest posture classifier, e.g. models/posture_random_forest.joblib")
    ps.add_argument("--zones", default=None, help="feeder zone polygon yaml for --ml-model, e.g. configs/zones/feeder_zones_ch005.yaml")
    ps.add_argument("--rf-threshold", type=float, default=0.5, help="RandomForest 'eating' probability cutoff -- lower is more lenient (e.g. 0.4)")

    pc = sub.add_parser("score", help="compute MAE / precision / recall per method from a filled-in labels.csv")
    pc.add_argument("--labels", required=True)

    return p.parse_args()


def cmd_sample(args):
    config = load_config(args.config)
    probed_fps, width, height = probe_video(args.source)
    fps = args.fps if args.fps is not None else probed_fps
    total_frames = int(cv2.VideoCapture(args.source).get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise SystemExit(f"{args.source}의 프레임 수를 읽지 못했습니다.")

    ml_model = zones = None
    if args.ml_model:
        if not args.zones:
            raise SystemExit("--ml-model을 쓰려면 --zones도 필요합니다.")
        from feeding_monitor.posture_classifier import MLPostureModel
        from feeding_monitor.posture_features import load_zones
        ml_model = MLPostureModel.load(args.ml_model)
        zones = load_zones(args.zones)
        print(f"[info] RandomForest 예측도 같이 기록: {args.ml_model}")

    random.seed(args.seed)
    # stratified: split the video into n equal-length segments, one random frame per segment,
    # so both busy and quiet moments have a chance of being sampled instead of clustering.
    segment_len = max(1, total_frames // args.n)
    target_frames = sorted(
        min(total_frames - 1, seg * segment_len + random.randint(0, max(0, segment_len - 1)))
        for seg in range(args.n)
    )

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    tracker = PoseTracker(config.model)
    cap = cv2.VideoCapture(args.source)
    rows = []
    skipped = 0
    for target in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            skipped += 1
            continue
        # single-frame inference (no tracking needed for a one-off count)
        result = tracker.model.predict(frame, conf=config.model.conf, imgsz=config.model.imgsz,
                                        device=tracker.device, verbose=False)[0]
        geo_count = 0
        dets = []
        if result.boxes is not None and len(result.boxes) > 0 and result.keypoints is not None:
            kxy = result.keypoints.xy.cpu().numpy()
            kconf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            boxes = result.boxes.xyxy.cpu().numpy()
            for i in range(len(result.boxes)):
                kpts = [
                    (float(kxy[i, j, 0]), float(kxy[i, j, 1]), float(kconf[i, j]) if kconf is not None else 1.0)
                    for j in range(kxy.shape[1])
                ]
                det = Detection(frame_idx=target, track_id=i, det_conf=1.0, box=tuple(boxes[i].tolist()), kpts=kpts)
                dets.append(det)
                if classify(det, config.feeder, config.feeding_rule).is_feeding:
                    geo_count += 1

        img_name = f"frame_{target:08d}_t{target / fps:07.1f}s.jpg"
        cv2.imwrite(str(frames_dir / img_name), frame)
        row = {"frame": target, "elapsed_sec": round(target / fps, 1), "image": img_name,
               f"{METHOD_PREFIX}geometric": geo_count}
        if ml_model is not None:
            ml_preds = ml_model.predict_batch(dets, zones, threshold=args.rf_threshold)
            row[f"{METHOD_PREFIX}randomforest"] = sum(ml_preds)
        row["human_count"] = ""
        rows.append(row)
    cap.release()

    if skipped:
        print(f"[warn] {skipped}개 프레임을 못 읽어서 건너뜀 (seek 실패 -- 헤더 손상된 파일이면 흔함)")

    fieldnames = ["frame", "elapsed_sec", "image", f"{METHOD_PREFIX}geometric"]
    if ml_model is not None:
        fieldnames.append(f"{METHOD_PREFIX}randomforest")
    fieldnames.append("human_count")

    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[saved] {len(rows)}장 -> {frames_dir}")
    print(f"[saved] {labels_path}")
    print(f"다음 단계: {frames_dir} 안 이미지를 보면서 labels.csv의 human_count 열을 채운 뒤, "
          f"'python scripts/validate_counts.py score --labels {labels_path}' 실행")


def cmd_score(args):
    with open(args.labels, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.labels}에 행이 없습니다.")

    unlabeled = [r["image"] for r in rows if r["human_count"].strip() == ""]
    if unlabeled:
        raise SystemExit(f"human_count이 비어있는 행이 {len(unlabeled)}개 있습니다 (예: {unlabeled[:3]}). 전부 채운 뒤 다시 실행하세요.")

    methods = [c[len(METHOD_PREFIX):] for c in rows[0].keys() if c.startswith(METHOD_PREFIX)]
    if not methods:
        raise SystemExit(f"{args.labels}에 {METHOD_PREFIX}* 컬럼이 없습니다.")

    print(f"n = {len(rows)}")
    for method in methods:
        col = f"{METHOD_PREFIX}{method}"
        errors, tp, fp, fn, tn = [], 0, 0, 0, 0
        for r in rows:
            pred, human = int(r[col]), int(r["human_count"])
            errors.append(abs(pred - human))
            pred_pos, human_pos = pred > 0, human > 0
            tp += pred_pos and human_pos
            fp += pred_pos and not human_pos
            fn += (not pred_pos) and human_pos
            tn += (not pred_pos) and (not human_pos)

        mae = sum(errors) / len(errors)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")

        print(f"\n===== {method} =====")
        print(f"MAE (count) = {mae:.2f}")
        print(f"'is anyone feeding' 이진 분류: precision={precision:.2f}  recall={recall:.2f}  "
              f"(tp={tp} fp={fp} fn={fn} tn={tn})")
        worst = sorted(rows, key=lambda r: abs(int(r[col]) - int(r["human_count"])), reverse=True)[:5]
        print("가장 많이 틀린 5개:")
        for r in worst:
            print(f"  {r['image']}: predicted={r[col]} human={r['human_count']}")

    if len(methods) > 1:
        print(f"\n위 MAE/precision/recall을 비교해서 이 데이터에서 더 정확한 쪽을 고르세요.")


def main():
    args = parse_args()
    if args.mode == "sample":
        cmd_sample(args)
    else:
        cmd_score(args)


if __name__ == "__main__":
    main()
