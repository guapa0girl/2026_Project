"""Compare this project's geometric 3-condition classifier (feeding_classifier
.classify -- distance/direction/confidence against a circular ROI, already
baked into the "feeding" column of a *_tracks.csv from run_analysis.py) against
the ported RandomForest posture classifier (posture_classifier.py -- 6
features against polygon feeder zones, pretrained on ~940 hand-labeled frames
from the same farm camera).

Neither one has real human ground truth on this specific footage yet, so this
reports *agreement* between the two methods (where they disagree is exactly
where a human should look first -- see scripts/validate_counts.py) rather than
declaring a winner. Run validate_counts.py against both methods' predictions
for the actual accuracy comparison the two methods can't give each other.

Usage:
    python scripts/run_analysis.py --config configs/farm_ch005_20260610.yaml \
        --source "data/raw/[CH005] 202606100923_44.avi" --fps 200 --max-frames 2000 \
        --no-video --run-name farm_cmp
    python scripts/compare_classifiers.py --tracks-csv outputs/farm_cmp_tracks.csv \
        --zones configs/feeder_zones_ch005.yaml --fps 200
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_monitor.pose_tracker import Detection
from feeding_monitor.posture_classifier import MLPostureModel
from feeding_monitor.posture_features import load_zones


def parse_args():
    p = argparse.ArgumentParser(description="Compare the geometric and RandomForest feeding classifiers on an existing tracks CSV")
    p.add_argument("--tracks-csv", required=True, help="*_tracks.csv from run_analysis.py (needs frame-csv enabled, i.e. no --no-frame-csv)")
    p.add_argument("--zones", default=str(ROOT / "configs" / "zones" / "feeder_zones_ch005.yaml"))
    p.add_argument("--model", default=str(ROOT / "models" / "posture_random_forest.joblib"))
    p.add_argument("--fps", type=float, required=True, help="the fps run_analysis.py used for this CSV (for the time-series plot)")
    p.add_argument("--bin-sec", type=float, default=5.0)
    p.add_argument("--rf-threshold", type=float, default=0.5, help="RandomForest 'eating' probability cutoff -- lower is more lenient (e.g. 0.4)")
    p.add_argument("--out-prefix", default=None, help="default: <tracks-csv stem>_compare")
    return p.parse_args()


def load_rows(path, n_keypoints=18):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("track_id", "") == "":
                continue
            kpts = [(float(row[f"kpt{i}_x"]), float(row[f"kpt{i}_y"]), float(row[f"kpt{i}_conf"])) for i in range(n_keypoints)]
            det = Detection(
                frame_idx=int(row["frame"]), track_id=int(float(row["track_id"])), det_conf=float(row["det_conf"]),
                box=(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])), kpts=kpts,
            )
            geometric_feeding = row["feeding"].strip().lower() == "true"
            rows.append((det, geometric_feeding))
    return rows


def main():
    args = parse_args()
    zones = load_zones(args.zones)
    ml_model = MLPostureModel.load(args.model)
    rows = load_rows(args.tracks_csv)
    if not rows:
        raise SystemExit(f"{args.tracks_csv}에서 유효한 row를 찾지 못했습니다.")
    print(f"[info] {len(rows)}개 detection 로드됨")

    dets = [d for d, _ in rows]
    print("[info] RandomForest 일괄 예측 중...")
    ml_preds = ml_model.predict_batch(dets, zones, threshold=args.rf_threshold)

    both_yes = both_no = geo_only = ml_only = 0
    per_frame_geo = {}
    per_frame_ml = {}
    for (det, geo), ml in zip(rows, ml_preds):
        both_yes += geo and ml
        both_no += (not geo) and (not ml)
        geo_only += geo and not ml
        ml_only += ml and not geo
        if geo:
            per_frame_geo.setdefault(det.frame_idx, set()).add(det.track_id)
        if ml:
            per_frame_ml.setdefault(det.frame_idx, set()).add(det.track_id)

    n = len(rows)
    agree = both_yes + both_no
    print(f"\n===== 인스턴스 단위 일치도 =====")
    print(f"둘 다 섭취    : {both_yes:6d}  ({both_yes/n*100:5.1f}%)")
    print(f"둘 다 비섭취  : {both_no:6d}  ({both_no/n*100:5.1f}%)")
    print(f"기하학만 섭취 : {geo_only:6d}  ({geo_only/n*100:5.1f}%)  <- RandomForest는 아니라고 봄")
    print(f"RF만 섭취     : {ml_only:6d}  ({ml_only/n*100:5.1f}%)  <- 기하학은 아니라고 봄")
    print(f"전체 일치율   : {agree/n*100:.1f}%")

    all_frames = sorted(set(per_frame_geo) | set(per_frame_ml) | {d.frame_idx for d, _ in rows})
    geo_counts = [len(per_frame_geo.get(f, ())) for f in all_frames]
    ml_counts = [len(per_frame_ml.get(f, ())) for f in all_frames]

    bin_frames = max(1, round(args.fps * args.bin_sec))
    bins_x, geo_binned, ml_binned = [], [], []
    for start in range(0, len(all_frames), bin_frames):
        chunk_g = geo_counts[start:start + bin_frames]
        chunk_m = ml_counts[start:start + bin_frames]
        bins_x.append(all_frames[start] / args.fps)
        geo_binned.append(sum(chunk_g) / len(chunk_g))
        ml_binned.append(sum(chunk_m) / len(chunk_m))

    out_prefix = Path(args.out_prefix) if args.out_prefix else Path(args.tracks_csv).with_name(f"{Path(args.tracks_csv).stem}_compare")
    plt.figure(figsize=(10, 4.5))
    plt.plot(bins_x, geo_binned, marker="o", markersize=3, label="geometric (this project)", color="#2a78d6")
    plt.plot(bins_x, ml_binned, marker="o", markersize=3, label="RandomForest (ported)", color="#d6702a")
    plt.xlabel("elapsed time (s)")
    plt.ylabel("avg pigs feeding")
    plt.title("Geometric vs RandomForest feeding count")
    plt.legend()
    plt.grid(axis="y", color="#e1e0d9")
    plt.tight_layout()
    png_path = out_prefix.with_suffix(".png")
    plt.savefig(png_path, dpi=130)

    csv_path = out_prefix.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin_start_sec", "avg_pigs_geometric", "avg_pigs_randomforest"])
        for x, g, m in zip(bins_x, geo_binned, ml_binned):
            w.writerow([round(x, 1), round(g, 3), round(m, 3)])

    print(f"\n[saved] {png_path}")
    print(f"[saved] {csv_path}")
    print("\n참고: 이 결과는 두 방법이 서로 얼마나 일치하는지를 보여줄 뿐, 어느 쪽이 "
          "실제로 더 정확한지는 알려주지 않습니다. 사람이 직접 센 정답과 비교하려면 "
          "scripts/validate_counts.py를 쓰세요.")


if __name__ == "__main__":
    main()
