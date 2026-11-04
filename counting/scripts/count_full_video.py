"""Count feeding pigs across an entire long video by sampling one frame per
second via seek + single-frame inference, rather than decoding every frame
through a continuous tracking stream.

Why not just run run_analysis.py / render_comparison_video.py end to end:
their frame-by-frame streaming reader (ultralytics model.track(..., stream=
True)) stops early on this project's problem AVI files (confirmed: it quit
after ~188s of a requested 300s test, no error, likely a decode hiccup deep
in the corrupted-header file) even though direct seeks (cv2 VideoCapture.set
(POS_FRAMES)) reach much further into the same file without trouble (see
validate_counts.py's sampling, which reached t=1205s/20min fine). So this
seeks to one frame per second instead of decoding continuously -- robust to
that failure mode, and fast enough to cover a long file in a few minutes
instead of well over an hour.

Trade-off: no ByteTrack track IDs (each sampled frame is independent
single-frame inference), which is fine here -- a per-second *count* doesn't
need persistent per-pig identity, only "how many right now".

Two outputs (2026-08-25, requested: 1s-resolution raw data + coarser graph):
  <out-prefix>_1s.csv   one row per second, geo/rf count at that instant
  <out-prefix>.csv/.png the usual --bin-sec-binned summary + graph, now
                         aggregated FROM the 1s data via --bin-agg
                         (mean, the project's established "avg pigs
                         feeding" metric everywhere else -- or max, which
                         doesn't get diluted by a bin's quiet moments but
                         changes what the number means; not a drop-in
                         swap for downstream comparisons that assume mean)

Usage:
    python scripts/count_full_video.py --config configs/farm_ch005_20260610.yaml \
        --source "data/raw/[CH005] 202606100923_44.avi" --fps 200 --bin-sec 10 \
        --start-time 2026-06-10T10:09:23 \
        --zones configs/zones/feeder_zones_ch005.yaml --model models/posture_random_forest.joblib \
        --out-prefix outputs/0824/farm_ch005_full20min
"""

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from feeding_monitor.config import load_config
from feeding_monitor.feeding_classifier import classify
from feeding_monitor.pose_tracker import Detection, PoseTracker
from feeding_monitor.posture_classifier import MLPostureModel
from feeding_monitor.posture_features import load_zones


def parse_args():
    p = argparse.ArgumentParser(description="1-second-sampled feeding count over a full long video (geometric + RandomForest)")
    p.add_argument("--config", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--fps", type=float, required=True, help="real fps (see run_analysis.py --fps -- some DVR AVI headers lie)")
    p.add_argument("--bin-sec", type=float, default=10.0, help="graph/summary bin size in seconds -- the raw CSV is always 1s regardless")
    p.add_argument("--bin-agg", choices=["mean", "max"], default="mean",
                    help="how to collapse 1s samples into each --bin-sec bin for the summary/graph -- "
                         "mean matches the project's 'avg pigs feeding' metric used everywhere else "
                         "(compare_stages.py's shape normalization, the README's stated deliverable); "
                         "max keeps a bin's peak instead of averaging it away, but is then a different "
                         "metric, not directly comparable to other mean-based numbers")
    p.add_argument("--start-time", required=True, help="ISO datetime the recording started, e.g. 2026-06-10T10:09:23")
    p.add_argument("--zones", required=True)
    p.add_argument("--model", default=None,
                   help="RandomForest .joblib (optional). Omit -> rf column is left empty; "
                        "RF can be (re)trained later from <out-prefix>_dets.csv")
    p.add_argument("--rf-threshold", type=float, default=0.5, help="RandomForest 'eating' probability cutoff -- lower is more lenient (e.g. 0.4)")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--save-dets", action="store_true",
                   help="also write <out-prefix>_dets.csv: every detection's box + all keypoints (x,y,conf) per 1s sample, "
                        "so geo/RF/RF+Geo can be recomputed later (other ROI, new features) without re-running YOLO")
    return p.parse_args()


def detection_from_result(result, frame_idx):
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
            dets.append(Detection(frame_idx=frame_idx, track_id=i, det_conf=1.0,
                                   box=tuple(boxes[i].tolist()), kpts=kpts))
    return dets


def main():
    args = parse_args()
    config = load_config(args.config)
    zones = load_zones(args.zones)
    ml_model = MLPostureModel.load(args.model) if args.model else None
    start_time = datetime.fromisoformat(args.start_time)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.source}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        raise SystemExit(f"{args.source}의 프레임 수를 읽지 못했습니다.")

    total_sec = total_frames / args.fps
    n_seconds = int(total_sec) + 1
    print(f"[info] 총 {total_frames}프레임 / {args.fps}fps = {total_sec/60:.1f}분 -> "
          f"{n_seconds}개 1초 샘플 (그래프는 {args.bin_sec:.0f}초 단위로 {args.bin_agg} 집계)")

    tracker = PoseTracker(config.model)
    cap = cv2.VideoCapture(args.source)

    dets_f = dets_w = None
    if args.save_dets:
        dets_path = Path(str(args.out_prefix) + "_dets.csv")
        dets_path.parent.mkdir(parents=True, exist_ok=True)
        dets_f = open(dets_path, "w", newline="", encoding="utf-8")
        dets_w = csv.writer(dets_f)
        header = ["elapsed_sec", "wall_time", "det", "x1", "y1", "x2", "y2"]
        for j in range(config.model.n_keypoints):
            header += [f"k{j}_x", f"k{j}_y", f"k{j}_c"]
        dets_w.writerow(header + ["geo_feeding"])

    per_second = []  # one row per second: (sec, wall_time, geo_count, rf_count) or None on read failure
    n_read_fail = 0
    for sec in range(n_seconds):
        target = min(total_frames - 1, round(sec * args.fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:
            n_read_fail += 1
            per_second.append(None)
            continue
        result = tracker.model.predict(frame, conf=config.model.conf, imgsz=config.model.imgsz,
                                        device=tracker.device, verbose=False)[0]
        dets = detection_from_result(result, target)
        geo_flags = [classify(d, config.feeder, config.feeding_rule).is_feeding for d in dets]
        geo_count = sum(geo_flags)
        if ml_model is None:
            rf_count = float("nan")
        else:
            rf_count = sum(ml_model.predict_batch(dets, zones, threshold=args.rf_threshold)) if dets else 0
        wall_time = start_time + timedelta(seconds=sec)
        per_second.append({"sec": sec, "wall_time": wall_time, "geo": geo_count, "rf": rf_count})
        if dets_w is not None:
            for i, d in enumerate(dets):
                row = [sec, wall_time.isoformat(), i] + [round(v, 1) for v in d.box]
                for (x, y, c) in d.kpts:
                    row += [round(x, 1), round(y, 1), round(c, 3)]
                dets_w.writerow(row + [int(geo_flags[i])])
            if sec % 60 == 0:
                dets_f.flush()
        if sec % 120 == 0:
            print(f"  [{sec+1}/{n_seconds}] {wall_time:%H:%M:%S}  geo={geo_count}  rf={rf_count}")

    cap.release()
    if dets_f is not None:
        dets_f.close()
    if n_read_fail:
        print(f"[warn] {n_read_fail}개 프레임 seek/read 실패 (건너뜀)")

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # --- 1s raw CSV -----------------------------------------------------
    csv_1s_path = Path(str(out_prefix) + "_1s.csv")
    with open(csv_1s_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_sec", "wall_time", "pigs_geometric", "pigs_randomforest"])
        for r in per_second:
            if r is None:
                continue
            w.writerow([r["sec"], r["wall_time"].isoformat(), r["geo"], r["rf"]])

    # --- bin-sec summary (mean or max of the 1s samples in each bin) ----
    agg = max if args.bin_agg == "max" else (lambda vals: sum(vals) / len(vals))
    n_bins = int(total_sec // args.bin_sec) + 1
    rows = []
    for b in range(n_bins):
        bin_start_sec = b * args.bin_sec
        bin_secs = [r for r in per_second if r is not None and bin_start_sec <= r["sec"] < bin_start_sec + args.bin_sec]
        if not bin_secs:
            continue
        rows.append({
            "bin_start_sec": bin_start_sec,
            "wall_time": start_time + timedelta(seconds=bin_start_sec),
            "n_samples_ok": len(bin_secs),
            "avg_geo": agg([r["geo"] for r in bin_secs]),
            "avg_rf": agg([r["rf"] for r in bin_secs]),
        })

    csv_path = out_prefix.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin_start_sec", "wall_time", "n_samples_ok", "avg_pigs_geometric", "avg_pigs_randomforest"])
        for r in rows:
            w.writerow([r["bin_start_sec"], r["wall_time"].isoformat(), r["n_samples_ok"],
                        round(r["avg_geo"], 3), round(r["avg_rf"], 3)])

    times = [r["wall_time"] for r in rows]
    geo_vals = [r["avg_geo"] for r in rows]
    rf_vals = [r["avg_rf"] for r in rows]

    plt.figure(figsize=(14, 4.8))
    plt.plot(times, geo_vals, marker="o", markersize=2, label="geometric (this project)", color="#2a78d6")
    plt.plot(times, rf_vals, marker="o", markersize=2, label="RandomForest (ported)", color="#d6702a")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=2))
    ax.grid(which="major", axis="x", color="#c7c6bd", linewidth=0.9)
    ax.grid(which="minor", axis="x", color="#e8e7e0", linewidth=0.6)
    ax.grid(which="major", axis="y", color="#e1e0d9")
    plt.gcf().autofmt_xdate()
    plt.xlabel("time of day")
    plt.ylabel(f"pigs feeding ({args.bin_agg} per {args.bin_sec:.0f}s)")
    plt.title(f"Feeding count over ~{total_sec/60:.0f} min -- {config.pen_id} ({args.bin_sec:.0f}s bins, {args.bin_agg})")
    plt.legend()
    plt.tight_layout()
    png_path = out_prefix.with_suffix(".png")
    plt.savefig(png_path, dpi=130)

    valid_geo = [v for v in geo_vals if v == v]
    valid_rf = [v for v in rf_vals if v == v]
    print(f"\n===== 요약 ({len(rows)}개 {args.bin_sec:.0f}초 구간, {total_sec/60:.1f}분) =====")
    print(f"geometric : peak={max(valid_geo):.2f}  mean={sum(valid_geo)/len(valid_geo):.2f}")
    if valid_rf:
        print(f"randomforest : peak={max(valid_rf):.2f}  mean={sum(valid_rf)/len(valid_rf):.2f}")
    else:
        print("randomforest : (no --model given, skipped)")
    print(f"\n[saved] {csv_1s_path}  (1초 단위 원본)")
    print(f"[saved] {csv_path}  ({args.bin_sec:.0f}초 {args.bin_agg} 집계)")
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()
