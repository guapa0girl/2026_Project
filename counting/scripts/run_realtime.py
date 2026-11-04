"""Live entrypoint: run the same pipeline over an RTSP stream or webcam,
showing a live "Count : N" overlay, printing one line per closed bin to the
terminal, and appending that same line to a continuously-growing .txt log
(scripts/plot_live_log.py later turns that log into a feeding-by-time-of-day
graph). This is the intended deployment path once a pen camera is
available -- offline video (run_analysis.py) exists to develop and validate
the pipeline before pointing it at a live feed.

Set aggregation.bin_size: 1s in the pen config (see configs/pen01_realtime.yaml)
so a bin closes -- and a log line is written -- once per second.

Every bin is timestamped with the real wall clock (datetime.now()) at the
moment it closes, NOT the aggregator's frame-count/assumed-fps elapsed time
-- an RTSP stream's actual delivery/processing rate will not match a guessed
--fps, and that mismatch compounds over a day into a multi-hour timestamp
drift that would silently move this study's actual result (which hour pigs
feed most). See src/feeding_monitor/live_log.py's docstring.

Usage:
    python scripts/run_realtime.py --config configs/pen01_realtime.yaml --source rtsp://192.168.0.10/stream1
    python scripts/run_realtime.py --config configs/pen01_realtime.yaml --source 0 --fps 15   # webcam, index 0
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from feeding_monitor.config import load_config
from feeding_monitor.feeder_roi import draw_calibration
from feeding_monitor.live_log import LiveLogWriter
from feeding_monitor.pipeline import run_pipeline
from feeding_monitor.report import plot_timeseries, save_csv


def parse_args():
    p = argparse.ArgumentParser(description="Run the feeding-time pipeline over a live stream")
    p.add_argument("--config", required=True)
    p.add_argument("--source", required=True, help="RTSP URL, or an integer webcam index")
    p.add_argument("--fps", type=float, default=None, help="assumed fps, only used to size the smoothing window -- NOT used for log timestamps (those are real wall-clock)")
    p.add_argument("--out-dir", default=str(ROOT / "outputs" / "realtime"))
    p.add_argument("--run-name", default=None, help="default: <config_name>__<start_timestamp>")
    p.add_argument("--log-file", default=None, help="default: <out-dir>/<run-name>_live_log.txt")
    p.add_argument("--no-video", action="store_true", help="skip writing the annotated video file")
    p.add_argument("--no-show", action="store_true", help="don't open a live preview window (headless)")
    p.add_argument("--replot-every", type=int, default=6, help="regenerate the trend PNG every N closed bins")
    p.add_argument("--calib-snapshot-every", type=int, default=0,
                    help="save a feeder-ROI overlay snapshot every N frames, 0 to disable "
                         "(sanity-check that the ROI hasn't drifted from a bumped/cleaned camera)")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    source = int(args.source) if args.source.isdigit() else args.source
    fps = args.fps or 15.0
    run_start = datetime.now()
    run_name = args.run_name or f"{Path(args.config).stem}__{run_start:%Y%m%d_%H%M%S}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file else out_dir / f"{run_name}_live_log.txt"
    calib_dir = out_dir / f"{run_name}_calib_snapshots"

    print(f"[info] pen={config.pen_id} stage={config.growth_stage} source={source} fps(smoothing only)={fps}")
    print(f"[info] bin_size={config.aggregation.bin_size} -- one log line per closed bin (real wall-clock timestamp), Ctrl+C (or 'q' in the preview window) to stop")
    print(f"[info] live log: {log_path}")

    all_bins = []
    all_bin_times = []  # true wall-clock per bin -- see live_log.py docstring on why frame-count timing isn't trustworthy live
    live_logger = LiveLogWriter(log_path, start_wall_time=run_start)

    def on_bin(b):
        now = datetime.now()
        all_bins.append(b)
        all_bin_times.append(now)
        live_logger.write(b, ts=now)
        if len(all_bins) % args.replot_every == 0:
            save_csv(all_bins, out_dir / f"{run_name}_feeding_binned.csv", timestamps=all_bin_times)
            plot_timeseries(
                all_bins, out_dir / f"{run_name}_feeding_count.png",
                title=f"Feeding count (live) -- {config.pen_id} ({config.growth_stage})",
                timestamps=all_bin_times,
            )

    on_frame = None
    if args.calib_snapshot_every > 0:
        calib_dir.mkdir(parents=True, exist_ok=True)

        def on_frame(frame_idx, frame):
            if frame_idx % args.calib_snapshot_every == 0:
                snap = draw_calibration(frame, config.feeder)
                cv2.imwrite(str(calib_dir / f"frame{frame_idx:08d}_{datetime.now():%Y%m%d_%H%M%S}.png"), snap)

    try:
        run_pipeline(
            config=config,
            source=source,
            fps=fps,
            out_dir=out_dir,
            run_name=run_name,
            save_video=not args.no_video,
            save_frame_csv=False,
            show=not args.no_show,
            window_name=f"{config.pen_id} -- live feeding count",
            on_bin=on_bin,
            on_frame=on_frame,
        )
    except KeyboardInterrupt:
        print("\n[info] stopped by user")
    finally:
        live_logger.close()
        if all_bins:
            save_csv(all_bins, out_dir / f"{run_name}_feeding_binned.csv", timestamps=all_bin_times)
            plot_timeseries(
                all_bins, out_dir / f"{run_name}_feeding_count.png",
                title=f"Feeding count (live) -- {config.pen_id} ({config.growth_stage})",
                timestamps=all_bin_times,
            )
            print(f"[saved] {out_dir / f'{run_name}_feeding_binned.csv'}")
        print(f"[saved] {log_path}")


if __name__ == "__main__":
    main()
