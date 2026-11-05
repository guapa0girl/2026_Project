"""Offline entrypoint: run the full pipeline over a recorded video file.

Also useful to validate the live-monitoring setup (the "Count : N" overlay,
the per-bin .txt log) against a recorded clip before pointing
run_realtime.py at an actual camera -- pass --show and/or --live-log with a
config whose aggregation.bin_size is 1s.

Usage:
    python scripts/run_analysis.py --config configs/pen01.yaml --source data/raw/pen01_2026-08-19.mp4
    python scripts/run_analysis.py --config configs/pen01.yaml --source data/raw/clip.mp4 --no-video
    python scripts/run_analysis.py --config configs/pen01_realtime.yaml --source data/raw/clip.mp4 --show --live-log
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_monitor.config import load_config
from feeding_monitor.live_log import LiveLogWriter
from feeding_monitor.pipeline import probe_video, run_pipeline
from feeding_monitor.report import plot_timeseries, save_csv, summarize


def parse_args():
    p = argparse.ArgumentParser(description="Run the feeding-time pipeline over a recorded video")
    p.add_argument("--config", required=True, help="pen config YAML, e.g. configs/pen01.yaml")
    p.add_argument("--source", required=True, help="video file to analyze")
    p.add_argument("--out-dir", default=str(ROOT / "outputs"))
    p.add_argument("--run-name", default=None, help="default: <config_name>__<video_stem>")
    p.add_argument("--no-video", action="store_true", help="skip writing the annotated video (faster)")
    p.add_argument("--no-frame-csv", action="store_true", help="skip the per-frame/per-keypoint CSV")
    p.add_argument("--show", action="store_true", help="open a live preview window with the Count : N overlay")
    p.add_argument("--live-log", action="store_true", help="also write the per-bin .txt log (see run_realtime.py)")
    p.add_argument("--fps", type=float, default=None,
                    help="override the container's reported fps -- some DVR/CCTV AVI exports have a corrupted "
                         "header that reports a nonsense fps (e.g. 200); pass the camera's real fps here")
    p.add_argument("--max-frames", type=int, default=None,
                    help="stop after N frames (a quick test run before committing to the whole file)")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    source = Path(args.source)
    run_name = args.run_name or f"{Path(args.config).stem}__{source.stem}"
    out_dir = Path(args.out_dir)

    probed_fps, width, height = probe_video(str(source))
    fps = args.fps if args.fps is not None else probed_fps
    if args.fps is not None and abs(args.fps - probed_fps) > 0.5:
        print(f"[warn] overriding container fps={probed_fps:.1f} with --fps={args.fps:.1f}")
    print(f"[info] pen={config.pen_id} stage={config.growth_stage} source={source.name} fps={fps:.1f} size={width}x{height}")
    print(f"[info] model={config.model.weights}")
    print(f"[info] bin_size={config.aggregation.bin_size} ({config.aggregation.bin_size_sec:.0f}s)  "
          f"smooth={config.aggregation.smooth_sec}s")

    live_logger = None
    on_bin = None
    if args.live_log:
        log_path = out_dir / f"{run_name}_live_log.txt"
        # bin_start_sec is accurate here (a recorded file's fps is real
        # metadata, not a live guess -- see live_log.py's docstring), so
        # anchoring it to the video's real start time (if configured) gives
        # correct absolute timestamps for testing the hour-of-day analysis
        # against known footage. Falls back to "now" if that's not set --
        # fine for just eyeballing the mechanics, not for real analysis.
        start = (
            datetime.fromisoformat(config.aggregation.video_start_time)
            if config.aggregation.video_start_time
            else datetime.now()
        )
        live_logger = LiveLogWriter(log_path, start_wall_time=start)
        on_bin = live_logger.write
        print(f"[info] live log: {log_path}  (timestamps anchored to "
              f"{'video_start_time' if config.aggregation.video_start_time else 'now() -- set aggregation.video_start_time in the config for accurate timestamps'})")

    try:
        bins = run_pipeline(
            config=config,
            source=str(source),
            fps=fps,
            out_dir=out_dir,
            run_name=run_name,
            save_video=not args.no_video,
            save_frame_csv=not args.no_frame_csv,
            show=args.show,
            frame_size=(height, width),
            on_bin=on_bin,
            max_frames=args.max_frames,
        )
    finally:
        if live_logger is not None:
            live_logger.close()

    csv_path = out_dir / f"{run_name}_feeding_binned.csv"
    png_path = out_dir / f"{run_name}_feeding_count.png"
    save_csv(bins, csv_path)
    plot_timeseries(
        bins, png_path,
        title=f"Feeding count over time -- {config.pen_id} ({config.growth_stage})",
        video_start_time=config.aggregation.video_start_time,
    )

    stats = summarize(bins, fps)
    if config.expected_pig_count is not None:
        print(f"[check] expected pig count: {config.expected_pig_count} (compare against the tracked video / unique IDs)")
    print("\n===== summary =====")
    print(f"peak avg pigs feeding : {stats['peak_avg_pigs']}")
    print(f"total pig-seconds     : {stats['pig_seconds']}")
    print(f"bins                  : {stats['n_bins']}")
    print(f"\n[saved] {csv_path}")
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()
