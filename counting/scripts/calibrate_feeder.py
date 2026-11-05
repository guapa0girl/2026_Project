"""Visual feeder-ROI calibration: pull one frame from a video, auto-detect
the feeder's circular lid (red, by default), draw the ring used by the
feeding classifier on top, and print a YAML snippet to paste into a pen
config. Re-run with --cx/--cy/--r to override the auto-detection by hand
and preview the result before committing it to configs/*.yaml.

Usage:
    python scripts/calibrate_feeder.py --source "data/raw/[CH010] ....avi" --at-sec 15
    python scripts/calibrate_feeder.py --source "data/raw/[CH010] ....avi" --at-sec 15 --cx 925 --cy 535 --r 338
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_monitor.config import FeederROI
from feeding_monitor.feeder_roi import detect_feeder_circle, draw_calibration


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate the circular feeder ROI against a sample frame")
    p.add_argument("--source", required=True, help="video to grab a frame from")
    p.add_argument("--at-sec", type=float, default=15.0, help="timestamp to sample")
    p.add_argument("--cx", type=float, default=None, help="manual override: center x")
    p.add_argument("--cy", type=float, default=None, help="manual override: center y")
    p.add_argument("--r", type=float, default=None, help="manual override: radius")
    p.add_argument("--ring-inner-ratio", type=float, default=0.20)
    p.add_argument("--ring-outer-ratio", type=float, default=1.10)
    p.add_argument("--out", default=str(ROOT / "outputs" / "feeder_calib.png"))
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.at_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"{args.at_sec}초 지점 프레임을 읽지 못했습니다.")

    if args.cx is not None and args.cy is not None and args.r is not None:
        feeder = FeederROI(cx=args.cx, cy=args.cy, r=args.r,
                            ring_inner_ratio=args.ring_inner_ratio, ring_outer_ratio=args.ring_outer_ratio)
        print("[info] 수동 값 사용")
    else:
        detected = detect_feeder_circle(frame)
        if detected is None:
            raise SystemExit(
                "자동 검출 실패 (빨간 뚜껑을 못 찾음). --cx/--cy/--r 로 수동 지정하세요."
            )
        feeder = FeederROI(cx=detected.cx, cy=detected.cy, r=detected.r,
                            ring_inner_ratio=args.ring_inner_ratio, ring_outer_ratio=args.ring_outer_ratio)
        print(f"[info] 자동 검출: cx={feeder.cx:.1f} cy={feeder.cy:.1f} r={feeder.r:.1f}")
        print("       (빗나갔으면 --cx/--cy/--r 로 다시 지정해서 확인)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), draw_calibration(frame, feeder))
    print(f"[saved] {out_path}  -- 열어서 노란 링이 급이 판정 영역과 맞는지 확인하세요")

    print("\n이 값을 pen config의 feeder: 항목에 붙여넣으세요:")
    print("feeder:")
    print(f"  cx: {feeder.cx:.1f}")
    print(f"  cy: {feeder.cy:.1f}")
    print(f"  r: {feeder.r:.1f}")
    print(f"  ring_inner_ratio: {feeder.ring_inner_ratio}")
    print(f"  ring_outer_ratio: {feeder.ring_outer_ratio}")


if __name__ == "__main__":
    main()
