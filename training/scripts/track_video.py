"""영상(기본값: 프로젝트 루트의 test_video.avi)에 대해 n/s/m 각 모델로 돼지
탐지 + 추적(tracking) + 포즈 추정(18키포인트 스켈레톤)을 수행하고, 결과를
video_result/<size>/ 에 샘플 영상(mp4)으로 저장한다.

Ultralytics 내장 트래커(기본 ByteTrack)로 프레임 간 같은 개체에 고유
track id(#N)를 부여해서 박스 위에 함께 표시한다. 원본 영상이 매우 길어서
(테스트 영상 기준 약 995초) 3개 모델 전체 길이를 다 돌리면 오래 걸리므로,
기본적으로 앞부분 --max-frames 프레임만 샘플로 뽑아 처리한다.

사용법:
    python scripts/track_video.py                                # n/s/m 전부, 기본 900프레임 샘플
    python scripts/track_video.py --sizes m                      # medium만
    python scripts/track_video.py --max-frames 1800 --start-frame 3000
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils.paths import DATA_YAML, RESULTS_DIR, ROOT, SIZES, VIDEO_RESULT_DIR
from utils.skeleton import load_pose_spec
from utils.viz import draw_pose, extract_pose_arrays

# 일부 소스 영상이 헤더 파싱 문제로 비정상적인 fps(예: 200)를 보고하는 경우가
# 있어, 이 범위를 벗어나면 출력 영상은 안전한 기본값으로 인코딩한다.
FPS_FALLBACK = 25.0
FPS_SANE_RANGE = (1.0, 60.0)


def process_one(size, source, max_frames, start_frame, conf, imgsz, tracker):
    weights = RESULTS_DIR / size / "weights" / "best.pt"
    if not weights.exists():
        print(f"[track] {size}: 가중치가 없습니다 ({weights}) - 건너뜁니다. 먼저 scripts/train.py를 실행하세요.")
        return None

    spec = load_pose_spec(DATA_YAML)
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = src_fps if FPS_SANE_RANGE[0] <= src_fps <= FPS_SANE_RANGE[1] else FPS_FALLBACK
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_dir = VIDEO_RESULT_DIR / size
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(source).stem}_{size}_sample.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    n_written, n_with_detections = 0, 0
    while n_written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.track(
            frame, persist=True, conf=conf, imgsz=imgsz, tracker=tracker, verbose=False,
        )[0]
        boxes_xyxy, box_conf, kpts_xy, kpts_conf, track_ids = extract_pose_arrays(result, spec["num_keypoints"])
        if len(boxes_xyxy):
            n_with_detections += 1
            frame = draw_pose(frame, boxes_xyxy, kpts_xy, kpts_conf, spec["skeleton"], box_conf, track_ids)

        writer.write(frame)
        n_written += 1

    cap.release()
    writer.release()

    print(
        f"[track] {size}: {n_written}프레임 처리(탐지 있던 프레임 {n_with_detections}) "
        f"-> {out_path} (fps={out_fps:g})"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="test_video를 n/s/m 모델로 추적+포즈 추정해서 샘플 영상 저장")
    parser.add_argument("--source", type=Path, default=ROOT / "test_video.avi")
    parser.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    parser.add_argument("--max-frames", type=int, default=900, help="모델당 처리할 최대 프레임 수 (샘플 길이)")
    parser.add_argument("--start-frame", type=int, default=0, help="영상의 몇 번째 프레임부터 샘플링할지")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tracker", default="bytetrack.yaml", help="ultralytics 내장 트래커 설정 (bytetrack.yaml / botsort.yaml)")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"소스 영상을 찾을 수 없습니다: {args.source}")

    for size in args.sizes:
        process_one(size, args.source, args.max_frames, args.start_frame, args.conf, args.imgsz, args.tracker)


if __name__ == "__main__":
    main()
