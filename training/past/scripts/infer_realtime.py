"""실시간(웹캠 / 영상 / RTSP) 돼지 자세 추정 데모.

스켈레톤 시각화는 scripts/utils/viz.py를 공유해서 사용한다(영상 파일 배치
처리용 scripts/track_video.py와 동일 로직).

사용법:
    python scripts/infer_realtime.py --source 0                     # 웹캠
    python scripts/infer_realtime.py --source path/to/video.mp4
    python scripts/infer_realtime.py --source rtsp://...
    python scripts/infer_realtime.py --size m                       # medium 모델 사용 (기본값)
"""
import argparse

import cv2
from ultralytics import YOLO

from utils.paths import DATA_YAML, RESULTS_DIR, SIZES
from utils.skeleton import load_pose_spec
from utils.viz import draw_pose, extract_pose_arrays


def main():
    parser = argparse.ArgumentParser(description="실시간 돼지 포즈 추정 데모")
    parser.add_argument("--source", default="0", help="웹캠 인덱스(0,1,...), 영상 파일 경로, 또는 RTSP URL")
    parser.add_argument("--size", default="m", choices=list(SIZES), help="사용할 모델 사이즈 (기본: medium)")
    parser.add_argument("--weights", default=None, help="직접 가중치 경로 지정 (지정 시 --size 무시)")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    weights = args.weights or str(RESULTS_DIR / args.size / "weights" / "best.pt")
    spec = load_pose_spec(DATA_YAML)
    model = YOLO(weights)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"영상 소스를 열 수 없습니다: {args.source}")

    print("[infer] 'q' 키를 누르면 종료합니다.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes_xyxy, box_conf, kpts_xy, kpts_conf, _ = extract_pose_arrays(result, spec["num_keypoints"])
        if len(boxes_xyxy):
            frame = draw_pose(frame, boxes_xyxy, kpts_xy, kpts_conf, spec["skeleton"], box_conf)

        cv2.imshow("Pig Pose - realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
