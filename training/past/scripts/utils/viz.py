"""탐지/추적/포즈 결과를 프레임 위에 그리는 공용 시각화 헬퍼.
infer_realtime.py(웹캠)와 track_video.py(영상 파일) 둘 다 이 함수를 사용한다.
Ultralytics 기본 시각화는 COCO 17-keypoint 스켈레톤을 기준으로 하기 때문에,
우리 프로젝트의 18-keypoint 커스텀 스켈레톤(configs/pig_pose.yaml)을 직접 그린다."""
import cv2

KPT_CONF_THRES = 0.5


def draw_pose(frame, boxes_xyxy, kpts_xy, kpts_conf, skeleton, box_conf, track_ids=None):
    """frame 위에 bbox + 18키포인트 스켈레톤을 그려서 반환한다.
    track_ids를 주면 박스 라벨에 추적 ID(#N)를 함께 표시한다."""
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = map(int, boxes_xyxy[i])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

        tag = f" #{int(track_ids[i])}" if track_ids is not None else ""
        cv2.putText(frame, f"pig{tag} {box_conf[i]:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        pts = kpts_xy[i]
        confs = kpts_conf[i]

        for a, b in skeleton:
            if confs[a] < KPT_CONF_THRES or confs[b] < KPT_CONF_THRES:
                continue
            pa = tuple(map(int, pts[a]))
            pb = tuple(map(int, pts[b]))
            cv2.line(frame, pa, pb, (60, 220, 60), 2)

        for k in range(len(pts)):
            if confs[k] < KPT_CONF_THRES:
                continue
            cv2.circle(frame, tuple(map(int, pts[k])), 4, (0, 0, 255), -1)
    return frame


def extract_pose_arrays(result, num_keypoints):
    """ultralytics Results 객체 -> (boxes_xyxy, box_conf, kpts_xy, kpts_conf, track_ids) numpy 배열.
    탐지된 개체가 없으면 빈 배열을 반환한다."""
    import numpy as np

    if result.boxes is None or len(result.boxes) == 0:
        empty2 = np.zeros((0, num_keypoints, 2))
        return np.zeros((0, 4)), np.zeros((0,)), empty2, np.zeros((0, num_keypoints)), None

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    box_conf = result.boxes.conf.cpu().numpy()
    kpts_xy = result.keypoints.xy.cpu().numpy()
    kpts_conf = (
        result.keypoints.conf.cpu().numpy()
        if result.keypoints.conf is not None
        else result.keypoints.data[..., 2].cpu().numpy()
    )
    track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
    return boxes_xyxy, box_conf, kpts_xy, kpts_conf, track_ids
