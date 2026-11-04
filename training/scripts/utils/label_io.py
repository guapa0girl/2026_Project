"""YOLO-Pose txt 라벨 파일 읽기/쓰기 헬퍼."""
import numpy as np


def _clip01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def clip_label_line(line, num_keypoints):
    """CVAT export 좌표 중 일부가 이미지 경계를 살짝 벗어나 있어(occluded/잘린
    부위를 프레임 밖으로 찍은 경우 등) 0~1 범위를 벗어나는 경우가 많다.
    Ultralytics 로더는 이런 인스턴스를 통째로 "corrupt"로 버리므로, bbox와
    키포인트 x/y를 [0,1]로 clip해서 최대한 학습에 활용한다(visibility 값은
    그대로 유지). 반환값은 재구성된 라벨 한 줄(문자열), 빈 줄이면 None."""
    parts = line.split()
    if not parts:
        return None
    cls = parts[0]
    vals = list(map(float, parts[1:]))
    cx, cy, w, h = vals[:4]
    kpt_flat = vals[4:]

    x1, y1 = _clip01(cx - w / 2), _clip01(cy - h / 2)
    x2, y2 = _clip01(cx + w / 2), _clip01(cy + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None  # 클립 후 면적이 사라지는 인스턴스는 제외
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    out = [cls, f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for i in range(num_keypoints):
        x, y, v = kpt_flat[i * 3:i * 3 + 3]
        out += [f"{_clip01(x):.6f}", f"{_clip01(y):.6f}", str(int(v))]
    return " ".join(out)


def clip_yolo_pose_label_file(src_path, dst_path, num_keypoints):
    """src_path의 라벨을 읽어 좌표를 클립한 뒤 dst_path에 저장."""
    lines = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            clipped = clip_label_line(line.strip(), num_keypoints)
            if clipped is not None:
                lines.append(clipped)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def read_yolo_pose_label(txt_path, num_keypoints):
    """반환: boxes (N,4) cx,cy,w,h / kpts (N,K,2) x,y / vis (N,K) 0/1/2"""
    boxes, kpts, vis = [], [], []
    if not txt_path.exists():
        return (
            np.zeros((0, 4)),
            np.zeros((0, num_keypoints, 2)),
            np.zeros((0, num_keypoints)),
        )

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            vals = list(map(float, parts[1:]))  # class_id 제외
            box = vals[:4]
            kp_flat = vals[4:]
            expected = num_keypoints * 3
            if len(kp_flat) != expected:
                raise ValueError(
                    f"{txt_path}: 키포인트 개수가 kpt_shape({num_keypoints})와 "
                    f"다릅니다 ({len(kp_flat)} / {expected} 값)"
                )
            kp = np.array(kp_flat, dtype=np.float32).reshape(num_keypoints, 3)
            boxes.append(box)
            kpts.append(kp[:, :2])
            vis.append(kp[:, 2])

    if not boxes:
        return (
            np.zeros((0, 4)),
            np.zeros((0, num_keypoints, 2)),
            np.zeros((0, num_keypoints)),
        )

    return np.array(boxes, dtype=np.float32), np.array(kpts, dtype=np.float32), np.array(vis, dtype=np.float32)
