"""돼지가 급식 중인지(섭식 자세) 판단하기 위한 특징 추출 + 규칙 기반 분류.

키포인트(18개, configs/pig_pose.yaml 순서)에서 급식 관련 기하학적 특징을
뽑아내고, 간단한 규칙(코가 급식 구역 안/근처 + 고개를 숙임)으로 1차 분류한다.
이 규칙 기반 결과는 RandomForest 등 ML 분류기의 부트스트랩 학습 라벨로
쓰인다 (scripts/label_posture_rule_based.py -> scripts/train_posture_classifier.py).

좌표는 전부 픽셀 단위(원본 이미지 크기 기준)를 입력으로 받는다.
"""
import numpy as np
import yaml

# configs/pig_pose.yaml 순서와 반드시 일치해야 함 (0-indexed)
NOSE, NECK = 12, 11
L_HIP, R_HIP = 2, 3
L_SHOULDER, R_SHOULDER = 8, 9

FEATURE_NAMES = [
    "nose_in_zone",
    "nose_zone_dist_norm",
    "head_dip_deg",
    "spine_dip_deg",
    "neck_rel_height",
    "key_kpt_conf",
]


def load_zones(zones_yaml):
    with open(zones_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [[tuple(pt) for pt in zone] for zone in data["zones"]]


def point_in_polygon(point, polygon):
    """레이캐스팅으로 점이 다각형 안에 있는지 판정."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def polygon_center(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def angle_from_vertical(vec):
    """벡터가 수직 아래(화면 기준 +y) 방향과 이루는 각도(도). 0=완전히 아래를 향함, 180=위."""
    vx, vy = vec
    norm = float(np.hypot(vx, vy))
    if norm < 1e-6:
        return 90.0
    cos_a = np.clip(vy / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def extract_features(kpts_xy, kpts_vis, bbox_xyxy, zones):
    """
    kpts_xy: (18,2) 픽셀 좌표
    kpts_vis: (18,) 0/1/2 visibility
    bbox_xyxy: (4,) x1,y1,x2,y2 (픽셀)
    zones: load_zones()의 반환값 — list[list[(x,y)]]
    반환: FEATURE_NAMES 순서와 일치하는 dict
    """
    kpts_xy = np.asarray(kpts_xy, dtype=np.float64)
    kpts_vis = np.asarray(kpts_vis, dtype=np.float64)

    nose = kpts_xy[NOSE]
    neck = kpts_xy[NECK]
    hips_center = (kpts_xy[L_HIP] + kpts_xy[R_HIP]) / 2
    shoulders_center = (kpts_xy[L_SHOULDER] + kpts_xy[R_SHOULDER]) / 2

    x1, y1, x2, y2 = bbox_xyxy
    bbox_diag = float(np.hypot(x2 - x1, y2 - y1)) + 1e-6

    nose_in_zone = 0.0
    min_dist = None
    for zone in zones:
        if point_in_polygon(nose, zone):
            nose_in_zone = 1.0
        cx, cy = polygon_center(zone)
        d = float(np.hypot(nose[0] - cx, nose[1] - cy)) / bbox_diag
        if min_dist is None or d < min_dist:
            min_dist = d
    if min_dist is None:
        min_dist = 10.0  # 급식 구역이 정의 안 됐으면 매우 먼 것으로 취급

    head_dip_deg = angle_from_vertical(nose - neck)
    spine_dip_deg = angle_from_vertical(neck - hips_center)
    neck_rel_height = float((neck[1] - shoulders_center[1]) / bbox_diag)

    key_idx = [NOSE, NECK, L_HIP, R_HIP, L_SHOULDER, R_SHOULDER]
    key_kpt_conf = float((kpts_vis[key_idx] > 0).mean())

    return {
        "nose_in_zone": nose_in_zone,
        "nose_zone_dist_norm": min_dist,
        "head_dip_deg": head_dip_deg,
        "spine_dip_deg": spine_dip_deg,
        "neck_rel_height": neck_rel_height,
        "key_kpt_conf": key_kpt_conf,
    }


def rule_based_classify(features, dip_thresh_deg=55.0, dist_thresh=0.35):
    """코가 급식 구역 안(또는 충분히 근처) + 고개를 숙였으면 '섭식(True)'.

    두 threshold는 대략적인 기본값이다 — 실제 라벨(있다면)과 비교해서
    scripts/label_posture_rule_based.py 실행 후 confusion matrix를 보고
    조정하는 것을 권장한다."""
    near_zone = features["nose_in_zone"] > 0.5 or features["nose_zone_dist_norm"] < dist_thresh
    head_down = features["head_dip_deg"] < dip_thresh_deg
    return bool(near_zone and head_down)
