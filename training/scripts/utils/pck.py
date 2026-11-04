"""PCK(Percentage of Correct Keypoints) 계산.

Ultralytics의 pose mAP(OKS 기반)는 model.val() 이 기본 제공하지만, PCK는
제공하지 않기 때문에 이 모듈에서 별도로 구현한다.

정의: 예측 키포인트와 GT 키포인트 사이의 유클리드 거리가
      threshold * norm_factor 보다 작으면 "정답"으로 카운트한다.
      norm_factor 는 해당 GT 인스턴스 bbox의 max(width, height) 를 사용한다
      (PCKh 대신 bbox 기준 PCK@bbox, 흔히 쓰는 방식).

한 이미지 안에 돼지가 여러 마리 있을 수 있으므로, 예측 인스턴스와 GT
인스턴스를 bbox IoU로 그리디 매칭한 뒤 매칭된 쌍에 대해서만 키포인트 오차를
계산한다. IoU 임계값 미만으로 매칭 실패한 GT 인스턴스는 모든 키포인트를
오답으로 카운트한다(탐지 자체를 놓친 것이므로).
"""
from dataclasses import dataclass, field

import numpy as np


def box_iou_xywhn(box_a, box_b):
    """box_a, box_b: (cx, cy, w, h) normalized. 단일 값끼리 IoU."""
    def to_xyxy(b):
        cx, cy, w, h = b
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    ax1, ay1, ax2, ay2 = to_xyxy(box_a)
    bx1, by1, bx2, by2 = to_xyxy(box_b)

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_instances(pred_boxes, gt_boxes, iou_thres=0.5):
    """그리디 IoU 매칭. 반환: list[(pred_idx, gt_idx)], list[unmatched_gt_idx]"""
    pairs = []
    for gi, gt_b in enumerate(gt_boxes):
        for pi, pred_b in enumerate(pred_boxes):
            iou = box_iou_xywhn(pred_b, gt_b)
            if iou >= iou_thres:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_pred, used_gt = set(), set()
    matches = []
    for iou, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi))

    unmatched_gt = [gi for gi in range(len(gt_boxes)) if gi not in used_gt]
    return matches, unmatched_gt


@dataclass
class PCKAccumulator:
    num_keypoints: int
    thresholds: tuple = (0.05, 0.1, 0.2)
    correct: dict = field(default_factory=dict)
    total: dict = field(default_factory=dict)
    matched_instances: int = 0
    missed_instances: int = 0

    def __post_init__(self):
        for t in self.thresholds:
            self.correct[t] = np.zeros(self.num_keypoints, dtype=np.int64)
            self.total[t] = np.zeros(self.num_keypoints, dtype=np.int64)

    def add_image(self, pred_boxes, pred_kpts, gt_boxes, gt_kpts, gt_vis, iou_thres=0.5):
        """
        pred_boxes: (Np, 4) normalized cx,cy,w,h
        pred_kpts:  (Np, K, 2) normalized x,y
        gt_boxes:   (Ng, 4) normalized cx,cy,w,h
        gt_kpts:    (Ng, K, 2) normalized x,y
        gt_vis:     (Ng, K) 0/1/2 visibility (0=미라벨, 이 키포인트는 평가에서 제외)
        """
        matches, unmatched_gt = match_instances(pred_boxes, gt_boxes, iou_thres)
        self.matched_instances += len(matches)
        self.missed_instances += len(unmatched_gt)

        for pi, gi in matches:
            box_w, box_h = gt_boxes[gi][2], gt_boxes[gi][3]
            norm = max(box_w, box_h, 1e-6)
            dists = np.linalg.norm(pred_kpts[pi] - gt_kpts[gi], axis=-1)  # (K,)
            visible = gt_vis[gi] > 0
            for t in self.thresholds:
                self.total[t][visible] += 1
                self.correct[t][visible & (dists < t * norm)] += 1

        # 탐지를 놓친 GT 인스턴스는 라벨된 키포인트를 전부 오답 처리
        for gi in unmatched_gt:
            visible = gt_vis[gi] > 0
            for t in self.thresholds:
                self.total[t][visible] += 1
                # correct 는 더하지 않음 (오답)

    def summary(self, keypoint_names=None):
        out = {
            "matched_instances": self.matched_instances,
            "missed_instances": self.missed_instances,
            "by_threshold": {},
        }
        for t in self.thresholds:
            total = self.total[t]
            correct = self.correct[t]
            overall = float(correct.sum() / total.sum()) if total.sum() > 0 else None
            per_kp = {}
            for k in range(self.num_keypoints):
                name = keypoint_names[k] if keypoint_names else str(k)
                per_kp[name] = (
                    float(correct[k] / total[k]) if total[k] > 0 else None
                )
            out["by_threshold"][f"PCK@{t}"] = {
                "overall": overall,
                "per_keypoint": per_kp,
            }
        return out
