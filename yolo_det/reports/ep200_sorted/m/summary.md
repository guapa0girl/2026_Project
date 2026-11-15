# YOLO 직접 판정 — ep200_sorted / m (test = 000, 사진 86장)

- 분할: sorted, 가중치: `models/ep200_sorted/m/weights/best.pt`, visiting conf ≥ 0.85 (valid 31장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|
| YOLO (m) | 0.186 (-0.163) | 81.4% | 100.0% | -10.5% | 0.130 |

- 총 visiting: YOLO 119 / 사람 133, 많이 센 사진 1 / 적게 센 사진 15
- 진단(IoU ≥ 0.5): 못 찾은 visiting 14, 찾았지만 other로 판정 3, other를 visiting으로 2, 사람 라벨에 없는 박스를 visiting으로 1
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-15T18:50:35, git b3ce24b, 시작 가중치 coco (5d4a90cdc7a21786), best.pt f31b9b631ddd4ec3
- 참고 box mAP (test): box_mAP50 0.884, box_mAP50-95 0.614, box_mAP50-95_other 0.652, box_mAP50-95_visiting 0.576
