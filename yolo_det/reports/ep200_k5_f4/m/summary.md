# YOLO 직접 판정 — ep200_k5_f4 / m (test = 000, 사진 86장)

- 분할: kfold_k5_f4, 가중치: `models/ep200_k5_f4/m/weights/best.pt`, visiting conf ≥ 0.70 (valid 40장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|
| YOLO (m) | 0.209 (-0.047) | 79.1% | 100.0% | -3.0% | 0.061 |

- 총 visiting: YOLO 129 / 사람 133, 많이 센 사진 7 / 적게 센 사진 11
- 진단(IoU ≥ 0.5): 못 찾은 visiting 8, 찾았지만 other로 판정 5, other를 visiting으로 6, 사람 라벨에 없는 박스를 visiting으로 3
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-15T19:26:25, git b3ce24b, 시작 가중치 coco (5d4a90cdc7a21786), best.pt 9b4ca4410869aa5d
- 참고 box mAP (test): box_mAP50 0.878, box_mAP50-95 0.621, box_mAP50-95_other 0.618, box_mAP50-95_visiting 0.625
