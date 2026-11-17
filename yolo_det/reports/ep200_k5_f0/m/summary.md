# YOLO 직접 판정 — ep200_k5_f0 / m (test = 000, 사진 86장)

- 분할: kfold_k5_f0, 가중치: `models/ep200_k5_f0/m/weights/best.pt`, visiting conf ≥ 0.65 (valid 42장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | RMSE | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|---|
| YOLO (m) | 0.151 (-0.105) | 0.389 | 84.9% | 100.0% | -6.8% | 0.119 |

- valid 42장 (같은 임계값, 임계값을 여기서 골라 낙관적): MAE 0.167 (+0.024), RMSE 0.408, 정확 일치 83.3%, box_P 0.941, box_R 0.912, box_mAP50 0.945, box_mAP50-95 0.645
- 총 visiting: YOLO 124 / 사람 133, 많이 센 사진 2 / 적게 센 사진 11
- 진단(IoU ≥ 0.5): 못 찾은 visiting 6, 찾았지만 other로 판정 9, other를 visiting으로 5, 사람 라벨에 없는 박스를 visiting으로 1
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-15T19:17:39, git b3ce24b, 시작 가중치 coco (5d4a90cdc7a21786), best.pt c61a43c3b805aeb0
- 참고 box (test): box_P 0.920, box_R 0.847, box_mAP50 0.879, box_mAP50-95 0.603, box_mAP50-95_other 0.648, box_mAP50-95_visiting 0.558
- 추론 시간 (NVIDIA RTX 6000 Ada Generation, batch 1, 640, 예열 5장 제외, 86장 × 3회, end-to-end): 중앙값 21.3 ms, p95 23.7 ms
