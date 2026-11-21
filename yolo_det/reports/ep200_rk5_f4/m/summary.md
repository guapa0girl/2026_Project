# YOLO 직접 판정 — ep200_rk5_f4 / m (test = 000, 사진 86장)

- 분할: rkfold_k5_f4, 가중치: `models/ep200_rk5_f4/m/weights/best.pt`, visiting conf ≥ 0.65 (valid 40장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | RMSE | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|---|
| YOLO (m) | 0.209 (-0.070) | 0.482 | 80.2% | 98.8% | -4.5% | 0.120 |

- valid 40장 (같은 임계값, 임계값을 여기서 골라 낙관적): MAE 0.250 (-0.100), RMSE 0.500, 정확 일치 75.0%, box_P 0.925, box_R 0.942, box_mAP50 0.950, box_mAP50-95 0.703
- 총 visiting: YOLO 127 / 사람 133, 많이 센 사진 5 / 적게 센 사진 12
- 진단(IoU ≥ 0.5): 못 찾은 visiting 5, 찾았지만 other로 판정 8, other를 visiting으로 5, 사람 라벨에 없는 박스를 visiting으로 2
- valid 진단(IoU ≥ 0.5): 못 찾은 visiting 5, 찾았지만 other로 판정 2, other를 visiting으로 1, 사람 라벨에 없는 박스를 visiting으로 2
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-18T00:15:31, git c1df046, 시작 가중치 coco (5d4a90cdc7a21786), best.pt d5ac6dd8864e46a6
- 참고 box (test): box_P 0.912, box_R 0.835, box_mAP50 0.869, box_mAP50-95 0.590, box_mAP50-95_other 0.640, box_mAP50-95_visiting 0.540
- 추론 시간 (NVIDIA RTX 6000 Ada Generation, batch 1, 640, 예열 5장 제외, 86장 × 3회, end-to-end): 중앙값 22.3 ms, p95 24.1 ms
