# YOLO 직접 판정 — ep200_rk5_f1 / m (test = 000, 사진 86장)

- 분할: rkfold_k5_f1, 가중치: `models/ep200_rk5_f1/m/weights/best.pt`, visiting conf ≥ 0.45 (valid 34장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | RMSE | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|---|
| YOLO (m) | 0.174 (-0.035) | 0.418 | 82.6% | 100.0% | -2.3% | 0.037 |

- valid 34장 (같은 임계값, 임계값을 여기서 골라 낙관적): MAE 0.088 (-0.088), RMSE 0.383, 정확 일치 94.1%, box_P 0.948, box_R 0.926, box_mAP50 0.966, box_mAP50-95 0.728
- 총 visiting: YOLO 130 / 사람 133, 많이 센 사진 6 / 적게 센 사진 9
- 진단(IoU ≥ 0.5): 못 찾은 visiting 6, 찾았지만 other로 판정 6, other를 visiting으로 5, 사람 라벨에 없는 박스를 visiting으로 4
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-17T23:51:03, git c1df046, 시작 가중치 coco (5d4a90cdc7a21786), best.pt 42e544068f0a9601
- 참고 box (test): box_P 0.941, box_R 0.886, box_mAP50 0.899, box_mAP50-95 0.622, box_mAP50-95_other 0.675, box_mAP50-95_visiting 0.569
- 추론 시간 (NVIDIA RTX 6000 Ada Generation, batch 1, 640, 예열 5장 제외, 86장 × 3회, end-to-end): 중앙값 22.1 ms, p95 24.3 ms
