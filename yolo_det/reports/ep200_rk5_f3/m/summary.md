# YOLO 직접 판정 — ep200_rk5_f3 / m (test = 000, 사진 86장)

- 분할: rkfold_k5_f3, 가중치: `models/ep200_rk5_f3/m/weights/best.pt`, visiting conf ≥ 0.55 (valid 37장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | RMSE | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|---|
| YOLO (m) | 0.477 (+0.221) | 0.707 | 53.5% | 98.8% | +14.3% | 0.379 |

- valid 37장 (같은 임계값, 임계값을 여기서 골라 낙관적): MAE 0.297 (-0.189), RMSE 0.593, 정확 일치 73.0%, box_P 0.962, box_R 0.896, box_mAP50 0.944, box_mAP50-95 0.742
- 총 visiting: YOLO 152 / 사람 133, 많이 센 사진 29 / 적게 센 사진 11
- 진단(IoU ≥ 0.5): 못 찾은 visiting 6, 찾았지만 other로 판정 8, other를 visiting으로 7, 사람 라벨에 없는 박스를 visiting으로 26
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-18T00:07:07, git c1df046, 시작 가중치 coco (5d4a90cdc7a21786), best.pt 465b1109c6843ea7
- 참고 box (test): box_P 0.848, box_R 0.867, box_mAP50 0.892, box_mAP50-95 0.620, box_mAP50-95_other 0.676, box_mAP50-95_visiting 0.565
- 추론 시간 (NVIDIA RTX 6000 Ada Generation, batch 1, 640, 예열 5장 제외, 86장 × 3회, end-to-end): 중앙값 22.3 ms, p95 24.2 ms
