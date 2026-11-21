# YOLO 직접 판정 — ep200_rk5_f2 / m (test = 000, 사진 86장)

- 분할: rkfold_k5_f2, 가중치: `models/ep200_rk5_f2/m/weights/best.pt`, visiting conf ≥ 0.50 (valid 37장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | RMSE | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|---|
| YOLO (m) | 0.419 (-0.372) | 0.647 | 58.1% | 100.0% | -24.1% | 0.362 |

- valid 37장 (같은 임계값, 임계값을 여기서 골라 낙관적): MAE 0.432 (-0.162), RMSE 0.771, 정확 일치 64.9%, box_P 0.914, box_R 0.913, box_mAP50 0.945, box_mAP50-95 0.679
- 총 visiting: YOLO 101 / 사람 133, 많이 센 사진 2 / 적게 센 사진 34
- 진단(IoU ≥ 0.5): 못 찾은 visiting 6, 찾았지만 other로 판정 33, other를 visiting으로 5, 사람 라벨에 없는 박스를 visiting으로 2
- valid 진단(IoU ≥ 0.5): 못 찾은 visiting 6, 찾았지만 other로 판정 5, other를 visiting으로 3, 사람 라벨에 없는 박스를 visiting으로 2
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-17T23:59:05, git c1df046, 시작 가중치 coco (5d4a90cdc7a21786), best.pt 7a0e7e3bc1f572df
- 참고 box (test): box_P 0.870, box_R 0.792, box_mAP50 0.836, box_mAP50-95 0.565, box_mAP50-95_other 0.645, box_mAP50-95_visiting 0.486
- 추론 시간 (NVIDIA RTX 6000 Ada Generation, batch 1, 640, 예열 5장 제외, 86장 × 3회, end-to-end): 중앙값 21.9 ms, p95 23.7 ms
