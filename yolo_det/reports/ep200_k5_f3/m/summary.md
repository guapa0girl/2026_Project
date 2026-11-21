# YOLO 직접 판정 — ep200_k5_f3 / m (test = 000, 사진 86장)

- 분할: kfold_k5_f3, 가중치: `models/ep200_k5_f3/m/weights/best.pt`, visiting conf ≥ 0.25 (valid 37장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | RMSE | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|---|
| YOLO (m) | 0.244 (-0.058) | 0.494 | 75.6% | 100.0% | -3.8% | 0.097 |

- valid 37장 (같은 임계값, 임계값을 여기서 골라 낙관적): MAE 0.297 (+0.027), RMSE 0.593, 정확 일치 73.0%, box_P 0.920, box_R 0.873, box_mAP50 0.956, box_mAP50-95 0.648
- 총 visiting: YOLO 128 / 사람 133, 많이 센 사진 8 / 적게 센 사진 13
- 진단(IoU ≥ 0.5): 못 찾은 visiting 5, 찾았지만 other로 판정 11, other를 visiting으로 7, 사람 라벨에 없는 박스를 visiting으로 4
- valid 진단(IoU ≥ 0.5): 못 찾은 visiting 4, 찾았지만 other로 판정 4, other를 visiting으로 7, 사람 라벨에 없는 박스를 visiting으로 2
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-15T19:18:11, git b3ce24b, 시작 가중치 coco (5d4a90cdc7a21786), best.pt 6e43d6493d9c8624
- 참고 box (test): box_P 0.922, box_R 0.862, box_mAP50 0.871, box_mAP50-95 0.585, box_mAP50-95_other 0.648, box_mAP50-95_visiting 0.522
- 추론 시간 (NVIDIA RTX 6000 Ada Generation, batch 1, 640, 예열 5장 제외, 86장 × 3회, end-to-end): 중앙값 21.9 ms, p95 24.3 ms
