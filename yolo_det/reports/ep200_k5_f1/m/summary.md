# YOLO 직접 판정 — ep200_k5_f1 / m (test = 000, 사진 86장)

- 분할: kfold_k5_f1, 가중치: `models/ep200_k5_f1/m/weights/best.pt`, visiting conf ≥ 0.50 (valid 34장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | RMSE | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|---|
| YOLO (m) | 0.279 (-0.047) | 0.550 | 73.3% | 98.8% | -3.0% | 0.176 |

- valid 34장 (같은 임계값, 임계값을 여기서 골라 낙관적): MAE 0.588 (-0.118), RMSE 0.939, 정확 일치 55.9%, box_P 0.883, box_R 0.756, box_mAP50 0.843, box_mAP50-95 0.570
- 총 visiting: YOLO 129 / 사람 133, 많이 센 사진 10 / 적게 센 사진 13
- 진단(IoU ≥ 0.5): 못 찾은 visiting 5, 찾았지만 other로 판정 15, other를 visiting으로 14, 사람 라벨에 없는 박스를 visiting으로 2
- valid 진단(IoU ≥ 0.5): 못 찾은 visiting 10, 찾았지만 other로 판정 8, other를 visiting으로 10, 사람 라벨에 없는 박스를 visiting으로 4
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-15T19:25:50, git b3ce24b, 시작 가중치 coco (5d4a90cdc7a21786), best.pt 2453a0eddca649ec
- 참고 box (test): box_P 0.853, box_R 0.896, box_mAP50 0.876, box_mAP50-95 0.610, box_mAP50-95_other 0.658, box_mAP50-95_visiting 0.562
- 추론 시간 (NVIDIA RTX 6000 Ada Generation, batch 1, 640, 예열 5장 제외, 86장 × 3회, end-to-end): 중앙값 21.3 ms, p95 22.4 ms
