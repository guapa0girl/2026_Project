# YOLO 직접 판정 — ep200_rk5_f0 / m (test = 000, 사진 86장)

- 분할: rkfold_k5_f0, 가중치: `models/ep200_rk5_f0/m/weights/best.pt`, visiting conf ≥ 0.25 (valid 42장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | RMSE | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|---|
| YOLO (m) | 0.221 (-0.058) | 0.470 | 77.9% | 100.0% | -3.8% | 0.092 |

- valid 42장 (같은 임계값, 임계값을 여기서 골라 낙관적): MAE 0.190 (+0.048), RMSE 0.436, 정확 일치 81.0%, box_P 0.948, box_R 0.944, box_mAP50 0.958, box_mAP50-95 0.714
- 총 visiting: YOLO 128 / 사람 133, 많이 센 사진 7 / 적게 센 사진 12
- 진단(IoU ≥ 0.5): 못 찾은 visiting 7, 찾았지만 other로 판정 10, other를 visiting으로 7, 사람 라벨에 없는 박스를 visiting으로 5
- valid 진단(IoU ≥ 0.5): 못 찾은 visiting 1, 찾았지만 other로 판정 3, other를 visiting으로 5, 사람 라벨에 없는 박스를 visiting으로 1
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-17T23:42:57, git c1df046, 시작 가중치 coco (5d4a90cdc7a21786), best.pt ca0f0a8f616e2afe
- 참고 box (test): box_P 0.927, box_R 0.833, box_mAP50 0.839, box_mAP50-95 0.588, box_mAP50-95_other 0.659, box_mAP50-95_visiting 0.517
- 추론 시간 (NVIDIA RTX 6000 Ada Generation, batch 1, 640, 예열 5장 제외, 86장 × 3회, end-to-end): 중앙값 22.0 ms, p95 23.9 ms
