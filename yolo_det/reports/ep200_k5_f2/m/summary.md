# YOLO 직접 판정 — ep200_k5_f2 / m (test = 000, 사진 86장)

- 분할: kfold_k5_f2, 가중치: `models/ep200_k5_f2/m/weights/best.pt`, visiting conf ≥ 0.35 (valid 37장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|
| YOLO (m) | 0.233 (-0.070) | 76.7% | 100.0% | -4.5% | 0.104 |

- 총 visiting: YOLO 127 / 사람 133, 많이 센 사진 7 / 적게 센 사진 13
- 진단(IoU ≥ 0.5): 못 찾은 visiting 3, 찾았지만 other로 판정 14, other를 visiting으로 10, 사람 라벨에 없는 박스를 visiting으로 1
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-15T19:34:22, git b3ce24b, 시작 가중치 coco (5d4a90cdc7a21786), best.pt 035c29ff9d79e827
- 참고 box mAP (test): box_mAP50 0.892, box_mAP50-95 0.583, box_mAP50-95_other 0.659, box_mAP50-95_visiting 0.508
