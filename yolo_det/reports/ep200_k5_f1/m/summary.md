# YOLO 직접 판정 — ep200_k5_f1 / m (test = 000, 사진 86장)

- 분할: kfold_k5_f1, 가중치: `models/ep200_k5_f1/m/weights/best.pt`, visiting conf ≥ 0.50 (valid 34장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|
| YOLO (m) | 0.279 (-0.047) | 73.3% | 98.8% | -3.0% | 0.176 |

- 총 visiting: YOLO 129 / 사람 133, 많이 센 사진 10 / 적게 센 사진 13
- 진단(IoU ≥ 0.5): 못 찾은 visiting 5, 찾았지만 other로 판정 15, other를 visiting으로 14, 사람 라벨에 없는 박스를 visiting으로 2
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-15T19:25:50, git b3ce24b, 시작 가중치 coco (5d4a90cdc7a21786), best.pt 2453a0eddca649ec
- 참고 box mAP (test): box_mAP50 0.876, box_mAP50-95 0.610, box_mAP50-95_other 0.658, box_mAP50-95_visiting 0.562
