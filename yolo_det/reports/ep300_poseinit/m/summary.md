# YOLO 직접 판정 — ep300_poseinit / m (test = 000, 사진 86장)

- 분할: random, 가중치: `models/ep300_poseinit/m/weights/best.pt`, visiting conf ≥ 0.25 (valid 28장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|
| YOLO (m) | 0.209 (-0.116) | 80.2% | 98.8% | -7.5% | 0.075 |

- 총 visiting: YOLO 123 / 사람 133, 많이 센 사진 3 / 적게 센 사진 14
- 진단(IoU ≥ 0.5): 못 찾은 visiting 4, 찾았지만 other로 판정 11, other를 visiting으로 1, 사람 라벨에 없는 박스를 visiting으로 4
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-15T19:39:33, git b3ce24b, 시작 가중치 pose (5e23e8ccc106557a), best.pt 8bd3bd14697680ee
- 참고 box mAP (test): box_mAP50 0.904, box_mAP50-95 0.623, box_mAP50-95_other 0.670, box_mAP50-95_visiting 0.576
