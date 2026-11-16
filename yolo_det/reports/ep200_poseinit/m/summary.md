# YOLO 직접 판정 — ep200_poseinit / m (test = 000, 사진 86장)

- 분할: random, 가중치: `models/ep200_poseinit/m/weights/best.pt`, visiting conf ≥ 0.75 (valid 28장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|
| YOLO (m) | 0.174 (-0.081) | 82.6% | 100.0% | -5.3% | 0.094 |

- 총 visiting: YOLO 126 / 사람 133, 많이 센 사진 4 / 적게 센 사진 11
- 진단(IoU ≥ 0.5): 못 찾은 visiting 11, 찾았지만 other로 판정 4, other를 visiting으로 5, 사람 라벨에 없는 박스를 visiting으로 3
- 버전: ultralytics 8.4.108, torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 6000 Ada Generation, 학습 2026-11-14T22:49:31.157958+09:00, git 323b212 (추정), 시작 가중치 pose (5e23e8ccc106557a), best.pt 913752abc90a46e8
- 참고 box mAP (test): box_mAP50 0.901, box_mAP50-95 0.645, box_mAP50-95_other 0.687, box_mAP50-95_visiting 0.603
