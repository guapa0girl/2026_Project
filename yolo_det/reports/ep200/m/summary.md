# YOLO 직접 판정 — ep200 / m (test = 000, 사진 86장)

- 가중치: `models/ep200/m/weights/best.pt`, visiting conf ≥ 0.55 (valid 28장에서 사진당 MAE 최소), NMS agnostic, iou 0.7

| 방법 | 사진당 MAE (bias) | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |
|---|---|---|---|---|---|
| YOLO (m) | 0.198 (-0.151) | 80.2% | 100.0% | -9.8% | 0.150 |

- 총 visiting: YOLO 120 / 사람 133, 많이 센 사진 2 / 적게 센 사진 15
- 진단(IoU ≥ 0.5): 못 찾은 visiting 8, 찾았지만 other로 판정 11, other를 visiting으로 2, 사람 라벨에 없는 박스를 visiting으로 4
- 참고 box mAP (test): box_mAP50 0.888, box_mAP50-95 0.601, box_mAP50-95_other 0.644, box_mAP50-95_visiting 0.559
