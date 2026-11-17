# 추론 시간 (end-to-end, 사진 1장)

환경: Python 3.14.7 / torch 2.11.0+cu128 / ultralytics 8.4.108 / CPU AMD Ryzen 5 7500F 6-Core Processor / GPU NVIDIA GeForce RTX 5060

조건: batch 1, imgsz 640, 예열 5장 제외, 000 86장 × 3회, 포즈 = `ep200_k5_f0` best.pt (conf 0.4), RF = 같은 분할 train 사진으로 학습. 사진 읽기부터 마릿수까지.

| 장비 | 방법 | 중앙값 (ms) | p95 (ms) | 읽기 | 모델 추론 | 특징 계산 | 판정 | 1 fps (≤1000 ms) | 30 fps (≤33.3 ms) |
|---|---|---|---|---|---|---|---|---|---|
| GPU (NVIDIA GeForce RTX 5060) | Geo | 20.7 | 22.6 | 8.7 | 11.3 | 0.6 | 0.00 | ✅ | ✅ |
| GPU (NVIDIA GeForce RTX 5060) | RF | 26.2 | 28.6 | 8.7 | 11.3 | 0.6 | 5.33 | ✅ | ✅ |
| GPU (NVIDIA GeForce RTX 5060) | RF+Geo | 26.0 | 28.9 | 8.7 | 11.3 | 0.6 | 5.12 | ✅ | ✅ |
| GPU (NVIDIA GeForce RTX 5060) | RF+Geo+Leg | 26.0 | 28.7 | 8.7 | 11.3 | 0.6 | 5.08 | ✅ | ✅ |
| CPU (AMD Ryzen 5 7500F 6-Core Processor) | Geo | 151.4 | 171.3 | 10.2 | 140.4 | 0.8 | 0.00 | ✅ | ❌ |
| CPU (AMD Ryzen 5 7500F 6-Core Processor) | RF | 158.5 | 178.6 | 10.2 | 140.4 | 0.8 | 7.31 | ✅ | ❌ |
| CPU (AMD Ryzen 5 7500F 6-Core Processor) | RF+Geo | 158.2 | 178.2 | 10.2 | 140.4 | 0.8 | 7.01 | ✅ | ❌ |
| CPU (AMD Ryzen 5 7500F 6-Core Processor) | RF+Geo+Leg | 158.2 | 178.4 | 10.2 | 140.4 | 0.8 | 6.95 | ✅ | ❌ |

단계별 칸은 중앙값(ms). '모델 추론' = 포즈 또는 YOLO predict(전처리 · NMS · 후처리 포함). '특징 계산' = 검출된 돼지 전부의 특징 + Geo 판정. '판정' = RF predict (Geo는 특징 계산에 포함, 0).
실시간 판정은 p95 기준.

YOLO는 이 장비에서 재지 않음 (`--yolo-weights` 없음). 같은 장비 비교를 위해 YOLO 가중치로 다시 재거나, 수아 서버에서 이 스크립트로 Geo · RF 계열을 같이 재야 함.

