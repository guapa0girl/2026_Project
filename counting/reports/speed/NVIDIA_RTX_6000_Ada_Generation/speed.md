# 추론 시간 (end-to-end, 사진 1장) — Random 5-fold / Time 5-fold

환경: Python 3.10.12 / torch 2.11.0+cu128 / ultralytics 8.4.108 / CPU x86_64 / GPU NVIDIA RTX 6000 Ada Generation

조건: batch 1, imgsz 640, 예열 5장 제외, 000 86장 × 3회, fold마다 그 fold의 모델 (포즈 `ep200_<fold>` best.pt conf 0.4, RF = 그 fold train 사진으로 학습). 사진 읽기부터 마릿수까지.

값 = fold 5개(모델 5개) 각각의 중앙값 · p95를 평균 ± 표본 표준편차. 단계별 칸은 fold 중앙값의 평균.

| 분할 | 장비 | 방법 | 중앙값 (ms) | p95 (ms) | p95 최대 (fold) | 읽기 | 모델 추론 | 특징 계산 | 판정 | 1 fps (≤1000 ms) | 30 fps (≤33.3 ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Random 5-fold | GPU (NVIDIA RTX 6000 Ada Generation) | Geo | 23.8 ± 0.5 | 24.5 ± 0.3 | 24.8 | 13.0 | 10.1 | 0.8 | 0.00 | ✅ | ✅ |
| Random 5-fold | GPU (NVIDIA RTX 6000 Ada Generation) | RF | 28.0 ± 0.5 | 28.6 ± 0.3 | 28.9 | 13.0 | 10.1 | 0.8 | 4.14 | ✅ | ✅ |
| Random 5-fold | GPU (NVIDIA RTX 6000 Ada Generation) | RF+Geo | 27.9 ± 0.5 | 28.6 ± 0.3 | 28.9 | 13.0 | 10.1 | 0.8 | 4.09 | ✅ | ✅ |
| Random 5-fold | GPU (NVIDIA RTX 6000 Ada Generation) | RF+Geo+Leg | 27.9 ± 0.5 | 28.6 ± 0.3 | 28.9 | 13.0 | 10.1 | 0.8 | 4.08 | ✅ | ✅ |
| Random 5-fold | GPU (NVIDIA RTX 6000 Ada Generation) | YOLO | 21.3 ± 0.1 | 21.4 ± 0.2 | 21.6 | 12.6 | 8.6 | 0.0 | 0.00 | ✅ | ✅ |
| Random 5-fold | CPU (x86_64) | Geo | 185.8 ± 4.1 | 192.7 ± 4.6 | 197.4 | 12.9 | 171.5 | 1.0 | 0.00 | ✅ | ❌ |
| Random 5-fold | CPU (x86_64) | RF | 190.4 ± 4.0 | 197.4 ± 4.6 | 202.2 | 12.9 | 171.5 | 1.0 | 4.66 | ✅ | ❌ |
| Random 5-fold | CPU (x86_64) | RF+Geo | 190.0 ± 4.1 | 196.9 ± 4.6 | 201.6 | 12.9 | 171.5 | 1.0 | 4.23 | ✅ | ❌ |
| Random 5-fold | CPU (x86_64) | RF+Geo+Leg | 190.0 ± 4.1 | 196.8 ± 4.6 | 201.5 | 12.9 | 171.5 | 1.0 | 4.17 | ✅ | ❌ |
| Random 5-fold | CPU (x86_64) | YOLO | 179.5 ± 3.2 | 186.4 ± 2.3 | 189.8 | 13.0 | 166.5 | 0.0 | 0.00 | ✅ | ❌ |
| Time 5-fold | GPU (NVIDIA RTX 6000 Ada Generation) | Geo | 23.7 ± 0.1 | 24.1 ± 0.3 | 24.5 | 13.0 | 10.0 | 0.8 | 0.00 | ✅ | ✅ |
| Time 5-fold | GPU (NVIDIA RTX 6000 Ada Generation) | RF | 27.8 ± 0.1 | 28.2 ± 0.3 | 28.7 | 13.0 | 10.0 | 0.8 | 4.13 | ✅ | ✅ |
| Time 5-fold | GPU (NVIDIA RTX 6000 Ada Generation) | RF+Geo | 27.8 ± 0.1 | 28.2 ± 0.2 | 28.6 | 13.0 | 10.0 | 0.8 | 4.08 | ✅ | ✅ |
| Time 5-fold | GPU (NVIDIA RTX 6000 Ada Generation) | RF+Geo+Leg | 27.8 ± 0.1 | 28.2 ± 0.3 | 28.6 | 13.0 | 10.0 | 0.8 | 4.07 | ✅ | ✅ |
| Time 5-fold | GPU (NVIDIA RTX 6000 Ada Generation) | YOLO | 21.3 ± 0.1 | 21.5 ± 0.1 | 21.6 | 12.6 | 8.6 | 0.0 | 0.00 | ✅ | ✅ |
| Time 5-fold | CPU (x86_64) | Geo | 191.6 ± 4.0 | 200.4 ± 8.1 | 211.2 | 12.9 | 177.3 | 1.1 | 0.00 | ✅ | ❌ |
| Time 5-fold | CPU (x86_64) | RF | 196.3 ± 4.0 | 205.2 ± 8.0 | 215.9 | 12.9 | 177.3 | 1.1 | 4.69 | ✅ | ❌ |
| Time 5-fold | CPU (x86_64) | RF+Geo | 195.9 ± 4.0 | 204.7 ± 8.1 | 215.4 | 12.9 | 177.3 | 1.1 | 4.24 | ✅ | ❌ |
| Time 5-fold | CPU (x86_64) | RF+Geo+Leg | 195.8 ± 4.0 | 204.6 ± 8.1 | 215.4 | 12.9 | 177.3 | 1.1 | 4.16 | ✅ | ❌ |
| Time 5-fold | CPU (x86_64) | YOLO | 183.3 ± 4.9 | 189.5 ± 2.6 | 191.4 | 13.1 | 169.9 | 0.0 | 0.00 | ✅ | ❌ |

'모델 추론' = 포즈 또는 YOLO predict(전처리 · NMS · 후처리 포함). '특징 계산' = 검출된 돼지 전부의 특징 + Geo 판정. '판정' = RF predict (Geo는 특징 계산에 포함, 0).
실시간 판정은 fold 5개 p95 평균 기준. fold별 값은 `speed_folds.csv`.

