# 2026_Project — 돼지 사료통 점유(visiting) 마리 수 추정

돈방 CCTV 영상에서 **시간대별로 사료통을 차지한(visiting) 돼지 수**를 자동으로 세는 연구 코드.
2026 MIX_Lab 학부연구생 「육돈 사료통 점유 시간 계측 연구」 과제용.

- 학습: 001(CH005) + 002(CH010) 190장
- test: 000(1분 샘플 영상) 86장 — 학습에 쓰지 않은 다른 카메라
- 비교: 돼지마다 visiting 판정하는 방법 5가지 — **Geo / RF / RF+Geo / RF+Geo+Leg / YOLO**

진행 내용과 결과는 [PROGRESS.md](PROGRESS.md)에 단계별로 자세히 기록.

---

## 현재 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| ① 포즈 추정 | YOLOv8-pose n/s/m/l × epoch 100·200 학습·비교 | ✅ 완료 — epoch 200이 전 사이즈에서 나음, **medium ep200** 사용(안) |
| ② visiting 판정 | Geo / RF / RF+Geo / RF+Geo+Leg 학습·평가 | ⏳ 다음 단계 |
| ② visiting 판정 | YOLO 직접 판정 | ⏳ 수아 진행 |
| ③·④ 마릿수 집계 | 영상 1초 추론 → 1분 / 10분 집계, DM 정답지 검증 | ⏳ 예정 |

### 포즈 추정 결과 요약 (test = 000)

| 사이즈 | epoch 100 → 200: pose mAP50-95 | PCK@0.1 | PCK@0.2 | 학습 시간(분) |
|---|---|---|---|---|
| nano | 0.079 → 0.101 | 0.667 → 0.685 | 0.861 → 0.867 | 3.3 → 6.2 |
| small | 0.060 → 0.103 | 0.701 → 0.689 | 0.850 → 0.856 | 4.6 → 8.6 |
| **medium** | 0.091 → 0.100 | 0.684 → **0.710** | 0.865 → **0.870** | 7.9 → 15.3 |
| large | 0.059 → **0.107** | 0.673 → 0.668 | 0.866 → 0.834 | 16.1 → 31.5 |

- epoch 200에서 mAP가 모든 사이즈에서 오름. PCK(관절 점이 정답 근처에 들어온 비율)는 차이 작음.
- medium ep200: test PCK 최고, mAP도 최고(large)와 0.007 차이 → visiting 판정에 쓸 모델(안).

---

## 파이프라인

```
CCTV 사진/영상 ─→ ① 포즈 추정 (YOLOv8-pose, 돼지마다 관절 18개)
                ─→ ② 돼지마다 visiting / other 판정 (5가지 방법 비교)
                ─→ ③ 사진(1초)마다 visiting 마릿수
                ─→ ④ 1분 / 10분 집계 → 시간대별 섭식 패턴   (③·④ = DM 파트)
```

```
2026_Project/
├── training/      ① 포즈 모델 학습·평가 (CVAT XML 변환 → 분할 → 학습 → mAP·PCK)
├── counting/      ②③④ visiting 판정 방법 평가, 영상 추론·집계, DM 정답지 추출
└── PROGRESS.md    단계별 방법·결과 상세
```

| 폴더 | 하는 일 | 설명 |
|---|---|---|
| `training/` | CVAT XML → YOLO-Pose 라벨 변환 → 분할 → 학습 → 평가 → 사이즈·epoch 비교 | [training/README.md](training/README.md) |
| `counting/` | 판정 방법 평가(Stage 1·2), 영상 1초 샘플링 추론, 1분/10분 집계, DM 정답지 추출 | [counting/README.md](counting/README.md) |

---

## 데이터

| task | 카메라 | 사진 | 돼지 (visiting / other) | 용도 |
|---|---|---|---|---|
| 001 | CH005 (1920×1080) | 90 | 899 (136 / 763) | **학습** — 포즈 모델, RF 계열 |
| 002 | CH010 (1920×1080) | 100 | 990 (179 / 811) | **학습** — 포즈 모델, RF 계열 |
| 000 | 1분 샘플 영상 (960×720) | 86 | 860 (133 / 727) | **test** — 학습에 전혀 안 씀 |

- 모든 사진에 사람이 CVAT로 스켈레톤(관절 18개)과 State(visiting / other)를 라벨링.
- 포즈 모델 분할: 001+002를 train 162 / valid 28로 나누고, 000 86장은 통째로 test.
- test(000)는 학습(CH005·CH010)과 **다른 카메라·해상도** → 처음 보는 화면에서의 성능을 잼.

---

## 비교하는 판정 방법 (②)

| 방법 | 어떻게 판정 | 학습 | 담당 |
|---|---|---|---|
| **Geo (규칙 기반)** | 코·목 신뢰도 ≥ 0.5, 코가 사료통 원 안, 목→코 방향이 사료통 중심에서 60° 이내 → visiting | 없음 | |
| **RF** | 자세 특징 6개 → RandomForest | 001+002 | |
| **RF+Geo** | RF 특징 + Geo 값 5개(코-사료통 거리 ÷ r, 머리 방향 각도 등) | 001+002 | |
| **RF+Geo+Leg** | RF+Geo + 다리 모양 누움 특징 5개(다리-척추 각도, 다리 길이, 발 쏠림 등) | 001+002 | |
| **YOLO** | 포즈 없이 YOLO가 돼지를 visiting / other로 바로 검출 | 001+002 | 수아 |

- 특징 정의: `counting/src/feeding_monitor/` (`feeding_classifier.py`, `posture_features.py`, `leg_features.py`, `method_features.py`).
- 사료통 원(cx, cy, r)은 카메라마다 따로 잡음(`counting/configs/eval_cameras.yaml` — CH005, CH010, SAMPLE).
  특징은 r·몸통 길이로 나눈 비율이나 각도라 카메라 높이·사료통 크기 차이에 영향을 덜 받음.

### 평가

| | Stage 1 | Stage 2 (최종 test) |
|---|---|---|
| 데이터 | 001+002 | 000 |
| 관절 | 사람이 찍은 관절 | **포즈 모델이 찍은 관절** (실제 운용과 같음) |
| 학습/시험 | 클립 단위 교차검증 (같은 클립 사진이 학습·시험에 나뉘지 않게) | 001+002 전체로 학습 → 000에 적용 |
| 지표 | visiting F1 (precision, recall) | visiting F1 + 사진당 마릿수 오차(MAE)·치우침(bias) |

---

## 설치 (Windows, NVIDIA GPU)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
# RTX 50 시리즈는 CUDA 12.8 이상 빌드가 필요
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r training\requirements.txt -r counting\requirements.txt
```

확인된 환경: torch 2.11+cu128, ultralytics 8.4, RTX 5060 8GB.

---

## 이 레포에 없는 것

- **사진·라벨·원본 영상**: 농장 데이터라 올리지 않음. 각 폴더의 `data/`는 빈 구조(`.gitkeep`)만 있음.
- **가중치·학습 산출물**: 용량이 커서 git 제외. 지표(json)·비교표(csv)·그래프(png)만 `training/reports/`에 올림.
