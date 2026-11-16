# 2026_Project — 돼지 사료통 점유(visiting) 마리 수 추정

돈방 CCTV 영상에서 **시간대별로 사료통을 차지한(visiting) 돼지 수**를 자동으로 세는 연구 코드.
2026 MIX_Lab 학부연구생 「육돈 사료통 점유 시간 계측 연구」 과제용.

- **결과: [RESULTS.md](RESULTS.md)** — 판정 방법 5가지(Geo / RF / RF+Geo / RF+Geo+Leg / YOLO)를 같은 train 사진, 같은 test(000)로 비교
- 진행 과정 전체 기록: [PROGRESS.md](PROGRESS.md)

---

## 현재 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| ① 포즈 추정 | YOLOv8-pose 사이즈·epoch 비교 → medium ep200, 분할마다 학습 | ✅ 완료 |
| ② visiting 판정 | 5가지 방법을 분할 3가지(랜덤 / 시간대 / 시간대 5-fold)에서 동일 조건 비교 | ✅ 완료 |
| ③·④ 영상 집계 | 영상 1초 추론 → 1분 / 10분 집계 | ⏳ 예정 |

### 결과 요약 (test = 000, 시간대 5-fold 평균)

| 방법 | 사진당 마릿수 MAE | 정확 일치 | 1분 총 점유량 오차 |
|---|---|---|---|
| Geo | 0.288 ± 0.013 | 72.3% | −17.4% |
| RF | 0.140 ± 0.022 | 86.0% | −2.4% |
| RF+Geo | 0.142 ± 0.021 | 85.8% | −4.4% |
| **RF+Geo+Leg** | **0.119 ± 0.030** | **88.1%** | −3.8% |
| YOLO | 0.223 ± 0.048 | 77.9% | −4.2% |

- 모든 분할에서 오차 순서는 **RF 계열 < YOLO < Geo**. RF+Geo+Leg가 가장 좋고 안정적임.
- 분할별 결과, train / valid 비율, 진행 방법은 [RESULTS.md](RESULTS.md).

---

## 파이프라인

```
CCTV 사진/영상 ─→ ① 포즈 추정 (YOLOv8-pose, 돼지마다 관절 18개)
                ─→ ② 돼지마다 visiting / other 판정 (5가지 방법 비교)
                ─→ ③ 사진(1초)마다 visiting 마릿수
                ─→ ④ 1분 / 10분 집계 → 시간대별 섭식 패턴
```

| 폴더 | 하는 일 | 설명 |
|---|---|---|
| `training/` | CVAT XML → YOLO-Pose 라벨 변환 → 분할 → 포즈 모델 학습·평가 | [training/README.md](training/README.md) |
| `counting/` | Geo · RF 계열 판정 평가, 분할 방식별 비교, 영상 추론·집계 | [counting/README.md](counting/README.md) |
| `yolo_det/` | YOLOv8 detect로 visiting / other 직접 검출 (수아) | [yolo_det/README.md](yolo_det/README.md) |

---

## 데이터

| task | 카메라 | 사진 | 돼지 (visiting / other) | 용도 |
|---|---|---|---|---|
| 001 | CH005 (1920×1080) | 90 | 899 (136 / 763) | train / valid |
| 002 | CH010 (1920×1080) | 100 | 990 (179 / 811) | train / valid |
| 000 | 1분 샘플 영상 (960×720) | 86 | 860 (133 / 727) | **test** (학습에 전혀 안 씀) |

- 모든 사진에 사람이 CVAT로 스켈레톤(관절 18개)과 State(visiting / other)를 라벨링함.
- 001+002 190장을 train / valid로 나눔(기본 랜덤 분할 162 : 28 = 85 : 15). test는 항상 000.

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
- **가중치·학습 산출물**: 용량이 커서 git 제외. 지표(json)·비교표(csv)·그래프(png)만 `reports/`에 올림.
