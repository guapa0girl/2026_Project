# 2026_Project — 돼지 사료통 점유(visiting) 마리 수 추정

돈방 CCTV 사진에서 사료통을 차지한(visiting) 돼지 수를 세는 연구 코드.
2026 MIX_Lab 학부연구생 「육돈 사료통 점유 시간 계측 연구」 과제용.

```
CCTV 사진 ─→ 포즈 추정 (YOLOv8-pose, 관절 18개) ─→ 돼지마다 visiting / other 판정 ─→ 사진당 visiting 마릿수
```

판정 방법 5가지(Geo / RF / RF+Geo / RF+Geo+Leg / YOLO)를 같은 train 사진으로 학습하고, 같은 test(000)로 비교함.

## 결과

test = 000(1분 샘플 영상 86장, 사람 visiting 133마리). 11/16 통일 기준(같은 train 사진 · seed 42 · val만으로 선택)으로 학습한 랜덤 / 시간대 5-fold의 fold 5개 평균 ± 표본 표준편차.

| 방법 | 사진당 마릿수 MAE (랜덤) | MAE (시간대) | 정확 일치 (랜덤 / 시간대) | 1분 총 점유량 오차 (랜덤 / 시간대) |
|---|---|---|---|---|
| Geo | 0.293 ± 0.028 | 0.293 ± 0.022 | 71.9% / 71.9% | −18.6% / −17.7% |
| RF | 0.114 ± 0.015 | 0.116 ± 0.032 | 88.8% / 88.4% | −2.3% / −1.2% |
| RF+Geo | 0.133 ± 0.033 | 0.126 ± 0.019 | 86.7% / 87.4% | −4.4% / −3.3% |
| **RF+Geo+Leg** (RF best) | 0.121 ± 0.033 | 0.109 ± 0.021 | 87.9% / 89.1% | −3.3% / −1.4% |
| YOLO | 0.300 ± 0.137 | 0.223 ± 0.048 | 70.5% / 77.9% | −4.1% / −4.2% |

- RF 계열 3가지가 가장 좋음. RF best(5-fold 평균 val MAE 최소)는 두 분할 모두 RF+Geo+Leg이고, Geo · YOLO보다 test 사진 단위로 유의하게 좋음(Wilcoxon p ≤ 0.007). RF 계열 3가지 사이 차이는 유의하지 않음(p > 0.3).
- YOLO는 랜덤 5-fold에서 fold마다 크게 흔들림(MAE 0.174\~0.477). Geo는 test에서 약 18% 적게 셈.
- 이전(통일 기준 전) 결과와 비교 과정: [RESULTS.md](RESULTS.md) 4\~5장, 통일 기준 결과 · 조건 · 한계점: [RESULTS.md 6\~8장](RESULTS.md), 표 전체: [fold_study.md](counting/reports/fold_study/fold_study.md), 논문용 지표: [paper_metrics.md](counting/reports/fold_study/paper/paper_metrics.md).
- 데이터, train / valid 비율, 진행 방법, 분할별 결과: **[RESULTS.md](RESULTS.md)**
- 진행 과정 전체 기록: [PROGRESS.md](PROGRESS.md)

## 데이터

| task | 카메라 | 사진 | 돼지 (visiting / other) | 용도 |
|---|---|---|---|---|
| 001 | CH005 | 90 | 899 (136 / 763) | train / valid |
| 002 | CH010 | 100 | 990 (179 / 811) | train / valid |
| 000 | 1분 샘플 영상 | 86 | 860 (133 / 727) | test |

사진·라벨·가중치는 레포에 없음(농장 데이터, 용량). 지표·비교표·그래프만 `reports/`에 있음.

## 폴더

| 폴더 | 하는 일 |
|---|---|
| [training/](training/README.md) | 라벨 변환, 분할, 포즈 모델 학습·평가 |
| [counting/](counting/README.md) | Geo · RF 계열 판정 평가, 분할별 비교 |
| [yolo_det/](yolo_det/README.md) | YOLO 직접 판정 (수아) |

## 설치 (Windows, NVIDIA GPU)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r training\requirements.txt -r counting\requirements.txt
```

확인된 환경: torch 2.11+cu128, ultralytics 8.4, RTX 5060 8GB.
