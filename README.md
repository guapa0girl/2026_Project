# 2026_Project — 돼지 사료통 점유(visiting) 마리 수 추정

돈방 CCTV 사진에서 사료통을 차지한(visiting) 돼지 수를 세는 연구 코드.
2026 MIX_Lab 학부연구생 「육돈 사료통 점유 시간 계측 연구」 과제용.

```
CCTV 사진 ─→ 포즈 추정 (YOLOv8-pose, 관절 18개) ─→ 돼지마다 visiting / other 판정 ─→ 사진당 visiting 마릿수
```

판정 방법 5가지(Geo / RF / RF+Geo / RF+Geo+Leg / YOLO)를 같은 train 사진으로 학습하고, 같은 test(000)로 비교함.

## 결과

test = 000(1분 샘플 영상 86장), 시간대 5-fold 평균.

| 방법 | 사진당 마릿수 MAE | 정확 일치 | 1분 총 점유량 오차 |
|---|---|---|---|
| Geo | 0.288 ± 0.013 | 72.3% | −17.4% |
| RF | 0.140 ± 0.022 | 86.0% | −2.4% |
| RF+Geo | 0.142 ± 0.021 | 85.8% | −4.4% |
| **RF+Geo+Leg** | **0.119 ± 0.030** | **88.1%** | −3.8% |
| YOLO | 0.223 ± 0.048 | 77.9% | −4.2% |

- 모든 분할에서 오차 순서는 RF 계열 < YOLO < Geo. RF+Geo+Leg가 가장 좋고 안정적임.
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
