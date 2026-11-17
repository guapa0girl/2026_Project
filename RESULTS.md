# 결과 — visiting 판정 5가지 방법 비교 (동일 조건)

돈방 CCTV 사진에서 사료통을 차지한(visiting) 돼지 수를 세는 방법 5가지를 **같은 train 사진, 같은 test(000)**로 비교한 결과.
진행 과정 전체 기록은 [PROGRESS.md](PROGRESS.md).

---

## 1. 데이터

| task | 카메라 | 사진 | 돼지 (visiting / other) | 용도 |
|---|---|---|---|---|
| 001 | CH005 (1920×1080) | 90 | 899 (136 / 763) | train / valid |
| 002 | CH010 (1920×1080) | 100 | 990 (179 / 811) | train / valid |
| 000 | 1분 샘플 영상 (960×720) | 86 | 860 (133 / 727) | **test** (학습·선택에 전혀 안 씀) |

- 모든 돼지에 사람이 CVAT로 관절 18개 스켈레톤과 State(visiting / other)를 라벨링함.
- test(000)는 학습 카메라(CH005·CH010)와 다른 카메라·해상도임 → 처음 보는 화면에서의 성능.
- 000은 사진당 visiting 0\~2마리(평균 1.55).

## 2. 분할 (train / valid / test)

test는 세 분할 모두 **000 86장으로 고정**. 001+002 190장만 train / valid로 나눔.
분할 목록은 YOLO 실험(`yolo_det/scripts/prepare_data.py`)과 같은 함수로 만들어(`training/scripts/make_study_splits.py`), 포즈 모델·RF·YOLO가 **한 장까지 같은 train 사진**으로 학습함.

| 분할 | 나누는 방법 | train : valid (장) | 비율 |
|---|---|---|---|
| 랜덤 | 001+002를 seed 42로 섞어 15%를 valid | 162 : 28 | 85 : 15 |
| 시간대 | 001·002 각각 마지막 클립을 통째로 valid | 159 : 31 | 84 : 16 |
| 시간대 5-fold | 001·002 각각 시간순 5구간, fold마다 한 구간이 valid (5번 학습) | 148\~155 : 34\~42 | 78\~82 : 18\~22 |

- 랜덤 분할은 같은 클립의 비슷한 사진이 train과 valid에 갈림. 시간대 분할은 이를 막으려고 클립 단위로 뗌.
- **5-fold 평균을 최종 비교 기준으로 씀**: 모델 5개의 평균이라 train 사진에 따른 흔들림이 반영됨.

각 모델이 사진을 쓰는 방식:

| 모델 | train | valid | test (000) |
|---|---|---|---|
| 포즈 모델 (Geo · RF 계열이 사용) | 학습 | best.pt(가장 좋은 epoch) 선택 | 채점 |
| RF 계열 | 학습 | **쓰지 않음** (고를 하이퍼파라미터가 없음) | 채점 |
| YOLO | 학습 | best.pt 선택 + conf 임계값 선택 | 채점 |

## 3. 진행 방법

### 3-1. 포즈 모델 (YOLOv8-pose)

Geo와 RF 계열은 돼지마다 찍힌 관절 좌표로 판정하므로 포즈 모델이 먼저 필요함. **분할마다 따로 학습**함(랜덤 1개, 시간대 1개, 5-fold 5개, 모두 7개).

| 항목 | 값 |
|---|---|
| 모델 | YOLOv8m-pose (medium), 관절 18개 |
| 시작 가중치 | COCO 사전학습 `yolov8m-pose.pt` |
| epoch | 200, 조기 종료 끔 (`--patience 0`) |
| 입력 크기 / batch | 640 / 8 |
| 옵티마이저 | `auto` → AdamW (lr 0.002) |
| best.pt | valid pose mAP50-95가 가장 높은 epoch |
| 추론 | 검출 conf 0.4, imgsz 640 |

- **medium · epoch 200을 고른 이유**: 랜덤 분할에서 nano / small / medium / large × epoch 100 / 200 / 300을 비교함. medium ep200이 test PCK@0.1(0.710)·PCK@0.2(0.870)·box mAP50-95(0.672)에서 가장 좋았음. epoch 300은 valid만 오르고 000에서는 이득이 없었음. 비교표는 [PROGRESS.md](PROGRESS.md) 2장.

분할별 포즈 모델 성능 (test = 000):

| 분할 | train : valid | pose mAP50-95 | PCK@0.1 | PCK@0.2 | box mAP50-95 |
|---|---|---|---|---|---|
| 랜덤 | 162 : 28 | 0.100 | 0.710 | 0.870 | 0.672 |
| 시간대 | 159 : 31 | 0.079 | 0.652 | 0.839 | 0.657 |
| 5-fold 평균 (f0\~f4) | 148\~155 : 34\~42 | 0.108 | 0.696 | 0.859 | 0.641 |

- PCK@0.1 / 0.2 = 찍은 관절이 정답에서 돼지 박스 긴 변의 10% / 20% 안에 들어온 비율.
- 관절별로는 목·몸통이 가장 정확하고(PCK@0.2 0.95 이상), 코 0.87, 발이 가장 부정확함(앞발 PCK@0.1 약 0.27, 랜덤 분할 모델 기준).

### 3-2. 판정 방법 5가지

| 방법 | 입력 | 판정 | 학습 |
|---|---|---|---|
| **Geo** | 포즈 관절 + 사료통 원 | 규칙: ① 코·목 관절 신뢰도 ≥ 0.5 ② 코가 사료통 원 안 ③ 목→코 벡터와 코→사료통 중심 벡터 사이 각도 ≤ 60° | 없음 |
| **RF** | 포즈 관절 + 사료통 구역 | 자세 특징 6개 → RandomForest | train 사진 |
| **RF+Geo** | 〃 | RF 특징 6개 + Geo 값 5개 → RandomForest | train 사진 |
| **RF+Geo+Leg** | 〃 | RF+Geo 특징 11개 + 다리 특징 5개 → RandomForest | train 사진 |
| **YOLO** | 사진만 | YOLOv8m detect가 돼지를 visiting / other 클래스로 바로 검출 | train 사진 (수아, `yolo_det/`) |

- **RF 특징 6개**: 코가 사료통 구역 안인지, 코-구역 거리(박스 대각선으로 나눔), 머리 숙임 각도, 척추 기울기, 목 상대 높이, 주요 관절 신뢰도.
- **Geo 값 5개**: 코-사료통 중심 거리 ÷ 반지름 r, 머리 방향 각도(Geo 규칙 ③의 각도), confidence 기반 누움 신호 3개.
- **Leg 특징 5개**: 다리-척추 각도, 다리 길이 ÷ 몸통 길이, 발-척추 거리 ÷ 몸통 길이, 네 발이 한쪽에 몰린 정도, 목-몸통-꼬리 각도.
- **RandomForest**: 나무 100개, `class_weight="balanced"`, seed 42, 판정 기준 확률 0.5. 하이퍼파라미터 튜닝은 하지 않음.
  - train 사진의 **사람이 찍은 관절과 라벨**로 학습함. 000에서는 포즈 모델이 찍은 관절에 적용함.
- **YOLO**: YOLOv8m detect, COCO 사전학습 `yolov8m.pt`, epoch 200, 조기 종료 끔, imgsz 640, batch 8, seed 42, NMS class-agnostic(IoU 0.7). 사진당 마릿수 = conf ≥ 임계값인 visiting 박스 수.
  - conf 임계값은 valid에서 사진당 MAE가 가장 작은 값: 랜덤 0.55, 시간대 0.85, 5-fold f0\~f4 0.65 / 0.50 / 0.35 / 0.25 / 0.70.
- 사료통 원(cx, cy, r)은 카메라마다 따로 지정함(`counting/configs/eval_cameras.yaml`). 000은 (453.4, 354.3, 169.2).

### 3-3. 평가 (`counting/scripts/evaluate_methods.py`, `count_metrics()`)

- **정답**: 000의 사람 라벨. 연구의 최종 출력이 "사진마다 visiting 몇 마리"라서 **마릿수**로 채점함.
- Geo · RF 계열: 포즈 모델이 찾은 돼지 전체를 판정해서 셈. 포즈 모델이 못 찾은 돼지는 visiting이 아닌 것으로 처리함(실제 운용과 같음).
- YOLO: 같은 000 사진, 같은 `count_metrics()`로 채점함.

| 지표 | 뜻 |
|---|---|
| **사진당 MAE** | 사진마다 \|모델 마릿수 − 사람 마릿수\|의 평균. 기본 지표. 86장이라 **0.012 ≈ 사진 1장이 1마리 틀림** |
| bias | 모델 − 사람의 평균. + 면 많이 셈, − 면 적게 셈 |
| 정확 일치 | 마릿수가 정확히 맞은 사진 비율 |
| 1분 총 점유량 오차 | 86장 visiting 합의 차이 ÷ 사람 합 |
| 10초 구간 오차 | 1분을 6구간으로 나눠 구간별 평균 마릿수의 차이(MAE) |

- 오차 원인은 개체 단위로 셈: 모델 돼지와 사람 돼지를 관절 거리로 짝지음(헝가리안, 기준 0.25). YOLO는 박스 IoU ≥ 0.5로 짝지음(헝가리안).

```powershell
cd training; python scripts\make_study_splits.py                                   # 분할 목록
powershell -ExecutionPolicy Bypass -File scripts\run_split_study.ps1              # 포즈 모델 (시간대, 5-fold)
cd ..\counting; powershell -ExecutionPolicy Bypass -File scripts\run_split_eval.ps1  # Geo·RF 계열 평가 + YOLO와 합쳐 비교
```

YOLO 학습·평가 명령은 [yolo_det/README.md](yolo_det/README.md).

---

## 4. 결과 (test = 000, 86장, 사람 visiting 133마리)

> 4장은 11/16 통일 기준 적용 **전** 결과임(포즈 seed·best.pt 기준이 다름). 통일 기준으로 다시 학습한 랜덤 · 시간대 5-fold 결과는 **[6장](#6-랜덤-vs-시간대-5-fold--1116-통일-기준으로-재학습)**.

### 4-1. 사진당 마릿수

5-fold는 fold 5개의 평균 ± 표준편차.

| 분할 | 방법 | 사진당 MAE | bias | 정확 일치 | 1분 총 점유량 오차 | 10초 구간 오차 |
|---|---|---|---|---|---|---|
| **시간대 5-fold** | Geo | 0.288 ± 0.013 | −0.270 | 72.3 ± 1.3% | −17.4 ± 1.3% | 0.220 ± 0.018 |
| | RF | 0.140 ± 0.022 | −0.037 | 86.0 ± 2.2% | −2.4 ± 2.2% | 0.084 ± 0.027 |
| | RF+Geo | 0.142 ± 0.021 | −0.067 | 85.8 ± 2.1% | −4.4 ± 2.7% | 0.087 ± 0.014 |
| | **RF+Geo+Leg** | **0.119 ± 0.030** | −0.058 | **88.1 ± 3.0%** | −3.8 ± 2.4% | **0.081 ± 0.014** |
| | YOLO | 0.223 ± 0.048 | −0.065 | 77.9 ± 4.4% | −4.2 ± 1.6% | 0.111 ± 0.042 |
| 랜덤 | Geo | 0.279 | −0.279 | 73.3% | −18.0% | 0.234 |
| | RF | 0.105 | −0.012 | 89.5% | −0.8% | 0.062 |
| | RF+Geo | 0.105 | −0.012 | 89.5% | −0.8% | 0.062 |
| | **RF+Geo+Leg** | **0.093** | **0.000** | **90.7%** | **0.0%** | 0.069 |
| | YOLO | 0.198 | −0.151 | 80.2% | −9.8% | 0.150 |
| 시간대 | Geo | 0.279 | −0.279 | 73.3% | −18.0% | 0.217 |
| | **RF** | **0.093** | −0.047 | **90.7%** | −3.0% | 0.069 |
| | RF+Geo | 0.151 | −0.058 | 84.9% | −3.8% | 0.075 |
| | RF+Geo+Leg | 0.140 | −0.047 | 86.0% | −3.0% | 0.069 |
| | YOLO | 0.186 | −0.163 | 81.4% | −10.5% | 0.130 |

![분할 방식별 비교](counting/reports/split_comparison/split_comparison.png)

5-fold의 fold별 사진당 MAE:

| fold | Geo | RF | RF+Geo | RF+Geo+Leg | YOLO |
|---|---|---|---|---|---|
| f0 | 0.302 | 0.174 | 0.174 | 0.163 | **0.151** |
| f1 | 0.302 | 0.140 | 0.128 | **0.116** | 0.279 |
| f2 | 0.279 | 0.140 | 0.128 | **0.105** | 0.233 |
| f3 | 0.279 | 0.128 | 0.151 | **0.081** | 0.244 |
| f4 | 0.279 | **0.116** | 0.128 | 0.128 | 0.209 |

### 4-2. 오차 원인 (개체 단위, 5-fold는 평균 마릿수)

| 분할 | 방법 | 못 찾은 visiting | 찾았지만 other로 판정 | other를 visiting으로 | 라벨 없는 검출을 visiting으로 |
|---|---|---|---|---|---|
| 시간대 5-fold | Geo | 6.4 | 19.2 | 1.0 | 1.4 |
| | RF | 6.4 | 3.2 | 4.0 | 2.4 |
| | RF+Geo | 6.4 | 4.0 | 3.0 | 1.6 |
| | RF+Geo+Leg | 6.4 | 3.0 | 2.4 | 2.0 |
| | YOLO | 5.4 | 10.8 | 8.4 | 2.2 |
| 랜덤 | Geo | 9 | 18 | 0 | 3 |
| | RF | 9 | 1 | 4 | 5 |
| | RF+Geo | 9 | 0 | 4 | 4 |
| | RF+Geo+Leg | 9 | 0 | 4 | 5 |
| | YOLO | 8 | 11 | 2 | 4 |
| 시간대 | Geo | 4 | 21 | 0 | 1 |
| | RF | 4 | 4 | 3 | 1 |
| | RF+Geo | 4 | 6 | 4 | 1 |
| | RF+Geo+Leg | 4 | 5 | 4 | 1 |
| | YOLO | 14 | 3 | 2 | 1 |

- "못 찾은 visiting"은 Geo · RF 계열이 같은 포즈 모델을 쓰므로 4개 방법이 같은 값임. YOLO는 자기 검출기라 값이 다름.

### 4-3. 방법을 바꿨을 때 MAE 변화

"A → B"는 A 대신 B를 썼을 때를 뜻함. 칸 값은 B의 MAE − A의 MAE. − 면 좋아짐. fold별은 5-fold의 fold 5개 중 B가 좋아짐 / 같음 / 나빠짐인 횟수.

| 비교 | 랜덤 | 시간대 | 5-fold 평균 | fold별 |
|---|---|---|---|---|
| Geo → RF | −0.174 | −0.186 | −0.148 | 5 / 0 / 0 |
| RF → RF+Geo | 0.000 | +0.058 | +0.002 | 2 / 1 / 2 |
| RF+Geo → RF+Geo+Leg | −0.012 | −0.011 | −0.023 | 4 / 1 / 0 |
| YOLO → RF+Geo+Leg | −0.105 | −0.046 | −0.104 | 4 / 0 / 1 |

---

## 5. 결론

- **오차 순서는 모든 분할에서 RF 계열 < YOLO < Geo.** 5-fold 평균 사진당 MAE는 RF+Geo+Leg 0.119, RF 0.140, RF+Geo 0.142, YOLO 0.223, Geo 0.288.
- **RF+Geo+Leg가 가장 좋고 안정적임.** 5-fold 평균 1위, 랜덤 1위, 시간대 2위. RF+Geo에 Leg를 더하면 나빠진 fold가 없음(4 / 1 / 0).
  - 다만 RF / RF+Geo와의 차이는 사진 1\~3장 수준(MAE 0.01\~0.03)임. 5-fold 표준편차(±0.02\~0.03)와 비슷해서 분할 하나만 보면 순위가 바뀜(시간대 분할에서는 RF가 1위).
- **YOLO는 돼지를 찾는 능력은 비슷하지만 visiting / other 구분에서 많이 틀림.** 5-fold에서 판정 오류(찾았지만 other로 10.8 + other를 visiting으로 8.4)가 RF+Geo+Leg(3.0 + 2.4)의 약 3.5배임. 사료통 위치를 쓰지 않고 사진만 보고 판정하기 때문으로 보임. train 사진에 따른 흔들림도 가장 큼(±0.048).
- **Geo(60° 규칙)는 항상 약 18% 적게 셈.** 찾은 visiting 중 18\~21마리를 방향 조건에서 탈락시킴. 학습이 없어서 분할과 상관없이 결과가 거의 같음.
- **Geo · RF 계열에 남은 오차의 상당 부분은 포즈 모델 단계에서 생김**(못 찾은 visiting 4\~9마리, 라벨 없는 검출). 판정보다 검출을 개선하는 것이 다음 과제임.
- **한계**: test가 1분 영상 하나(000, 86장)라 장면이 다양하지 않음. 학습 사진도 190장이라 train이 조금만 바뀌어도 결과가 흔들림.

결과 파일: `counting/reports/split_comparison/` (`split_comparison.csv` / `.md` / `.png`, `split_folds.csv`, `pose_by_split.csv`), YOLO는 `yolo_det/reports/ep200*/m/summary.json`.

---

## 6. 랜덤 vs 시간대 5-fold — 11/16 통일 기준으로 재학습

### 6-1. 조건 — 5가지 방법을 어떻게 같은 조건으로 맞췄나

**분할 (랜덤 · 시간대 공통)**

| 항목 | 값 |
|---|---|
| 시간대 `k5_f0~f4` | 001 · 002를 각각 시간순 5구간으로 나눠 fold마다 한 구간이 valid |
| 랜덤 `rk5_f0~f4` | task마다 seed 42로 섞고, fold마다 train · valid 장수를 `k5_f<i>`와 task별로 똑같이 맞춤 |
| train / valid 장수 | f0 148 / 42, f1 155 / 34, f2 151 / 37, f3 151 / 37, f4 149 / 40 (두 분할 동일) |
| test | 000 86장 (visiting 133마리) 고정. 학습 · 선택에 쓰지 않음 |

랜덤과 시간대는 **valid를 어떻게 고르는지만 다름**.

**YOLO와 Geo · RF 계열이 같은 항목**

| 항목 | YOLO (수아) | 포즈 모델 → Geo · RF 계열 (서인) | 확인 방법 |
|---|---|---|---|
| 분할 목록 | `prepare_data.py --split kfold\|rkfold` | `make_study_splits.py` (같은 함수 import) | 정렬된 `<task>__<사진>` 목록의 SHA-256 앞 12자리, 10개 fold 모두 일치 |
| test 정답 | 000 86장 | 000 86장 | 수아 000 사진별 csv의 gt가 우리 라벨과 86장 모두 일치 |
| 모델 크기 · 시작 | YOLOv8m detect, COCO `yolov8m.pt` | YOLOv8m-pose, COCO `yolov8m-pose.pt` | 같은 크기, 둘 다 COCO에서 처음부터 |
| epoch / 조기 종료 | 200 / 끔 | 200 / 끔 (`--patience 0`) | 학습 코드 기본값 |
| imgsz / batch / seed / optimizer | 640 / 8 / 42 / auto | 640 / 8 / 42 / auto | 학습 코드 기본값 · `train_medium.yaml` |
| val로만 고르는 것 | conf 임계값 (val MAE 최소) | best.pt (val pose mAP50-95 최고), RF best (5-fold 평균 val MAE 최소) | test는 선택에 미사용 |
| 지표 | `count_metrics()` | `count_metrics()` | YOLO `evaluate.py`가 같은 함수를 import |
| 라이브러리 | torch 2.11.0+cu128, ultralytics 8.4.108 | 같음 | |

**방법이 달라서 생기는 차이 (맞출 대상이 아님)**

| 항목 | YOLO | Geo · RF 계열 |
|---|---|---|
| 판정에 쓰는 정보 | 사진만 | 포즈 관절 18개 + 카메라별 사료통 좌표 (`configs/eval_cameras.yaml`) |
| 학습 라벨 | 스켈레톤 점의 최소 · 최대 좌표로 만든 박스 + visiting / other | 관절 18개 (포즈), visiting / other (RF). 같은 CVAT 라벨에서 만듦 |
| 임계값 | conf를 val에서 고름 | 포즈 검출 conf 0.4, RF 기준 0.5 고정 |
| best.pt 기준 | box (detect 기본) | pose mAP50-95 (box 제외) |
| valid 사진 사용 | best.pt · conf 선택 | 포즈 best.pt 선택. RF는 train 사진으로만 학습 (나무 100개, balanced, seed 42) |

**실행 환경 (결과 비교에는 영향 거의 없음)**: YOLO는 RTX 6000 Ada · Python 3.10.12, 포즈 · RF는 RTX 5060 8GB · Python 3.14. GPU가 달라 seed가 같아도 비트 단위로 같은 결과는 아님. 추론 시간은 GPU가 달라 직접 비교하지 않음.

- YOLO 시간대 `ep200_k5_f*`는 11/14\~15 학습 후 11/17 재평가, 랜덤 `ep200_rk5_f*`는 11/17 학습 · 평가. 값은 수아 노션 표와 000 사진별 카운트 csv(`counting/reports/fold_study/inputs/`)에서 가져옴.
- 확인 필요: 수아 노션 YOLO **val** 표에서 랜덤 f2 · f3 · f4의 MAE · RMSE가 시간대와 소수점까지 같음(conf · mAP는 다름). 옮겨 적다 생긴 실수로 보이며 test 값 · 결론에는 영향 없음.

### 6-2. 사진당 마릿수 MAE / RMSE

> 논문용 지표 전체(MAE · RMSE · ME · Acc · Acc±1 · TCE · 10초 구간 MAE · visiting Precision / Recall / F1, test MAE 95% bootstrap CI, Wilcoxon 짝비교, 오류 분해, 포즈 box / pose mAP · PCK@0.05 / 0.1 / 0.2, YOLO 검출 지표)는 **[paper_metrics.md](counting/reports/fold_study/paper/paper_metrics.md)** (`counting/scripts/paper_tables.py`로 생성).

| 방법 | 랜덤 val MAE | 랜덤 test MAE | 랜덤 test RMSE | 랜덤 test 정확 일치 | 시간대 val MAE | 시간대 test MAE | 시간대 test RMSE | 시간대 test 정확 일치 |
|---|---|---|---|---|---|---|---|---|
| Geo | 0.566 ± 0.045 | 0.293 ± 0.028 | 0.562 ± 0.025 | 71.9 ± 2.8% | 0.575 ± 0.148 | 0.293 ± 0.022 | 0.562 ± 0.020 | 71.9 ± 2.2% |
| RF | 0.389 ± 0.136 | **0.114 ± 0.015** | **0.343 ± 0.032** | **88.8 ± 1.3%** | 0.632 ± 0.252 | 0.116 ± 0.032 | 0.338 ± 0.046 | 88.4 ± 3.2% |
| RF+Geo | 0.259 ± 0.139 | 0.133 ± 0.033 | 0.362 ± 0.047 | 86.7 ± 3.3% | 0.349 ± 0.122 | 0.126 ± 0.019 | 0.353 ± 0.028 | 87.4 ± 1.9% |
| **RF+Geo+Leg** (RF best) | **0.209 ± 0.133** | 0.121 ± 0.033 | 0.345 ± 0.051 | 87.9 ± 3.3% | **0.284 ± 0.104** | **0.109 ± 0.021** | **0.329 ± 0.032** | **89.1 ± 2.1%** |
| YOLO | 0.251 ± 0.128 | 0.300 ± 0.137 | 0.545 ± 0.125 | 70.5 ± 13.6% | 0.347 ± 0.166 | 0.223 ± 0.048 | 0.475 ± 0.059 | 77.9 ± 4.4% |

- YOLO val은 conf를 같은 valid에서 골라 낙관적임. 랜덤 val MAE 0.251은 fold별 값(소수 3자리)의 평균이라 수아 노션 0.252와 반올림 차이.

YOLO 검출 성능 (box, 평균 ± 표본 표준편차):

| 분할 | conf | val mAP50-95 | test P | test R | test mAP50 | test mAP50-95 |
|---|---|---|---|---|---|---|
| 랜덤 5-fold | 0.48 ± 0.15 | 0.713 ± 0.024 | 0.900 ± 0.039 | 0.843 ± 0.036 | 0.867 ± 0.029 | 0.597 ± 0.024 |
| 시간대 5-fold | 0.49 ± 0.19 | 0.610 ± 0.042 | 0.910 ± 0.033 | 0.859 ± 0.022 | 0.879 ± 0.008 | 0.600 ± 0.016 |

fold별 값은 [fold_study.md 8장](counting/reports/fold_study/fold_study.md).

![MAE 랜덤 vs 시간대](counting/reports/fold_study/graph_B_mae_random_vs_time.png)

### 6-3. 실제로 센 마릿수 (fold 5개 합계, GT = 사람)

| 분할 · 세트 | GT | Geo | RF | RF+Geo | RF+Geo+Leg | YOLO |
|---|---|---|---|---|---|---|
| 랜덤 · val | 315 | 281 (−10.8%) | 357 (+13.3%) | 331 (+5.1%) | **317 (+0.6%)** | – |
| 랜덤 · test | 665 (133 × 5) | 541 (−18.6%) | 650 (−2.3%) | 636 (−4.4%) | 643 (−3.3%) | 638 (−4.1%) |
| 시간대 · val | 315 | 273 (−13.3%) | 400 (+27.0%) | 340 (+7.9%) | **322 (+2.2%)** | – |
| 시간대 · test | 665 (133 × 5) | 547 (−17.7%) | 657 (−1.2%) | 643 (−3.3%) | 656 (−1.4%) | 637 (−4.2%) |

- YOLO 랜덤 test는 합계로는 −4.1%지만 fold마다 크게 흔들림: f2 101마리(−24.1%), f3 152마리(+14.3%). YOLO val 총 마릿수는 수아 자료에 없어 '–'.
- fold별 값은 [fold_study.md 6장](counting/reports/fold_study/fold_study.md).
- 000 사진별 카운트 csv: `counting/reports/fold_study/per_frame_000/<geo|rf|rf_geo|rf_geo_leg|yolo>.csv`. 열은 frame_idx, image, gt, k5_f0\~f4, rk5_f0\~f4로 수아 `000_per_frame_counts.csv`와 같은 형식임.

**test 사진별 카운트 (모델 × 사진 86장, GT와 함께)**: [per_image_000.md](counting/reports/fold_study/per_image/per_image_000.md) (`counting/scripts/per_image_000.py`로 생성)
- 모델별 csv · 그래프: `counting/reports/fold_study/per_image/by_model/<geo|rf|rf_geo|rf_geo_leg|yolo>.csv` · `.png` (칸 숫자 = 센 마릿수, 칸 색 = 오차, 맨 위 = GT)
- 전체 csv: `per_image_000_long.csv`(사진 × 분할 × fold × 방법 한 줄씩), `per_image_000_wide.csv`(사진 한 줄)
- 공통으로 틀리는 사진: 56\~58번(GT 2, 대부분 모델이 1로 셈), 34번 · 65번(GT 1, RF 계열이 2로 셈). Geo는 54\~61번 · 64\~74번에서 계속 덜 셈.

![사진별 카운트 (방법별)](counting/reports/fold_study/per_image/graph_E_per_image_counts.png)
![사진별 오차 (모델 50개)](counting/reports/fold_study/per_image/graph_F_per_image_error_heatmap.png)

![fold별 총 마릿수](counting/reports/fold_study/graph_C_total_counts.png)
![000 사진별 마릿수](counting/reports/fold_study/graph_A_frame_counts_000.png)

### 6-4. 포즈 모델 (keypoint)

| 분할 | val pose mAP50-95 | val PCK@0.1 | test pose mAP50-95 | test PCK@0.1 | test PCK@0.2 |
|---|---|---|---|---|---|
| 랜덤 5-fold | 0.287 ± 0.024 | 0.784 ± 0.018 | 0.111 ± 0.021 | 0.691 ± 0.029 | 0.865 ± 0.009 |
| 시간대 5-fold | 0.167 ± 0.020 | 0.695 ± 0.025 | 0.087 ± 0.010 | 0.692 ± 0.024 | 0.865 ± 0.011 |

- keypoint별로는 몸통(tail · hip · flank · shoulder · neck · ear)이 test PCK@0.1 0.72\~0.94, 다리 끝·귀 끝(foot · hand · ear_tip)이 0.26\~0.69로 낮음. keypoint별 표는 [fold_study.md 7장](counting/reports/fold_study/fold_study.md).

![keypoint별 PCK](counting/reports/fold_study/graph_D_keypoint_pck.png)

### 6-5. 정리

- **test(000)에서 RF 계열은 랜덤과 시간대 차이가 거의 없음** (0.11\~0.13). Geo도 0.293으로 같음.
- **YOLO는 랜덤 5-fold test에서 흔들림이 큼** (MAE 0.300 ± 0.137, rk5_f2 0.419, rk5_f3 0.477). box mAP50-95는 두 분할 모두 0.60으로 안정적이라, 흔들림은 검출이 아니라 conf 임계값과 visiting / other 판정에서 생김.
- test 순서: 시간대는 RF 계열 < YOLO < Geo, 랜덤은 RF 계열 < Geo ≈ YOLO.
- **val은 시간대가 랜덤보다 나쁘고 fold마다 크게 흔들림** (RF 0.389 → 0.632, k5_f3는 1.054). 시간대 분할은 학습 때 못 본 시간대를 valid로 두기 때문임. 포즈 val mAP50-95도 0.287 → 0.167로 떨어지지만 test PCK는 같음.
- **RF best는 두 분할 모두 RF+Geo+Leg**이고, 실제 센 마릿수도 val에서 GT와 가장 가까움(+0.6%, +2.2%). RF 단독은 val에서 13\~27% 많이 셈.
- Geo는 val · test 모두 약 11\~19% 적게 셈.

결과 파일: `counting/reports/fold_study/` (`fold_study.md`, `fold_metrics.csv`, `fold_summary.csv`, `pose_folds.csv`, `pose_keypoints.csv`, `fold_composition.csv`, `frames/ep200_<split>/<방법>_<val|test>.csv`, 그래프 A\~D). 명령: `training/scripts/run_split_study.ps1`, `counting/scripts/run_split_eval.ps1 -Valid`.

---

## 7. 추론 시간 (end-to-end, 사진 1장)

`counting/scripts/benchmark_speed.py` (11/16 통일 조건: 같은 장비 한 대, batch 1, imgsz 640, 예열 5장 제외, 000 86장 × 3회, 중앙값 · p95). 사진 읽기부터 마릿수까지. 포즈 = `ep200_k5_f0` best.pt, RF = 같은 분할 train 사진으로 학습. 측정 중 센 마릿수가 평가 결과(`per_frame_000`)와 86장 모두 같음을 확인함.

장비: GPU NVIDIA GeForce RTX 5060 (8GB), CPU AMD Ryzen 5 7500F, torch 2.11.0+cu128, ultralytics 8.4.108.

| 장비 | 방법 | 중앙값 (ms) | p95 (ms) | 사진 읽기 | 포즈 추론 | 특징 계산 | 판정 | 1 fps (≤1000 ms) | 30 fps (≤33.3 ms) |
|---|---|---|---|---|---|---|---|---|---|
| GPU | Geo | 20.7 | 22.6 | 8.7 | 11.3 | 0.6 | 0.00 | ✅ | ✅ |
| GPU | RF | 26.2 | 28.6 | 8.7 | 11.3 | 0.6 | 5.33 | ✅ | ✅ |
| GPU | RF+Geo | 26.0 | 28.9 | 8.7 | 11.3 | 0.6 | 5.12 | ✅ | ✅ |
| GPU | RF+Geo+Leg | 26.0 | 28.7 | 8.7 | 11.3 | 0.6 | 5.08 | ✅ | ✅ |
| CPU | Geo | 151.4 | 171.3 | 10.2 | 140.4 | 0.8 | 0.00 | ✅ | ❌ |
| CPU | RF | 158.5 | 178.6 | 10.2 | 140.4 | 0.8 | 7.31 | ✅ | ❌ |
| CPU | RF+Geo | 158.2 | 178.2 | 10.2 | 140.4 | 0.8 | 7.01 | ✅ | ❌ |
| CPU | RF+Geo+Leg | 158.2 | 178.4 | 10.2 | 140.4 | 0.8 | 6.95 | ✅ | ❌ |

- 단계별 칸은 중앙값(ms). 실시간 판정은 p95 기준.
- **시간 대부분은 포즈 추론**(GPU 11 ms, CPU 140 ms). RF 판정은 사진당 5\~7 ms(나무 100개, 그 사진의 돼지 전부를 한 번에), Geo 판정 · 특징 계산은 1 ms 미만.
- 사진 읽기 8.7 ms는 000이 PNG라서임(디코딩). 영상 스트림에서 프레임을 받으면 이 시간은 줄어듦.
- **YOLO**: 수아 측정 RTX 6000 Ada 중앙값 21.3\~22.3 ms, p95 22.4\~24.3 ms. 장비가 달라 위 표와 직접 비교하지 않음. 같은 장비 비교는 같은 스크립트에 `--yolo-weights <best.pt> --yolo-conf <그 fold conf>`를 주고 한 장비에서 다시 재야 함.

결과 파일: `counting/reports/speed/NVIDIA_GeForce_RTX_5060/` (`speed.md`, `speed_summary.csv`, `speed_raw.csv`).

---

## 8. 한계점 근거 (11/18 재학습 결과 기준)

**데이터 규모와 fold 간 분산**
- 학습 + valid 190장(001 90 + 002 100)뿐이라 train 사진이 조금만 바뀌어도 결과가 흔들림. test MAE의 fold 간 범위(최소\~최대):

| 방법 | 랜덤 5-fold | 시간대 5-fold |
|---|---|---|
| Geo | 0.256\~0.326 | 0.256\~0.314 |
| RF | 0.093\~0.128 | 0.081\~0.163 |
| RF+Geo | 0.093\~0.174 | 0.093\~0.140 |
| RF+Geo+Leg | 0.070\~0.151 | 0.081\~0.140 |
| YOLO | 0.174\~0.477 | 0.151\~0.279 |

- val은 더 흔들림: 시간대 RF 0.400\~1.054, 시간대 RF+Geo+Leg 0.167\~0.441.
- 포즈 test PCK@0.1: 랜덤 0.650\~0.726, 시간대 0.657\~0.719. val pose mAP50-95는 랜덤 0.287, 시간대 0.167 (못 본 시간대에서 크게 떨어짐).
- RF / RF+Geo / RF+Geo+Leg 사이 차이(test MAE 0.01\~0.02)는 fold 간 범위보다 작고 Wilcoxon p > 0.3 → 셋의 순위는 확정할 수 없음.

**test가 1분 영상 1개**
- 000 86장, 학습에 없는 카메라(SAMPLE, 960×720). 장면 하나라 다양한 상황을 대표하지 못함.
- 사진끼리 연속 프레임이라 독립이 아님 → 95% CI · p값은 낙관적(paper Table 3 · 4).

**촬영 시간대 · 카메라 편중** ([time_distribution.md](training/reports/data/time_distribution.md))
- 001 = CH005, 7/8 아침 06:15\~07:44(클립 4개, 약 1.5시간), 002 = CH010, 7/16 10:26\~18:32(클립 17개, 약 8시간).
- 002는 클립 17개 중 11개가 사진 1\~4장이고, 17:34 클립 하나가 26장 → 사진 수가 클립마다 크게 다름. 밤 시간대와 다른 날짜는 없음.

**가림 · 밀집 오차** ([error_cases.md](counting/reports/fold_study/error_cases/error_cases.md), `counting/scripts/error_cases.py`)
- Geo · RF 계열의 남은 오차 상당 부분은 포즈 단계: test에서 사람 visiting 133마리 중 fold당 6.2\~8.6마리를 포즈 모델이 매칭하지 못함.
- 가장 많이 놓친 사진은 56\~58번(사람 visiting 2마리, 10개 분할 모델 합 20개 중 11개 놓침). 사료통 급이 파이프 아래로 머리를 넣은 돼지는 검출 자체가 안 되고(no detection), 사료통 옆에 누운 · 겹친 돼지는 검출은 되지만 keypoint가 어긋나 매칭 실패(keypoints off).

![가림 · 밀집 오차 사례](counting/reports/fold_study/error_cases/error_cases.png)

**향후 계획 근거**
- 데이터 추가: 다른 날짜 · 밤 시간대 · 다른 카메라 (val pose mAP 0.287 → 0.167, 시간대 val RF 0.632).
- 검출 개선: 급이 파이프 가림 · 밀집 상황 (포즈 단계 미검출 fold당 6\~9마리).
- YOLO conf 임계값 안정화: conf가 fold마다 0.25\~0.70으로 바뀌고 랜덤 rk5_f3는 152마리(+14.3%)로 과대 집계.
- 실시간: GPU에서는 Geo · RF 계열 모두 30 fps 기준(p95 ≤ 33.3 ms) 통과, CPU는 1 fps만 통과 → 엣지 장비는 포즈 모델 경량화(n · s 크기) 필요.
