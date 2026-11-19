# yolo_det — YOLO 직접 판정 (visiting / other)

포즈(관절) 없이 YOLOv8 **detect** 모델이 돼지를 `other(0) / visiting(1)`로 바로 검출한다.
루트 README 비교표의 5번째 방법 **YOLO**. Geo / RF 계열과 같은 데이터(학습 001+002, test 000)·같은 지표(사진당 visiting 마릿수)로 비교한다.

| | 기존 (Geo / RF 계열) | YOLO 직접 판정 |
|---|---|---|
| 구조 | pose 모델로 관절 18개 → 특징 → 규칙 / RF 판정 (2단계) | 사진 → 박스 + visiting/other (1단계) |
| 판정 근거 | 사람이 설계한 특징 (코 위치, 머리 방향, 다리 모양) | 모델이 이미지에서 학습 |

---

## 1. 폴더 구조

```
yolo_det/
├── configs/pig_det.yaml      클래스 이름 (0 other / 1 visiting — evaluate_methods.py STATE_MAP과 같은 번호)
├── scripts/
│   ├── common.py             경로·상수
│   ├── prepare_data.py       CVAT zip → YOLO detect 라벨 + 분할 + 사람 정답 마릿수
│   ├── train.py              학습
│   └── evaluate.py           test(000) 평가 → reports/
├── data/                     (자동 생성, git 제외)
├── models/<run>/<size>/      (자동 생성, git 제외) weights/best.pt
├── outputs/<run>/<size>/     (자동 생성, git 제외) 평가 상세 csv
└── reports/<run>/<size>/     평가 요약 summary.json / summary.md (git에 올림)
```

원본 데이터는 서인 레포 규칙대로 `counting/data/labels/source/<task>_sorted.zip`(CVAT for images 1.1)에서 읽는다.

---

## 2. 실행 (레포 루트에서)

```powershell
# 1) 데이터 준비 (CPU, 몇 초)
python yolo_det\scripts\prepare_data.py

# 2) 학습 (GPU) — pose 실험과 같은 조건: COCO 사전학습, 640, epoch 200, 조기 종료 끔
python yolo_det\scripts\train.py --sizes m --run ep200 --device 0
python yolo_det\scripts\train.py --sizes n s m l --run ep200 --device 0      # 사이즈 전부
python yolo_det\scripts\train.py --sizes m --run ep200_poseinit --init yolo_det\pig_pose_best.pt --device 0   # 서인 pose 가중치에서 시작

# 3) 평가 (test = 000)
python yolo_det\scripts\evaluate.py --run ep200 --size m
```

- 동작만 확인: `train.py --sizes n --epochs 1 --run smoke --device cpu` → `evaluate.py --run smoke --size n --device cpu`
- 끊긴 학습 이어 가기: `train.py --sizes m --run ep200 --resume`
- `--init`: 모양이 맞는 층(backbone·neck)만 옮기고 detect head는 새로 학습. 가중치 크기와 `--sizes`가 다르면 멈춤. 평가는 `evaluate.py --run ep200_poseinit --size m`.
- Windows에서 데이터로더 오류가 나면 `--workers 0`
- 리눅스 서버에서 `ConnectionResetError: [Errno 104]` 또는 존재하는 사진에 `Image Not Found`가 나면 열 수 있는 파일 수 제한(`ulimit -n` 1024) 때문 → 학습 전에 `ulimit -n $(ulimit -Hn)`, 그래도 나면 `--workers 2`

---

## 3. 데이터 준비 규칙 (`prepare_data.py`)

- **박스**: pose 라벨과 같은 규칙 — skeleton 점(화면 밖 제외)의 최소·최대 범위 → 0~1 정규화 → 이미지 경계로 clip.
  가짜 데이터로 `training/` 파이프라인 결과와 비교해 **박스 완전히 동일** 확인.
- **클래스**: State 속성 → `visiting(feeding)`, `visitig(feeding)`(CH010 오타), `visiting` = 1 / `other` = 0.
  State가 없거나 Unlabeling인 돼지가 있으면 중단(그 돼지가 배경으로 학습되는 것 방지).
- **분할**: `training/scripts/split_dataset.py --test-zip ... --valid 0.15 --seed 42` 와 **같은 사진으로** 나뉘도록 재현
  → train 162 / valid 28 / test 86(000). pose 모델과 같은 사진으로 학습해서 비교가 공정함.
- **시간대 분할 (`--split sorted`)**: random 분할은 같은 영상 클립의 비슷한 사진이 train과 valid에 갈려서
  valid 점수와 conf 임계값이 낙관적으로 나올 수 있음 → 001·002를 각각 시간순으로 정렬하고 **마지막 클립을 통째로** valid로 뗌
  (떼는 클립 수는 valid 장수가 15%에 가장 가깝게). 결과: 001 마지막 1클립 20장 + 002 마지막 2클립 11장
  → train 159 / valid 31 / test 86(000), train과 valid에 겹치는 클립 없음.
  ```bash
  python yolo_det/scripts/prepare_data.py --split sorted
  python yolo_det/scripts/train.py --sizes m --run ep200_sorted --device 0
  python yolo_det/scripts/evaluate.py --run ep200_sorted --size m
  ```
  - 서인 pose 가중치는 random 분할로 학습됐으므로 `--split sorted` 상태에서 `--init`을 주면 train.py가 멈춤(valid 누수 방지).
  - 분할 방식은 `data/split.json`에 기록되고 train.py가 `models/<run>/<size>/split.json`으로 복사함.
    evaluate.py는 지금 `data/`가 학습 때와 다른 분할이면 멈춤 → 예전 random run을 다시 평가하려면 `--split random`으로 다시 준비.
- **사진 이름의 시각 = 실제 촬영 시각 − 9시간** (실제 KST = 이름 시각 + 9시간. 예: 002 `20260716012647` → 실제 10:26:47,
  001 `20260707211528` → 실제 7/8 06:15:28). 분할은 이름 시각을 정렬에만 쓰므로 결과 영향 없음.
  task별 실제 시간대 분포와 분할별 valid 위치: `training/reports/data/time_distribution.md` / `.png`.
- **시간대 5-fold (`--split kfold`)**: 001·002를 각각 시간순으로 5개 연속 구간으로 나누고 fold마다 한 구간이 valid
  (경계는 가까운 클립 경계에 맞추고, 클립 중간에서 잘리면 valid와 붙은 사진 1장을 train에서 뺌). test는 모든 fold에서 000.
  fold마다 데이터 폴더를 따로 써서(`YOLO_DET_DATA=data_k5f<i>`) GPU 여러 개로 동시에 돌려도 안전.
  ```bash
  bash yolo_det/scripts/run_kfold.sh 0 0 1 2      # GPU 0에서 fold 0,1,2
  bash yolo_det/scripts/run_kfold.sh 1 3 4        # GPU 1에서 fold 3,4
  python yolo_det/scripts/summarize_folds.py --prefix ep200_k5 --k 5   # fold별 표 + 평균 ± 표준편차
  ```
  | fold | train | valid (visiting) |
  |---|---|---|
  | 0 | 148 | 42 (104) |
  | 1 | 155 | 34 (55) |
  | 2 | 151 | 37 (44) |
  | 3 | 151 | 37 (61) |
  | 4 | 149 | 40 (51) |
- **재평가·추론 시간 (`run_kfold.sh`)**:
  ```bash
  EVAL_ONLY=1 bash yolo_det/scripts/run_kfold.sh 0 0 1 2 3 4                 # 시간대 재평가 (재학습 없음, val 결과 추가)
  SPLIT=rkfold TIMING=1 bash yolo_det/scripts/run_kfold.sh 0 0 1 2 3 4       # 랜덤 5-fold 학습·평가 + 추론 시간
  python yolo_det/scripts/summarize_folds.py --prefix ep200_k5 --k 5           # val 표 + test(000) 표
  python yolo_det/scripts/summarize_folds.py --prefix ep200_rk5 --k 5
  ```
  `TIMING=1`: batch 1, 640, 예열 5장 제외, 000 × 3회, 사진 읽기→마릿수 end-to-end, 중앙값·p95(ms). 같은 GPU 한 대로 잴 것.
- **사람 정답 마릿수**: clip 전 XML 기준으로 `data/gt_counts.csv`에 저장(evaluate.py가 사용).

---

## 4. 평가 규칙 (`evaluate.py`)

| 항목 | 내용 |
|---|---|
| 사진당 visiting 마릿수 | conf ≥ 임계값인 visiting 박스 수 |
| NMS | **agnostic** (한 마리에 visiting·other 박스가 겹쳐 남지 않게) |
| conf 임계값 | **valid 28장**에서 사진당 MAE가 가장 작은 값을 골라 test에 고정 (test를 보고 고르지 않음) |
| 지표 | `counting/scripts/evaluate_methods.py`의 `count_metrics()`를 그대로 import — MAE, bias, 정확 일치, ±1 이내, 총 점유량 오차 + 구간(6등분 ≈ 10초) 오차 |
| 진단 | 사람 박스와 IoU ≥ 0.5로 짝지어 못 찾은 visiting / other로 판정 / 잘못 visiting / 라벨에 없는 박스 |
| 참고 | Ultralytics box mAP50, mAP50-95 (클래스별) |

평가는 valid에도 같은 임계값을 적용해 val 결과(P, R, mAP50, mAP50-95, MAE, RMSE, 정확 일치율)를 `summary.json`의 `valid`에 남기고
(임계값을 valid에서 골랐으므로 낙관적), 000 사진별 카운트를 팀 통일 형식(`image, frame_idx, gt, pred, error=pred−gt`)으로
`reports/<run>/<size>/test_frames.csv`에 저장한다(git에 올림).

결과의 `reports/<run>/<size>/summary.md` 표 한 줄을 루트 README·PROGRESS.md 비교표의 "YOLO (수아)" 줄에 옮긴다.

- **버전 기록**: train.py가 `models/<run>/<size>/env.json`에 python·torch·CUDA·ultralytics 버전, GPU, git 커밋,
  시작 가중치 sha256(앞 16자리), 학습 설정을 저장 → evaluate.py가 `summary.json`(`env_train`, `env_eval`)과
  `summary.md`의 "버전" 줄에 옮겨 적음. 기록 기능 전 run(ep200, ep200_poseinit)은 나중에 같은 형식으로 채워 넣음(`"backfilled": true`).

---

## 5. 결과

test = 000(1분 샘플 영상 86장, 학습에 없는 카메라). 모든 run 공통: YOLOv8m detect, imgsz 640, 조기 종료 끔, seed 42.
conf 임계값은 각 run의 valid에서 사진당 MAE가 가장 작은 값(000은 보지 않음). 요약: `reports/<run>/m/summary.md` / `.json`.
환경(연구실 서버 RTX 6000 Ada): python 3.10.12, torch 2.11.0+cu128, CUDA 12.8, ultralytics 8.4.108. 학습 1회 ep200 약 8분, ep300 약 12~20분.
서인 pose 가중치 `pig_pose_best.pt`(YOLOv8m-pose, ep200, seed 42 random 분할 train 162장으로 학습 — 서인 확인) sha256 앞 16자리 `5e23e8ccc106557a`.
(seed 42는 **분할** seed임. 이 모델의 학습 seed는 0(Ultralytics 기본, `training/models/results/ep200/m/args.yaml`), best.pt는 valid pose + box mAP50-95 합 기준 — 11/18 서인 확인)
`--init` 시 backbone·neck만 옮기고 detect head는 새로 학습(Transferred 469/475).

### 사진당 visiting 마릿수 (000)

| run | 시작 가중치 | 분할 (train / valid) | epoch | conf | 사진당 MAE (bias) | 정확 일치 | ±1 이내 | 1분 총 점유량 오차 | 10초 구간 평균 오차 | box mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|---|
| `ep200` | COCO | 랜덤 (162 / 28) | 200 | 0.55 | 0.198 (−0.151) | 80.2% | 100.0% | −9.8% | 0.150 | 0.601 |
| `ep200_poseinit` | 서인 pose | 랜덤 (162 / 28) | 200 | 0.75 | 0.174 (−0.081) | 82.6% | 100.0% | −5.3% | 0.094 | 0.645 |
| `ep300` | COCO | 랜덤 (162 / 28) | 300 | 0.45 | 0.221 (+0.012) | 77.9% | 100.0% | +0.8% | 0.057 | 0.598 |
| `ep300_poseinit` | 서인 pose | 랜덤 (162 / 28) | 300 | 0.25 | 0.209 (−0.116) | 80.2% | 98.8% | −7.5% | 0.075 | 0.623 |
| `ep200_sorted` | COCO | 시간대 (159 / 31) | 200 | 0.85 | 0.186 (−0.163) | 81.4% | 100.0% | −10.5% | 0.130 | 0.614 |
| `ep300_sorted` | COCO | 시간대 (159 / 31) | 300 | 0.65 | 0.221 (+0.012) | 77.9% | 100.0% | +0.8% | 0.081 | 0.592 |
| `ep200_k5` (5-fold 평균 ± 표준편차) | COCO | 시간대 5-fold (148~155 / 34~42) | 200 | 0.49 ± 0.19 | 0.223 ± 0.048 (−0.065) | 77.9% ± 4.4% | 99.8% | −4.2 ± 1.6% | 0.111 ± 0.042 | 0.601 ± 0.016 |

기존 방법 (루트 README 표, 같은 000·같은 `count_metrics()`):

| 방법 | 사진당 MAE (bias) | 정확 일치 | 1분 총 점유량 오차 | 10초 구간 평균 오차 |
|---|---|---|---|---|
| Geo (방향 60°) | 0.279 (−0.279) | 73.3% | −18.0% | 0.234 |
| Geo (방향 90°) | 0.093 (−0.047) | 90.7% | −3.0% | 0.069 |
| RF | 0.105 (−0.012) | 89.5% | −0.8% | 0.062 |
| RF+Geo(+Leg) | **0.093 (0.000)** | **90.7%** | **0.0%** | 0.069 |

### 오류 내역 (visiting, 사람 박스와 IoU ≥ 0.5로 짝지음)

| run | 총 visiting (YOLO / 사람) | 많이 센 / 적게 센 사진 | 못 찾음 | 찾았지만 other로 판정 | other를 visiting으로 | 라벨 없는 박스를 visiting으로 | best.pt sha256 앞 16자리 |
|---|---|---|---|---|---|---|---|
| `ep200` | 120 / 133 | 2 / 15 | 8 | 11 | 2 | 4 | b6b947b0dc7b8d0a |
| `ep200_poseinit` | 126 / 133 | 4 / 11 | 11 | 4 | 5 | 3 | 913752abc90a46e8 |
| `ep300` | 134 / 133 | 10 / 9 | 4 | 6 | 7 | 4 | 32ecf88b1eba6b36 |
| `ep300_poseinit` | 123 / 133 | 3 / 14 | 4 | 11 | 1 | 4 | 8bd3bd14697680ee |
| `ep200_sorted` | 119 / 133 | 1 / 15 | 14 | 3 | 2 | 1 | f31b9b631ddd4ec3 |
| `ep300_sorted` | 134 / 133 | 10 / 9 | 10 | 5 | 9 | 7 | c2dfceed51502b00 |

`ep200`, `ep200_poseinit`은 버전 기록 기능 전(11/14 22:50 / 22:49) 학습 → 11/15에 `env.json`·`split.json`을 채워 넣고 다시 평가(결과 숫자 동일).
학습 설정(epoch 200, batch 8, seed 42)과 ultralytics 버전은 best.pt 안 기록에서, 환경은 같은 서버·같은 .venv 기준. git 커밋만 추정(학습 직후 커밋 323b212).

### 시간대 5-fold (`ep200_k5_f0`~`f4`, COCO 시작, epoch 200)

| fold | conf | 사진당 MAE (bias) | 정확 일치 | ±1 이내 | 못 찾음 | other로 판정 | other→visiting | 라벨 없는 박스 | box mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.65 | 0.151 (−0.105) | 84.9% | 100.0% | 6 | 9 | 5 | 1 | 0.603 |
| 1 | 0.50 | 0.279 (−0.047) | 73.3% | 98.8% | 5 | 15 | 14 | 2 | 0.610 |
| 2 | 0.35 | 0.233 (−0.070) | 76.7% | 100.0% | 3 | 14 | 10 | 1 | 0.583 |
| 3 | 0.25 | 0.244 (−0.058) | 75.6% | 100.0% | 5 | 11 | 7 | 4 | 0.585 |
| 4 | 0.70 | 0.209 (−0.047) | 79.1% | 100.0% | 8 | 5 | 6 | 3 | 0.621 |

전체: `reports/ep200_k5/m/folds_summary.md` / `.json`.

### 랜덤 5-fold (`ep200_rk5_f0`~`f4`, COCO 시작, epoch 200, 11/17)

`SPLIT=rkfold bash yolo_det/scripts/run_kfold.sh 0 0 1 2 3 4`. fold별 train / valid 장수는 시간대 5-fold와 같음(148/42, 155/34, 151/37, 151/37, 149/40), 분할 목록 해시 10개 서인 쪽과 일치.

| fold | conf | 사진당 MAE (bias) | RMSE | 정확 일치 | ±1 이내 | 총 visiting (사람 133) | 못 찾음 | other로 판정 | box mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.25 | 0.221 (−0.058) | 0.470 | 77.9% | 100.0% | 128 | 7 | 10 | 0.588 |
| 1 | 0.45 | 0.174 (−0.035) | 0.418 | 82.6% | 100.0% | 130 | 6 | 6 | 0.622 |
| 2 | 0.50 | 0.419 (−0.372) | 0.647 | 58.1% | 100.0% | 101 | 6 | 33 | 0.565 |
| 3 | 0.55 | 0.477 (+0.221) | 0.707 | 53.5% | 98.8% | 152 | 6 | 8 | 0.620 |
| 4 | 0.65 | 0.209 (−0.070) | 0.482 | 80.2% | 98.8% | 127 | 5 | 8 | 0.590 |
| 평균 ± SD | 0.48 ± 0.15 | 0.300 ± 0.137 | 0.545 ± 0.125 | 70.5 ± 13.6% | 99.5% | 127.6 ± 18.1 | 6.0 | 13.0 | 0.597 ± 0.024 |

- 시간대 5-fold 평균: MAE 0.223 ± 0.048, RMSE 0.475 ± 0.059, 정확 일치 77.9 ± 4.4%, box mAP50-95 0.601 ± 0.016.
- 랜덤은 box mAP50-95가 시간대와 같은데 f2(덜 셈) · f3(더 셈)에서 크게 틀림 → 흔들림은 검출이 아니라 conf 임계값 · visiting / other 판정에서 생김.
- val(conf를 같은 valid에서 골라 낙관적): 시간대 MAE 0.347 ± 0.166, 랜덤 0.252 ± 0.128, box mAP50-95 0.610 / 0.713.
- 값 출처: 수아 노션(11/17), 000 사진별 카운트 csv(`counting/reports/fold_study/inputs/`). RMSE · bias · 정확 일치 · ±1 이내는 그 csv에서 계산.
- Geo · RF 계열과의 비교(같은 분할, 통일 기준): [RESULTS.md](../RESULTS.md) 4장.

### 해석

- **YOLO는 모든 조건에서 RF 계열보다 나쁨.** 가장 좋은 fold(MAE 0.151)도 RF+Geo(0.093)에 못 미침.
  (통일 기준 랜덤 · 시간대 5-fold 평균으로도 같음. fold 단위 예외는 시간대 f0 하나로, YOLO 0.151이 같은 fold RF 단독 0.163보다 좋음.) 단, Geo / RF는 카메라마다
  사료통 원 좌표를 받고(`counting/configs/eval_cameras.yaml`, 000은 SAMPLE) YOLO는 사진만으로 판정 → 처음 보는 카메라에서 불리한 조건.
- **epoch 300은 이득 없음.** COCO·랜덤, COCO·시간대, pose·랜덤 세 조건 모두 사진당 MAE가 나빠짐 → ep200 유지
  (서인 pose 실험의 ep300 결론과 같음). 총 점유량·구간 오차가 좋아 보이는 run(`ep300`, `ep300_sorted`)은 많이 센 사진과
  적게 센 사진이 비슷해서 상쇄된 결과. 두 run은 지표가 우연히 같지만 best.pt 해시와 오류 내역이 다른 별개 모델.
- **pose 시작 vs COCO 시작, 랜덤 vs 시간대 분할 차이는 확정 불가.** 5-fold에서 valid 구간만 바꿔도 MAE가 0.151~0.279
  (틀린 사진 13~23장)로 흔들림 → run 하나씩의 1~3장 차이(예: COCO 0.198 vs pose 0.174)는 우연 범위.
- **흔들림의 원인은 판정·임계값.** box mAP50-95는 fold 사이에 안정적(±0.016)인데 conf 임계값은 valid에 따라
  0.25~0.85로 크게 바뀌고, 오류 대부분이 visiting/other 혼동(other로 판정 평균 11, other→visiting 평균 8 vs 못 찾음 평균 5).

### Geo / RF 계열과 같은 분할로 비교

Geo·RF 계열도 이 분할 3가지(같은 사진 목록, `training/scripts/make_study_splits.py`가 이 폴더의 분할 함수를 그대로 씀)로 다시 학습함.
포즈 모델과 RF를 각 분할의 train으로만 학습. 결과는 `counting/reports/split_comparison/`, 해석은 [PROGRESS.md](../PROGRESS.md) 3장 결과 4.

| 분할 | Geo | RF | RF+Geo | RF+Geo+Leg | YOLO |
|---|---|---|---|---|---|
| 랜덤 | 0.279 | 0.105 | 0.105 | **0.093** | 0.198 |
| 시간대 | 0.279 | **0.093** | 0.151 | 0.140 | 0.186 |
| 시간대 5-fold | 0.288 ± 0.013 | 0.140 ± 0.022 | 0.142 ± 0.021 | **0.119 ± 0.030** | 0.223 ± 0.048 |

(사진당 visiting 마릿수 MAE, test = 000) → 같은 조건에서도 YOLO는 RF 계열보다 나쁨. 5-fold 기준 fold 5개 중 4개에서 RF+Geo+Leg가 앞섬.

> 위 표의 Geo · RF 계열 시간대 5-fold 값은 11/18 포즈 재학습 **전** 값임. 11/16 통일 기준(포즈 seed 42 · best.pt = val pose mAP50-95, 랜덤 5-fold 추가)으로 다시 돌린 비교: 시간대 / 랜덤 5-fold test MAE RF+Geo+Leg 0.109 / 0.121, RF 0.116 / 0.114, RF+Geo 0.126 / 0.133, Geo 0.293 / 0.293, YOLO 0.223 / 0.300 → [RESULTS.md](../RESULTS.md) 4장.

### 주의 / 남은 일

- 임계값 안정화: 5-fold valid 예측을 모아 임계값 하나로 고르거나 고정값(예: 0.5)으로 재평가.
- pose vs COCO 확정: 랜덤 분할에서 seed를 바꿔 반복(pose 가중치가 seed 42 random 분할로 학습돼서 시간대 분할에서는
  `--init`을 쓸 수 없음 → 쓰려면 서인이 같은 분할로 pose 재학습).
- ~~RF 계열도 같은 방식의 변동 폭(교차검증)을 확인해야 같은 기준의 비교가 됨.~~ → 완료(위 "Geo / RF 계열과 같은 분할로 비교"). RF 계열 5-fold 표준편차 ±0.02\~0.03, YOLO ±0.05.
- 메모의 "8/1/1 분할" 의미(000 포함 여부) 확인 필요.