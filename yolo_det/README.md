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

결과의 `reports/<run>/<size>/summary.md` 표 한 줄을 루트 README·PROGRESS.md 비교표의 "YOLO (수아)" 줄에 옮긴다.

---

## 5. 결과

test = 000(1분 샘플 영상 86장, 학습에 없는 카메라). medium, epoch 200, 조기 종료 끔, seed 42, 분할 train 162 / valid 28.
연구실 서버 RTX 6000 Ada에서 학습 1회 약 8분. 요약: `reports/<run>/m/summary.md` / `.json`.

| run | 시작 가중치 |
|---|---|
| `ep200` | COCO 사전학습 `yolov8m.pt` |
| `ep200_poseinit` | 서인 pose 모델 `pig_pose_best.pt`(YOLOv8m-pose, ep200)의 backbone·neck (`--init`) |

### 사진당 visiting 마릿수 (000)

| 방법 | 사진당 MAE (bias) | 정확 일치 | ±1 이내 | 1분 총 점유량 오차 | 10초 구간 평균 오차 |
|---|---|---|---|---|---|
| Geo (방향 60°) | 0.279 (−0.279) | 73.3% | — | −18.0% | 0.234 |
| Geo (방향 90°) | 0.093 (−0.047) | 90.7% | — | −3.0% | 0.069 |
| RF | 0.105 (−0.012) | 89.5% | — | −0.8% | 0.062 |
| RF+Geo(+Leg) | **0.093 (0.000)** | **90.7%** | 100% | **0.0%** | 0.069 |
| YOLO (COCO 시작) | 0.198 (−0.151) | 80.2% | 100.0% | −9.8% | 0.150 |
| **YOLO (pose 시작)** | 0.174 (−0.081) | 82.6% | 100.0% | −5.3% | 0.094 |

- Geo / RF 줄은 루트 README 표에서 가져옴(같은 000, 같은 `count_metrics()`). ±1 이내는 루트 README에 RF+Geo(+Leg)만 적혀 있음.
- conf 임계값은 valid 28장에서 사진당 MAE가 가장 작은 값: COCO 시작 0.55, pose 시작 0.75 (000은 보지 않음).

### 오류 내역 (visiting, 사람 박스와 IoU ≥ 0.5로 짝지음)

| | COCO 시작 | pose 시작 |
|---|---|---|
| 총 visiting (YOLO / 사람) | 120 / 133 | 126 / 133 |
| 많이 센 사진 / 적게 센 사진 | 2 / 15 | 4 / 11 |
| 못 찾은 visiting (검출) | 8 | 11 |
| 찾았지만 other로 판정 (판정) | **11** | **4** |
| other를 visiting으로 | 2 | 5 |
| 사람 라벨에 없는 박스를 visiting으로 | 4 | 3 |
| box mAP50 / mAP50-95 (000) | 0.888 / 0.601 | 0.901 / 0.645 |
| box mAP50-95 visiting (000) | 0.559 | 0.603 |
| 참고: valid box mAP50 / visiting AP50 | 0.946 / 0.912 | 0.965 / 0.955 |

### 해석

- 두 YOLO 모두 Geo(60°)보다 좋고 RF 계열보다 나쁨. 틀린 사진은 모두 1마리 차이(±1 이내 100%).
- 둘 다 visiting을 **적게 셈**(bias 음수). COCO 시작은 빠진 visiting 19마리 중 11마리가 "찾았지만 other로 판정" → 판정 쪽 오류가 큼.
- pose 시작은 "other로 판정"이 11 → 4로 줄어 MAE·점유량·구간 오차가 모두 나아짐. pose 학습에서 배운 머리·코 위치가 판정에 도움이 된 것으로 보임. 대신 임계값이 0.75로 높아져 못 찾은 visiting은 8 → 11로 늘어남.
- Geo / RF는 카메라마다 사료통 원 좌표를 받음(`counting/configs/eval_cameras.yaml`, 000은 SAMPLE). YOLO는 사료통 정보 없이 사진만으로 판정 → 처음 보는 카메라에서 판정이 불리한 조건.

### 주의

- **seed 1개 결과.** COCO 시작과 pose 시작의 차이는 틀린 사진 17장 vs 15장(2장 차이)이라 우연일 수 있음 → seed를 바꿔 반복 필요.
- **pose 시작의 누수 여부**: `pig_pose_best.pt`가 같은 seed 42 분할(train 162장)로 학습한 것이어야 valid 28장이 깨끗함 → 서인 확인 필요.
- valid는 사진 단위 무작위 추출이라 같은 클립의 비슷한 사진이 train과 갈릴 수 있음(서인 pose 실험과 같은 조건). 시간대·클립 단위 분할은 서인과 다시 정하기로 함.
