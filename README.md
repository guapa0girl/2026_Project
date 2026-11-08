# 2026_Project — 돼지 사료 점유 마리 수 추정 (YOLO-Pose)

돈방 CCTV 영상에서 **시간대별 사료통 점유율**을 추정하는 연구 코드입니다.
2026 MIX_Lab에서 학부연구생으로 진행하는 「육돈 사료통 점유 시간 계측 연구」 과제를 위한 것입니다.

```
2026_Project/
├── training/   ① 돼지 18-keypoint 포즈 모델 학습 (CVAT 라벨 → YOLOv8-pose n/s/m/l)
│   └── past/   이전 연구에서 쓰던 파일 보관 (현재 연구에서는 사용 안 함)
└── counting/   ② 학습된 모델로 영상 추론 → 사료통 판정 → 시간대별 집계
    └── past/   이전 연구에서 쓰던 파일 보관 (현재 연구에서는 사용 안 함)
```

| 폴더 | 하는 일 | 자세한 설명 |
|---|---|---|
| `training/` | CVAT export 병합 → train/valid/test 분할 → 학습 → mAP·PCK 평가 | [training/README.md](training/README.md) |
| `counting/` | 클립별 1초 샘플링 추론(`count_day.py`) → 섭취 판정 → 1초/1분/10분 집계·그래프 | [counting/README.md](counting/README.md) |

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

## 전체 흐름

### ① 포즈 모델 학습 (`training/` 폴더에서)

```powershell
cd training
# CVAT export zip (<task>/img + <task>/label 구조)을 data\raw\data\ 바로 아래에 둔다
# 라벨만 export한 경우: scripts\build_raw_zip.py 로 원본 이미지와 묶어서 만든다
..\.venv\Scripts\python scripts\ingest_raw_zips.py
..\.venv\Scripts\python scripts\split_dataset.py --train 0.8 --valid 0.1 --test 0.1
..\.venv\Scripts\python scripts\train.py --sizes n s m --epochs 100
```

- **`eval_frames.txt`**: 섭취 판정 평가용 50장의 목록입니다. 이 50장은 **pose 학습에 절대 섞으면 안 됩니다.**
  `build_raw_zip.py --exclude eval_frames.txt`가 걸러냅니다.
- 학습 결과 지표는 `training/reports/metrics/<size>/`, 비교표는 `training/reports/comparison/`에 저장됩니다.

### ② 영상 추론 (`counting/` 폴더에서)

```powershell
cd counting
# 가중치를 models\pig_pose_ch010_m_best.pt 로 둔다 (configs\farm_ch010.yaml 의 weights:)
..\.venv\Scripts\python scripts\count_day.py --config configs\farm_ch010.yaml --videos "<영상폴더>" `
    --out-dir outputs\ch010_<날짜> --zones configs\zones\feeder_zones_ch010.yaml --bin-sec 600 --save-dets
..\.venv\Scripts\python scripts\plot_1min.py outputs\ch010_<날짜>\<클립>_1s.csv   # 1분 단위 그래프
```

---

## 연구 계획 — 섭취(visiting) 판정 방법 비교

같은 포즈 모델 출력(keypoint)을 두고, 돼지 한 마리가 사료통을 **visiting** 중인지 판정하는 네 가지 방법을 비교합니다.

| 방법 | 입력 | 설명 |
|---|---|---|
| **Geo 단독 (ROI 규칙)** | 코·목 keypoint + 사료통 원 | 규칙 기반. 코·목 신뢰도 ≥ 0.5, 코가 사료통 뚜껑 원 안, 목→코 방향이 중심에서 **60° 이내**이면 visiting (`feeding_classifier.py`). 학습 없음 |
| **RF 단독** | keypoint 기반 자세 특징 6개 | RandomForest 분류기 (`posture_features.py`) |
| **RF + Geo feature** | RF 특징 + Geo 값 5개(코-사료통 거리 ÷ r, 머리 방향 각도 등) | Geo 규칙의 값을 특징으로 넣어 RF가 함께 학습 |
| **RF + Geo + Leg** | RF + Geo 특징 + 다리 모양 누움 특징 5개 | 다리-척추 각도, 다리 길이, 발이 한쪽에 몰린 정도 등 관절 **위치**로 "옆으로 누운 자세"를 구분 (`leg_features.py`) |

- **Geo 설정**: 방향 조건 **60°로 확정** (평가 50장에서 마릿수 오차가 가장 작고 치우침이 거의 없음. 90°는 F1이 조금 높지만 사진당 0.5마리 과대 집계).
- **정답**: 사람이 CVAT에서 붙인 State 라벨(visiting / other). 라벨러 일치도: 수아가 라벨링한 20장을 검토자가 확인해 **100% 일치**.
- **카메라**: 빨간 원형 사료통이 화면에 통째로 보이는 채널만 사용 — CH010, CH005, 1분 샘플(SAMPLE). CH002는 급이기가 화면에서 잘려 제외.
  사료통 원(cx, cy, r)은 채널마다 `calibrate_feeder.py`로 잡아 `counting/configs/eval_cameras.yaml`에 연결합니다.
  특징은 r·몸통 길이로 나눈 비율이나 각도라 카메라 높이·사료통 크기 차이에 영향을 덜 받습니다.
- **평가 2단계** (`counting/scripts/evaluate_methods.py`):
  - **Stage 1** — 사람이 찍은 keypoint + State 라벨로 방법 비교. 같은 클립의 프레임이 train/test에 나뉘지 않도록 **클립 단위 StratifiedGroupKFold**.
  - **Stage 2** — pose 학습에서 제외한 **평가 50장**(`training/eval_frames.txt`)에서 **포즈 모델이 실제로 낸 keypoint**로 평가(실제 운용 성능).
    모델 검출과 사람 라벨은 개체 단위로 매칭(헝가리안). RF는 Stage 1 라벨 전체로 학습.
- **지표**: **visiting F1** (보조: precision, recall) + 그래프에 쓰이는 **사진당 마릿수 오차(MAE)와 치우침(bias)**.
- 영상에는 `count_day.py --save-dets`로 모든 검출의 keypoint를 저장해 두므로, 네 방법 모두 YOLO 재실행 없이 다시 계산합니다(`apply_methods.py`).

### 현재 결과 요약 (2026-11-07, Stage 1 = CH005 라벨 90장)

| 방법 | Stage 1 F1 | Stage 2 F1 | Stage 2 사진당 마릿수 오차 (bias) |
|---|---|---|---|
| Geo (60°) | 0.772 | 0.675 | 0.56 (+0.04) |
| RF | 0.930 | 0.685 | 1.24 (+1.20) |
| RF+Geo | 0.930 | 0.698 | 1.12 (+1.04) |
| **RF+Geo+Leg** | **0.949** | **0.900** | **0.22 (−0.02)** |

- RF·RF+Geo는 사료통 옆에 **누워 있는 돼지**를 visiting으로 세어 과대 집계하고, 다리 특징을 넣으면 이 오류가 거의 사라집니다.
- CH005로만 학습해 CH010에서 시험한 결과라, 카메라가 달라도 동작한다는 근거가 됩니다.
- 남은 라벨(약 140장)을 받아 다시 돌릴 예정입니다. 날짜별 진행·그림·해석은 [PROGRESS.md](PROGRESS.md)에 있습니다.

---

## 현재 결과 — pose 모델 (2026-11-05)

데이터는 CVAT task 4개(000~003)에서 287장이고, 평가용 50장은 제외했습니다.
채널 구성은 CH005 90장 · CH010 50장 · CH002 61장 · 채널 미표기 86장입니다.
분할은 train 230 / valid 29 / test 28장이며, YOLOv8-pose를 COCO 사전학습에서 시작해 100 epoch, imgsz 640으로 학습했습니다.

| 모델 | test pose mAP50 | test pose mAP50-95 | test box mAP50-95 | test PCK@0.1 | test PCK@0.2 |
|---|---|---|---|---|---|
| nano | 0.703 | 0.377 | 0.836 | 0.800 | 0.923 |
| small | 0.790 | 0.420 | 0.861 | 0.808 | 0.923 |
| **medium** | **0.795** | **0.436** | **0.870** | **0.828** | **0.934** |
| large | 학습 예정 | | | | |

- 추론에는 **medium**을 씁니다(`counting/configs/farm_ch010.yaml`).
- test가 28장이라 사이즈 간 차이 0.01~0.02는 오차 범위로 봐야 합니다.

## 현재 결과 — CH010 추론 (2026-07-16 10:26~13:32 KST, 클립 8개, 약 3시간)

- 1초 샘플 11,063개, 클립당 4~12분 처리. 실패 클립 없음.
- 1초 평균 visiting 마릿수: Geo(60°) 1.19 · RF+Geo+Leg 0.71마리. 11:30~11:40에는 모든 방법이 3마리 안팎으로 일치.
- 11:40~12:40에는 Geo가 사료통 옆에서 자는 돼지까지 세어 RF+Geo+Leg보다 1마리 이상 많음(장면 확인).
- 시간대별 마릿수 자체는 사람 정답이 없는 **추정치**이며, 정확도는 평가 50장으로 간접 확인했습니다.

---

## 이 레포에 없는 것

- **이미지·라벨 데이터, 원본 영상**: 농장 데이터입니다. 각 폴더의 `data/`는 빈 구조(`.gitkeep`)만 있습니다.
- **가중치·추론 결과·학습 산출물**: [Releases](../../releases)에 첨부합니다.

| Release 파일 | 받은 뒤 놓을 위치 |
|---|---|
| `pig_pose_ch010_m_best.pt` | `counting/models/` |
| `CH010.20260716012647-20260716015128.avi` | 아무 영상 폴더 (원래 이름 `[CH010] 20260716012647-20260716015128.avi`) |
| `ch010_20260716.zip` | `counting/outputs/` 에 압축 해제 |
| `results.zip` | `training/models/` 에 압축 해제 (n/s/m 학습 산출물 + 가중치) |
