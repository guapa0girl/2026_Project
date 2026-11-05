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

같은 포즈 모델 출력(keypoint)을 두고, 돼지 한 마리가 사료통을 **visiting** 중인지 판정하는 세 가지 방법을 비교합니다.

| 방법 | 입력 | 설명 |
|---|---|---|
| **Geo 단독 (ROI 규칙)** | 코·목 keypoint + 사료통 원 | 규칙 기반. 코가 사료통 뚜껑 원 안에 있고 목→코 방향이 중심을 향하면 visiting (`feeding_classifier.py`) |
| **RF 단독** | keypoint 기반 자세 특징 6개 | RandomForest 분류기 (`posture_features.py`, `posture_classifier.py`) |
| **RF + Geo feature** | RF 특징 + Geo 특징(코-사료통 거리, 방향 각도 등) 추가 | Geo 규칙의 값을 특징으로 넣어 RF가 함께 학습 |

- **정답**: 사람이 CVAT에서 붙인 State 라벨(visiting / other). 평가 50장(`training/eval_frames.txt`)은 pose 학습에서 제외되어 있습니다.
- **입력 keypoint**: 사람이 고친 keypoint가 아니라 **이 포즈 모델이 실제로 낸 keypoint**로 평가합니다(실제 운용 성능).
  모델 검출과 사람 라벨은 개체(돼지) 단위로 매칭합니다.
- **검증**: 같은 클립의 프레임이 train/test에 나뉘어 들어가지 않도록 **클립 단위 StratifiedGroupKFold**.
- **주 지표**: **visiting F1** (보조: precision, recall).
- 24시간 영상에는 `count_day.py --save-dets`로 모든 검출의 keypoint를 저장해 두므로, 세 방법 모두 YOLO 재실행 없이 다시 계산할 수 있습니다.

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

## 현재 결과 — CH010 추론 (2026-07-16 10:26~10:51 KST, 25분 클립)

- 1초 샘플 1,474개(읽기 실패 7개), 처리 시간 약 12분.
- geometric 기준 1분 평균 0~2.4마리, 10분 평균 0.95 → 1.19 → 1.59마리.
- 사람 라벨 대비 정확도는 아직 검증 전입니다(평가 50장으로 확인 예정).

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
