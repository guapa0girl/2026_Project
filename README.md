# 2026_Project — 돼지 사료 섭취 마리 수 추정 (YOLO-Pose)

돈방 CCTV 영상에서 **시간대별 사료 섭취 마리 수**를 추정하는 연구 코드입니다.
ETRI 2026 연구연수 「이유자돈·육성돈·비육돈 사료 섭취 시간 추정」 과제를 위한 것입니다.

이 레포는 두 단계로 나뉩니다.

```
2026_Project/
├── training/   ① 돼지 18-keypoint 포즈 모델 학습 (CVAT 라벨 → YOLOv8-pose n/s/m)
└── counting/   ② 학습된 모델로 영상 추론 → 사료통 ROI + 머리 방향(Geo) + 자세 판정 → 시간대별 집계
```

| 폴더 | 하는 일 | 자세한 설명 |
|---|---|---|
| `training/` | CVAT export 병합 → train/valid/test 분할 → n/s/m 학습 → mAP·PCK 평가 | [training/README.md](training/README.md) |
| `counting/` | 24시간 영상 추론(`count_day.py`) → 섭취 판정 → 10분 단위 집계·그래프 | [counting/README.md](counting/README.md) |

> 두 폴더는 원래 별도 레포(`Piggy_Auto_Labeling_`, `Piggy_Feeding_Counting`)였던 것을 합친 것입니다.
> 하위 README에 옛 레포 이름이나 `counting/training/`(옛 CVAT 부트스트랩 폴더, 이 레포에는 미포함)이
> 언급되는 부분이 남아 있을 수 있습니다.

---

## 설치 (Windows, NVIDIA GPU)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
# RTX 50 시리즈는 CUDA 12.8 이상 빌드가 필요
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r training\requirements.txt -r counting\requirements.txt
```

확인된 환경: Python 3.14, torch 2.11+cu128, ultralytics 8.4, RTX 5060 8GB.

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

- **`eval_frames.txt`**: 자세 판정 평가용 50장의 목록입니다. 이 50장은 **pose 학습에 절대 섞으면 안 됩니다.**
  `build_raw_zip.py --exclude eval_frames.txt`가 파일명 또는 `[CH010] <영상>_f<프레임>` ID로 걸러냅니다.
- 학습 결과 지표는 `training/reports/metrics/<size>/`, 비교표는 `training/reports/comparison/`에 저장됩니다.

### ② 24시간 추론 (`counting/` 폴더에서)

```powershell
cd counting
# 학습된 best.pt를 models\pig_pose_ch010_m_best.pt 로 복사 (configs\farm_ch010.yaml 의 weights:)
..\.venv\Scripts\python scripts\count_day.py --config configs\farm_ch010.yaml --videos "<영상폴더>" `
    --out-dir outputs\ch010_<날짜> --zones configs\zones\feeder_zones_ch010.yaml --bin-sec 600 --save-dets
```

---

## 현재 결과 — CH010 pose 모델 (2026-11-05)

데이터는 CVAT task 4개(000~003)에서 287장이고, 평가용 50장은 제외했습니다.
분할은 train 230 / valid 29 / test 28장이며, YOLOv8-pose를 COCO 사전학습에서 시작해 100 epoch, imgsz 640으로 학습했습니다.

| 모델 | test pose mAP50 | test pose mAP50-95 | test box mAP50-95 | test PCK@0.1 | test PCK@0.2 |
|---|---|---|---|---|---|
| nano | 0.703 | 0.377 | 0.836 | 0.800 | 0.923 |
| small | 0.790 | 0.420 | 0.861 | 0.808 | 0.923 |
| **medium** | **0.795** | **0.436** | **0.870** | **0.828** | **0.934** |

- 추론에는 **medium**을 씁니다(`counting/configs/farm_ch010.yaml`).
- test가 28장이라 사이즈 간 차이 0.01~0.02는 오차 범위로 봐야 합니다.
- `training/README.md` 8장의 표는 이전 937장 데이터셋(다른 test set) 결과라 위 표와 직접 비교할 수 없습니다.

---

## 이 레포에 없는 것

공개 레포라 아래 항목은 올리지 않습니다.

- **이미지·라벨 데이터, 원본 영상, 영상 캡처 이미지**: 농장 데이터입니다. 각 폴더의 `data/`는 빈 구조(`.gitkeep`)만 있습니다.
- **학습된 가중치(`*.pt`)**: [Releases](../../releases)에 첨부합니다. 받은 파일은 `counting/models/`에 넣으세요.
- **학습 중간 산출물**: `training/models/results/`, `training/past_train/`.
