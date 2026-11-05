# training/past — 이전 연구에서 사용하던 파일 (보관용)

이 폴더는 **이전 연구**(오토라벨링 학습 프로젝트, 937장 데이터셋 시기)에서 쓰던 파일을 지우지 않고 보관해 둔 곳입니다.
**현재 CH010 연구에서는 사용하지 않습니다.** 현재 코드는 상위 폴더(`training/`)를 보세요.

- 원래 `training/` 안에 있던 위치를 그대로 유지했습니다(`past/scripts/track_video.py` ← 원래 `scripts/track_video.py`).
- 이 폴더에서 바로 실행되도록 맞춰 두지 않았습니다. 다시 쓰려면 원래 위치로 옮겨야 하고,
  일부는 `training/scripts/utils/`의 현재 모듈(`paths.py` 등)을 import합니다. `paths.py`에서 지운 상수
  (`VIDEO_RESULT_DIR`, `RAW_ANNOTATIONS_DIR`, xlarge 항목)도 다시 넣어야 합니다.
- `OLD_README.md`: 정리하기 전의 training README 원문입니다(937장 데이터셋 결과, 사이즈별 keypoint PCK 표 등).

## 파일 목록

| 분류 | 파일 | 이전 용도 |
|---|---|---|
| 설정 | `configs/train_xlarge.yaml` | yolov8x-pose 하이퍼파라미터 (8GB GPU에서 너무 오래 걸려 사용 중단) |
| | `configs/feeder_zones.yaml` | `test_video.avi` 기준 사료통 8각형 구역 |
| 데모 | `scripts/infer_realtime.py` | 웹캠/영상 실시간 스켈레톤 오버레이 |
| | `scripts/track_video.py` | 탐지 + ByteTrack 추적 + 포즈 샘플 영상 저장 |
| | `scripts/utils/viz.py` | 위 두 데모의 그리기 함수 |
| 이전 RF | `scripts/define_feeder_zone.py` | 마우스로 사료통 구역 정의 |
| | `scripts/label_posture_rule_based.py` | 정답 keypoint에서 특징 6개 추출 + 규칙으로 1차 섭식 라벨 |
| | `scripts/train_posture_classifier.py` | 규칙 라벨로 RandomForest 학습 |
| | `scripts/utils/posture.py` | 위 RF 특징·규칙 함수 |
| | `reports/posture/` | 규칙 라벨 데이터셋 CSV, RF 분류 리포트 |
| 변환 | `scripts/convert_annotations.py` | COCO-Keypoints json → YOLO-Pose txt |

> 이전 RF는 **사람 라벨이 아니라 규칙으로 만든 라벨**로 학습한 것이라, 현재 연구의 RF(사람이 붙인 visiting/other 라벨,
> 클립 단위 StratifiedGroupKFold)와는 다릅니다.

---

## 다시 사용하려면 — 코드별로 옮길 위치

**규칙**: 경로에서 `past/`만 빼면 원래 위치입니다. 명령은 모두 `training/` 폴더에서 실행합니다.

```
training/past/scripts/X.py          → training/scripts/X.py
training/past/scripts/utils/X.py    → training/scripts/utils/X.py
training/past/configs/X.yaml        → training/configs/X.yaml
training/past/reports/posture/      → training/reports/posture/
```

스크립트는 `training/scripts/utils/`의 모듈을 import합니다. `paths.py`, `skeleton.py`, `label_io.py`, `pck.py`는 이미
현재 폴더에 있고, `viz.py`·`posture.py`는 아래 표대로 같이 옮겨야 합니다.
일부는 **`training/scripts/utils/paths.py`에 한 줄을 다시 넣어야** 합니다(정리하면서 지운 상수).

### 스크립트별

| 스크립트 → `training/scripts/` | 같이 옮길 파일 | `utils/paths.py`에 다시 넣을 것 | 필요한 데이터 |
|---|---|---|---|
| `infer_realtime.py` | `utils/viz.py` → `scripts/utils/` | — | `models/results/<size>/weights/best.pt` |
| `track_video.py` | `utils/viz.py` → `scripts/utils/` | `VIDEO_RESULT_DIR = ROOT / "video_result"` | 위 가중치, 영상 |
| `convert_annotations.py` | — | `RAW_ANNOTATIONS_DIR = RAW_DIR / "annotations"` | `data/raw/annotations/*.json` (COCO-Keypoints) |
| `define_feeder_zone.py` | — | — | 영상 (화면이 있는 환경) → `configs/feeder_zones.yaml` 생성 |
| `label_posture_rule_based.py` | `utils/posture.py` → `scripts/utils/`, `configs/feeder_zones.yaml` → `configs/` | — | `data/processed/` (split_dataset.py 결과) → `reports/posture/rule_based_dataset.csv` 생성 |
| `train_posture_classifier.py` | `utils/posture.py` → `scripts/utils/` | — | `reports/posture/rule_based_dataset.csv` (위 스크립트 결과, 또는 `past/reports/posture/`를 `reports/posture/`로 옮김) → `models/posture/random_forest.joblib` 생성 |

### xlarge 학습을 다시 하려면

1. `past/configs/train_xlarge.yaml` → `training/configs/`
2. `scripts/utils/paths.py`: `SIZES`에 `"x"`, `SIZE_LABELS`에 `"x": "xlarge"`, `PRETRAINED_WEIGHTS`에 `"x": "yolov8x-pose.pt"` 추가
3. `scripts/train.py`와 `scripts/training_status.py`의 사이즈별 설정 목록에 `"x": CONFIGS_DIR / "train_xlarge.yaml",` 추가
4. `python scripts\train.py --sizes x` (8GB GPU에서는 매우 오래 걸립니다 — `launch_detached.py`로 띄우기 권장)

### 예시: 추적 샘플 영상 다시 만들기

```powershell
cd training
move past\scripts\track_video.py scripts\
move past\scripts\utils\viz.py scripts\utils\
# scripts\utils\paths.py 에  VIDEO_RESULT_DIR = ROOT / "video_result"  한 줄 추가
..\.venv\Scripts\python scripts\track_video.py --source "<영상.avi>" --sizes m
```
