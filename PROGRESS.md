# 진행 상황 (파이프라인 순서)

**목표**: 돈방 CCTV에서 사료통을 visiting 중인 돼지 수를 자동으로 세고, 판정 방법 5가지(Geo / RF / RF+Geo / RF+Geo+Leg / YOLO)를 비교.

| 단계 | 내용 | 상태 |
|---|---|---|
| 1. 데이터 준비 | 001·002 학습용, 000 test용 정리 → YOLO-Pose 라벨 변환 → 분할 | ✅ 완료 |
| 2. 포즈 추정 | YOLOv8-pose n/s/m/l × epoch 100·200 학습·비교 → 사용할 모델 선택 | ✅ 학습 완료 (모델: medium ep200 안) |
| 3. visiting 판정 | Geo / RF / RF+Geo / RF+Geo+Leg 학습·평가, YOLO 직접 판정(수아) | ⏳ 예정 |
| 4. 마릿수 집계 | 영상 1초 추론 → 1분 / 10분 집계, DM 정답지로 검증 | ⏳ 예정 |

**표시 규칙** — ✅ **정답 평가 완료**: 사람이 만든 정답과 비교한 결과 / ⚠️ **정답 확인 필요**: 모델 추정치(사람 정답 없음, DM 프로젝트에서 실측 예정)

---

## 1. 데이터 준비

### 데이터

| task | 카메라 | 사진 | 돼지 | visiting | other | 용도 |
|---|---|---|---|---|---|---|
| 001 | CH005 (1920×1080) | 90 | 899 | 136 | 763 | 학습 |
| 002 | CH010 (1920×1080) | 100 | 990 | 179 | 811 | 학습 |
| 000 | 1분 샘플 영상 (960×720) | 86 | 860 | 133 | 727 | **test** |

- 원본: CVAT "for images 1.1" export(`<task>_sorted.zip`) — 사진 + `annotations.xml`(돼지마다 관절 18개 스켈레톤 + State).
- 모든 돼지에 State가 붙어 있음(Unlabeling 0). 관절 18개 모두 있음.
- zip 안에는 같은 사진이 예전 이름으로 한 번 더 들어 있음(000 86장, 002 100장, 바이트까지 동일) → XML에 적힌 사진만 사용.
- test(000)는 학습(CH005·CH010)과 **다른 카메라·다른 해상도** → 처음 보는 화면에서의 성능을 잼.
- 보관 위치(git 제외): `counting/data/labels/source/<task>_sorted.zip`, XML은 `counting/data/labels/state_<task>.xml`.

### YOLO-Pose 라벨 변환 (`training/scripts/cvat_xml_to_raw_zip.py`)

- **방법**: XML의 돼지 스켈레톤마다 한 줄 `0 cx cy w h x1 y1 v1 ... x18 y18 v18`(0-1 정규화)로 변환.
  - 박스 = 스켈레톤 점들의 최소·최대 범위
  - v = 2 보임 / 1 가려짐 / 0 화면 밖
  - 관절 순서 = CVAT skeleton 순서(`configs/pig_pose.yaml`, 코 12 · 목 11)
- **검증**: 같은 사진의 CVAT 자체 YOLO export와 비교 → 돼지 299마리에서 **박스·표시 완전히 동일**.
- **결과**: 000 86장(860마리), 001 90장(899마리), 002 100장(990마리) 변환.

### 분할 (`split_dataset.py --test-zip data/raw/data/000.zip --valid 0.15 --seed 42`)

| split | 사진 | 구성 |
|---|---|---|
| train | 162 | 001 77 + 002 85 |
| valid | 28 | 001 13 + 002 15 |
| test | 86 | 000 86 |

- 001+002를 seed 42로 섞어 15%를 valid(학습 중 가장 좋은 가중치 고르는 데 사용), 000은 통째로 test.
- 분할 스크립트에 `--test-zip` 옵션을 추가해 test task를 고정.

---

## 2. 포즈 추정 (YOLOv8-pose)

### 방법

- **모델**: YOLOv8-pose nano / small / medium / large, COCO 사전학습에서 시작, imgsz 640.
- **epoch 100 vs 200 비교**: 데이터가 적어(학습 162장) epoch를 늘리면 나아지는지 확인.
  - 두 설정 모두 **조기 종료를 끄고**(`--patience 0`) 지정한 epoch를 끝까지 학습. best.pt는 valid 성능 기준.
  - 결과는 run 이름별 폴더에 따로 저장(`POSE_RUN=ep100` / `ep200` → `models/results/<run>/<size>/`).
- **평가**: train / valid / test 각각 pose mAP50·mAP50-95, PCK@0.1·0.2(관절 점이 정답에서 돼지 크기 10%·20% 안에 들어온 비율).
  **test(000)가 최종 비교 기준.**

```powershell
cd training
$env:POSE_RUN = "ep100"; python scripts\train.py --sizes n s m l --epochs 100 --patience 0
$env:POSE_RUN = "ep200"; python scripts\train.py --sizes n s m l --epochs 200 --patience 0
python scripts\compare_runs.py --runs ep100 ep200
```

### 결과

test = 000(학습에 안 쓴 다른 카메라 86장). 배치: nano 16 / small 16 / medium 8 / large 8. GPU RTX 5060 8GB.

| 사이즈 | epoch | best epoch | 학습 시간(분) | test pose mAP50 | test pose mAP50-95 | test PCK@0.1 | test PCK@0.2 | valid pose mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| nano | 100 | 100 | 3.3 | 0.259 | 0.079 | 0.667 | 0.861 | 0.169 |
| nano | 200 | 188 | 6.2 | 0.278 | 0.101 | 0.685 | 0.867 | 0.245 |
| small | 100 | 95 | 4.6 | 0.204 | 0.060 | 0.701 | 0.850 | 0.201 |
| small | 200 | 186 | 8.6 | 0.348 | 0.103 | 0.689 | 0.856 | 0.284 |
| medium | 100 | 100 | 7.9 | 0.258 | 0.091 | 0.684 | 0.865 | 0.217 |
| **medium** | **200** | 190 | 15.3 | 0.330 | 0.100 | **0.710** | **0.870** | **0.294** |
| large | 100 | 100 | 16.1 | 0.252 | 0.059 | 0.673 | 0.866 | 0.202 |
| large | 200 | 200 | 31.5 | **0.413** | **0.107** | 0.668 | 0.834 | 0.289 |

표·그림: `training/reports/comparison/runs_ep100_vs_ep200.csv` / `.png`, 사이즈별 상세는 `training/reports/metrics/<run>/<size>/`.

**정리**

- **epoch 200이 모든 사이즈에서 나음**: valid pose mAP50-95 +0.06\~0.09, test pose mAP50-95 +0.01\~0.05. epoch 100은 best epoch가 대부분 마지막 epoch(100) → 덜 학습된 상태였음.
- **PCK@0.1·0.2는 epoch·사이즈 차이 작음**(±0.03): 관절 위치 자체는 epoch 100에서도 대부분 맞고, 200에서 정밀도가 조금 올라감.
- **valid보다 test가 크게 낮음**(mAP50-95 약 0.29 → 0.10): 000은 다른 카메라(960×720)라 학습 카메라(CH005·CH010)와 화면이 달라서 생기는 차이로 보임.
- large ep200은 mAP는 가장 높지만 PCK@0.2가 가장 낮고, best epoch가 200(아직 수렴 안 함), 학습 시간 2배.
- **사용 모델(안): medium ep200** — test PCK@0.1·0.2 최고, mAP50-95도 최고(large)와 0.007 차이, valid mAP50-95 최고. visiting 판정 특징은 관절 위치를 쓰므로 PCK를 우선함.
- large 학습 중 문제: 자동 배치(`batch: -1`)가 앞 사이즈(medium)가 잡고 있던 GPU 메모리 때문에 계산에 실패 → 기본값 16으로 떨어져 8GB를 넘김 → 시스템 메모리로 넘쳐 epoch당 약 220초. `batch: 8` 고정 후 epoch당 약 10\~18초로 정상화.

---

## 3. visiting 판정

### 비교하는 방법

| 방법 | 어떻게 판정 | 학습 데이터 | 담당 |
|---|---|---|---|
| **Geo (규칙 기반)** | ① 코·목 관절 신뢰도 ≥ 0.5 ② 코가 사료통 원 안(중심 거리 ≤ r) ③ 목→코 방향이 사료통 중심에서 60° 이내 | 없음 | |
| **RF** | 자세 특징 6개 → RandomForest | 001+002 | |
| **RF+Geo** | RF 특징 + Geo 값 5개 → RandomForest | 001+002 | |
| **RF+Geo+Leg** | RF+Geo 특징 + 다리 모양 누움 특징 5개 → RandomForest | 001+002 | |
| **YOLO** | 포즈 없이 YOLO가 돼지를 visiting / other로 바로 검출 | 001+002 | 수아 |

- **RF 특징 6개**: 코가 사료통 구역 안인지, 코-구역 거리(돼지 박스 대각선으로 나눔), 머리 숙임 각도, 척추 기울기, 목 상대 높이, 주요 관절 신뢰도.
- **Geo 값 5개**: 코-사료통 중심 거리 ÷ 사료통 반지름 r, 목→코 방향과 사료통 중심 사이 각도, confidence 기반 누움 신호 3개.
- **Leg 특징 5개**(관절 위치만 사용): 다리-척추 각도(누우면 90°에 가까움), 다리 길이 ÷ 몸통 길이, 발-척추 거리 ÷ 몸통 길이, 네 발이 한쪽에 몰린 정도("ㄷ" 자로 누우면 1), 목-몸통-꼬리 각도.
- RandomForest: 나무 100개, `class_weight=balanced`, seed 42, 계산 안 되는 특징은 −1.
- 사료통 원은 카메라마다 따로 잡음(`counting/configs/eval_cameras.yaml`):

| 카메라 | 사료통 원 (cx, cy, r) | 설정 |
|---|---|---|
| CH005 | 908.3, 529.1, 338.0 | `farm_ch005.yaml` |
| CH010 | 925.3, 535.1, 337.8 | `farm_ch010.yaml` |
| SAMPLE (000) | 453.4, 354.3, 169.2 | `farm_sample.yaml` |

### 평가 계획 (`counting/scripts/evaluate_methods.py`)

| | Stage 1 | Stage 2 (최종 test) |
|---|---|---|
| 데이터 | 001+002 | 000 |
| 관절 | 사람이 찍은 관절 | 2단계에서 고른 **포즈 모델이 찍은 관절** |
| 학습/시험 | 클립 단위 교차검증 | 001+002 전체로 학습 → 000에 적용 |
| 지표 | visiting F1 (precision, recall) | visiting F1 + 사진당 마릿수 오차(MAE)·치우침(bias) |

- Stage 2 매칭: 모델이 찾은 돼지와 사람 라벨 돼지를 관절 거리로 짝지음(헝가리안, 기준 0.25). 모델이 못 찾은 돼지는 "visiting 아님"으로 계산.
- YOLO 직접 판정(수아)도 같은 000에서 같은 지표로 비교.
- 할 일: 2단계에서 고른 가중치를 `counting/models/pig_pose_best.pt`로 복사(설정 파일 `weights:`가 이 이름을 가리킴).
  Geo 조건 비교용 `configs/ablation/`은 CH010만 들어 있어 000(SAMPLE)으로 평가하려면 SAMPLE 설정 추가 필요.

### 결과

예정.

---

## 4. 마릿수 집계 (DM 파트)

- **방법**: 영상을 1초에 1장씩 추론(`count_day.py --save-dets`, 모든 검출의 관절 좌표 저장) → 판정 방법을 YOLO 재실행 없이 적용(`apply_methods.py`) → 1분 / 10분 집계.
- **검증**: DM 정답지(사람이 1초 간격 사진 600장씩 visiting 마릿수를 직접 셈)로 초·분·10분 단위 오차와 집계 방식 비교.
  - A 구간(CH010 7/16 KST 10:30-10:40) 600장 추출 완료(`extract_dm_frames.py`, 모델이 본 프레임과 같은 프레임).
- **결과**: 예정.
