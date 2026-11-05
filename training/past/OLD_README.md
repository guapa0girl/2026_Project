# Piggy Auto Labeling — 돼지 자세 추정 오토라벨링 학습 (YOLO-Pose)

돼지의 18개 키포인트 스켈레톤을 추정하는 YOLO-Pose(Ultralytics) 모델을
**학습**시키는 프로젝트입니다. 이 저장소의 핵심 범위는 원본 데이터 업로드
→ 라벨 병합 → train/valid/test 분할 → `nano / small / medium / large /
xlarge` 학습 → 성능 평가/비교까지의 **오토라벨링 학습 파이프라인**이고,
**bbox 탐지 성능보다 포즈(스켈레톤) 성능을 최우선 지표**로 관리합니다.
(`xlarge`는 8GB급 GPU에서 200에폭에 2일 가까이 걸려서 현재는 `n/s/m/l`
네 가지만 실제로 학습·비교하고 있습니다 — 12번 섹션 참고)

추적(tracking)/실시간 영상 추론/행동 분류처럼 **학습된 모델을 가지고
그다음에 하는 작업**은 이 저장소의 핵심 범위가 아니며, 13번 섹션에 별도
"추가 작업"으로 분리해서 정리했습니다.

---

## 1. 디렉토리 구조

```
auto_labeling_model/
├── data/
│   ├── raw/
│   │   ├── data/                # ★ CVAT task export zip은 여기에 업로드 ★ (000.zip, 001_.zip ...)
│   │   ├── img/                 #   (자동 생성) ingest_raw_zips.py가 zip에서 꺼내 모아주는 원본 이미지
│   │   ├── label/                #  (자동 생성) ingest_raw_zips.py가 zip에서 꺼내 모아주는 YOLO-Pose txt 라벨
│   │   └── annotations/         #   (선택) COCO-Keypoints json 등, zip이 아닌 원본 라벨 export본
│   ├── processed/                # split_dataset.py가 raw를 train/valid/test로 나눠서 채우는 곳 (자동 생성, 직접 건드리지 않음)
│   │   ├── train/{images,labels}
│   │   ├── valid/{images,labels}
│   │   └── test/{images,labels}
│   └── past_train/               # (선택) 데이터셋 자체를 버전별로 스냅샷 보관하고 싶을 때 사용
│
├── configs/
│   ├── pig_pose.yaml             # ★ 18개 키포인트 이름/스켈레톤/flip_idx 정의 (CVAT raw에서 파생) ★
│   ├── cvat_skeleton_raw.json    # CVAT skeleton 라벨 원본(Raw) — pig_pose.yaml의 출처, 참고용 보관
│   ├── train_nano.yaml           # yolov8n-pose 하이퍼파라미터
│   ├── train_small.yaml          # yolov8s-pose 하이퍼파라미터
│   ├── train_medium.yaml         # yolov8m-pose 하이퍼파라미터
│   ├── train_large.yaml          # yolov8l-pose 하이퍼파라미터 (batch: -1 autobatch)
│   ├── train_xlarge.yaml         # yolov8x-pose 하이퍼파라미터 (batch: -1 autobatch)
│   └── feeder_zones.yaml         # 급식 구역 다각형 정의 (13.2번 섹션 — 추가 작업)
│
├── models/
│   ├── pretrained/               # COCO 사전학습 가중치 (yolov8n/s/m/l/x-pose.pt) — 없으면 자동 다운로드
│   ├── results/                  # ★ 가장 최근 학습된 모델만 존재 (n/s/m/l/x 각각) ★
│   │   ├── n/weights/best.pt
│   │   ├── s/weights/best.pt
│   │   ├── m/weights/best.pt
│   │   ├── l/weights/best.pt
│   │   └── x/weights/best.pt
│   └── posture/random_forest.joblib   # 급식 자세 분류기 (13.2번 섹션 — 추가 작업)
│
├── past_train/                   # 이전 학습 결과(모델+지표)가 자동으로 이동되는 아카이브
│   └── run_20260821_153000/
│       ├── n/{model,metrics}
│       ├── s/{model,metrics}
│       ├── m/{model,metrics}
│       ├── l/{model,metrics}
│       └── x/{model,metrics}
│
├── reports/
│   ├── metrics/<size>/{train,valid,test}_metrics.json, summary.json
│   ├── comparison/latest_comparison.{csv,png}   # n/s/m/l/x 성능 비교 (매 학습마다 갱신)
│   └── posture/       # rule_based_dataset.csv, classifier_report.json (13.2번 섹션 — 추가 작업)
│
├── scripts/
│   ├── ingest_raw_zips.py        # ★ data/raw/data/*.zip -> data/raw/{img,label} 로 병합 ★
│   ├── convert_annotations.py    # COCO json -> YOLO-Pose txt 변환 (필요한 경우만)
│   ├── split_dataset.py          # raw 전체 -> train/valid/test 분할
│   ├── train.py                  # ★ n/s/m/l/x 학습 + 평가 + 비교까지 한 번에 실행하는 메인 스크립트 ★
│   ├── evaluate.py               # 학습된 모델을 split별로 평가 (mAP, PCK)
│   ├── compare_models.py         # n/s/m/l/x 비교 리포트 생성
│   ├── archive_previous_run.py   # 이전 결과를 past_train/으로 이동 (train.py가 자동 호출)
│   ├── training_status.py        # n/s/m/l/x 학습 진행률을 %로 확인 (12번 섹션 참고)
│   ├── infer_realtime.py         # (추가 작업, 13.1번) 웹캠/영상 실시간 추론 데모
│   ├── track_video.py            # (추가 작업, 13.1번) 탐지+추적+포즈 샘플 영상 저장 (video_result/<size>/)
│   ├── define_feeder_zone.py     # 마우스로 급식 구역 정의 (13.2번 섹션 — 추가 작업)
│   ├── label_posture_rule_based.py  # 규칙 기반 급식 자세 1차 라벨링 (13.2번 섹션 — 추가 작업)
│   ├── train_posture_classifier.py  # RandomForest 급식 자세 분류기 학습 (13.2번 섹션 — 추가 작업)
│   └── utils/                    # 공용 헬퍼 (경로, 스켈레톤 정의, PCK 계산, 라벨 파싱, 시각화, 급식 자세 특징)
│
├── runs/                         # ultralytics 학습 중 생성되는 로그/그래프 (용량 큼, git 제외)
├── video_result/<size>/          # track_video.py가 만드는 추적 샘플 영상 (용량 큼, git 제외)
├── launch_detached.py            # 장시간 학습을 세션과 분리된 프로세스로 띄우는 도구 (12번 섹션 참고)
├── requirements.txt
└── README.md
```

---

## 2. 파일은 어디에 올려야 하나요?

| 올릴 파일 | 위치 | 비고 |
|---|---|---|
| CVAT에서 task 단위로 내보낸 export zip (내부에 `img/`, `label/` 폴더 포함) | `data/raw/data/` | **압축을 풀지 말고 zip 파일 그대로** 올리면 됨. 파일명은 자유(예: `005.zip`). 누적 저장(현재 약 750장, 계속 추가) |
| (다른 툴에서) **COCO-Keypoints json**으로만 export되는 경우 | `data/raw/annotations/` | 이후 `scripts/convert_annotations.py`로 `data/raw/label/`에 변환 생성 |
| COCO 사전학습 pose 가중치(`yolov8n/s/m/l/x-pose.pt`) | `models/pretrained/` | 없으면 `train.py` 실행 시 자동 다운로드되어 여기 저장됨 (직접 올릴 필요 없음) |
| CVAT skeleton 라벨 설정(Raw JSON)이 바뀐 경우 | `configs/cvat_skeleton_raw.json`을 갱신하고 `configs/pig_pose.yaml`도 함께 수정 | **라벨링 시작 전에 반드시 확정** (아래 3번 참고) |

> `data/raw/img/`, `data/raw/label/`, `data/processed/`, `models/results/`,
> `reports/`, `past_train/`는 전부 스크립트가 자동으로 채우는 산출물 폴더입니다.
> 여기에는 직접 파일을 올리지 마세요 — `data/raw/data/`에 zip만 올리면
> `scripts/ingest_raw_zips.py`가 `img/`, `label/`을 알아서 채워줍니다.

### zip 파일 여러 개에 같은 task가 중복/충돌하는 경우
`scripts/ingest_raw_zips.py`는 같은 task_id가 여러 zip에 들어있으면(예: 개별
export와 전체 프로젝트 백업 export가 겹치는 경우) 자동으로 하나를 조용히
덮어쓰지 않습니다. zip 파일명이 task_id와 정확히 같은 "단독 export"를
우선 사용하고, 무엇을 선택/제외했는지를 `data/raw/_ingest_conflicts.log`에
기록합니다(충돌이 없으면 이 로그 파일은 생성되지 않거나 삭제됩니다). 실행 후
콘솔에 `[ingest][경고]`가 뜨면 반드시 이 로그를 확인하세요.

### 라벨 파일(txt) 한 줄 형식
```
<class_id> <cx> <cy> <w> <h> <x1> <y1> <v1> ... <x18> <y18> <v18>
```
- `class_id`는 항상 `0`(pig, 1종 클래스).
- `cx,cy,w,h`, `x,y`는 이미지 폭/높이로 정규화된 0~1 값.
- `v`(visibility): `0`=라벨 안 함, `1`=라벨했지만 가려짐(occluded), `2`=라벨하고 보임.
- 키포인트 18개의 **순서**는 반드시 `configs/pig_pose.yaml`의 `keypoint_names` 순서와 일치해야 합니다.

---

## 3. 18개 키포인트 스켈레톤

CVAT 프로젝트의 `pig` skeleton 라벨 설정(Raw JSON, `configs/cvat_skeleton_raw.json`)에서
파생된 실제 스펙입니다. **실제 라벨 txt 파일의 키포인트 순서는 raw json의
`sublabels` 배열 표시 순서가 아니라, svg 안 각 포인트의 `data-node-id`
오름차순(스켈레톤 템플릿을 처음 만들 때의 생성 순서)을 따릅니다** — 처음에
sublabels 순서로 만들었다가 실제 데이터와 어긋나는 것을 발견하고(스켈레톤
연결선이 이상하게 그려짐, 좌우 반전 augmentation도 잘못 매칭됨) 바로잡았습니다.

| # | 이름 | 설명 |
|---|---|---|
| 0 | tail_tip | 꼬리 끝 |
| 1 | tail | 꼬리 시작점 |
| 2 | l_hip | 왼쪽 엉덩이 |
| 3 | r_hip | 오른쪽 엉덩이 |
| 4 | l_foot | 왼쪽 뒷발 |
| 5 | r_foot | 오른쪽 뒷발 |
| 6 | r_flank | 오른쪽 옆구리 |
| 7 | l_flank | 왼쪽 옆구리 |
| 8 | l_shoulder | 왼쪽 어깨 |
| 9 | r_shoulder | 오른쪽 어깨 |
| 10 | l_hand | 왼쪽 앞발 |
| 11 | neck | 목 |
| 12 | nose | 코 |
| 13 | l_ear | 왼쪽 귀 |
| 14 | l_ear_tip | 왼쪽 귀 끝 |
| 15 | r_ear | 오른쪽 귀 |
| 16 | r_ear_tip | 오른쪽 귀 끝 |
| 17 | r_hand | 오른쪽 앞발 |

스켈레톤 연결선(18쌍, `configs/pig_pose.yaml`의 `skeleton`)과 좌우 반전
매핑(`flip_idx`)도 CVAT raw json의 svg 엣지를 node-id 순서 기준으로 파싱해서
생성했습니다.

> CVAT에서 키포인트를 추가/삭제하면 반드시 `configs/cvat_skeleton_raw.json`을
> 새로 받아서 `configs/pig_pose.yaml`(`keypoint_names`, `skeleton`,
> `flip_idx`, `kpt_shape`)도 함께 수정하세요 — 이때도 sublabels 표시 순서가
> 아니라 svg의 `data-node-id` 오름차순을 기준으로 삼아야 합니다. 순서가
> 어긋나면 지표가 조용히 틀어집니다 — 실제 export된 라벨 txt 파일을 1개 열어서
> 값 54개(18×3)와 순서가 맞는지 한 번은 직접 확인해보길 권장합니다.

---

## 4. 설치

```bash
pip install -r requirements.txt
```

GPU 학습을 원하면 CUDA 버전에 맞는 PyTorch를 [pytorch.org](https://pytorch.org)
안내에 따라 먼저 설치한 뒤 `requirements.txt`를 설치하세요.

`requirements.txt`에 `albumentations`가 포함되어 있는데, 이 프로젝트 코드가
직접 호출하지는 않습니다 — Ultralytics가 albumentations 설치 여부를 자동
감지해서, 설치돼 있으면 학습 파이프라인에 낮은 확률(`p=0.01`)의
Blur/MedianBlur/ToGray/CLAHE 변형을 자동으로 추가합니다(기존 mosaic·좌우반전·
HSV 증강 위에 얹히는 형태). 빼고 싶으면 `pip uninstall albumentations`만
하면 됩니다.

---

## 5. 학습 파이프라인

### Step 1. 원본 데이터 업로드
CVAT task export zip을 `data/raw/data/`에 그대로 올립니다. (COCO json만 있다면
`python scripts/convert_annotations.py --coco data/raw/annotations/xxx.json` 먼저 실행)

### Step 2. zip 병합
```bash
python scripts/ingest_raw_zips.py
```
`data/raw/data/`의 모든 zip을 열어 `data/raw/img/`, `data/raw/label/`에
평평하게 모읍니다. 이미 병합된 적 있는 zip을 다시 실행해도 안전합니다.

### Step 3. train/valid/test 분할
```bash
python scripts/split_dataset.py --train 0.8 --valid 0.1 --test 0.1
```
`data/raw/img`, `data/raw/label`에 있는 **전체 누적 데이터**를 기준으로 매번
새로 분할하여 `data/processed/`를 다시 채웁니다. (아래 6번 정책 때문에 매번
재분할이 맞습니다.)

### Step 4. 학습 (nano / small / medium / large / xlarge 한 번에)
```bash
python scripts/train.py
```
- 다섯 사이즈(`n`, `s`, `m`, `l`, `x`)를 순서대로 학습합니다.
- 학습 직전 자동으로 이전 결과를 `past_train/`으로 옮깁니다.
- 학습 직후 자동으로 train/valid/test 각각 평가하고, 모델 비교 리포트까지 생성합니다.

옵션 예시:
```bash
python scripts/train.py --sizes n s              # medium/large/xlarge 제외하고 학습
python scripts/train.py --epochs 100             # configs/train_*.yaml 값을 덮어쓰기
python scripts/train.py --sizes l x --resume     # 중단된 학습을 이어서 진행 (아래 11번 참고)
```

진행 상황이 궁금하면(특히 large/xlarge처럼 오래 걸리는 학습):
```bash
python scripts/training_status.py
```
사이즈별 완료 에폭/전체 에폭을 막대그래프(%)로 보여줍니다.

### Step 5. 결과 확인
- `reports/metrics/<size>/{train,valid,test}_metrics.json` — split별 상세 지표
- `reports/comparison/latest_comparison.csv` / `.png` — n/s/m/l/x 비교표·그래프
- `models/results/<size>/weights/best.pt` — 최종 학습된 가중치

학습된 모델로 실시간 추론/영상 추적을 해보고 싶으면 13.1번 섹션을
참고하세요 — 이 저장소의 핵심 학습 범위와는 별도로 분리해뒀습니다.

---

## 6. 데이터 추가 시 정책 (파인튜닝 아님, 항상 재학습)

라벨링된 데이터가 추가로 들어오면(예: 750장 → 1,200장):

1. 새 CVAT export zip을 `data/raw/data/`에 **추가**로 올립니다.
2. `python scripts/ingest_raw_zips.py` 를 다시 실행해 `data/raw/img/`, `data/raw/label/`을 전체 zip 기준으로 재생성합니다.
3. `python scripts/split_dataset.py` 를 다시 실행해 `data/processed/`를 전체 데이터 기준으로 재생성합니다.
4. `python scripts/train.py` 를 다시 실행합니다.
   - 기존 `models/results/<size>`와 `reports/metrics/<size>`는 학습 시작 전
     **자동으로** `past_train/run_<timestamp>/<size>/`로 이동됩니다.
   - 각 모델은 이전 가중치에서 이어서 학습(fine-tuning)하지 않고, **COCO
     사전학습 가중치(`yolov8n/s/m/l/x-pose.pt`)에서 매번 처음부터** 다시 학습합니다.
   - 학습이 끝나면 `models/results/<size>/`에는 **새로 학습된 모델만** 남습니다.
   - `reports/comparison/latest_comparison.*` 도 최신 결과로만 갱신됩니다.

과거 학습이 궁금하면 `past_train/run_.../`에서 언제든 꺼내볼 수 있습니다.

> **주의**: 이 정책은 "새 데이터가 들어왔을 때 재학습"에 대한 것입니다.
> "같은 데이터로 학습하던 중 컴퓨터/세션 문제로 중간에 끊긴 경우"는 다른
> 상황입니다 — 그건 파인튜닝이 아니라 **이어하기(resume)** 이고, 11번
> 섹션의 `--resume` 옵션을 쓰면 됩니다.

---

## 7. 성능 지표 — Pose 우선

이 프로젝트의 목적은 자세(스켈레톤) 추정이므로, **bbox 탐지 지표는 참고용**이고
**pose 관련 지표를 가장 중요하게** 봅니다. `scripts/evaluate.py`가 학습된
모델마다 train/valid/test 세 split에 대해 아래 지표를 각각 계산합니다.

| 지표 | 설명 | 우선순위 |
|---|---|---|
| **Pose mAP50 / mAP50-95** | OKS(Object Keypoint Similarity) 기반 키포인트 mAP. Ultralytics `model.val()`이 기본 제공 | ★★★ 최우선 |
| **PCK@0.05 / 0.1 / 0.2** | 예측 키포인트와 정답 키포인트 사이 거리가 (bbox 최대 변의 threshold배) 이내면 정답으로 카운트. `scripts/utils/pck.py`에서 직접 구현. 전체 평균 + 18개 키포인트별 breakdown 모두 제공 | ★★★ 최우선 |
| Box mAP50 / mAP50-95 | 돼지 탐지(bbox) 자체의 정확도 | ★ 참고용 |

- mAP만으로는 "어떤 부위가 잘 안 잡히는지" 알기 어렵기 때문에, PCK는
  키포인트별(예: `l_hand`, `r_ear_tip` ...)로도 따로 저장됩니다 —
  `reports/metrics/<size>/<split>_metrics.json`의 `pck.by_threshold.PCK@0.2.per_keypoint`.
- `compare_models.py`는 test split의 `pose mAP50-95`를 기준으로 n/s/m/l/x 중
  최고 모델을 콘솔에 출력합니다.

---

## 8. 모델(n/s/m/l, xlarge 제외) 비교

`train.py`(또는 단독으로 `python scripts/compare_models.py`)를 실행할 때마다
`reports/comparison/latest_comparison.csv`, `.png`가 **현재 존재하는 사이즈
기준으로** 최신 결과로 갱신됩니다. 정확도(pose mAP, PCK)와 함께 실시간 처리
요구사항(FPS/지연시간)을 고려해서 실제 배포 모델을 선택하면 됩니다.

### 실측 결과 (937장 데이터셋 — 10개 태스크, train 750 / valid 94 / test 93, 200에폭)

test split 기준:

| 사이즈 | pose mAP50 | pose mAP50-95 | box mAP50-95 | PCK@0.2 |
|---|---|---|---|---|
| nano | 0.810 | 0.396 | 0.821 | 0.895 |
| small | 0.849 | 0.455 | 0.847 | 0.908 |
| medium | 0.859 | **0.490** | 0.848 | **0.909** |
| large | 0.859 | 0.478 | 0.847 | 0.907 |

### 이전 학습(752장) 대비 향상 폭

직전까지 최신이었던 752장 데이터셋(n/s/m/l, 동일하게 200에폭) 결과와
937장 결과를 test split 기준으로 비교하면:

| 사이즈 | pose mAP50-95 (이전→현재) | 변화 | PCK@0.2 (이전→현재) | 변화 | box mAP50-95 (이전→현재) | 변화 |
|---|---|---|---|---|---|---|
| nano | 0.400 → 0.396 | -0.004 | 0.885 → 0.895 | +0.010 | 0.832 → 0.821 | -0.011 |
| small | 0.439 → 0.455 | **+0.016** | 0.896 → 0.908 | +0.012 | 0.850 → 0.847 | -0.003 |
| medium | 0.485 → 0.490 | +0.005 | 0.900 → 0.909 | +0.009 | 0.856 → 0.848 | -0.008 |
| large | 0.477 → 0.478 | +0.001 | 0.898 → 0.907 | +0.009 | 0.851 → 0.847 | -0.004 |

**PCK@0.2는 네 사이즈 모두 확실히 향상**됐습니다(+0.009~+0.012) — 데이터가
185장(752→937장) 늘면서 키포인트 위치 정확도 자체는 꾸준히 좋아졌다는 뜻입니다.
반면 **pose mAP50-95는 사이즈별로 갈렸습니다**: small이 가장 크게
좋아졌고(+0.016), medium·large는 소폭 개선(+0.005 / +0.001)에 그쳤고,
nano는 오히려 살짝 떨어졌습니다(-0.004). box mAP50-95도 네 사이즈 모두
소폭 하락했는데, 새로 추가된 이미지(003/005 zip 등)에 겹침·가려짐이 있는
장면이 더 많아 bbox 자체의 IoU 기준이 살짝 더 까다로워진 것으로 추정됩니다
(포즈/PCK가 함께 나빠지지 않은 걸 보면 라벨 품질 문제는 아닌 것으로 보임).

**`large`가 다시 `medium`보다 낮게** 나왔습니다(0.478 vs 0.490) — 752장 때
(0.477 vs 0.485)와 같은 패턴이라 데이터를 늘린 것만으로는 해결되지
않았습니다. 이번 학습에서는 autobatch가 `batch=1`이 아니라 `batch=16`을
골랐는데도(GPU 메모리 추정치 자체는 "7.96GB 중 40.51GB 필요(509%)"로
앞뒤가 안 맞는 경고가 떴습니다 — autobatch의 메모리 추정이 이번엔 신뢰하기
어려운 상태였던 것으로 보임) 여전히 medium을 못 넘은 걸 보면, 단순 배치
크기 문제만은 아니고 large 자체가 이 정도 데이터 규모(~900장)에서는
medium 대비 이점이 크지 않은 것으로 보입니다.
**현재 최종 순위는 실시간성 중요하면 small, 정확도 우선이면 medium**입니다.

### 스켈레톤(키포인트)별 정확도 — PCK@0.2, test split

**코(`nose`)**와 **목(`neck`)**은 급식 자세 판단(13.2번 섹션)에서 가장
중요하게 쓰이는 키포인트라 굵게 표시했습니다.

| 부위 | 키포인트 | nano | small | medium | large |
|---|---|---|---|---|---|
| 코·머리 | **nose** | **0.890** | **0.899** | **0.912** | **0.906** |
| 머리 | l_ear | 0.912 | 0.914 | 0.924 | 0.919 |
| 머리 | r_ear | 0.912 | 0.920 | 0.925 | 0.917 |
| 머리 | l_ear_tip | 0.869 | 0.886 | 0.891 | 0.890 |
| 머리 | r_ear_tip | 0.867 | 0.880 | 0.889 | 0.882 |
| 목 | **neck** | **0.920** | **0.925** | **0.927** | **0.930** |
| 몸통 | l_shoulder | 0.914 | 0.927 | 0.926 | 0.926 |
| 몸통 | r_shoulder | 0.916 | 0.926 | 0.924 | 0.924 |
| 몸통 | l_flank | 0.927 | 0.933 | 0.930 | 0.931 |
| 몸통 | r_flank | 0.923 | 0.932 | 0.930 | 0.926 |
| 몸통 | l_hip | 0.917 | 0.925 | 0.919 | 0.924 |
| 몸통 | r_hip | 0.908 | 0.922 | 0.919 | 0.917 |
| 다리 | l_hand(왼쪽 앞발) | 0.817 | 0.863 | 0.853 | 0.862 |
| 다리 | r_hand(오른쪽 앞발) | 0.828 | 0.854 | 0.866 | 0.860 |
| 다리 | l_foot(왼쪽 뒷발) | 0.883 | 0.900 | 0.893 | 0.899 |
| 다리 | r_foot(오른쪽 뒷발) | 0.882 | 0.898 | 0.897 | 0.887 |
| 꼬리 | tail | 0.916 | 0.926 | 0.921 | 0.923 |
| 꼬리 | tail_tip | 0.917 | 0.924 | 0.921 | 0.920 |

**패턴**: 네 사이즈 모두 **다리(특히 앞발 `l_hand`/`r_hand`)가 가장 약한
부위**입니다(0.82~0.87) — 다리는 자주 가려지거나 빠르게 움직여서 라벨/예측
둘 다 어려운 부위이기 때문으로 보입니다. 반대로 **코와 목은 네 사이즈 다
0.89 이상으로 안정적**이라, 급식 자세 판단(13.2번 섹션)이 이 두 키포인트에
의존하는 것은 근거가 있는 선택입니다. `large`는 `medium`과 거의 비슷한
수준(코 0.906 vs 0.912, 목은 오히려 0.930으로 최고)이라, 위 종합 지표에서
`large`가 뒤처지는 것은 특정 부위가 아니라 전반적으로 골고루 조금씩 낮은
결과입니다.

---

## 9. 데이터/모델 용량 관련 참고

`.gitignore`에 의해 다음은 원격 저장소(GitHub)에 올라가지 않습니다:
`data/raw/data/*.zip`, `data/raw/img/*`, `data/raw/label/*`, `data/processed/*`,
`data/past_train/*`, `models/pretrained/*.pt`, `models/results/**/*.pt`,
`past_train/**/*.pt`, `runs/*`, `*.avi`, `video_result/*/*.mp4`. (zip이나 영상
하나가 수백 MB~1GB에 달할 수 있으므로 반드시 이 목록에 포함되어 있는지
커밋 전에 `git status`로 확인하세요.)

이미지·라벨·가중치처럼 용량이 큰 파일을 팀과 공유해야 한다면 Git 저장소에
직접 커밋하지 말고 별도 스토리지(예: Git LFS, 사내 NAS, 클라우드 스토리지)를
사용하는 것을 권장합니다. 코드/설정(`scripts/`, `configs/`, `*.yaml`, 지표
json/csv)만 이 Git 저장소에 커밋됩니다.

---

## 10. 알려진 이슈 / 주의사항

- **라벨 좌표 클리핑**: CVAT에서 가려지거나 프레임 밖으로 나간 부위의
  키포인트를 이미지 경계 밖 좌표로 저장하는 경우가 있습니다(실측 데이터
  기준 전체 인스턴스의 약 35%). Ultralytics는 0~1 범위를 벗어난 좌표가 하나라도
  있으면 해당 인스턴스 전체를 조용히 학습에서 제외하므로,
  `scripts/split_dataset.py`가 `data/raw` -> `data/processed`로 복사할 때
  bbox/키포인트 좌표를 [0,1]로 클리핑합니다(`data/raw`의 원본은 그대로 둠).
  `python scripts/split_dataset.py` 실행 후 "N corrupt" 경고가 0이 아니면
  이 클리핑이 제대로 적용되지 않은 것이니 확인하세요.
- **데이터셋 경로는 항상 절대경로로 해석됨**: `configs/pig_pose.yaml`의
  `path: ../data/processed`는 사람이 읽기 위한 상대경로일 뿐, 실제 학습/평가
  호출에는 항상 `scripts/utils/skeleton.py`의 `resolve_data_yaml()`이 생성하는
  절대경로 사본(`configs/_pig_pose.resolved.yaml`, git에 올라가지 않음)을
  사용합니다. Ultralytics는 yaml의 `path`가 상대경로인데 현재 작업 디렉터리
  기준으로 존재하지 않으면 **전역 Ultralytics 설정(`datasets_dir`)** 기준으로
  다시 해석하는데, 이 전역 설정이 이 프로젝트가 아닌 다른 프로젝트를 가리키고
  있으면 엉뚱한 폴더를 읽거나 그 폴더에 결과물을 쓰는 사고가 조용히 발생할 수
  있습니다. 새 스크립트를 추가할 때 Ultralytics의 `model.train()`/`model.val()`을
  직접 호출한다면 `data=str(DATA_YAML)`이 아니라 반드시
  `data=str(resolve_data_yaml())`을 사용하고, `project=`/`name=`도 이
  프로젝트 안의 경로로 명시하세요.

---

## 11. 학습 파라미터 조절 가이드

### 에폭(epoch)이 뭔가요?

**에폭 1회 = 갖고 있는 학습 이미지 전체(현재 602장)를 모델이 한 번씩 다 보고
배우는 것**입니다. 문제집에 비유하면:

- 문제집(학습 데이터) 602문제를 처음부터 끝까지 한 바퀴 푸는 게 **1 에폭**
- 200 에폭 = 같은 문제집을 **200번 반복해서** 풀면서 점점 실력을 키움
- 매번 완전히 똑같이 풀리는 건 아니고, 회전/반전/색상 변화 등을 조금씩 섞어서
  (augmentation) 매번 살짝 다른 버전으로 보여줌 — 그래야 "이 각도, 이
  조명에서만" 맞히는 게 아니라 진짜 실력이 붙음

### 조절 가능한 요소들

`configs/train_*.yaml` 안에 있는 값들입니다. 기본 원칙: 대부분 값을 키우면
**정확도는 오를 가능성이 크지만 학습 시간·자원 소모도 같이 늘어납니다.**

| 요소 | 늘리면 | 줄이면 |
|---|---|---|
| **epochs**(현재 200) | 더 많이 반복 학습 → 보통 더 정확해짐. 너무 많으면 학습 데이터를 통째로 외워버려서(과적합) 처음 보는 사진(valid/test)에서는 오히려 성능이 떨어질 수 있음 | 학습 시간은 짧아지지만 덜 배운 상태로 끝남 |
| **model size**(n/s/m/l/x) | 모델이 커질수록 표현력이 좋아져 정확도↑. 학습·추론 속도가 느려지고 GPU 메모리도 훨씬 많이 씀 | 작을수록 빠르고 가벼워서 실시간(CCTV 등)에 유리하지만 정확도는 떨어짐 |
| **batch**(한 번에 몇 장씩 보고 배울지) | 크면 학습이 안정적이고 GPU 병렬처리를 잘 활용해 빠름 | GPU 메모리가 부족하면 어쩔 수 없이 작아짐(large/xlarge는 `batch: -1` autobatch로 1까지 내려갈 수 있음) — 너무 작으면 학습이 다소 불안정(그래디언트 노이즈가 큼) |
| **imgsz**(입력 해상도, 현재 640) | 크면 작은 부위(귀 끝, 발 등) 인식이 좋아짐 | 작으면 빠르지만 디테일을 놓치기 쉬움 |
| **patience**(조기 종료, 현재 30) | 크면 개선이 더뎌도 오래 기다려줌(끝까지 밀어붙임) | 작으면 지정한 에폭 동안 성능이 안 오르면 바로 중단(시간 절약) |
| **lr0**(학습률, 현재 0.01) | 크면 빨리 배우지만 너무 크면 불안정(널뛰기)해서 오히려 성능이 나빠질 수 있음 | 작으면 안정적이지만 같은 에폭 수로는 덜 배움(학습이 느림) |
| **train/valid/test 비율**(현재 80/10/10, `split_dataset.py --train/--valid/--test`) | valid/test를 늘리면 성능 측정을 더 믿을 만하게 잴 수 있지만, 학습에 쓸 데이터가 줄어 모델이 덜 배울 수 있음 | train을 늘리면 더 많이 배우지만 valid/test 데이터가 적어져 성능 측정이 덜 정확해짐 |
| **PCK threshold**(0.05/0.1/0.2, 평가 지표) | 크게 잡으면(0.2) 대충 맞아도 정답 처리 → 점수가 후하게 나옴 | 작게 잡으면(0.05) 아주 정밀하게 맞아야 정답 → 점수가 박하게 나와서 실제 정밀도를 더 엄격히 봄 |
| **workers**(DataLoader 워커 수) | 크면 데이터 로딩이 빨라져 GPU가 노는 시간이 줄어듦(기본 8) | 작으면(0) 메인 프로세스에서 직접 로딩 — 12번 섹션의 이유로 large/xlarge는 0으로 고정해둠 |

**실전 팁**: n/s/m/l(+선택적으로 x)을 다 학습해서 비교하는 이유가 바로
"모델 크기 vs 속도/정확도" 트레이드오프를 직접 눈으로 확인하기 위한
것입니다(8번 섹션의 large vs medium 사례 참고 — 클수록 항상 좋은 건 아님).
실시간 처리가
중요하면 nano/small, 정확도가 최우선이면 large/xlarge를 고르면 됩니다.

---

## 12. 장시간 학습이 중간에 끊길 때 (Windows 백그라운드 실행 주의사항)

large/xlarge처럼 몇 시간씩 걸리는 학습을 돌리다 보면, 이 프로젝트를 다루는
세션(터미널/원격 환경)이 재시작되면서 학습 프로세스가 같이 죽는 경우가 있을 수
있습니다(코드 버그가 아니라 실행 환경 쪽 문제). 겪었던 증상과 대처법을
정리합니다.

### 증상 1: 그냥 background로 띄운 프로세스가 죽음
세션이 재시작되면 그 세션에 종속된 하위 프로세스도 함께 종료됩니다.

### 증상 2: Windows 작업 스케줄러(Task Scheduler)로 띄워도 죽음
`schtasks`로 등록해도 "대화형 로그온 사용자" 컨텍스트로 실행하면 현재 세션의
콘솔에 묶여서, 그 콘솔에 전달되는 신호(Ctrl+C 등, 종료 코드
`STATUS_CONTROL_C_EXIT` = `-1073741510`)를 같이 받아 죽을 수 있습니다.

### 증상 3: 콘솔을 아예 안 갖는(`DETACHED_PROCESS`) 방식은 DataLoader가 멈춤
콘솔 신호 문제를 피하려고 콘솔 자체를 할당하지 않으면, PyTorch DataLoader가
`workers > 0`일 때 필요한 멀티프로세싱 워커 프로세스를 스폰하지 못하고
그대로 멈춰버립니다(CPU 사용량이 거의 0으로 정지 — `Get-Process`로 CPU 시간이
거의 안 늘어나는 걸로 확인 가능).

### 대처법
1. **`launch_detached.py`**(프로젝트 루트)로 띄운다 — `CREATE_NEW_CONSOLE`(숨김) +
   `CREATE_NEW_PROCESS_GROUP` + `CREATE_BREAKAWAY_FROM_JOB` 조합을 씁니다.
   이 세션의 콘솔과는 분리되면서도(신호를 같이 안 받음) DataLoader 워커가
   필요로 하는 정상적인 콘솔 컨텍스트는 유지됩니다.
   ```bash
   python launch_detached.py -- train.py --sizes l x --resume
   ```
2. large/xlarge처럼 GPU 메모리 여유가 없어 `batch`가 1~2로 내려가는
   모델은 `workers: 0`으로 설정해두는 것을 권장합니다(데이터 로딩이
   병목이 되지 않을 만큼 배치가 작기 때문에 속도 손해는 미미함).
3. 그래도 죽었다면, `python scripts/train.py --sizes <size> --resume`으로
   `models/results/<size>/weights/last.pt`부터 이어서 진행합니다(11번 섹션
   위쪽의 "새 데이터 재학습"과는 다른, 중단 복구 전용 옵션).
4. 지금 어디까지 진행됐는지는 `python scripts/training_status.py`로 확인하세요.

---

## 13. 추가 작업 (핵심 오토라벨링 학습 범위 밖)

아래는 전부 **"학습된 포즈 모델을 가지고 그다음에 무엇을 할지"**에 관한
작업입니다 — 이 저장소 본연의 목적(오토라벨링 학습 파이프라인)과는
별개로 붙여둔 것이라 여기 따로 모았습니다. 시계열(연속 프레임 시퀀스를
보는 모델, 예: ST-GCN/LSTM 같은 동작 인식 모델)은 검토는 했지만 **선택하지
않았고 구현하지도 않았습니다** — 아래는 실제로 선택/구현한 것만 정리한
것입니다.

### 13.1 영상 추적 / 실시간 추론 도구

학습된 모델(`models/results/<size>/weights/best.pt`)을 영상/웹캠에
바로 적용해보는 데모 스크립트입니다.

```bash
# 웹캠/영상에 실시간으로 스켈레톤 오버레이
python scripts/infer_realtime.py --source 0 --size m
python scripts/infer_realtime.py --source farm_cam.mp4 --size s

# 영상 파일에 대해 탐지+추적(ByteTrack)+포즈를 한 번에 돌려 샘플 영상 저장
python scripts/track_video.py --source test_video.avi --sizes n s m l x
```
`track_video.py`는 Ultralytics 내장 트래커(ByteTrack)로 개체별 추적
ID(#N)를 부여하면서 스켈레톤을 그린 샘플 영상을 `video_result/<size>/`에
저장합니다. 영상이 길면 `--max-frames`(기본 900)로 앞부분만 샘플링합니다.

### 13.2 급식(섭식) 자세 분류 — 규칙 기반 + RandomForest

"돼지가 지금 급식 중인가"를 키포인트로 판단하는 2단계 분류 작업입니다.
**규칙 기반**과 **머신러닝(RandomForest)**을 결합합니다 — 규칙 기반이
(라벨링 없이) 1차 라벨을 만들고, 그 라벨로 RandomForest를 학습시켜서
여러 특징을 더 유연하게 조합하게 합니다.

#### 1) 급식 구역 정의
`configs/feeder_zones.yaml`에 `test_video.avi` 기준 추정 다각형이 이미
들어있습니다(원형 급식기를 8각형으로 근사, 중심 약 (905, 502)). 다른
영상/카메라 앵글이면 직접 클릭해서 다시 잡으세요(화면 있는 환경에서 실행):
```bash
python scripts/define_feeder_zone.py --source test_video.avi
```
조작: 클릭으로 점 추가, `n`=새 구역, `z`=undo, `s`=저장, `q`=취소.

#### 2) 특징 추출 + 규칙 기반 1차 라벨링
```bash
python scripts/label_posture_rule_based.py
```
`data/processed/{train,valid,test}`의 **정답 키포인트**(모델 추론이 아님 —
이미 라벨링된 데이터라 GPU 없이 즉시 실행됨)에서 6개 특징을 뽑고, 규칙
(코가 급식 구역 안/근처 + 고개를 숙임)으로 섭식 여부를 1차 판정해서
`reports/posture/rule_based_dataset.csv`에 저장합니다.

| 특징 | 의미 |
|---|---|
| `nose_in_zone` | 코가 급식 구역 다각형 안에 있는지 (0/1) |
| `nose_zone_dist_norm` | 코-급식구역 중심 거리 (bbox 대각선 길이로 정규화) |
| `head_dip_deg` | 목→코 벡터가 수직 아래와 이루는 각도 (0=완전히 숙임) |
| `spine_dip_deg` | 엉덩이→목 벡터가 수직 아래와 이루는 각도 (몸을 낮췄는지) |
| `neck_rel_height` | 목 높이 - 어깨 높이 (상대적, bbox로 정규화) |
| `key_kpt_conf` | 핵심 키포인트(코/목/엉덩이/어깨)가 실제로 라벨된 비율 |

`--dip-thresh`(기본 55도), `--dist-thresh`(기본 0.35)로 규칙 민감도를
조절할 수 있습니다. 지금 기본값 기준 실측 데이터에서는 약 7~8%가
"섭식"으로 분류됩니다.

#### 3) RandomForest 학습
```bash
python scripts/train_posture_classifier.py
```
위 CSV의 `rule_label`을 정답 삼아 RandomForest를 학습합니다(split 컬럼을
그대로 써서 train/valid/test 데이터 누수 없음). 결과는
`models/posture/random_forest.joblib`(모델), `reports/posture/classifier_report.json`
(정확도/precision/recall/confusion matrix/feature importance)에 저장됩니다.

> **주의**: 지금은 RandomForest가 "규칙 기반 라벨을 얼마나 잘 흉내내는지"만
> 검증된 상태입니다(같은 특징으로 계산된 라벨이라 정확도가 매우 높게
> 나오는 게 당연함). **진짜 검증은 사람이 실제로 "섭식/비섭식"을 보고 붙인
> 라벨과 비교해야** 합니다 — 그런 라벨이 생기면
> `--labels-csv path/to/human_labels.csv` (컬럼: `split,image,instance,<6개 특징>,label`)
> 로 다시 학습시키면 규칙 기반의 한계(threshold로 못 잡는 애매한 자세)를
> ML이 보완하기 시작합니다.

#### 다음 단계로 남겨둔 것
학습된 분류기를 14.1의 `track_video.py`/실시간 추론에 연결해서 영상에
"섭식 중" 라벨을 오버레이하는 것은 아직 구현하지 않았습니다.
