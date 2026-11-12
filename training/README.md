# training — 돼지 18-keypoint 포즈 모델 학습 (YOLOv8-pose)

CVAT에서 라벨링한 돼지 스켈레톤(18개 keypoint)으로 YOLOv8-pose `nano / small / medium / large`를 학습하고
pose mAP·PCK로 비교. **bbox보다 포즈(keypoint) 성능을 우선 지표**로 봄.

---

## 1. 데이터 구성

| task | 카메라 | 사진 | 돼지 | 용도 |
|---|---|---|---|---|
| 001 | CH005 (1920×1080) | 90 | 899 | 학습 (train/valid) |
| 002 | CH010 (1920×1080) | 100 | 990 | 학습 (train/valid) |
| 000 | 1분 샘플 영상 (960×720) | 86 | 860 | **test 전용** (학습에 안 씀) |

- 001+002 190장을 seed 42로 섞어 **train 162 / valid 28**(valid 15%), 000 86장은 통째로 **test**.
- test가 학습에 없는 다른 카메라라, test 점수 = 처음 보는 화면에서의 성능.
- 원본: CVAT "for images 1.1" export(`<task>_sorted.zip` = 사진 + `annotations.xml`, 스켈레톤 + State 라벨).

---

## 2. 폴더 구조

```
training/
├── data/
│   ├── raw/data/        학습용 zip (000.zip, 001.zip, 002.zip — cvat_xml_to_raw_zip.py가 만듦)
│   ├── raw/img, label/  (자동) ingest_raw_zips.py가 zip에서 꺼내 모은 이미지/라벨 (<task>__<이름>)
│   └── processed/       (자동) split_dataset.py가 만든 train/valid/test
├── configs/
│   ├── pig_pose.yaml          18개 keypoint 이름 / 스켈레톤 / flip_idx
│   ├── cvat_skeleton_raw.json CVAT skeleton 원본 (pig_pose.yaml의 출처)
│   └── train_{nano,small,medium,large}.yaml  사이즈별 하이퍼파라미터
├── models/
│   ├── pretrained/            COCO 사전학습 가중치 (없으면 자동 다운로드)
│   └── results/<run>/<size>/  학습 결과 (weights/best.pt 등, git 제외)
├── past_train/                같은 run을 다시 학습할 때 이전 결과 자동 보관 (git 제외)
├── reports/
│   ├── metrics/<run>/<size>/  split별 지표 json
│   └── comparison/            사이즈 비교(<run>/), run 간 비교(runs_<a>_vs_<b>)
├── scripts/
│   ├── cvat_xml_to_raw_zip.py CVAT XML + 사진 zip → data/raw/data/<task>.zip (YOLO-Pose 라벨)
│   ├── ingest_raw_zips.py     data/raw/data/*.zip → data/raw/{img,label}
│   ├── split_dataset.py       raw → train/valid/test (--test-zip 으로 test task 고정)
│   ├── train.py               학습 + 평가 + 사이즈 비교를 한 번에
│   ├── evaluate.py            split별 mAP·PCK 평가
│   ├── compare_models.py      사이즈 비교 리포트
│   ├── compare_runs.py        run 간 비교 (예: epoch 100 vs 200)
│   ├── archive_previous_run.py  이전 결과를 past_train/으로 이동 (train.py가 자동 호출)
│   ├── training_status.py     사이즈별 학습 진행률(%)
│   ├── build_raw_zip.py       (CVAT YOLO 형식 export용) 라벨 zip + 원본 사진 → raw zip
│   └── utils/                 경로, 스켈레톤, PCK, 라벨 입출력
└── launch_detached.py         긴 학습을 터미널과 분리해서 띄우기 (Windows)
```

---

## 3. 학습 순서

```powershell
cd training
# 1) CVAT XML export → 학습용 zip (task마다)
python scripts\cvat_xml_to_raw_zip.py --src "<경로>\001_sorted.zip" --task 001 --out data\raw\data\001.zip
python scripts\cvat_xml_to_raw_zip.py --src "<경로>\002_sorted.zip" --task 002 --out data\raw\data\002.zip
python scripts\cvat_xml_to_raw_zip.py --src "<경로>\000_sorted.zip" --task 000 --out data\raw\data\000.zip
# 2) 병합 → 분할 (000 = test 고정, 나머지에서 valid 15%)
python scripts\ingest_raw_zips.py
python scripts\split_dataset.py --test-zip data\raw\data\000.zip --valid 0.15 --seed 42
# 3) 학습 + 평가 + 비교 (run 이름별로 결과 폴더가 나뉨)
$env:POSE_RUN = "ep100"; python scripts\train.py --sizes n s m l --epochs 100 --patience 0
$env:POSE_RUN = "ep200"; python scripts\train.py --sizes n s m l --epochs 200 --patience 0
python scripts\compare_runs.py --runs ep100 ep200
# 진행률: $env:POSE_RUN = "ep100"; python scripts\training_status.py
```

- **CVAT XML → YOLO-Pose 변환 규칙**: 박스 = 스켈레톤 점들의 최소·최대 범위, v = 2 보임 / 1 가려짐 / 0 화면 밖.
  CVAT 자체 YOLO export와 비교해 돼지 299마리에서 박스·표시가 완전히 같음을 확인.
- **`POSE_RUN`**: 같은 데이터로 설정만 바꾼 학습을 덮어쓰지 않고 나란히 저장(`results/<run>/`, `metrics/<run>/`).
- **`--patience 0`**: 조기 종료를 꺼서 지정한 epoch를 끝까지 학습. best.pt는 valid 성능 기준으로 저장됨.
- **데이터가 늘면 항상 처음부터 재학습**(COCO 사전학습에서 시작, 이전 가중치에서 이어 학습하지 않음).
- **중간에 끊긴 학습**은 `--resume`으로 이어 감(같은 `POSE_RUN`으로).
- 오래 걸리는 학습은 `python launch_detached.py -- train.py --sizes l`로 띄우면 터미널을 닫아도 계속 돎.

---

## 4. 결과

학습 중 — 끝나면 epoch 100 / 200 × n/s/m/l 결과를 여기에 정리.

### 지표

| 지표 | 설명 |
|---|---|
| **Pose mAP50 / mAP50-95** | OKS 기반 keypoint mAP (Ultralytics `model.val()`) — 최우선 |
| **PCK@0.05 / 0.1 / 0.2** | 예측-정답 keypoint 거리가 bbox 긴 변 × threshold 이내인 비율 (`utils/pck.py`) — 최우선 |
| Box mAP50 / mAP50-95 | bbox 탐지 정확도 — 참고용 |

---

## 5. 18개 keypoint

라벨 txt의 keypoint 순서는 CVAT skeleton svg의 **`data-node-id` 오름차순**(raw json의 `sublabels` 표시 순서가 아님).
순서가 어긋나면 지표와 섭취 판정(코=12, 목=11)이 조용히 틀어지므로, CVAT skeleton을 바꾸면
`configs/pig_pose.yaml`과 `counting/src/feeding_monitor/pose_tracker.py`의 `KEYPOINT_NAMES`를 함께 고칠 것.

| # | 이름 | # | 이름 | # | 이름 |
|---|---|---|---|---|---|
| 0 | tail_tip | 6 | r_flank | 12 | **nose** |
| 1 | tail | 7 | l_flank | 13 | l_ear |
| 2 | l_hip | 8 | l_shoulder | 14 | l_ear_tip |
| 3 | r_hip | 9 | r_shoulder | 15 | r_ear |
| 4 | l_foot | 10 | l_hand (왼 앞발) | 16 | r_ear_tip |
| 5 | r_foot | 11 | **neck** | 17 | r_hand (오른 앞발) |

라벨 한 줄 형식: `0 cx cy w h x1 y1 v1 ... x18 y18 v18` (좌표는 0\~1 정규화, v: 0=화면 밖 / 1=가려짐 / 2=보임)

---

## 6. 주의사항

- **좌표 클리핑**: CVAT가 이미지 밖 좌표를 저장하는 경우가 있고, Ultralytics는 0\~1 밖 좌표가 하나라도 있으면
  그 인스턴스를 통째로 버림. `split_dataset.py`가 복사할 때 [0,1]로 클리핑(원본은 그대로).
- **데이터 경로는 항상 절대경로**: 학습/평가는 `configs/pig_pose.yaml` 대신 런타임에 만드는
  `configs/_pig_pose.resolved.yaml`(절대경로, git 제외)을 사용. 새 스크립트에서 `model.train()`/`model.val()`을
  부를 때도 `data=str(resolve_data_yaml())`을 쓸 것. 상대경로를 쓰면 Ultralytics 전역 설정의 다른 폴더를 읽을 수 있음.
