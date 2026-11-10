# training — 돼지 18-keypoint 포즈 모델 학습 (YOLOv8-pose)

CVAT에서 라벨링한 돼지 스켈레톤(18개 keypoint)으로 YOLOv8-pose `nano / small / medium / large`를 학습하고
pose mAP·PCK로 비교합니다. **bbox보다 포즈(keypoint) 성능을 우선 지표**로 봅니다.

---

## 1. 폴더 구조

```
training/
├── data/
│   ├── raw/data/        ★ CVAT task export zip을 여기에 (000.zip, 001.zip ...) ★
│   ├── raw/img, label/  (자동) ingest_raw_zips.py가 zip에서 꺼내 모은 이미지/라벨
│   └── processed/       (자동) split_dataset.py가 만든 train/valid/test
├── configs/
│   ├── pig_pose.yaml          18개 keypoint 이름 / 스켈레톤 / flip_idx
│   ├── cvat_skeleton_raw.json CVAT skeleton 원본 (pig_pose.yaml의 출처)
│   └── train_{nano,small,medium,large}.yaml  사이즈별 하이퍼파라미터
├── models/
│   ├── pretrained/      COCO 사전학습 가중치 (없으면 자동 다운로드)
│   └── results/<size>/  최근 학습 결과 (weights/best.pt 등, git 제외)
├── past_train/          이전 학습 결과 자동 보관 (git 제외)
├── reports/
│   ├── metrics/<size>/      split별 지표 json
│   └── comparison/          사이즈 비교표·그래프
├── scripts/
│   ├── build_raw_zip.py     CVAT 라벨 zip + 원본 이미지 → data/raw/data용 zip (평가 50장 제외)
│   ├── ingest_raw_zips.py   data/raw/data/*.zip → data/raw/{img,label}
│   ├── split_dataset.py     raw 전체 → train/valid/test
│   ├── train.py             학습 + 평가 + 비교를 한 번에
│   ├── evaluate.py          split별 mAP·PCK 평가
│   ├── compare_models.py    사이즈 비교 리포트
│   ├── archive_previous_run.py  이전 결과를 past_train/으로 이동 (train.py가 자동 호출)
│   ├── training_status.py   사이즈별 학습 진행률(%)
│   └── utils/               경로, 스켈레톤, PCK, 라벨 입출력
├── eval_frames.txt      섭취 판정 평가용 50장 목록 (pose 학습에서 제외)
└── launch_detached.py   긴 학습을 터미널과 분리해서 띄우기 (Windows)
```

---

## 2. 학습 순서

```powershell
# (라벨만 export한 경우) 원본 이미지와 묶어서 zip 만들기 -- 평가 50장은 자동 제외
python scripts\build_raw_zip.py --labels-zip "CVAT_export.zip" --images "원본이미지폴더" `
    --exclude eval_frames.txt --task CH010_431 --out data\raw\data\CH010_431.zip

python scripts\ingest_raw_zips.py                                  # zip 병합
python scripts\split_dataset.py --train 0.8 --valid 0.1 --test 0.1 # 분할 (seed 42)
python scripts\train.py --sizes n s m --epochs 100                 # 학습 + 평가 + 비교
python scripts\training_status.py                                  # 진행률 확인
```

- **평가 50장 제외**: `eval_frames.txt`의 프레임은 pose 학습에 섞이면 안 됩니다. `build_raw_zip.py --exclude`가
  파일명 또는 `[CH010] <영상>_f<프레임>` ID로 걸러냅니다.
- **데이터가 늘면 항상 처음부터 재학습**합니다(이전 가중치에서 이어 학습하지 않음, COCO 사전학습에서 시작).
  이전 결과는 학습 직전에 `past_train/run_<시각>/`으로 자동 이동됩니다.
- **중간에 끊긴 학습**은 `--resume`으로 이어갑니다: `python scripts\train.py --sizes l --resume`
- 오래 걸리는 학습은 `python launch_detached.py -- train.py --sizes l`로 띄우면 터미널을 닫아도 계속 돕니다.

---

## 3. 현재 결과 (2026-11-05, 287장, 100 epoch, imgsz 640)

| 사이즈 | test pose mAP50 | test pose mAP50-95 | test box mAP50-95 | test PCK@0.1 | test PCK@0.2 |
|---|---|---|---|---|---|
| nano | 0.703 | 0.377 | 0.836 | 0.800 | 0.923 |
| small | 0.790 | 0.420 | 0.861 | 0.808 | 0.923 |
| medium | **0.795** | 0.436 | 0.870 | 0.828 | 0.934 |
| large | 0.778 | **0.437** | **0.874** | **0.831** | **0.938** |

- 데이터: CVAT task 000~003, 287장 (CH005 90 · CH010 50 · CH002 61 · 채널 미표기 86). 분할 train 230 / valid 29 / test 28.
- test가 28장이라 사이즈 간 0.01~0.02 차이는 오차 범위입니다. large(2026-11-10 학습)는 medium과 거의 같아 추론은 medium을 씁니다.
- 굵은 값 = 지표별 1등.
- 키포인트별 PCK는 `reports/metrics/<size>/test_metrics.json`의 `pck.by_threshold.PCK@0.2.per_keypoint`에 있습니다.

### 지표

| 지표 | 설명 |
|---|---|
| **Pose mAP50 / mAP50-95** | OKS 기반 keypoint mAP (Ultralytics `model.val()`) — 최우선 |
| **PCK@0.05 / 0.1 / 0.2** | 예측-정답 keypoint 거리가 bbox 긴 변 × threshold 이내인 비율 (`utils/pck.py`) — 최우선 |
| Box mAP50 / mAP50-95 | bbox 탐지 정확도 — 참고용 |

---

## 4. 18개 keypoint

라벨 txt의 keypoint 순서는 CVAT skeleton svg의 **`data-node-id` 오름차순**입니다(raw json의 `sublabels` 표시 순서가 아님).
순서가 어긋나면 지표와 섭취 판정(코=12, 목=11)이 조용히 틀어지므로, CVAT skeleton을 바꾸면
`configs/pig_pose.yaml`과 `counting/src/feeding_monitor/pose_tracker.py`의 `KEYPOINT_NAMES`를 함께 고치세요.

| # | 이름 | # | 이름 | # | 이름 |
|---|---|---|---|---|---|
| 0 | tail_tip | 6 | r_flank | 12 | **nose** |
| 1 | tail | 7 | l_flank | 13 | l_ear |
| 2 | l_hip | 8 | l_shoulder | 14 | l_ear_tip |
| 3 | r_hip | 9 | r_shoulder | 15 | r_ear |
| 4 | l_foot | 10 | l_hand (왼 앞발) | 16 | r_ear_tip |
| 5 | r_foot | 11 | **neck** | 17 | r_hand (오른 앞발) |

라벨 한 줄 형식: `0 cx cy w h x1 y1 v1 ... x18 y18 v18` (좌표는 0~1 정규화, v: 0=라벨 없음 / 1=가려짐 / 2=보임)

---

## 5. 주의사항

- **좌표 클리핑**: CVAT가 이미지 밖 좌표를 저장하는 경우가 있고, Ultralytics는 0~1 밖 좌표가 하나라도 있으면
  그 인스턴스를 통째로 버립니다. `split_dataset.py`가 복사할 때 [0,1]로 클리핑합니다(원본은 그대로).
- **데이터 경로는 항상 절대경로**: 학습/평가는 `configs/pig_pose.yaml` 대신 런타임에 만드는
  `configs/_pig_pose.resolved.yaml`(절대경로, git 제외)을 씁니다. 새 스크립트에서 `model.train()`/`model.val()`을
  부를 때도 `data=str(resolve_data_yaml())`을 쓰세요. 상대경로를 쓰면 Ultralytics 전역 설정의 다른 폴더를 읽을 수 있습니다.
