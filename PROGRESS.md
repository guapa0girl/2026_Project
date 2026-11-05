# 진행 상황 (요일별)

이번 주 목표: **Geo 결과 완성 + 토요일에 RF 추가** → 월요일 미팅

| 날짜 | 요일 | 계획 | 상태 |
|---|---|---|---|
| 11/04 | 수 | pose 모델 학습(n/s/m) → 가중치, 영상 시간·fps 확인, 레포 생성 | ✅ 완료 |
| 11/05 | 목 | PC: 25분 클립 추론, 레포 정리, 주간 추론, Geo 조건별 비교(Stage 2), 주간 Geo 그래프 | ✅ 완료 |
| 11/06 | 금 | 휴무 | — |
| 11/07 | 토 | PC: 수아 라벨 → Stage 1, RF 학습 → Stage 2 재실행, 세 방법 그래프 비교, 슬라이드 초안 | ⏳ 예정 |
| 11/08 | 일 | 휴무 | — |
| 11/09 | 월 | 미팅 | ⏳ 예정 |

---

## 11/04 (수) — pose 모델 학습

### 한 일
- [x] CVAT task 4개(000~003) 병합 → **287장** (평가 50장 `training/eval_frames.txt`는 제외)
  - 채널: CH005 90 · CH010 50 · CH002 61 · 채널 미표기 86
  - 분할: train 230 / valid 29 / test 28
- [x] YOLOv8-pose **n / s / m** 학습 (COCO 사전학습, 100 epoch, imgsz 640) — 밤새 돌려 11/05 00:49 종료
- [x] 추론용 가중치: medium `best.pt` → `counting/models/pig_pose_ch010_m_best.pt`
- [x] DVR 영상 확인: 파일명 시각은 **UTC (KST = +9h)**, 헤더와 무관하게 **fps = 200** (ffmpeg로 확인)
- [x] GitHub 레포 생성 (training + counting 합침), Release `1.0.0`에 가중치·학습 산출물·25분 영상 첨부 (데이터셋은 비공개)
- [ ] large(l) 학습 — 예정

### 결과 — pose 모델 (test 28장)

| 모델 | pose mAP50 | pose mAP50-95 | box mAP50-95 | PCK@0.1 | PCK@0.2 |
|---|---|---|---|---|---|
| nano | 0.703 | 0.377 | 0.836 | 0.800 | 0.923 |
| small | 0.790 | 0.420 | 0.861 | 0.808 | 0.923 |
| **medium** | **0.795** | **0.436** | **0.870** | **0.828** | **0.934** |
| large | 학습 예정 | | | | |

- test가 28장이라 사이즈 간 0.01~0.02 차이는 오차 범위. 추론은 medium 사용.
- 산출물: `training/models/results/<size>/`, 지표 `training/reports/metrics/<size>/`

---

## 11/05 (목) — PC

### 한 일 (오전 · 레포 정리)
- [x] `count_day.py` / `count_full_video.py`: RF `--model` 선택 사항으로, `--save-dets`(검출별 keypoint 저장) 추가
- [x] 25분 클립 추론 (KST 10:26~10:51) → 1초 샘플 1,474개, 약 12분 소요
  - geo 1분 평균 0~2.4마리, 10분 평균 0.95 → 1.19 → 1.59마리
  - 1분 단위 그래프 스크립트 `counting/scripts/plot_1min.py` 추가
- [x] 레포 정리: 현재 CH010 연구 파일만 남기고 이전 연구 파일은 `counting/past/`, `training/past/`로 보관 (옮겨서 다시 쓰는 방법 README에 정리)
- [x] README에 연구 계획 추가: Geo 단독 / RF 단독 / RF+Geo feature 비교, visiting F1, 클립 단위 StratifiedGroupKFold
- [x] 추론 결과(`counting/outputs/`)는 git에서 제외, Release에만 첨부
- [x] 커밋 기록 정리 (Contributors에 본인만 표시)

### 한 일 (저녁 · 평가)
- [x] 평가 스크립트·설정 반영 후 커밋·푸시
  - `counting/scripts/evaluate_methods.py` (Stage 1 / Stage 2, Geo / RF / RF+Geo)
  - `counting/configs/eval_cameras.yaml` (CH010만 활성)
  - `counting/configs/ablation/` (설정 10개) + `counting/scripts/run_geo_ablation.ps1`
  - `counting/.gitignore`에 `data/labels/` 추가 (라벨 XML·평가 이미지가 공개 레포에 안 올라가게)
- [x] 주간 추론 — 앞쪽 3개 클립, **KST 10:26~11:23 (약 57분)**
- [x] Stage 2 — Geo 조건별 비교 (평가 50장, 모델 keypoint)
- [x] 주간 Geo 10분 단위 그래프 (`counting/scripts/plot_day.py` 추가)
- [ ] 수아 라벨링 요청 → 라벨링용 zip을 토요일에 받기로 함

### 데이터 위치 (모두 git 제외)

| 내용 | 위치 |
|---|---|
| 평가 50장 정답 XML (스켈레톤 + State) | `counting/data/labels/ch010_eval50.xml` (원래 이름 `annotations.xml`) |
| 평가 50장 이미지 | `counting/data/labels/eval50_img/` |
| CH010 7/16 영상 8개 (KST 10:26~13:32) | `counting/data/raw/CH010_20260716/` |

- XML에는 이미지 100장이 있지만 평가 목록(`training/eval_frames.txt`) 50장만 사용 (나머지 50장은 전부 Unlabeling).
- 평가 50장: 사람 라벨 495마리 = visiting 83 / other 412.

### 결과 1 — Geo 조건별 비교 (Stage 2)

사람 개체 495 중 모델 검출과 매칭 **455 (91.9%)** → 80% 이상이라 `--match-thresh` 조정 불필요.

| 설정 | visiting F1 | precision | recall | 마릿수 MAE | 치우침(bias) |
|---|---|---|---|---|---|
| 방향 60° (현재) | 0.671 | 0.679 | 0.663 | **0.54** | **+0.02** |
| 방향 90° | **0.713** | 0.638 | 0.807 | 0.66 | +0.50 |
| 방향 조건 끔 | 0.688 | 0.551 | 0.916 | 1.24 | +1.20 |
| 60° + 누움 조건 | 0.000 | – | 0.000 | 1.66 | −1.66 |
| 방향 끔 + 누움 조건 | 0.000 | – | 0.000 | 1.66 | −1.66 |

- F1은 90°가 가장 높지만 프레임당 0.5마리 과대 집계. 마릿수(그래프 값) 정확도는 60°가 가장 좋음.
- 누움 조건 기본 임계값은 CH010에서 모든 돼지를 "누움"으로 제외 → **CH010용 보정 전에는 사용 불가**.
- 상세: `counting/outputs/eval/geo_ablation/<설정>/summary.md`

### 결과 2 — 주간 Geo 그래프 (방향 60°)

| 시간 (KST) | 평균 마릿수 | 최대 | 비고 |
|---|---|---|---|
| 10:20–10:30 | 1.42 | 2 | 부분 구간 (10:26~) |
| 10:30–10:40 | 0.69 | 2 | |
| 10:40–10:50 | **1.65** | 4 | 최고 |
| 10:50–11:00 | 1.14 | 3 | |
| 11:00–11:10 | 0.23 | 2 | |
| 11:10–11:20 | 0.91 | 2 | |
| 11:20–11:30 | 0.05 | 1 | 부분 구간 (~11:23) |

- 그림: `counting/outputs/ch010_20260716_day/day_10min.png` (막대 10분 평균, 선 1분 평균)
- 클립별 프레임 읽기 실패 7~9개씩 (건너뜀), 실패 클립 없음.

---

## 11/07 (토) — PC

- [ ] 수아 라벨 export (CVAT for images 1.1) → `counting/data/labels/state_287.xml`
- [ ] kappa용 10장 겹치기 확인 (수아·서인)
- [ ] Stage 1 (Geo / RF / RF+Geo, 클립 단위 StratifiedGroupKFold)
- [ ] Stage 1 전체로 RF 학습 → Stage 2 재실행 (RF·RF+Geo 포함)
- [ ] 가장 좋은 방법을 주간 `_dets.csv`에 적용 → 세 방법 그래프 비교 (적용 스크립트 필요)
- [ ] 방향 60° / 90° 중 그래프에 쓸 설정 결정
- [ ] 슬라이드 초안 (일요일 휴무 → 토요일에 결과물 전부 마무리)

라벨이 적게 들어오면 들어온 만큼으로 돌리고, 라벨 수를 한계점으로 적는다.

```powershell
cd C:\ETRI\2026_Project\counting
C:\ETRI\Piggy_Feeding_Counting\.venv\Scripts\python scripts\evaluate_methods.py --cameras configs\eval_cameras.yaml `
    --stage1-xml data\labels\state_287.xml --stage1-exclude ..\training\eval_frames.txt `
    --stage2-xml data\labels\ch010_eval50.xml --stage2-images data\labels\eval50_img `
    --stage2-only-list ..\training\eval_frames.txt --out outputs\eval
```

---

## 11/09 (월) — 미팅

- [ ] 발표: pose 모델 결과, Geo 조건별 비교, 주간 그래프, (토요일 결과) Geo vs RF vs RF+Geo
- 한계점으로 적을 것: 평가 50장 규모, 주간 영상은 앞쪽 1시간만 처리, 누움 조건 미보정, Stage 1 라벨 수
