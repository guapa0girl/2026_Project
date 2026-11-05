# counting — 사료통 점유 마리 수 추론

학습된 포즈 모델로 CCTV 클립을 1초에 한 프레임씩 추론하고, 사료통에서 먹고 있는 돼지 수를 세어
1초 / 1분 / 10분 단위로 집계합니다.

---

## 1. 폴더 구조

```
counting/
├── configs/
│   ├── farm_ch010.yaml                 CH010 설정 (가중치, 사료통 원 좌표, 판정 규칙)
│   └── zones/feeder_zones_ch010.yaml   같은 사료통 원을 48각형으로 만든 구역 (RF 특징용)
├── models/          가중치 (pig_pose_ch010_m_best.pt, git 제외 — Releases)
├── data/raw/        원본 영상 (git 제외)
├── outputs/         추론 결과 (git 제외)
├── scripts/
│   ├── count_day.py          폴더의 클립 전체를 순서대로 추론 (클립마다 KST 시작 시각 자동 계산, 이어하기 가능)
│   ├── count_full_video.py   클립 1개를 1초 샘플링으로 추론 → 1초 CSV + 집계 CSV/그래프 (+ _dets.csv)
│   ├── plot_1min.py          1초 CSV → 1분 평균/최대 CSV + 그래프 (YOLO 재실행 없음)
│   ├── calibrate_feeder.py   영상 한 프레임에서 사료통 원 좌표 잡기
│   └── make_circle_zone.py   사료통 원 → 다각형 구역 yaml
├── src/feeding_monitor/
│   ├── config.py             yaml → 설정 객체
│   ├── pose_tracker.py       YOLO-Pose 래퍼, KEYPOINT_NAMES (18개 순서)
│   ├── feeding_classifier.py geometric 섭취 판정
│   ├── posture.py            누운 자세 판정 (CH010에서는 꺼짐)
│   ├── posture_features.py   RF용 특징 6개 추출
│   ├── posture_classifier.py RandomForest 래퍼
│   └── feeder_roi.py         사료통 빨간 뚜껑 자동 검출 (calibrate_feeder.py 보조용)
└── tests/                    판정 로직 단위 테스트 (모델·영상 불필요)
```

---

## 2. 실행

```powershell
cd counting
# 테스트: 첫 클립 1개만
..\.venv\Scripts\python scripts\count_day.py --config configs\farm_ch010.yaml --videos "<영상폴더>" `
    --out-dir outputs\ch010_test --zones configs\zones\feeder_zones_ch010.yaml --bin-sec 600 --save-dets --limit 1

# 본 실행: --limit 빼고 out-dir만 바꾼다. 중간에 멈추면 같은 명령을 다시 실행하면 끝난 클립은 건너뛴다.
..\.venv\Scripts\python scripts\count_day.py --config configs\farm_ch010.yaml --videos "<영상폴더>" `
    --out-dir outputs\ch010_20260716 --zones configs\zones\feeder_zones_ch010.yaml --bin-sec 600 --save-dets

# 1분 단위 그래프
..\.venv\Scripts\python scripts\plot_1min.py outputs\ch010_20260716\<클립>_1s.csv
```

- **시각**: DVR 파일명(`[CH010] 20260716012647-20260716015128.avi`)의 시각은 UTC입니다. `count_day.py`가 +9시간 해서
  KST로 기록합니다.
- **fps**: CH010 AVI는 헤더와 상관없이 200fps로 계산합니다(`--fps 200`, 기본값).
- **RF**: `--model <rf.joblib>`을 주면 RandomForest 판정도 같이 셉니다. 없으면 RF 열은 `nan`입니다.
- 오래 돌릴 때는 Windows 절전 모드를 꺼 두세요.

### 출력 (클립마다, `<prefix>` = `CH010_<시작>-<끝>`)

| 파일 | 내용 |
|---|---|
| `<prefix>_1s.csv` | 1초마다 `wall_time`(KST), `pigs_geometric`, `pigs_randomforest` |
| `<prefix>.csv` / `.png` | `--bin-sec` 단위 평균(기본 mean, `--bin-agg max` 가능)과 그래프 |
| `<prefix>_1min.csv` / `.png` | `plot_1min.py` 결과: 1분 평균·최대 |
| `<prefix>_dets.csv` | `--save-dets`: 1초 샘플마다 검출된 돼지 전부의 bbox + 18 keypoint (x, y, conf) + geo 판정. 판정 규칙이나 RF 특징을 바꿔도 YOLO를 다시 돌리지 않고 재계산할 수 있습니다. |

---

## 3. 섭취 판정 (geometric)

라벨 정의(CVAT State attribute):
- **visiting** = 코가 빨간 사료통 뚜껑 안에 있고, 누워 있지 않음
- **other** = 나머지 (뚜껑에 머리만 얹고 누운 경우, 벽 급수기 포함)

`feeding_classifier.classify()`는 아래 조건을 모두 만족해야 "섭취"로 셉니다. 값은 `configs/farm_ch010.yaml`에 있습니다.

| # | 조건 | CH010 설정 |
|---|---|---|
| 1 | 코(12)·목(11) keypoint confidence ≥ 기준 | `keypoint_conf_min: 0.5` |
| 2 | 코가 사료통 원 안 | 중심 (925.3, 535.1), r 337.8, 링 0.0~1.0 (뚜껑 전체) |
| 3 | 목→코 방향이 사료통 중심을 향함 | `direction_angle_max_deg: 60` |
| 4 | 옆으로 누운 자세가 아님 | `posture_enabled: false` (CH010 라벨로 확인 전까지 꺼 둠) |

Geo 단독 / RF 단독 / RF + Geo feature 세 방법의 비교 계획은 [루트 README의 연구 계획](../README.md#연구-계획--섭취visiting-판정-방법-비교)을 참고하세요.

사료통 좌표를 다시 잡을 때:

```powershell
..\.venv\Scripts\python scripts\calibrate_feeder.py --source "<영상.avi>" --at-sec 15 --cx 925 --cy 535 --r 338
..\.venv\Scripts\python scripts\make_circle_zone.py --cx 925 --cy 535 --r 338 --out configs\zones\feeder_zones_ch010.yaml
```

---

## 4. 테스트

```powershell
..\.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
```

keypoint 순서(코=12, 목=11)가 `configs/*.yaml`과 맞는지, 판정 조건이 의도대로 동작하는지 확인합니다.
