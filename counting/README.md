# counting — 사료통 점유 마리 수 추론

학습된 포즈 모델로 CCTV 클립을 1초에 한 프레임씩 추론하고, 사료통에서 먹고 있는 돼지 수를 세어
1초 / 1분 / 10분 단위로 집계합니다.

---

## 1. 폴더 구조

```
counting/
├── configs/
│   ├── farm_ch010.yaml / farm_ch005.yaml / farm_sample.yaml   채널별 설정 (가중치, 사료통 원 좌표, 판정 규칙 — 방향 60°)
│   ├── eval_cameras.yaml               평가용 채널 → 설정·구역 연결 (CH010, CH005, SAMPLE; CH002 제외)
│   ├── ablation/                       Geo 조건 비교용 설정 5종 (방향 60°/90°/끔, 누움 조건)
│   └── zones/feeder_zones_*.yaml       같은 사료통 원을 48각형으로 만든 구역 (RF 특징용)
├── models/          가중치 (pig_pose_ch010_m_best.pt, git 제외 — Releases)
├── data/raw/        원본 영상 (git 제외)
├── data/labels/     CVAT 라벨 XML·평가 이미지 (git 제외)
├── outputs/         추론·평가 결과 (git 제외)
├── scripts/
│   ├── count_day.py          폴더의 클립 전체를 순서대로 추론 (클립마다 KST 시작 시각 자동 계산, 이어하기 가능)
│   ├── count_full_video.py   클립 1개를 1초 샘플링으로 추론 → 1초 CSV + 집계 CSV/그래프 (+ _dets.csv)
│   ├── plot_1min.py          1초 CSV → 1분 평균/최대 CSV + 그래프
│   ├── plot_day.py           클립별 1초 CSV를 이어 붙여 시계 기준 10분 그래프
│   ├── evaluate_methods.py   Stage 1·2 평가 (Geo / RF / RF+Geo / RF+Geo+Leg)
│   ├── run_geo_ablation.ps1  Geo 조건 5종 Stage 2 평가
│   ├── eval_split50.py       평가 50장을 클립 단위 40:10으로 나눠 4가지 방법 비교 (모델 keypoint)
│   ├── apply_methods.py      _dets.csv에 모든 방법 적용 → 방법별 10분 그래프 (YOLO 재실행 없음)
│   ├── plot_methods.py / plot_geo_ablation.py / plot_compare_runs.py   평가 결과 그림
│   ├── extract_dm_frames.py  DM 정답지용 구간 사진(1초 1장) + 기록 시트 추출 (모델과 같은 프레임)
│   ├── make_dm_examples.py   DM 라벨링 기준 예시 사진 (사람 라벨 기준)
│   ├── calibrate_feeder.py   사진·영상 한 프레임에서 사료통 원 좌표 잡기
│   └── make_circle_zone.py   사료통 원 → 다각형 구역 yaml
├── src/feeding_monitor/
│   ├── config.py             yaml → 설정 객체
│   ├── pose_tracker.py       YOLO-Pose 래퍼, KEYPOINT_NAMES (18개 순서)
│   ├── feeding_classifier.py Geo 규칙 판정
│   ├── posture.py            confidence 기반 누움 신호 (규칙에서는 꺼짐, RF+Geo 특징으로 사용)
│   ├── posture_features.py   RF 특징 6개
│   ├── leg_features.py       다리 모양 누움 특징 5개 (RF+Geo+Leg)
│   ├── method_features.py    모든 방법의 특징 계산 (평가·그래프 공통)
│   └── feeder_roi.py         사료통 빨간 뚜껑 자동 검출 (calibrate_feeder.py 보조용)
└── tests/                    판정 로직 단위 테스트 (모델·영상 불필요)
```

---

## 2. 실행

```powershell
cd counting
# 테스트: 첫 클립 1개만
..\.venv\Scripts\python scripts\count_day.py --config configs\farm_ch010.yaml --videos "<영상폴더>" `
    --out-dir outputs\ch010_test --bin-sec 600 --save-dets --limit 1

# 본 실행: --limit 빼고 out-dir만 바꾼다. 중간에 멈추면 같은 명령을 다시 실행하면 끝난 클립은 건너뛴다.
..\.venv\Scripts\python scripts\count_day.py --config configs\farm_ch010.yaml --videos "<영상폴더>" `
    --out-dir outputs\ch010_20260716_day --bin-sec 600 --save-dets

# 1분 단위 그래프
..\.venv\Scripts\python scripts\plot_1min.py outputs\ch010_20260716\<클립>_1s.csv
```

- **시각**: DVR 파일명(`[CH010] 20260716012647-20260716015128.avi`)의 시각은 UTC입니다. `count_day.py`가 +9시간 해서
  KST로 기록합니다.
- **fps**: CH010 AVI는 헤더와 상관없이 200fps로 계산합니다(`--fps 200`, 기본값).
- 추론 중에는 Geo 규칙으로만 셉니다. RF 계열은 `--save-dets`로 저장한 `_dets.csv`에 `apply_methods.py`로 나중에 적용합니다.
- 오래 돌릴 때는 Windows 절전 모드를 꺼 두세요.

### 출력 (클립마다, `<prefix>` = `CH010_<시작>-<끝>`)

| 파일 | 내용 |
|---|---|
| `<prefix>_1s.csv` | 1초마다 `elapsed_sec`, `wall_time`(KST), `pigs_geometric` (2026-11-08 이전 결과에는 빈 `pigs_randomforest` 열이 더 있음) |
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

Geo / RF / RF+Geo / RF+Geo+Leg 네 방법의 비교와 결과는 [루트 README의 연구 계획](../README.md#연구-계획--섭취visiting-판정-방법-비교)과 [PROGRESS.md](../PROGRESS.md)를 참고하세요.

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
