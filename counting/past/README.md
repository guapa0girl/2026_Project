# counting/past — 이전 연구에서 사용하던 파일 (보관용)

이 폴더는 **이전 연구**(CH005 돈방 분석, pen01 샘플 영상, RTSP 실시간 모니터링)에서 쓰던 파일을 지우지 않고 보관해 둔 곳입니다.
**현재 CH010 연구에서는 사용하지 않습니다.** 현재 코드는 상위 폴더(`counting/`)를 보세요.

- 원래 `counting/` 안에 있던 위치를 그대로 유지했습니다(`past/scripts/run_realtime.py` ← 원래 `scripts/run_realtime.py`).
- 이 폴더에서 바로 실행되도록 맞춰 두지 않았습니다. 다시 쓰려면 파일을 원래 위치로 옮겨야 하고,
  일부는 `counting/src/feeding_monitor/`의 현재 모듈을 import합니다.
- `OLD_README.md`, `OLD_models_README.md`: 정리하기 전의 counting README와 models README 원문입니다.

## 파일 목록

| 분류 | 파일 | 이전 용도 |
|---|---|---|
| 설정 | `configs/farm_ch005_*.yaml` (5개) | CH005 돈방 낮·밤 영상 설정 (2026-06-10, v2 모델, 자세 판정 보정값 포함) |
| | `configs/pen01.yaml`, `pen01_realtime.yaml` | 1분 샘플 영상 / 실시간 스트림용 예시 설정 |
| | `configs/zones/feeder_zones_ch005.yaml` | CH005 RandomForest용 사료통 다각형 구역 |
| 실행 | `scripts/run_analysis.py` | 녹화 영상 전체를 연속 추적(ByteTrack)으로 분석 |
| | `scripts/run_realtime.py` | RTSP/웹캠 실시간 분석, 화면 Count 오버레이 + 초당 로그 |
| | `scripts/plot_live_log.py` | 초당 로그 → 시간대별(하루 패턴) 그래프 |
| | `scripts/compare_stages.py` | 성장 단계별(이유자돈·육성돈·비육돈) 하루 패턴 비교 |
| 검증·보정 | `scripts/validate_counts.py` | 사람이 센 마릿수 대비 MAE/precision/recall |
| | `scripts/calibrate_posture.py` | CH005 야간 영상 34장으로 누운 자세 임계값 보정 |
| | `scripts/sensitivity_check.py` | 각도·링 반경 임계값 그리드 스윕 |
| | `scripts/compare_classifiers.py` | geo vs RF 판정 일치도 (tracks.csv 기준) |
| | `scripts/render_comparison_video.py` | geo·RF 판정을 좌우로 그린 비교 영상 |
| 모듈 | `src/feeding_monitor/aggregator.py` | 평활화 + bout 필터 + 시간 bin 집계 |
| | `src/feeding_monitor/pipeline.py`, `report.py` | run_analysis/run_realtime용 파이프라인, CSV·그래프 저장 |
| | `src/feeding_monitor/live_log.py`, `daily_pattern.py` | 초당 로그 읽기/쓰기, 하루 패턴 접기 |
| 테스트 | `tests/test_live_log.py`, `test_daily_pattern.py` | 위 두 모듈의 단위 테스트 |
| 기타 | `outputs/exclusion_events.csv` | 분석 제외 구간(이동·처치일 등) 수동 기록 템플릿 |

---

## 다시 사용하려면 — 코드별로 옮길 위치

**규칙**: 경로에서 `past/`만 빼면 원래 위치입니다. 명령은 모두 `counting/` 폴더에서 실행합니다.

```
counting/past/scripts/X.py                → counting/scripts/X.py
counting/past/src/feeding_monitor/X.py    → counting/src/feeding_monitor/X.py
counting/past/configs/X.yaml              → counting/configs/X.yaml
counting/past/configs/zones/X.yaml        → counting/configs/zones/X.yaml
counting/past/tests/X.py                  → counting/tests/X.py
counting/past/outputs/X.csv               → counting/outputs/X.csv
```

스크립트는 `counting/src/feeding_monitor/`의 모듈을 import하므로, 아래 표의 **같이 옮길 모듈**을
`counting/src/feeding_monitor/`로 옮겨야 실행됩니다. 표에 없는 모듈(`config`, `pose_tracker`, `feeding_classifier`,
`posture`, `posture_features`, `posture_classifier`, `feeder_roi`)은 이미 현재 폴더에 있습니다.

### 스크립트별

| 스크립트 → `counting/scripts/` | 같이 옮길 모듈 → `counting/src/feeding_monitor/` | 같이 옮길 설정 → `counting/configs/` | 따로 구해야 하는 파일 |
|---|---|---|---|
| `run_analysis.py` | `pipeline.py`, `aggregator.py`, `report.py`, `live_log.py` | `pen01.yaml` 또는 `farm_ch005_*.yaml` | 설정의 `weights:` 가중치, 영상 |
| `run_realtime.py` | `pipeline.py`, `aggregator.py`, `report.py`, `live_log.py` | `pen01_realtime.yaml` | `models/pig_pose_m_best.pt`, RTSP 카메라 |
| `plot_live_log.py` | `live_log.py`, `aggregator.py`, `daily_pattern.py` | — | `run_realtime.py`가 만든 `*_live_log.txt` |
| `compare_stages.py` | `live_log.py`, `aggregator.py`, `daily_pattern.py` | — | 단계별 `*_live_log.txt` |
| `validate_counts.py` | `pipeline.py`, `aggregator.py` | `pen01.yaml` 또는 `farm_ch005_20260610.yaml`, (RF 쓸 때) `zones/feeder_zones_ch005.yaml` | 가중치, 영상, (RF 쓸 때) `models/posture_random_forest.joblib` |
| `calibrate_posture.py` | — | `farm_ch005_v2m_night.yaml` 등 | `models/pig_pose_v2_m_best.pt`, CH005 영상 |
| `sensitivity_check.py` | — | `pen01.yaml` 등 | `run_analysis.py`가 만든 `*_tracks.csv` (그래서 사실상 `run_analysis.py`도 필요) |
| `compare_classifiers.py` | — | `zones/feeder_zones_ch005.yaml` | `*_tracks.csv`, `models/posture_random_forest.joblib` |
| `render_comparison_video.py` | — | `farm_ch005_20260610.yaml`, `zones/feeder_zones_ch005.yaml` | 가중치, 영상, `models/posture_random_forest.joblib` |

### 테스트

| 테스트 → `counting/tests/` | 같이 옮길 모듈 → `counting/src/feeding_monitor/` |
|---|---|
| `test_live_log.py` | `live_log.py`, `aggregator.py` |
| `test_daily_pattern.py` | `daily_pattern.py` |

옮긴 뒤 `python -m unittest discover -s tests -p "test_*.py"`로 확인합니다.

### 설정과 가중치

| 설정 | `weights:`가 가리키는 파일 |
|---|---|
| `pen01.yaml`, `pen01_realtime.yaml`, `farm_ch005_20260610.yaml` | `models/pig_pose_m_best.pt` |
| `farm_ch005_v2m_*.yaml` (4개) | `models/pig_pose_v2_m_best.pt` |

- 이 가중치들과 `posture_random_forest.joblib`, CH005·pen01 영상은 **이 저장소에 없습니다.** 이전 연구 자료에서 구해
  `counting/models/`, `counting/data/raw/`에 두세요. 자세한 계보는 `OLD_models_README.md`에 있습니다.
- 현재 CH010 가중치(`pig_pose_ch010_m_best.pt`)로 돌리려면 설정의 `weights:`를 바꾸면 됩니다. keypoint 순서는 같습니다.
- `outputs/exclusion_events.csv`는 `counting/outputs/`로 옮기면 `.gitignore` 때문에 git에서 빠집니다. 커밋하려면 `git add -f`를 쓰세요.

### 예시: 실시간 모니터링 다시 쓰기

```powershell
cd counting
move past\scripts\run_realtime.py scripts\
move past\src\feeding_monitor\pipeline.py src\feeding_monitor\
move past\src\feeding_monitor\aggregator.py src\feeding_monitor\
move past\src\feeding_monitor\report.py src\feeding_monitor\
move past\src\feeding_monitor\live_log.py src\feeding_monitor\
move past\configs\pen01_realtime.yaml configs\
# configs\pen01_realtime.yaml 의 weights: 를 실제 가중치로 맞춘 뒤
..\.venv\Scripts\python scripts\run_realtime.py --config configs\pen01_realtime.yaml --source rtsp://<camera-ip>/stream1
```
