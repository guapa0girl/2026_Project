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
