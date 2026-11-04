# pig_feeding_monitor

Pose Estimation(YOLO-Pose) 기반, 돈방(pen) CCTV 영상에서 **시간대별 평균 사료 섭취 마리 수**를
추정하는 연구용 파이프라인. ETRI 2026 하계 연구연수 「이유자돈·육성돈·비육돈 사료 섭취 시간 추정」
연구의 다운스트림(탐지·추적 → 섭취 판정 → 시계열 집계) 단계를 담당한다.

## 연구 배경 및 목적

- 이유자돈 → 육성돈 → 비육돈으로 이어지는 성장 단계마다 사료 섭취 패턴이 다르고, 이유 직후
  섭식 적응에 실패하면 성장 지연·소화기 질환으로 이어진다. 성장 단계별 섭취 시간을 정량화하는 것이
  이 연구의 목표다.
- 기존 RFID 기반 전자식 급이 측정은 개체별 태그 부착이 필요하고, 단순 ROI(사료통 영역) 진입만으로
  "방문"과 "실제 섭식"을 구분하지 못한다.
- 이 프로젝트는 Pose Estimation으로 각 돼지의 **몸통 방향(목→코 벡터)** 을 계산해, 사료통 영역에
  머리를 박고 있어도 방향이 맞지 않으면(예: 사료통 옆에 누워 자는 돼지) 섭취로 세지 않도록 한다.
- 누워있는 자세를 분류하여 자고 있는 돼지의 머리 방향이 사료통을 향해도 섭취로 세지 않도록 한다.
- 최종 산출물은 **시간대(X축) × 평균 섭취 마리 수(Y축)** 그래프이며, 여러 돈방을 성장 단계별로
  비교하는 것이 장기 목표다. 개체 ID 추적은 프레임 단위 카운팅에는 필요 없지만("누가"가 아니라
  "몇 마리가" 먹는가), bout(지속 섭취) 필터링과 향후 개체별 섭취 시간량 확장을 위해 이미 계측한다.

## 프로젝트 구성: 분석 파이프라인 + 라벨링/학습 서브트리

이 저장소는 이전에 별도 폴더(`ObjectDetectionTracking_GPU`)에서 진행하던 CVAT 라벨링 →
YOLO-Pose 부트스트랩 학습(`bootstrap_v1..v3`) → 1분 샘플 영상 트래킹 실험 전체를 흡수해서
하나로 합친 것이다(그 폴더는 삭제됨; 커밋 히스토리는 `ObjectDetectionTracking_GPU_history.bundle`에
보존 -- `git clone ObjectDetectionTracking_GPU_history.bundle`로 복원 가능).

- **분석 파이프라인**(`src/`, `scripts/run_analysis.py` 등, 이 README의 주 대상)은 그 실험의
  `aggregate_feeding.py`(거리·방향·신뢰도 3조건을 **1분짜리 영상 하나에 맞춘 하드코딩 상수**로
  구현했던 것)를 일반화했다: 돈방별 설정을 `configs/*.yaml`로 분리하고, 오프라인/실시간을 같은
  코드로 처리하고, 집계 단위를 자유롭게 바꿀 수 있게 했다.
- **라벨링/학습 서브트리**(`training/`)는 옛 실험 폴더의 CVAT 부트스트랩 루프를 그대로 옮겨온
  것이다 -- 향후 라벨링을 확장하거나 재학습할 때 계속 쓴다. 자세한 내용은
  [`training/README.md`](training/README.md) 참고.
- 기본 모델은 `models/pig_pose_m_best.pt`(YOLOv8m-pose, imgsz 640, `results/{n,s,m,l}` 사이즈
  스윕에서 pose mAP50-95 최고). 실제 농장 영상(CH005) config들은 별도 스윕 결과인
  `models/pig_pose_v2_m_best.pt`(`models_v2/results/m`, 같은 CVAT 라벨 스펙이라 keypoint 순서
  동일)를 쓴다 -- `configs/farm_ch005_v2m_*.yaml`이 이 체크포인트용 config. `models/trained/`,
  `models/pig_pose_v3_best.pt`는 그 이전 학습 라운드들의 참고용 보관본. 여러 체크포인트 계보에
  대한 상세 설명(특히 keypoint 순서 검증 이력)은 [`models/README.md`](models/README.md) 참고.

## 파이프라인 개요

```
video/stream ─▶ PoseTracker (YOLO-Pose + ByteTrack)
                   │  프레임별 [bbox, 18 keypoint, track_id]
                   ▼
             feeding_classifier (4조건: 신뢰도·거리·방향·자세)
                   │  프레임별 "섭취 중인 track_id 집합"
                   ▼
             FeedingAggregator (평활화 → bout 필터 → 시간 bin 집계)
                   │  BinResult(bin_start, avg_pigs, n_samples)
                   ▼
             report (CSV + 그래프, 시간대 or 시각 x축)
```

`FeedingAggregator`는 프레임을 하나씩 받아 bin이 닫히는 순간 결과를 내보내는 증분(incremental)
구조라서, 영상 전체를 한 번에 처리하는 오프라인 스크립트와 프레임이 실시간으로 들어오는 라이브
스크립트가 완전히 같은 코드를 쓴다.

## 급이(Feeding) 판정 로직

`src/feeding_monitor/feeding_classifier.py`의 `classify()`가 4조건을 순서대로 확인해 하나라도
실패하면 그 자리에서 "섭취 아님"으로 판정한다:

| 순서 | 조건 | 내용 | 걸러내는 것 |
|---|---|---|---|
| 1 | 신뢰도 | 코·목(인덱스 12/11) keypoint confidence ≥ 0.5 | 겹친 무리에서 잘못 찍힌 keypoint |
| 2 | 거리 | 코가 사료통 중심에서 링(안쪽 0.20r ~ 바깥쪽 0.70r, CH005 기준) 안 | 뚜껑 한복판(코가 못 감), 테두리에 코만 얹고 쉬는 돼지 |
| 3 | 방향 | 목→코 벡터와 코→사료통중심 벡터 사이 각도 ≤ 60° | 사료통 옆에 있지만 머리가 딴 데를 향한 돼지 |
| 4 | 자세 (`posture.py`, CH005는 켜짐) | 옆으로 누운 자세가 아님 | 거리·방향을 다 통과한 나머지 애매한 케이스 |

조건 1-3(1~3조건, 예전 "3조건" 판정)만으로는 사료통 테두리에 코를 얹고 자는 돼지를 걸러내지
못한다 -- 코는 링 안에 있고 목→코 벡터도 우연히 중심을 향할 수 있어서, 조건 3("방향")이 원래
막으려던 "누워 자는 돼지" 케이스 중 이 서브케이스는 위치·방향 정보만으로는 구분이 안 된다. 실제로
2026-08-25 야간(IR) 영상에서 이 케이스가 카운트되는 게 관찰됐다 (아래 "각도 완화 실험" 참고).
조건 4는 탑뷰에 없는 높이 정보 대신 2D 투영에 남는 세 신호(사지 가시성 · 몸통 종횡비 · 좌우
keypoint confidence 비대칭)를 램프 함수로 합산한 점수로 눕/서를 구분한다 -- 자세한 설계 근거(특히
"근거 부족이면 배제하지 않는다"는 fail-open 원칙)는 `posture.py` 모듈 docstring 참고.

**기본값(`config.py`의 `FeedingRule` 필드 기본값)은 여전히 꺼져 있다(`posture_enabled = False`,
`ring_outer_ratio = 1.10`).** 이건 다른 돈방/카메라에 안전한 제네릭 placeholder일 뿐이고, 실제
검증된 값은 아래처럼 CH005 config들(`configs/farm_ch005_*.yaml`)에서 개별적으로 덮어쓴다. 새 돈방을
추가할 땐 이 기본값을 그대로 믿지 말고 아래 "실측 보정" 과정을 그 돈방/카메라로 반복할 것.

### 각도 완화 실험, 그리고 실제 오탐 발견/보정 (2026-08-25)

1. `direction_angle_max_deg`를 프레임 단위 flicker(같은 자세인데 keypoint jitter로 경계값 근처에서
   섭취/비섭취가 뒤집히는 것) 완화 목적으로 잠시 62°로 올렸다가, 밤/IR 영상에서 그 완화 자체가 누운
   돼지 오탐을 만드는 걸 확인하고 60°로 되돌렸다.
2. 그런데도 오탐이 남아 있다는 실제 관찰이 이어졌다: 밤 영상에서 60초 내내 거의 같은 자세로 정지해
   있는 돼지가 계속 "FEEDING"으로 표시됨. 좌표로 확인해보니 nose=(655,324), neck=(502,319),
   사료통 중심=(908.9,530.8) → 각도 32~38°(60° 이내라 통과), 거리 328px(당시 ring_outer=371px 이내라
   통과) -- 코가 사료통 테두리에 얹혀 있을 뿐 몸은 바닥에 누워 있는, 조건 4가 원래 잡으려던 바로 그
   케이스였다. 그런데 조건 4를 켜도 안 걸러졌다 -- `posture_limb_low/high`, `posture_aspect_low/high`,
   `posture_asym_low/high` 세 ramp 구간이 전부 실측과 안 맞아서(사지 가시성은 항상 ~0.99로 포화,
   종횡비/비대칭은 항상 램프 구간보다 훨씬 낮은 값) 점수가 `posture_score_max=0.5`를 구조적으로
   못 넘었다(밤 영상 372개 표본 중 0개 거부).
3. **라벨링 기준**: `scripts/calibrate_posture.py sample`이 뽑아주는 프레임은 숫자(거리/각도/종횡비
   등)를 안 보여주고 이미지만 보여준다 -- 라벨은 순전히 이미지를 보고 판단한 것이고, 실제로 이 사료통
   (원형 "TopFeeder", 중앙 기둥 + 가장자리 트인 홈통 구조) 위에서 육안으로 뚜렷이 갈리는 두 패턴이
   있었다:
   - **eating**: 몸통이 사료통 중앙 기둥 쪽으로 코를 박고, 몸은 뚜껑 밖 바닥까지 방사형으로 뻗어
     나온다 -- 여러 마리가 동시에 먹을 때는 사료통을 중심으로 바큇살(spoke)처럼 여러 마리가 대칭으로
     늘어선 모습이 된다(각 마리 머리가 중앙에, 몸이 바깥으로).
   - **lying**: 몸이 사료통 뚜껑에 전혀 걸치지 않고 완전히 바닥에 있으며, 대개 다른 돼지들과 함께
     무리 지어 누워 있는 무더기(pile)의 일부다 -- 코가 사료통 가장자리 쪽을 향해도, 몸 전체가 사료통
     구조와 분리돼 있다는 게 핵심 구분점.
   - 야간 IR이라 대비가 낮아서(사료통 자체가 밝게 반사돼 배경과 저대비) 애매한 경우(`unsure`, 6건)도
     있었다 -- 몸이 뚜껑에 살짝 걸쳤는지 애매하거나, 자세가 잘려서 안 보이는 경우.
   - 사후에 보니 이 육안 기준이 곧 **거리** 신호와 거의 그대로 대응됐다(아래 참고) -- "뚜껑까지 몸이
     닿았는가"가 결국 "코가 중심에 얼마나 가까운가"로 수치화된 셈. 종횡비가 반대 방향으로 나온 것도
     이 각도에서 보면 설명이 된다: 방사형으로 뻗어 다가서는 자세는 카메라 쪽으로 몸이 기울어
     neck-tail 투영 길이가 짧아지는 반면(→종횡비 커짐), 바닥에 완전히 뻗어 누운 자세는 위에서 볼 때
     몸통이 더 길게 펴져 보인다(→종횡비 작아짐).
   - 라벨 파일: `outputs/calibration/night_posture/labels.csv` + 그 옆 `frames/*.jpg` (34장 라벨 +
     6장 unsure 원본 이미지가 그대로 보존돼 있어, 이 판단 기준을 다시 검토하거나 다음 라운드 라벨링
     기준을 맞출 때 참고할 수 있다).
4. `scripts/calibrate_posture.py fit`으로 이 34개(eating 12 / lying 22) 라벨을 근거로 재보정했다:
   - **거리**가 압도적으로 강한 신호였다 (eating 65~186px vs lying 279~448px, **100% 분리**). 그래서
     `ring_outer_ratio`를 1.10 -> **0.70**으로 낮췄다 -- "코가 뚜껑 언저리에 있다"는 것 자체가 이
     사료통(TopFeeder) 구조에서는 오히려 쉬는 자세 쪽 신호였다(먹는 자세는 뚜껑 위로 몸을 걸쳐 트인
     구멍 쪽으로 더 깊이 들어감).
   - `direction_angle_max_deg`는 35.6°가 단일 신호로는 최적(91%)이었지만, 거리 필터를 이미 0.70으로
     좁힌 상태에서 각도까지 좁히면 오히려 넓은 각도의 진짜 먹는 사례들을 잘못 걸러내 전체 정확도가
     떨어졌다(36°: 88% vs 원래 60° 유지: 94%) -- 그래서 **60°를 그대로 유지**했다. 거리와 각도는
     서로 독립적으로 최적화하면 안 되고, 결합 정확도로 같이 봐야 한다는 교훈.
   - `aspect_ratio`(종횡비)는 애초 가설과 **방향이 반대**였다: 먹을 때가 더 크고(median 0.326), 누울 때가
     더 작았다(median 0.156, 91% 분리) -- `posture.py`의 `body_aspect_ratio` docstring에 이유 추정 기록.
     `_ramp()`에 `reverse` 옵션을 추가해 대응.
   - `limb_visibility`(사지 가시성)는 65%로 사실상 무의미해서 가중치 0으로 뺐다 -- 이 모델은 다리가
     가려져도 confidence를 거의 항상 높게 예측해서, 서 있든 누워있든 구분이 안 됐다.
   - 최종(거리+각도+자세 결합) 라벨셋 정확도 **94%(32/34)**. 남은 오차 2건은 사료통 중심에 아주
     가까운 edge case 1건, 각도 67.5°짜리 진짜 먹는 outlier 1건.
5. 재검증 결과: 밤 전체 영상(59.4분) geo mean **0.94 -> 0.44**로 감소(그 정지 돼지 하나가 카운트를
   상당 부분 부풀리고 있었음), 낮 전체 영상(20.9분) geo mean 1.09 -> 0.78.

n=34는 작은 표본(RF의 ~940장에 비하면)이라 이 값들을 최종으로 여기지 말 것 -- 특히 낮 영상은 별도
라벨링 없이 밤 값을 그대로 재사용했다. `scripts/calibrate_posture.py sample/fit`으로 언제든 표본을
늘려 재보정할 수 있다.

flicker(경계값 근처 jitter로 인한 프레임 단위 뒤집힘) 자체를 줄이고 싶으면 각도를 널널하게 하는 대신
`FeedingRule.min_bout_sec`(현재 0.0, 즉 비활성)를 키워서 트랙별 N초 연속 섭취만 카운트하는
hysteresis를 쓰는 쪽이 안전하다(`aggregator.py`의 `_bouted_count`) -- 진짜 오탐은 여전히 걸러내면서
flicker만 완화한다.

### 코드 정리 메모 (2026-08-25)

조건 4를 추가하는 편집이 원래 `feeding_classifier.py`가 아니라 `posture_classifier.py`(원래
RandomForest 비교용 `MLPostureModel` 래퍼가 있던 파일)에 잘못 저장돼서, `MLPostureModel`이 통째로
사라지고 `compare_classifiers.py`/`render_comparison_video.py`/`count_full_video.py`/
`validate_counts.py`가 전부 import 에러로 깨지는 상태였다. 4조건 로직은 `feeding_classifier.py`로
옮기고 `posture_classifier.py`는 `MLPostureModel`로 복원했다 -- 두 파일 다 원래 목적대로 정리된
상태. 또한 `config.py`의 `FeedingRule`에 `posture_*` 필드가 아예 없어서(위 문단의 값들) 조건 4가
실행되면 무조건 `AttributeError`가 났는데, 이번에 다 추가했다(테스트 17개 통과, 실제 영상 15초
구간으로 크래시 없음 확인).

사료통은 top-down 시점의 원형 급이기를 전제한다. `feeder_roi.detect_feeder_circle()`(빨간 뚜껑 색
검출)은 **`scripts/calibrate_feeder.py`에서 사람이 좌표를 잡을 때 보조용으로만** 쓰이고, 실제
탐지·집계 파이프라인(`feeding_classifier.py`)은 항상 `configs/*.yaml`에 고정된 좌표를 쓴다 -- 즉
야간 IR 조명에서 색이 사라져도 실시간 판정 자체는 영향받지 않는다. 다만 카메라가 흔들리거나 청소 중
건드려져 ROI가 실제 사료통과 어긋나는 것은 별개 위험이라, `run_realtime.py --calib-snapshot-every N`으로
주기적으로 ROI 오버레이 스냅샷을 저장해 사후 확인할 수 있게 해뒀다. **카메라 설치 후 주간에 한 번
보정하고 그 좌표를 고정하는 것**을 권장 워크플로로 삼는다.

선택적으로 `feeder_capacity`(사료통의 물리적 동시 급이 공간 수 -- 돈방 전체 마릿수인
`expected_pig_count`와는 다른 값)를 설정하면, 관측된 count가 이 값을 넘을 때 겹침/오검출 가능성을
경고하고 화면에 `Count : N / capacity`로 표시한다.

## 실시간 모니터링: Count 오버레이 + 초당 로그

`scripts/run_realtime.py`로 실시간(RTSP/웹캠) 영상을 처리할 때:

- **화면**: 미리보기 창에 매 프레임 `Count : N`(`feeder_capacity` 설정 시 `Count : N / capacity`)이
  크게 오버레이된다(`q` 키로 종료).
- **터미널**: bin이 닫힐 때마다(설정에서 `aggregation.bin_size: 1s`면 매초) 같은 값을
  `YYYY-MM-DD HH:MM:SS  Count : N  (n=samples)` 형식으로 한 줄씩 출력한다. `n=`은 그 bin에 실제로
  몇 개의 프레임 샘플이 들어갔는지 -- 처리 속도가 흔들리는 구간(네트워크 지연 등)을 사후에 가려낼
  수 있는 커버리지 신호다.
- **텍스트 로그**: 같은 줄이 `outputs/realtime/..._live_log.txt`에 계속 append된다(파일은 매 줄마다
  flush되므로 `tail -f`로 실시간으로도 볼 수 있다). 이 로그가 "하루 중 언제 가장 활발히 섭취하는지"
  분석의 원자료다.
- **`Count : N`은 반올림이 아니라 올림(ceil)이다** (`live_log.py`, 2026-08-25) -- `avg_pigs`는 bin/평활화
  구간에 걸친 평균이라 소수가 나오는데(예: 0.4마리), "0.5마리"는 실재하지 않으므로 반올림으로 0에
  뭉개버리지 않고 조금이라도 섭취가 있었으면 1 이상으로 올린다. CSV/그래프에 남는 원본 `avg_pigs`,
  `pig_seconds` 같은 연속 통계치는 그대로 소수로 유지된다 -- 정수 "Count" 표시부만 이렇게 바뀐다.
- **타임스탬프는 실제 벽시계(`datetime.now()`) 기준이다** -- 프레임 수를 가정 fps로 나눈 값이 아니다.
  RTSP는 실제 처리 fps가 네트워크/GPU 부하에 따라 흔들리므로, 프레임 수 기반 시간은 하루 단위로
  누적되면 몇 시간 단위로 어긋날 수 있다(예: 실제 12fps인데 15로 가정하면 24시간 후 약 4.8시간
  어긋남). 이 연구의 핵심 결과물이 "몇 시에 가장 활발한가"이기 때문에, `--fps`는 오직 평활화 창
  크기를 정하는 데만 쓰이고 로그/그래프 타임스탬프에는 전혀 관여하지 않는다.

```bash
python scripts/run_realtime.py --config configs/pen01_realtime.yaml --source rtsp://<camera-ip>/stream1
# 화면 없이 서버에서 돌릴 때: --no-show
# 사료통 ROI가 안 어긋났는지 주기적으로 스냅샷 저장: --calib-snapshot-every 1800  (약 2분마다, 15fps 가정 시)
```

`configs/pen01_realtime.yaml`은 `pen01.yaml`과 같지만 `aggregation.bin_size: 1s`로 되어 있어
정확히 초당 1줄이 기록된다(5초/1분 단위 트렌드 그래프가 필요하면 `pen01.yaml`처럼 더 큰 bin을 쓰면
됨 -- 로그 주기는 순전히 `bin_size`로 결정된다).

로그가 쌓이는 동안(또는 다 끝난 뒤) 언제든 시간대별 패턴 그래프를 뽑을 수 있다:

```bash
python scripts/plot_live_log.py --log outputs/realtime/pen01_realtime__..._live_log.txt
# 기본 --mode hourly: 로그에 담긴 모든 날짜를 시간대(0~23시)별로 n_samples 가중평균 -> "하루 패턴" 막대그래프
#   기록이 아예 없는 시간대는 0이 아니라 빈 칸(회색 해칭)으로 남고, 그 아래 커버리지(시간대별 로그 줄 수)
#   패널이 같이 그려진다 -- 카메라/네트워크가 끊긴 시간대가 "그 시간엔 안 먹는다"로 둔갑하지 않도록.
# --mode timeline: 접지 않고 시간 순서 그대로 (짧은 세션 확인용)
```

카메라가 아직 없다면, `run_analysis.py --show --live-log`로 녹화 영상에 대해 똑같은 화면/로그
동작을 미리 검증해볼 수 있다(아래 사용법 참고). 오프라인 영상은 컨테이너 메타데이터의 fps가 실제
값이므로(라이브 스트림과 달리 추측이 아님), `aggregation.video_start_time`을 설정해두면 그 fps 기준
경과시간으로 정확한 절대 타임스탬프를 계산한다 -- 라이브 경로만 fps를 신뢰할 수 없는 것이지, 녹화
파일 경로는 원래도 문제없었다.

## 두 번째 판정 방식: RandomForest posture 분류기 (비교용)

별도 프로젝트(`auto_labeling_model`)에서 같은 문제(포즈 키포인트로 "지금 먹고 있나" 판정)를 다른
방식으로 푼 것을 가져와 이 프로젝트의 기하학적 3조건 판정과 나란히 비교할 수 있게 포팅했다:

- `posture_features.py` -- 코가 급이 구역(다각형) 안/근처인지 + 머리를 얼마나 숙였는지(각도) +
  등을 낮췄는지 + 목 높이 + keypoint 신뢰도, 6개 특징을 뽑는다. 같은 카메라(CH005)에서 사람이
  손으로 라벨링한 ~940장으로 학습된 `models/posture_random_forest.joblib`을 그대로 재사용한다.
- 이 ML 경로는 원형 ROI가 아니라 다각형 급이구역(`configs/zones/feeder_zones_ch005.yaml`)을 쓴다 --
  학습 당시 특징 분포와 어긋나지 않게 원본 방식 그대로 유지했다. 이 프로젝트 자체의 판정 로직
  (`feeding_classifier.py`)은 방침대로 원형 ROI만 쓴다 -- 이 비교에만 쓰이는 별도 경로.

```bash
# 기하학 판정은 이미 저장돼 있는 *_tracks.csv의 feeding 컬럼을 그대로 씀 -- 재계산 없음
python scripts/run_analysis.py --config configs/farm_ch005_20260610.yaml \
  --source "data/raw/[CH005] 202606100923_44.avi" --fps 200 --max-frames 2000 --no-video

python scripts/compare_classifiers.py --tracks-csv outputs/<run-name>_tracks.csv \
  --zones configs/zones/feeder_zones_ch005.yaml --fps 200
```

이건 두 방법이 서로 얼마나 일치하는지만 보여준다(10초 샘플 기준 인스턴스 단위 69% 일치) -- 어느
쪽이 실제로 맞는지는 알려주지 않는다. 진짜 비교는 `validate_counts.py sample`에 `--ml-model`/`--zones`를
같이 주면 두 방법의 예측을 한 CSV에 나란히 기록해주고, `score`가 사람이 채운 정답 대비 각각의
MAE/precision/recall을 따로 계산한다 -- 그 숫자로 고르면 된다.

RandomForest 쪽이 geo보다 과소검출하는 경향이 관찰돼서(20분 전체 영상 기준 mean 0.56 vs geo 1.09),
`MLPostureModel.predict`/`predict_batch`에 `threshold` 인자를 추가했다(`posture_classifier.py`) --
기본 0.5(sklearn `.predict()`와 동일)보다 낮추면 "섭취" 쪽으로 더 관대하게 판정한다. `compare_
classifiers.py`/`render_comparison_video.py`/`count_full_video.py`/`validate_counts.py sample` 모두
`--rf-threshold`로 노출돼 있다(예: `--rf-threshold 0.4`). 근본 원인(학습 클립과 이 클립 간 특징
분포 차이인지, 임계값 자체 문제인지)은 아직 조사 전이라 기본값은 그대로 0.5로 뒀다.

## 비교 영상 화질/용량

`render_comparison_video.py`는 이 환경에서 mp4v 코덱으로만 쓸 수 있다 -- H.264(avc1)는 openh264
DLL이 없어서 OpenCV `VideoWriter`가 못 연다. mp4v는 압축 효율이 낮아서, 원본 해상도(패널당 960px)로
30초 분량을 뽑으면 93MB가 나온다(파일 전달 한도 30MB를 훌쩍 넘음). `--panel-width`로 해상도를
낮추는 게(예: 480px -> 25.5MB) 지금 쓰는 유일한 완화책이고, 그만큼 화질이 흐려진다.

시스템에 ffmpeg 바이너리 자체가 없어서(`ffmpeg` PATH에 없음) CLI로 우회할 수도 없다. 화질을
근본적으로 개선하려면 `pip install imageio-ffmpeg`(정적 ffmpeg 바이너리를 같이 설치)로 libx264
인코딩 경로를 만들어야 한다 -- 아직 시도 안 함, 필요해지면 `render_comparison_video.py`의
`cv2.VideoWriter`를 이 바이너리를 호출하는 subprocess 파이프로 바꿔야 한다.

### 긴 영상 구간 렌더링 (2026-08-25)

10분 이상 분량을 자연 속도로 뽑으려다가(`--playback-fps`/`--frame-stride`로 배속 1x 유지, 더 긴
구간을 커버) 이 프로젝트 AVI 파일들 특유의 두 가지 조기 종료 버그를 마주쳤다:

- **스트리밍 리더가 ~188초에서 조용히 멈춤** (`count_full_video.py` docstring에 이미 기록된 것과
  동일 증상) -- `PoseTracker.track()`이 기본으로 쓰는 ultralytics `model.track(stream=True)`가
  이 파일 앞부분에서 에러 없이 그냥 멈춘다. `--start-frame`을 1 이상으로 주면(0이 아니면) 자동으로
  cv2 수동 seek 경로로 전환되는데, 이 경로는 안 멈춘다 -- 처음부터 렌더링할 때도 `--start-frame 1`
  정도로 우회 가능.
- **중간 지점에서 seek 자체가 실패**: 밤 영상 40분 지점부터 시작해도 34,871프레임(약 43분 지점)에서
  다시 멈췄다 -- 시작 seek이 아니라 순차 읽기 도중 `cap.read()`가 실패하는 경우. `PoseTracker.track()`의
  수동 경로에 재시도를 추가했다: 시작 seek도, 스트림 중간의 읽기 실패도 최대 30프레임 앞으로
  nudge하면서 재시도 -- 파일의 국지적 손상 구간 하나가 렌더링 전체를 끊지 않도록.

`--start-frame`은 긴 녹화의 임의 지점(예: 40~50분 구간)부터 렌더링할 때도 쓴다 -- 전체를 처음부터
디코딩+추론할 필요 없이 그 지점으로 바로 seek한다.

## 프로젝트 구조

```
configs/                 돈방/카메라별 설정 (모델 경로, 사료통 ROI, 판정 임계값, 집계 단위)
src/feeding_monitor/
  config.py               YAML → dataclass 설정 로더
  pose_tracker.py          YOLO-Pose + ByteTrack 래퍼 (영상 파일/스트림 공용, start_frame로 긴 영상 중간부터 시작 + seek/mid-stream 읽기 실패 재시도)
  feeder_roi.py            원형 사료통 자동 검출 + 보정 시각화
  feeding_classifier.py    4조건 급이 판정 (신뢰도·거리·방향·자세)
  posture.py               조건 4: 옆으로 누운 자세 판별 (종횡비/좌우 비대칭 -- CH005는 34장 라벨로 보정, 다른 돈방은 기본 꺼짐)
  aggregator.py            평활화 + bout 필터 + 시간 bin 집계 (증분 처리)
  live_log.py              초당 "Count : N (n=samples)" 텍스트 로그 작성/파싱 -- 실시간은 항상 실제 벽시계로 타임스탬프
  daily_pattern.py          시간대(0~23h) 접기: n_samples 가중평균 + 결측 NaN 처리(hourly_pattern), 날짜별 정규화(daily_shape)
  posture_features.py       (비교용) RandomForest 판정용 6개 특징 추출 -- 다각형 급이구역 기준
  posture_classifier.py     (비교용) 사전학습 RandomForest 래퍼, feeding_classifier.classify()와 같은 인터페이스
  report.py                CSV/그래프 저장
  pipeline.py              위 모듈을 엮는 오케스트레이션 (화면 미리보기, 급이기 용량 초과 경고 포함)
scripts/
  run_analysis.py          오프라인: 저장된 영상 파일 전체 분석 (--show/--live-log로 실시간 동작 검증)
  run_realtime.py           실시간: RTSP/웹캠 스트림, Count 오버레이 + 초당 로그(실제 벽시계) + ROI 스냅샷
  plot_live_log.py          live_log.txt -> 시간대별(하루 패턴, 결측 구간 표시) 또는 시계열 그래프
  compare_stages.py         여러 live_log.txt(성장 단계별)를 하루 패턴으로 접어서 비교 (--metric shape/per-pig/raw)
  calibrate_feeder.py       사료통 ROI 보정 도구
  calibrate_posture.py      (sample/fit) 조건 3/4 임계값을 사람 라벨 소표본으로 보정 -- validate_counts.py와 같은 2단계 구조, fit --labels는 여러 라운드의 labels.csv를 한 번에 합쳐서 볼 수 있음
  sensitivity_check.py      임계값(각도/링 반경) 그리드 스윕 -- 결과가 임계값에 강건한지 확인
  compare_classifiers.py    (비교용) 기하학 vs RandomForest 판정 일치도 (같은 tracks.csv 기준, --rf-threshold로 RF 완화 가능)
  validate_counts.py        사람이 라벨링한 정답 개수 대비 방법별 MAE/precision/recall 채점 (sample --ml-model 옵션으로 두 방법 동시 기록 / score)
  render_comparison_video.py (비교용) geo·RF 판정을 좌우 분할 화면에 같이 그린 주석 영상 (--playback-fps/--frame-stride로 배속 조절, --panel-width로 용량 조절, --start-frame으로 긴 영상 중간부터 시작)
  count_full_video.py       문제 있는 AVI(스트리밍 리더가 중간에 멈춤)에서도 안전한, seek 샘플링 기반 전체 영상 초당 카운트 (geo+RF 동시) -- `<out-prefix>_1s.csv`(1초 단위 원본) + `<out-prefix>.csv/.png`(`--bin-sec` 단위로 `--bin-agg mean`(기본, "평균 섭취 마리 수" 지표와 일치)/`max` 집계)
configs/zones/           (비교용) RandomForest 판정용 다각형 급이구역
models/                  배포용 체크포인트 (pig_pose_m_best.pt 등) + posture_random_forest.joblib -- 자세한 계보는 models/README.md
training/                CVAT 라벨링 부트스트랩 + 재학습 서브트리 (training/README.md 참고)
data/raw/                입력 영상 (1m_sample_video.mp4, CH005_clip_20s.mp4, 실제 농장 영상 [CH005] 202606100923_44.avi 포함)
data/processed/          파이프라인 중간 산출물
outputs/                 실행별 CSV·그래프·주석 영상
  pilot_1m_sample_video/  1분 샘플 영상 파일럿의 결과물(그래프·보정 이미지·로그) 보존본
  exclusion_events.csv    수동으로 채우는 제외 구간 기록 템플릿(이동/재군성/처치일 등, 아래 참고)
tests/                   급이 판정 로직 단위 테스트 (모델/영상 불필요)
venv/                    GPU torch(cu128)+ultralytics 설치된 Python 3.13 가상환경 (venv\Scripts\python.exe로 직접 실행)
ObjectDetectionTracking_GPU_history.bundle  삭제된 옛 저장소의 git 히스토리 백업
```

## 사용법

`venv/`에 GPU torch(cu128) + ultralytics가 이미 설치되어 있다. 새로 설치할 필요 없이 그
인터프리터를 직접 쓰면 된다(아래 `python`은 `venv\Scripts\python.exe`를 뜻함). 다른 환경에서
새로 만들 경우에만 `pip install -r requirements.txt` (+ torch는 별도 설치, 상단 주석 참고).

```bash
# 1) 사료통 ROI 보정 (돈방/카메라가 바뀌면 반드시 재보정; 1분 샘플 영상으로 시연)
python scripts/calibrate_feeder.py --source data/raw/1m_sample_video.mp4 --at-sec 15
# outputs/feeder_calib.png 확인 후, 출력된 YAML을 configs/pen01.yaml의 feeder: 에 반영

# 2) 오프라인 분석 (녹화 영상)
python scripts/run_analysis.py --config configs/pen01.yaml --source data/raw/1m_sample_video.mp4

# 2-1) 화면/초당 로그 동작을 카메라 없이 미리 검증
python scripts/run_analysis.py --config configs/pen01_realtime.yaml --source data/raw/1m_sample_video.mp4 --show --live-log

# 3) 실시간 분석 (RTSP) -- 화면에 Count : N, 터미널/파일에 초당 로그
python scripts/run_realtime.py --config configs/pen01_realtime.yaml --source rtsp://<camera-ip>/stream1

# 3-1) 로그로부터 시간대별(하루 패턴) 그래프
python scripts/plot_live_log.py --log outputs/realtime/pen01_realtime__..._live_log.txt

# 4) 성장 단계별 비교 (live_log.txt 기반, 하루 패턴 모양으로 정규화 -- 마릿수/급이기 크기 차이 상쇄)
python scripts/compare_stages.py \
  --run "이유자돈=outputs/realtime/pen01_live_log.txt" \
  --run "육성돈=outputs/realtime/pen02_live_log.txt" \
  --run "비육돈=outputs/realtime/pen03_live_log.txt"

# 5) 임계값 민감도 확인 (단계 비교 전에, 프레임 CSV만 있으면 재탐지 없이 빠르게)
python scripts/sensitivity_check.py --tracks-csv outputs/pen01__clip_tracks.csv --config configs/pen01.yaml

# 6) 사람이 직접 센 개수와 비교 검증
python scripts/validate_counts.py sample --config configs/pen01.yaml --source data/raw/clip.mp4 --n 20 --out-dir outputs/validation/pen01
# outputs/validation/pen01/frames/*.jpg 를 보면서 labels.csv의 human_count 열을 채운 뒤:
python scripts/validate_counts.py score --labels outputs/validation/pen01/labels.csv
```

단위 테스트(모델·영상 불필요):

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## 연구 방법론 관련 유의사항

코드 리뷰를 받고 나서 실제 데이터 수집을 시작하기 전에 확정해둘 것들:

- **모델 체크포인트 동결**: 본 연구용 데이터 수집을 시작하기 전에 `configs/*.yaml`의
  `model.weights`를 하나로 고정하고(`models/README.md`에 기록), 라벨링을 확장해 재학습한 새
  체크포인트(`train6_best.pt` 등)로 절대 조용히 바꿔치지 않는다. 바꿔야 한다면 그 시점 이전/이후
  데이터를 구분해서 다루거나, 이전 영상을 새 체크포인트로 전부 재처리한다.
- **단계 간 파라미터**: 가능하면 `direction_angle_max_deg`/`ring_outer_ratio`/`min_bout_sec`를 세
  단계에 동일하게 적용한다. 체구 차이 때문에 부득이 다르게 튜닝했다면 그 사실과 이유를 명시하고,
  `scripts/sensitivity_check.py`로 각 단계에서 결과(특히 단계 간 순위)가 임계값에 얼마나 민감한지
  확인해둔다. 세 단계 급이기 형상이 실제로 원형 top-down 전제를 만족하는지는 코드로 확인할 수 없는
  부분이라 사진으로 직접 확인이 필요하다.
- **`daily_shape()`의 한계**: `compare_stages.py`의 기본 지표(`--metric shape`)는 각 날짜의 24시간
  프로파일을 그날 총합으로 나눠 정규화하는데, 이때 "그 시간대에 기록이 아예 없음"과 "그 시간대에
  0마리"를 날짜 단위로는 구분하지 못한다(시간대별 결측은 `hourly_pattern`/`plot_live_log.py`가
  NaN으로 정확히 구분하지만, 날짜 단위 모양 정규화 안에서는 결측 시간대가 0으로 들어간 뒤 정규화된다).
  하루 중 일부만 결측인 로그로 단계를 비교할 때는 먼저 `plot_live_log.py`로 그 날짜의 커버리지를
  확인할 것.
- **제외 구간 기록**: `outputs/exclusion_events.csv`에 `start_time,end_time,pen_id,reason,note` 형식으로
  이동/재군성 직후, 백신·처치일처럼 분석에서 빼야 할 구간을 그때그때 적어둔다(자동화된 도구는 아니고,
  사후에 왜 특정 구간을 뺐는지 알 수 있게 하는 수동 기록용 템플릿).
- **화질 저하**: 렌즈 오염/결로로 화질이 나빠지면 탐지가 조용히 틀려질 뿐 에러가 나지 않는다.
  `--calib-snapshot-every`로 저장되는 스냅샷을 일일 점검에 같이 활용할 것(선명도 지표 자동 로깅은
  아직 구현하지 않았다 -- 필요하면 추가 가능).

## 향후 계획

- **대량 라벨링 확장**: `training/`의 CVAT 부트스트랩 루프로 1,000+ 프레임까지 라벨링 확장 →
  재학습 → `models/`에 새 체크포인트(`train6_best.pt` 등) 추가하고 `configs/*.yaml` 갱신.
- **지속시간(bout) 필터 튜닝**: `feeding_rule.min_bout_sec`을 실제 데이터로 튜닝해, 스쳐 지나가며
  코가 링에 잠깐 걸치는 프레임을 섭취로 오카운트하지 않도록 한다(구조는 이미 구현됨).
- **조건 3/4 라벨 표본 확장**: 2026-08-25 `scripts/calibrate_posture.py`로 CH005 밤 영상 34장만
  라벨링해서 재보정했다(94% 정확도) -- 낮 영상은 별도 검증 안 됨, 다른 돈방/성장단계는 전혀 없음.
  RF 경로(~940장)만큼 표본을 늘리고, 새 돈방을 추가할 때마다 그 카메라로 `sample`/`fit`을 반복할 것.
- **개체 ID 기반 섭취 시간량**: 현재는 "몇 마리가 먹는가"만 집계하지만, track_id별 feeding bout
  길이를 누적하면 개체별 총 섭취 시간 → 성장 데이터 연계로 확장 가능(`aggregator.py`의 track_id
  단위 상태를 그대로 활용).
- **다중 돈방/장기 운영**: `configs/`에 돈방별 YAML을 추가하고 `run_realtime.py`를 상시 구동해
  성장 단계별 섭취 패턴을 장기간 비교.
- **일주기(daily) 패턴 검증**: `live_log.txt`를 여러 날 누적해 `plot_live_log.py --mode hourly`로
  뽑은 실측 패턴을, 기존 문헌에 보고된 돼지 급이 활동 패턴과 대조 -- 비침습적 방식으로 이 방법의
  타당성을 검증하는 것이 목표.
