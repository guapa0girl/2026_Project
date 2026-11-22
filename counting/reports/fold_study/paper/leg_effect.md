# RF+Geo → RF+Geo+Leg: 무엇이 바뀌었나

평가와 같은 학습 · 추론 · 짝짓기를 다시 돌려 개체별 특징 값과 Gini 중요도를 저장함 (`scripts/analyze_leg_effect.py`).

## 재현 확인 (test(000), visiting → other, fold 평균 마릿수)

| 분할 | 방법 | 이번 계산 | fold_metrics.csv |
|---|---|---|---|
| Random 5-fold | geo | 20.0 | 20.0 |
| Random 5-fold | rf | 2.8 | 2.8 |
| Random 5-fold | rf_geo | 4.6 | 4.6 |
| Random 5-fold | rf_geo_leg | 3.4 | 3.4 |
| Time 5-fold | geo | 20.6 | 20.6 |
| Time 5-fold | rf | 2.2 | 2.2 |
| Time 5-fold | rf_geo | 4.4 | 4.4 |
| Time 5-fold | rf_geo_leg | 2.0 | 2.0 |

## Gini 중요도 (fold 10개 평균)

| 특징 | 그룹 | RF | RF+Geo | RF+Geo+Leg |
|---|---|---|---|---|
| nose_in_zone | RF | 0.513 | 0.256 | 0.172 |
| nose_zone_dist_norm | RF | 0.307 | 0.173 | 0.211 |
| head_dip_deg | RF | 0.065 | 0.018 | 0.002 |
| spine_dip_deg | RF | 0.057 | 0.015 | 0.003 |
| neck_rel_height | RF | 0.049 | 0.016 | 0.002 |
| key_kpt_conf | RF | 0.010 | 0.003 | 0.004 |
| geo_nose_dist_over_r | Geo | – | 0.328 | 0.270 |
| geo_head_angle_deg | Geo | – | 0.033 | 0.007 |
| geo_limb_visibility | Geo | – | 0.000 | 0.000 |
| geo_aspect_ratio | Geo | – | 0.158 | 0.055 |
| geo_lateral_asymmetry | Geo | – | 0.001 | 0.001 |
| leg_extension | Leg | – | – | 0.034 |
| leg_angle_deg | Leg | – | – | 0.098 |
| paw_offset | Leg | – | – | 0.092 |
| leg_side_bias | Leg | – | – | 0.045 |
| body_bend_deg | Leg | – | – | 0.006 |

그룹 합계: rf: RF 1.000; rf_geo: Geo 0.520, RF 0.480; rf_geo_leg: Geo 0.332, Leg 0.275, RF 0.393

신뢰도 기반 누움 신호 3개 합계: RF+Geo 0.159 → RF+Geo+Leg 0.055

## 사람 visiting 개체 중 두 방법의 판정이 갈린 경우 (fold 10개 합)

- RF+Geo는 other, RF+Geo+Leg는 visiting(고침): 18개
- RF+Geo는 visiting, RF+Geo+Leg는 other(망침): 0개
- 둘 다 visiting: 1211개

### 특징 값 비교 (중앙값)

| 특징 | 학습(사람 관절) visiting | test 둘 다 맞힘 | test RF+Geo만 틀림 |
|---|---|---|---|
| geo_limb_visibility | 1.000 | 0.958 | 0.959 |
| geo_aspect_ratio | 0.333 | 0.306 | 0.300 |
| geo_lateral_asymmetry | 0.000 | 0.004 | 0.007 |
| geo_nose_dist_over_r | 0.451 | 0.463 | 0.977 |
| nose_zone_dist_norm | 0.324 | 0.285 | 0.534 |
| leg_angle_deg | 10.348 | 8.445 | 18.733 |
| paw_offset | 0.128 | 0.111 | 0.098 |

학습은 사람 관절이라 보이는 관절의 신뢰도가 모두 1.0이고, test는 포즈 모델이 낸 신뢰도임.

## 학습 데이터(사람 관절)의 특징 분포 (중앙값)

| 특징 | visiting | other 전체 | 사료통 근처 other (거리 ÷ r < 1.2) | test에서 고친 visiting |
|---|---|---|---|---|
| geo_nose_dist_over_r | 0.451 | 1.863 | 0.812 | 0.977 |
| geo_aspect_ratio | 0.333 | 0.188 | 0.138 | 0.300 |
| leg_angle_deg | 10.348 | 65.540 | 73.779 | 18.733 |
| paw_offset | 0.128 | 0.418 | 0.507 | 0.098 |
| leg_side_bias | 0.000 | 1.000 | 1.000 | 0.000 |
| leg_extension | 0.197 | 0.476 | 0.531 | 0.182 |

## test(000) 코 위치별 판정 (fold 10개 합, 사람 개체와 짝지어진 검출)

| 구간 | 개체 수 | 지표 | RF+Geo | RF+Geo+Leg |
|---|---|---|---|---|
| visiting, 코가 사료통 안쪽 (거리 ÷ r < 0.85) | 1115 | visiting으로 맞힌 비율 | 1.000 | 1.000 |
| visiting, 코가 사료통 가장자리 (0.85~1.1) | 141 | visiting으로 맞힌 비율 | 0.681 | 0.809 |
| other, 코가 사료통 가장자리 (0.85~1.1) | 71 | visiting으로 잘못 센 비율 | 0.155 | 0.183 |

고친 18건의 분할별 수: k5 12, rk5 6 (fold당 Random 5-fold 1.2, Time 5-fold 2.4마리).
고친 개체의 visiting 확률 중앙값: RF+Geo 0.45 → RF+Geo+Leg 0.56 (판정 기준 0.5).

