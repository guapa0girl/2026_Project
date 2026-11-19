# Paper metrics — Random 5-fold vs Time 5-fold, 5 methods

fold 5개 평균 ± 표본 표준편차(ddof=1). test = 000 86장(visiting 133마리), val = 그 fold의 valid 사진(001+002). '–' = 값 없음. 지표 정의와 계산 방법은 `counting/scripts/paper_tables.py` 맨 위 설명.

## Table 1. Counting performance — test(000)

| Split | Method | MAE | RMSE | ME | Acc (%) | Acc±1 (%) | TCE (%) | 10-s MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Random 5-fold | Geo | 0.293 ± 0.028 | 0.562 ± 0.025 | -0.288 ± 0.025 | 71.9 ± 2.8 | 98.8 ± 0.0 | -18.6 ± 1.6 | 0.226 ± 0.027 | 0.964 ± 0.034 | 0.785 ± 0.041 | 0.865 ± 0.038 |
| Random 5-fold | RF | 0.114 ± 0.015 | 0.343 ± 0.032 | -0.035 ± 0.043 | 88.8 ± 1.3 | 99.8 ± 0.5 | -2.3 ± 2.8 | 0.075 ± 0.017 | 0.936 ± 0.035 | 0.914 ± 0.037 | 0.925 ± 0.033 |
| Random 5-fold | RF+Geo | 0.133 ± 0.033 | 0.362 ± 0.047 | -0.067 ± 0.032 | 86.7 ± 3.3 | 100.0 ± 0.0 | -4.4 ± 2.1 | 0.083 ± 0.019 | 0.941 ± 0.037 | 0.901 ± 0.053 | 0.920 ± 0.045 |
| Random 5-fold | RF+Geo+Leg † | 0.121 ± 0.033 | 0.345 ± 0.051 | -0.051 ± 0.037 | 87.9 ± 3.3 | 100.0 ± 0.0 | -3.3 ± 2.4 | 0.083 ± 0.010 | 0.940 ± 0.036 | 0.910 ± 0.053 | 0.925 ± 0.044 |
| Random 5-fold | YOLO | 0.300 ± 0.137 | 0.545 ± 0.125 | -0.063 ± 0.210 | 70.5 ± 13.6 | 99.5 ± 0.6 | -4.1 ± 13.6 | 0.198 ± 0.160 | 0.899 ± 0.066 | 0.857 ± 0.085 | 0.874 ± 0.053 |
| Time 5-fold | Geo | 0.293 ± 0.022 | 0.562 ± 0.020 | -0.274 ± 0.021 | 71.9 ± 2.2 | 98.8 ± 0.0 | -17.7 ± 1.4 | 0.218 ± 0.016 | 0.971 ± 0.026 | 0.798 ± 0.014 | 0.876 ± 0.018 |
| Time 5-fold | RF | 0.116 ± 0.032 | 0.338 ± 0.046 | -0.019 ± 0.018 | 88.4 ± 3.2 | 100.0 ± 0.0 | -1.2 ± 1.1 | 0.064 ± 0.011 | 0.948 ± 0.024 | 0.937 ± 0.032 | 0.942 ± 0.028 |
| Time 5-fold | RF+Geo | 0.126 ± 0.019 | 0.353 ± 0.028 | -0.051 ± 0.024 | 87.4 ± 1.9 | 100.0 ± 0.0 | -3.3 ± 1.6 | 0.076 ± 0.011 | 0.952 ± 0.024 | 0.920 ± 0.027 | 0.936 ± 0.025 |
| Time 5-fold | RF+Geo+Leg † | 0.109 ± 0.021 | 0.329 ± 0.032 | -0.021 ± 0.022 | 89.1 ± 2.1 | 100.0 ± 0.0 | -1.4 ± 1.4 | 0.066 ± 0.008 | 0.951 ± 0.024 | 0.938 ± 0.029 | 0.945 ± 0.026 |
| Time 5-fold | YOLO | 0.223 ± 0.048 | 0.475 ± 0.059 | -0.065 ± 0.024 | 77.9 ± 4.4 | 99.8 ± 0.5 | -4.2 ± 1.6 | 0.111 ± 0.042 | 0.917 ± 0.028 | 0.878 ± 0.019 | 0.897 ± 0.022 |

† RF best (5-fold 평균 val MAE 최소). ME = 평균(예측 − 정답), TCE = 총 마릿수 오차 / 정답 총 마릿수. Precision · Recall · F1 = visiting 개체 단위(사람 개체와 짝짓기: Geo · RF 계열 = 관절 평균 거리 ÷ 사람 박스 대각선 ≤ 0.25, YOLO = 박스 IoU ≥ 0.5).

## Table 2. Counting performance — Validation

| Split | Method | MAE | RMSE | ME | Acc (%) | Acc±1 (%) | TCE (%) | 10-s MAE | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Random 5-fold | Geo | 0.566 ± 0.045 | 0.883 ± 0.079 | -0.180 ± 0.128 | 53.8 ± 3.5 | 90.1 ± 3.8 | -11.3 ± 8.2 | – | 0.790 ± 0.042 | 0.700 ± 0.063 | 0.741 ± 0.044 |
| Random 5-fold | RF | 0.389 ± 0.136 | 0.696 ± 0.156 | +0.221 ± 0.133 | 66.8 ± 9.3 | 94.3 ± 5.1 | +13.1 ± 8.3 | – | 0.821 ± 0.033 | 0.928 ± 0.065 | 0.870 ± 0.038 |
| Random 5-fold | RF+Geo | 0.259 ± 0.139 | 0.524 ± 0.185 | +0.082 ± 0.097 | 76.2 ± 11.4 | 97.9 ± 3.5 | +4.5 ± 5.2 | – | 0.897 ± 0.043 | 0.937 ± 0.046 | 0.916 ± 0.038 |
| Random 5-fold | RF+Geo+Leg † | 0.209 ± 0.133 | 0.489 ± 0.175 | +0.009 ± 0.117 | 81.8 ± 11.0 | 97.3 ± 3.3 | -0.1 ± 7.0 | – | 0.932 ± 0.039 | 0.930 ± 0.054 | 0.930 ± 0.035 |
| Random 5-fold | YOLO | 0.252 ± 0.128 | 0.537 ± 0.152 | -0.098 ± 0.092 | 77.6 ± 10.9 | 97.2 ± 3.3 | -6.2 ± 5.7 | – | – | – | – |
| Time 5-fold | Geo | 0.575 ± 0.148 | 0.871 ± 0.157 | -0.212 ± 0.151 | 52.3 ± 9.7 | 90.8 ± 9.0 | -13.2 ± 8.2 | – | 0.769 ± 0.179 | 0.665 ± 0.147 | 0.712 ± 0.157 |
| Time 5-fold | RF | 0.632 ± 0.252 | 0.983 ± 0.275 | +0.454 ± 0.288 | 54.3 ± 10.5 | 84.5 ± 13.5 | +29.6 ± 18.5 | – | 0.694 ± 0.123 | 0.884 ± 0.081 | 0.774 ± 0.099 |
| Time 5-fold | RF+Geo | 0.349 ± 0.122 | 0.621 ± 0.161 | +0.140 ± 0.135 | 68.0 ± 8.5 | 97.1 ± 5.1 | +9.8 ± 7.9 | – | 0.816 ± 0.112 | 0.890 ± 0.084 | 0.851 ± 0.095 |
| Time 5-fold | RF+Geo+Leg † | 0.284 ± 0.104 | 0.552 ± 0.131 | +0.038 ± 0.061 | 73.3 ± 8.2 | 98.3 ± 2.6 | +2.7 ± 4.3 | – | 0.854 ± 0.087 | 0.876 ± 0.087 | 0.865 ± 0.086 |
| Time 5-fold | YOLO | 0.347 ± 0.166 | 0.642 ± 0.213 | -0.067 ± 0.128 | 70.4 ± 10.4 | 94.9 ± 6.3 | -5.5 ± 10.3 | – | – | – | – |

† RF best (5-fold 평균 val MAE 최소). ME = 평균(예측 − 정답), TCE = 총 마릿수 오차 / 정답 총 마릿수. Precision · Recall · F1 = visiting 개체 단위(사람 개체와 짝짓기: Geo · RF 계열 = 관절 평균 거리 ÷ 사람 박스 대각선 ≤ 0.25, YOLO = 박스 IoU ≥ 0.5). val은 모든 방법이 조금 낙관적임: 포즈 best.pt(Geo · RF 계열)와 YOLO best.pt · conf를 같은 valid로 골랐음. 10-s MAE는 1분 연속 영상(test(000))을 프레임 번호로 6등분해 구하는 지표라, 여러 클립에서 뽑은 val 사진에는 정의되지 않음(–). YOLO val Precision · Recall · F1은 YOLO 재평가(valid 개체 진단 추가) 후 채워짐.

## Table 3. Test MAE with 95% bootstrap CI (images resampled, 10,000×)

| Method | Random 5-fold | Time 5-fold |
|---|---|---|
| Geo | 0.293 [0.200, 0.393] | 0.293 [0.205, 0.391] |
| RF | 0.114 [0.065, 0.170] | 0.116 [0.065, 0.174] |
| RF+Geo | 0.133 [0.074, 0.198] | 0.126 [0.070, 0.188] |
| RF+Geo+Leg | 0.121 [0.067, 0.181] | 0.109 [0.056, 0.170] |
| YOLO | 0.300 [0.240, 0.365] | 0.223 [0.158, 0.295] |

사진별 |오차|는 fold 5개 평균. 86장이 1분 영상 하나에서 나와 사진끼리 독립이 아니므로 CI는 낙관적임.

## Table 4. Paired comparison on test(000) images (Wilcoxon signed-rank, n = 86)

| Comparison (A vs B) | Split | ΔMAE (B − A) | A better / same / B better (images) | p |
|---|---|---|---|---|
| RF+Geo+Leg vs Geo | Random 5-fold | +0.172 | 24 / 58 / 4 | <0.001 |
| RF+Geo+Leg vs RF | Random 5-fold | -0.007 | 4 / 75 / 7 | 0.580 |
| RF+Geo+Leg vs RF+Geo | Random 5-fold | +0.012 | 6 / 78 / 2 | 0.429 |
| RF+Geo+Leg vs YOLO | Random 5-fold | +0.179 | 54 / 29 / 3 | <0.001 |
| RF+Geo+Leg vs Geo | Time 5-fold | +0.184 | 20 / 61 / 5 | 0.007 |
| RF+Geo+Leg vs RF | Time 5-fold | +0.007 | 6 / 78 / 2 | 0.449 |
| RF+Geo+Leg vs RF+Geo | Time 5-fold | +0.016 | 8 / 75 / 3 | 0.344 |
| RF+Geo+Leg vs YOLO | Time 5-fold | +0.114 | 27 / 56 / 3 | <0.001 |
| Geo: Random 5-fold vs Time 5-fold | – | +0.000 | 6 / 73 / 7 | 0.859 |
| RF: Random 5-fold vs Time 5-fold | – | -0.002 | 6 / 73 / 7 | 0.861 |
| RF+Geo: Random 5-fold vs Time 5-fold | – | +0.007 | 5 / 78 / 3 | 0.696 |
| RF+Geo+Leg: Random 5-fold vs Time 5-fold | – | +0.012 | 9 / 70 / 7 | 0.713 |
| YOLO: Random 5-fold vs Time 5-fold | – | +0.077 | 42 / 35 / 9 | <0.001 |

사진별 |오차|(fold 5개 평균)로 비교. Random 5-fold vs Time 5-fold 줄은 A = Random 5-fold, B = Time 5-fold (Δ > 0이면 Random 5-fold가 더 나쁨). 사진끼리 독립이 아니라 p값은 참고용.

## Table 5. Error decomposition on test(000) (visiting pigs per fold)

| Split | Method | Counted / GT | Missed (undetected) | Missed (judged other) | Other → visiting | Unlabeled → visiting |
|---|---|---|---|---|---|---|
| Random 5-fold | Geo | 108.2 ± 2.2 / 133 | 8.6 ± 5.4 | 20.0 ± 2.3 | 0.6 ± 0.5 | 3.2 ± 3.8 |
| Random 5-fold | RF | 130.0 ± 3.7 / 133 | 8.6 ± 5.4 | 2.8 ± 2.0 | 3.4 ± 1.7 | 5.0 ± 4.8 |
| Random 5-fold | RF+Geo | 127.2 ± 2.8 / 133 | 8.6 ± 5.4 | 4.6 ± 2.1 | 2.8 ± 0.8 | 4.6 ± 4.6 |
| Random 5-fold | RF+Geo+Leg | 128.6 ± 3.2 / 133 | 8.6 ± 5.4 | 3.4 ± 2.3 | 3.0 ± 0.7 | 4.6 ± 4.6 |
| Random 5-fold | YOLO | 127.6 ± 18.1 / 133 | 6.0 ± 0.7 | 13.0 ± 11.3 | 5.8 ± 1.1 | 7.8 ± 10.3 |
| Time 5-fold | Geo | 109.4 ± 1.8 / 133 | 6.2 ± 2.8 | 20.6 ± 1.5 | 0.8 ± 0.4 | 2.4 ± 2.6 |
| Time 5-fold | RF | 131.4 ± 1.5 / 133 | 6.2 ± 2.8 | 2.2 ± 1.6 | 4.0 ± 1.2 | 2.8 ± 2.9 |
| Time 5-fold | RF+Geo | 128.6 ± 2.1 / 133 | 6.2 ± 2.8 | 4.4 ± 2.9 | 3.4 ± 0.5 | 2.8 ± 2.9 |
| Time 5-fold | RF+Geo+Leg | 131.2 ± 1.9 / 133 | 6.2 ± 2.8 | 2.0 ± 1.2 | 3.6 ± 0.5 | 2.8 ± 2.9 |
| Time 5-fold | YOLO | 127.4 ± 2.1 / 133 | 5.4 ± 1.8 | 10.8 ± 4.0 | 8.4 ± 3.6 | 2.2 ± 1.3 |

Geo · RF 계열은 같은 포즈 모델을 써서 'Missed (undetected)'가 같음.

## Table 6. Pose model (YOLOv8m-pose, 18 keypoints)

| Split | Set | box_mAP50 | box_mAP50-95 | pose_mAP50 | pose_mAP50-95 | PCK@0.05 | PCK@0.1 | PCK@0.2 |
|---|---|---|---|---|---|---|---|---|
| Random 5-fold | test(000) | 0.885 ± 0.008 | 0.652 ± 0.023 | 0.338 ± 0.062 | 0.111 ± 0.021 | 0.369 ± 0.042 | 0.691 ± 0.029 | 0.865 ± 0.009 |
| Random 5-fold | Validation | 0.986 ± 0.005 | 0.797 ± 0.011 | 0.742 ± 0.031 | 0.287 ± 0.024 | 0.518 ± 0.016 | 0.784 ± 0.018 | 0.923 ± 0.009 |
| Time 5-fold | test(000) | 0.884 ± 0.026 | 0.646 ± 0.023 | 0.274 ± 0.049 | 0.087 ± 0.010 | 0.373 ± 0.009 | 0.692 ± 0.024 | 0.865 ± 0.011 |
| Time 5-fold | Validation | 0.960 ± 0.019 | 0.694 ± 0.024 | 0.548 ± 0.046 | 0.167 ± 0.020 | 0.406 ± 0.011 | 0.695 ± 0.025 | 0.871 ± 0.026 |

PCK@α = 정답과 예측 keypoint 거리가 α × 정답 박스 긴 변(max(w, h)) 미만인 비율 (`training/scripts/utils/pck.py`; 예측 conf 0.25, 사람 박스와 IoU ≥ 0.5로 짝지은 돼지만). mAP는 Ultralytics val. best.pt = val pose mAP50-95 최고 epoch. keypoint별 PCK@0.1:

| Keypoint | Random 5-fold test(000) | Random 5-fold Validation | Time 5-fold test(000) | Time 5-fold Validation |
|---|---|---|---|---|
| tail_tip | 0.818 ± 0.020 | 0.884 ± 0.021 | 0.794 ± 0.052 | 0.797 ± 0.041 |
| tail | 0.836 ± 0.012 | 0.935 ± 0.015 | 0.839 ± 0.021 | 0.871 ± 0.025 |
| l_hip | 0.746 ± 0.110 | 0.925 ± 0.016 | 0.717 ± 0.065 | 0.853 ± 0.036 |
| r_hip | 0.776 ± 0.090 | 0.932 ± 0.018 | 0.784 ± 0.076 | 0.876 ± 0.023 |
| l_foot | 0.608 ± 0.080 | 0.667 ± 0.013 | 0.551 ± 0.071 | 0.529 ± 0.036 |
| r_foot | 0.394 ± 0.050 | 0.671 ± 0.022 | 0.368 ± 0.071 | 0.495 ± 0.043 |
| r_flank | 0.943 ± 0.012 | 0.918 ± 0.008 | 0.942 ± 0.019 | 0.846 ± 0.017 |
| l_flank | 0.858 ± 0.074 | 0.913 ± 0.015 | 0.846 ± 0.045 | 0.836 ± 0.034 |
| l_shoulder | 0.790 ± 0.071 | 0.890 ± 0.025 | 0.840 ± 0.061 | 0.825 ± 0.034 |
| r_shoulder | 0.895 ± 0.046 | 0.890 ± 0.024 | 0.925 ± 0.007 | 0.819 ± 0.018 |
| l_hand | 0.310 ± 0.006 | 0.526 ± 0.040 | 0.262 ± 0.063 | 0.405 ± 0.042 |
| neck | 0.807 ± 0.069 | 0.890 ± 0.018 | 0.807 ± 0.035 | 0.827 ± 0.027 |
| nose | 0.712 ± 0.046 | 0.673 ± 0.041 | 0.703 ± 0.047 | 0.607 ± 0.045 |
| l_ear | 0.843 ± 0.078 | 0.842 ± 0.024 | 0.868 ± 0.059 | 0.770 ± 0.037 |
| l_ear_tip | 0.569 ± 0.096 | 0.603 ± 0.021 | 0.688 ± 0.114 | 0.497 ± 0.059 |
| r_ear | 0.857 ± 0.047 | 0.857 ± 0.021 | 0.858 ± 0.048 | 0.779 ± 0.017 |
| r_ear_tip | 0.489 ± 0.050 | 0.613 ± 0.026 | 0.518 ± 0.065 | 0.527 ± 0.053 |
| r_hand | 0.271 ± 0.041 | 0.519 ± 0.053 | 0.271 ± 0.032 | 0.407 ± 0.048 |

## Table 7. YOLO detector (visiting / other boxes)

| Split | Set | conf | P | R | mAP50 | mAP50-95 | mAP50-95 visiting | mAP50-95 other |
|---|---|---|---|---|---|---|---|---|
| Random 5-fold | test(000) | 0.48 ± 0.15 | 0.899 ± 0.039 | 0.842 ± 0.036 | 0.867 ± 0.029 | 0.597 ± 0.024 | 0.536 ± 0.035 | 0.659 ± 0.016 |
| Random 5-fold | Validation | 0.48 ± 0.15 | 0.939 ± 0.019 | 0.924 ± 0.020 | 0.953 ± 0.009 | 0.713 ± 0.024 | 0.597 ± 0.042 | 0.829 ± 0.010 |
| Time 5-fold | test(000) | 0.49 ± 0.19 | 0.911 ± 0.032 | 0.859 ± 0.022 | 0.879 ± 0.008 | 0.601 ± 0.016 | 0.555 ± 0.045 | 0.646 ± 0.016 |
| Time 5-fold | Validation | 0.49 ± 0.19 | 0.921 ± 0.036 | 0.838 ± 0.066 | 0.899 ± 0.059 | 0.610 ± 0.042 | 0.516 ± 0.097 | 0.704 ± 0.043 |

conf = 그 fold valid에서 사진당 MAE 최소. 값 출처: `yolo_det/reports/ep200_<fold>/m/summary.json`.

## Table 8. Inference time (end-to-end per image)

| Split | Device | Method | Median (ms) | p95 (ms) | Read | Model | Features | RF | 1 fps (≤1000 ms) | 30 fps (≤33.3 ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| Random 5-fold | GPU (RTX 6000 Ada) | Geo | 23.8 ± 0.5 | 24.5 ± 0.3 | 13.0 | 10.1 | 0.8 | 0.0 | ✅ | ✅ |
| Random 5-fold | GPU (RTX 6000 Ada) | RF | 28.0 ± 0.5 | 28.6 ± 0.3 | 13.0 | 10.1 | 0.8 | 4.1 | ✅ | ✅ |
| Random 5-fold | GPU (RTX 6000 Ada) | RF+Geo | 27.9 ± 0.5 | 28.6 ± 0.3 | 13.0 | 10.1 | 0.8 | 4.1 | ✅ | ✅ |
| Random 5-fold | GPU (RTX 6000 Ada) | RF+Geo+Leg | 27.9 ± 0.5 | 28.6 ± 0.3 | 13.0 | 10.1 | 0.8 | 4.1 | ✅ | ✅ |
| Random 5-fold | GPU (RTX 6000 Ada) | YOLO | 21.3 ± 0.1 | 21.4 ± 0.2 | 12.6 | 8.6 | 0.0 | 0.0 | ✅ | ✅ |
| Random 5-fold | CPU (AMD EPYC 7513) | Geo | 185.8 ± 4.1 | 192.7 ± 4.6 | 12.9 | 171.5 | 1.0 | 0.0 | ✅ | ❌ |
| Random 5-fold | CPU (AMD EPYC 7513) | RF | 190.4 ± 4.0 | 197.4 ± 4.6 | 12.9 | 171.5 | 1.0 | 4.7 | ✅ | ❌ |
| Random 5-fold | CPU (AMD EPYC 7513) | RF+Geo | 190.0 ± 4.1 | 196.9 ± 4.6 | 12.9 | 171.5 | 1.0 | 4.2 | ✅ | ❌ |
| Random 5-fold | CPU (AMD EPYC 7513) | RF+Geo+Leg | 190.0 ± 4.1 | 196.8 ± 4.6 | 12.9 | 171.5 | 1.0 | 4.2 | ✅ | ❌ |
| Random 5-fold | CPU (AMD EPYC 7513) | YOLO | 179.5 ± 3.2 | 186.4 ± 2.3 | 13.0 | 166.5 | 0.0 | 0.0 | ✅ | ❌ |
| Time 5-fold | GPU (RTX 6000 Ada) | Geo | 23.7 ± 0.1 | 24.1 ± 0.3 | 13.0 | 10.0 | 0.8 | 0.0 | ✅ | ✅ |
| Time 5-fold | GPU (RTX 6000 Ada) | RF | 27.8 ± 0.1 | 28.2 ± 0.3 | 13.0 | 10.0 | 0.8 | 4.1 | ✅ | ✅ |
| Time 5-fold | GPU (RTX 6000 Ada) | RF+Geo | 27.8 ± 0.1 | 28.2 ± 0.2 | 13.0 | 10.0 | 0.8 | 4.1 | ✅ | ✅ |
| Time 5-fold | GPU (RTX 6000 Ada) | RF+Geo+Leg | 27.8 ± 0.1 | 28.2 ± 0.3 | 13.0 | 10.0 | 0.8 | 4.1 | ✅ | ✅ |
| Time 5-fold | GPU (RTX 6000 Ada) | YOLO | 21.3 ± 0.1 | 21.5 ± 0.1 | 12.6 | 8.6 | 0.0 | 0.0 | ✅ | ✅ |
| Time 5-fold | CPU (AMD EPYC 7513) | Geo | 191.6 ± 4.0 | 200.4 ± 8.1 | 12.9 | 177.3 | 1.1 | 0.0 | ✅ | ❌ |
| Time 5-fold | CPU (AMD EPYC 7513) | RF | 196.3 ± 4.0 | 205.2 ± 8.0 | 12.9 | 177.3 | 1.1 | 4.7 | ✅ | ❌ |
| Time 5-fold | CPU (AMD EPYC 7513) | RF+Geo | 195.9 ± 4.0 | 204.7 ± 8.1 | 12.9 | 177.3 | 1.1 | 4.2 | ✅ | ❌ |
| Time 5-fold | CPU (AMD EPYC 7513) | RF+Geo+Leg | 195.8 ± 4.0 | 204.6 ± 8.1 | 12.9 | 177.3 | 1.1 | 4.2 | ✅ | ❌ |
| Time 5-fold | CPU (AMD EPYC 7513) | YOLO | 183.3 ± 4.9 | 189.5 ± 2.6 | 13.1 | 169.9 | 0.0 | 0.0 | ✅ | ❌ |

조건: 5가지 방법 모두 같은 서버 한 대, batch 1, imgsz 640, 예열 5장 제외, test(000) 86장 × 3회, 사진 읽기부터 마릿수까지, fold마다 그 fold 모델(YOLO는 그 fold conf) → fold 5개 중앙값 · p95의 평균 ± 표본 표준편차. 단계별 칸(Read · Model · Features · RF)은 fold 중앙값의 평균(ms). 실시간 판정은 p95 평균 기준. 원자료: `counting/reports/speed/NVIDIA_RTX_6000_Ada_Generation/` (`benchmark_speed.py`).

## Table 9. Visiting pigs per image (human labels)

| Set | Images | 0 | 1 | 2 | 3+ | Mean ± SD | Total visiting |
|---|---|---|---|---|---|---|---|
| train + valid 001 (CH005) | 90 | 26 | 23 | 23 | 18 | 1.51 ± 1.42 | 136 |
| train + valid 002 (CH010) | 100 | 19 | 33 | 19 | 29 | 1.79 ± 1.44 | 179 |
| train + valid 001 + 002 | 190 | 45 | 56 | 42 | 47 | 1.66 ± 1.44 | 315 |
| test(000) | 86 | 5 | 29 | 52 | 0 | 1.55 ± 0.61 | 133 |

train + valid 190장은 5-fold valid를 합친 것(각 사진이 한 번씩 valid). 0 / 1 / 2 / 3+ = 사진 수.

## Table 10. Validation MAE by camera (fold 5개 평균 ± 표본 표준편차)

| Split | Method | CH005 (001, 아침) | CH010 (002, 낮~저녁) |
|---|---|---|---|
| Random 5-fold | Geo | 0.423 ± 0.092 | 0.690 ± 0.097 |
| Random 5-fold | RF | 0.298 ± 0.159 | 0.472 ± 0.142 |
| Random 5-fold | RF+Geo | 0.255 ± 0.140 | 0.262 ± 0.141 |
| Random 5-fold | RF+Geo+Leg | 0.212 ± 0.146 | 0.203 ± 0.147 |
| Time 5-fold | Geo | 0.360 ± 0.093 | 0.765 ± 0.281 |
| Time 5-fold | RF | 0.253 ± 0.094 | 0.960 ± 0.455 |
| Time 5-fold | RF+Geo | 0.224 ± 0.069 | 0.452 ± 0.184 |
| Time 5-fold | RF+Geo+Leg | 0.218 ± 0.148 | 0.342 ± 0.148 |

fold마다 그 fold valid 중 그 카메라 사진만으로 MAE를 계산 후 평균. fold별 장수는 `val_mae_by_camera.csv`.

## Table 11. RF+Geo+Leg feature importance (Gini, mean ± SD over 10 fold models)

| Rank | Feature | Group | Importance |
|---|---|---|---|
| 1 | geo_nose_dist_over_r | Geo | 0.270 ± 0.008 |
| 2 | nose_zone_dist_norm | RF | 0.211 ± 0.004 |
| 3 | nose_in_zone | RF | 0.172 ± 0.012 |
| 4 | leg_angle_deg | Leg | 0.098 ± 0.006 |
| 5 | paw_offset | Leg | 0.092 ± 0.005 |
| 6 | geo_aspect_ratio | Geo | 0.055 ± 0.006 |
| 7 | leg_side_bias | Leg | 0.045 ± 0.006 |
| 8 | leg_extension | Leg | 0.034 ± 0.007 |
| 9 | geo_head_angle_deg | Geo | 0.007 ± 0.001 |
| 10 | body_bend_deg | Leg | 0.006 ± 0.001 |
| 11 | key_kpt_conf | RF | 0.004 ± 0.001 |
| 12 | spine_dip_deg | RF | 0.003 ± 0.001 |
| 13 | neck_rel_height | RF | 0.002 ± 0.001 |
| 14 | head_dip_deg | RF | 0.002 ± 0.000 |
| 15 | geo_lateral_asymmetry | Geo | 0.001 ± 0.000 |
| 16 | geo_limb_visibility | Geo | 0.000 ± 0.000 |

그룹 합계: Geo 0.332 ± 0.012, Leg 0.275 ± 0.010, RF 0.393 ± 0.011. RF는 그 fold train 사진의 사람 관절로 학습(평가와 동일), 나무 100개 · balanced · seed 42.

## Figures

| Figure | File | Caption |
|---|---|---|
| A | `../graph_A_frame_counts_000.png` | Per-image visiting count on test(000) (GT, Geo, RF best, YOLO; mean over 5 folds, band = min–max) |
| B | `../graph_B_mae_random_vs_time.png` | Per-image count MAE by method, Random 5-fold vs Time 5-fold (mean ± SD) |
| C | `../graph_C_total_counts.png` | Total visiting pigs counted per fold vs human GT |
| D | `../graph_D_keypoint_pck.png` | Per-keypoint PCK@0.1, Random 5-fold vs Time 5-fold |
| E | `../per_image/graph_E_per_image_counts.png` | Visiting pigs counted per test(000) image, every method vs GT (mean over 5 folds, band = min–max) |
| F | `../per_image/graph_F_per_image_error_heatmap.png` | Per-image counting error of all 50 models on test(000) |
| G | `../error_cases/all_models_57.png`, `all_models_76.png` | Occlusion / crowding error cases: all 10 pose models side by side |
| J | `../graph_J_random_5fold_frames.png`, `../graph_J_time_5fold_frames.png` | Visiting pigs per test(000) image (frame 0–85), human GT vs Geo, RF+Geo+Leg, YOLO, one figure per split (mean of 5 fold models; lines offset by −0.06 / 0 / +0.06) |
| – | `../per_image/by_model/<method>.png` | Per-method counts on test(000): GT row + 10 fold models |

