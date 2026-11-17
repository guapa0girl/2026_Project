#!/usr/bin/env bash
# 시간대 K-fold: fold마다 데이터 준비 → 학습 → 평가 (COCO 시작). fold마다 data_k<K>f<i>/ 를 따로 써서 동시에 돌려도 안전.
#
# 사용법 (레포 루트에서, venv 켠 상태):
#   bash yolo_det/scripts/run_kfold.sh <GPU> <fold...>        예) GPU 0에서 fold 0 1 2
#   bash yolo_det/scripts/run_kfold.sh 0 0 1 2
#   bash yolo_det/scripts/run_kfold.sh 1 3 4
# 환경변수: SPLIT(kfold=시간대 기본 / rkfold=랜덤), K(기본 5), EPOCHS(기본 200),
#           RUN_PREFIX(기본 시간대 ep<EPOCHS>_k<K>, 랜덤 ep<EPOCHS>_rk<K>)
#   랜덤 5-fold:  SPLIT=rkfold bash yolo_det/scripts/run_kfold.sh 0 0 1 2 3 4
#   EVAL_ONLY=1 ... : 재학습 없이 평가만 다시 (val 결과 추가 등)
#   TIMING=1 ...    : 평가 때 추론 시간도 측정 (batch 1, 640, 예열 5장 제외, 000 × 3회)
# 학습 전 1회 대조: data_<tag>f<i>/split.json ↔ ${SPLITS_DIR:-training/data/splits}/<tag>_f<i>.json 가 다르면 중단
#   (먼저 만든 사람의 분할이 기준). 대조 생략: SKIP_COMPARE=1
# 다 끝나면: python yolo_det/scripts/summarize_folds.py --prefix ep200_k5 --k 5
set -euo pipefail
GPU=$1; shift
SPLIT=${SPLIT:-kfold}; K=${K:-5}; EPOCHS=${EPOCHS:-200}
case "$SPLIT" in kfold) TAG=k ;; rkfold) TAG=rk ;; *) echo "SPLIT=$SPLIT (kfold|rkfold)"; exit 1 ;; esac
PREFIX=${RUN_PREFIX:-ep${EPOCHS}_${TAG}${K}}
SPLITS_DIR=${SPLITS_DIR:-training/data/splits}
EVAL_ARGS=(); [[ "${TIMING:-0}" == 1 ]] && EVAL_ARGS+=(--timing)
for f in "$@"; do
  export YOLO_DET_DATA=data_${TAG}${K}f${f}
  RUN=${PREFIX}_f${f}
  echo "===== fold ${f}/${K} → run ${RUN}, data ${YOLO_DET_DATA}, GPU ${GPU} ====="
  python yolo_det/scripts/prepare_data.py --split "$SPLIT" --k "$K" --fold "$f"
  if [[ "${SKIP_COMPARE:-0}" != 1 ]]; then
    python yolo_det/scripts/compare_split.py --ref "${SPLITS_DIR}/${TAG}${K}_f${f}.json"
  fi
  if [[ "${EVAL_ONLY:-0}" != 1 ]]; then
    python yolo_det/scripts/train.py --sizes m --run "$RUN" --epochs "$EPOCHS" --device "$GPU"
  fi
  python yolo_det/scripts/evaluate.py --run "$RUN" --size m --device "$GPU" ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}
done