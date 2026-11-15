#!/usr/bin/env bash
# 시간대 K-fold: fold마다 데이터 준비 → 학습 → 평가 (COCO 시작). fold마다 data_k<K>f<i>/ 를 따로 써서 동시에 돌려도 안전.
#
# 사용법 (레포 루트에서, venv 켠 상태):
#   bash yolo_det/scripts/run_kfold.sh <GPU> <fold...>        예) GPU 0에서 fold 0 1 2
#   bash yolo_det/scripts/run_kfold.sh 0 0 1 2
#   bash yolo_det/scripts/run_kfold.sh 1 3 4
# 환경변수: K(기본 5), EPOCHS(기본 200), RUN_PREFIX(기본 ep<EPOCHS>_k<K>)
# 다 끝나면: python yolo_det/scripts/summarize_folds.py --prefix ep200_k5 --k 5
set -euo pipefail
GPU=$1; shift
K=${K:-5}; EPOCHS=${EPOCHS:-200}; PREFIX=${RUN_PREFIX:-ep${EPOCHS}_k${K}}
for f in "$@"; do
  export YOLO_DET_DATA=data_k${K}f${f}
  RUN=${PREFIX}_f${f}
  echo "===== fold ${f}/${K} → run ${RUN}, data ${YOLO_DET_DATA}, GPU ${GPU} ====="
  python yolo_det/scripts/prepare_data.py --split kfold --k "$K" --fold "$f"
  python yolo_det/scripts/train.py --sizes m --run "$RUN" --epochs "$EPOCHS" --device "$GPU"
  python yolo_det/scripts/evaluate.py --run "$RUN" --size m --device "$GPU"
done
