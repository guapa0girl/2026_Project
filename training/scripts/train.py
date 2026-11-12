"""돼지 포즈 추정 모델을 nano / small / medium 세 사이즈로 학습하고,
학습 직후 각 모델을 train/valid/test 세 split에 대해 자동으로 평가한 뒤
세 모델의 성능을 비교한 리포트까지 생성한다.

중요한 정책 (요청사항 그대로):
  - 새 라벨링 데이터가 추가되어도 "파인튜닝"하지 않는다.
    매번 COCO 사전학습 가중치(yolov8*-pose.pt)에서 처음부터 다시 학습한다.
  - 학습 직전, 이전에 남아있던 models/results/<size> 와
    reports/metrics/<size> 는 past_train/run_<timestamp>/ 로 자동 이동된다.
    -> models/results/ 에는 항상 "가장 최근 학습 결과"만 남는다.

사용법:
    python scripts/train.py                      # n, s, m, l 전부 학습 + 평가 + 비교
    python scripts/train.py --sizes n s           # 일부만
    python scripts/train.py --epochs 100          # configs/train_*.yaml의 epochs를 덮어씀
    python scripts/train.py --sizes l --resume  # 중단된 학습을 이어서 진행(가능한 것만), 나머지는 새로 시작
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

from archive_previous_run import archive_all
from compare_models import main as compare_main
from evaluate import evaluate_split, save_summary
from utils.paths import (
    CONFIGS_DIR,
    DATA_YAML,
    PRETRAINED_DIR,
    PRETRAINED_WEIGHTS,
    RESULTS_DIR,
    SIZE_LABELS,
    SIZES,
    SPLITS,
)
from utils.skeleton import load_pose_spec, resolve_data_yaml

SIZE_CONFIG_FILE = {
    "n": CONFIGS_DIR / "train_nano.yaml",
    "s": CONFIGS_DIR / "train_small.yaml",
    "m": CONFIGS_DIR / "train_medium.yaml",
    "l": CONFIGS_DIR / "train_large.yaml",
}


def ensure_pretrained_weight(size: str) -> Path:
    """COCO 사전학습 pose 가중치를 models/pretrained/ 에 준비(없으면 자동 다운로드)."""
    weight_name = PRETRAINED_WEIGHTS[size]
    weight_path = PRETRAINED_DIR / weight_name
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    if weight_path.exists():
        return weight_path

    print(f"[pretrained] {weight_name} 이 없어 자동 다운로드합니다...")
    tmp_model = YOLO(weight_name)  # ultralytics가 없으면 자동 다운로드 후 로드
    downloaded = Path(weight_name)
    if downloaded.exists():
        shutil.move(str(downloaded), str(weight_path))
    else:
        tmp_model.save(str(weight_path))
    return weight_path


def train_one_size(size: str, epochs_override, imgsz_override, batch_override, resume=False, patience_override=None):
    last_pt = RESULTS_DIR / size / "weights" / "last.pt"

    if resume and last_pt.exists():
        # 세션/환경이 중간에 끊겨 학습이 죽었을 때 이어서 재개하는 경로.
        # "새 데이터로 파인튜닝"과는 다르다 — 같은 데이터로 진행 중이던
        # 학습을 optimizer 상태/LR 스케줄까지 그대로 이어간다. resume=True일
        # 때는 체크포인트 옆 args.yaml에서 전체 설정을 그대로 읽어오므로
        # epochs/imgsz/batch 등 다른 인자를 같이 넘기면 안 된다.
        print(f"\n===== [{SIZE_LABELS[size]} / yolov8{size}-pose] 중단된 학습 이어서 진행 ({last_pt}) =====")
        model = YOLO(str(last_pt))
        model.train(resume=True)
        return model

    with open(SIZE_CONFIG_FILE[size], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    epochs = epochs_override or cfg["epochs"]
    imgsz = imgsz_override or cfg["imgsz"]
    batch = batch_override or cfg["batch"]

    weight_path = ensure_pretrained_weight(size)
    print(f"\n===== [{SIZE_LABELS[size]} / yolov8{size}-pose] 처음부터 학습 시작 (파인튜닝 아님) =====")

    model = YOLO(str(weight_path))
    model.train(
        data=str(resolve_data_yaml()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=cfg.get("patience", 30) if patience_override is None else patience_override,
        optimizer=cfg.get("optimizer", "auto"),
        lr0=cfg.get("lr0", 0.01),
        workers=cfg.get("workers", 8),
        project=str(RESULTS_DIR),
        name=size,
        exist_ok=True,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="nano/small/medium 돼지 포즈 모델 학습 + 평가 + 비교")
    parser.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None,
                        help="조기 종료 patience 덮어쓰기. 0 = 조기 종료 끔(지정한 epoch를 끝까지 학습, best.pt는 valid 기준)")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--skip-archive", action="store_true", help="이전 결과 자동 이동을 생략(디버깅용)")
    parser.add_argument(
        "--resume", action="store_true",
        help="models/results/<size>/weights/last.pt 가 있으면 처음부터 다시 하지 않고 이어서 진행 "
             "(세션/환경이 끊겨 학습이 중단됐을 때 복구용, 새 데이터 파인튜닝 용도 아님)",
    )
    args = parser.parse_args()

    resumable = {
        s for s in args.sizes
        if args.resume and (RESULTS_DIR / s / "weights" / "last.pt").exists()
    }
    if not args.skip_archive:
        archive_all(sizes=[s for s in args.sizes if s not in resumable])

    spec = load_pose_spec(DATA_YAML)

    for size in args.sizes:
        model = train_one_size(size, args.epochs, args.imgsz, args.batch, resume=size in resumable,
                               patience_override=args.patience)

        print(f"----- [{SIZE_LABELS[size]}] train/valid/test 평가 -----")
        summary = {}
        for split in args.splits:
            summary[split] = evaluate_split(
                model, size, split, args.imgsz or 640, conf=0.25, iou_thres=0.5,
                num_keypoints=spec["num_keypoints"], keypoint_names=spec["keypoint_names"],
            )
        save_summary(size, summary)

    print("\n===== 사이즈별 성능 비교 리포트 생성 =====")
    compare_main()


if __name__ == "__main__":
    main()
