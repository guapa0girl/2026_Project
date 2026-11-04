"""학습된 YOLO-Pose 모델을 train/valid/test 각 split에 대해 평가한다.

두 종류의 지표를 계산한다:
  1) Ultralytics 공식 검증(model.val): box/pose 각각의 mAP50, mAP50-95
     (pose mAP는 OKS 기반. bbox보다 pose 지표를 우선순위로 본다)
  2) PCK@0.05 / 0.1 / 0.2 (scripts/utils/pck.py, bbox 정규화 기준),
     전체 평균 + 키포인트별 breakdown

결과는 reports/metrics/<size>/<split>_metrics.json 에 저장되고,
reports/metrics/<size>/summary.json 에 3개 split을 모아서 저장한다.

사용법:
    python scripts/evaluate.py --size n
    python scripts/evaluate.py --size n --splits test
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from utils.label_io import read_yolo_pose_label
from utils.paths import DATA_YAML, METRICS_DIR, PROCESSED_DIR, RESULTS_DIR, SIZES, SPLITS
from utils.pck import PCKAccumulator
from utils.skeleton import load_pose_spec, resolve_data_yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def run_official_val(model, size, split, split_key, imgsz):
    """ultralytics 공식 val: box + pose mAP50 / mAP50-95.

    data=resolve_data_yaml()로 절대경로 yaml을 넘기고, project/name을 우리
    프로젝트 안(models/results/<size>/val_<split>)으로 명시적으로 고정한다.
    (전역 Ultralytics 설정의 datasets_dir/runs_dir이 이 프로젝트가 아닌 다른
    곳을 가리키고 있어도 영향받지 않도록 하기 위함 — README '알려진 이슈' 참고)"""
    metrics = model.val(
        data=str(resolve_data_yaml()),
        split=split_key,
        imgsz=imgsz,
        verbose=False,
        project=str(RESULTS_DIR / size),
        name=f"val_{split}",
        exist_ok=True,
    )
    return {
        "box": {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
        "pose": {
            "mAP50": float(metrics.pose.map50),
            "mAP50-95": float(metrics.pose.map),
        },
    }


def run_pck(model, split_dir_name, num_keypoints, keypoint_names, conf, iou_thres):
    img_dir = PROCESSED_DIR / split_dir_name / "images"
    lbl_dir = PROCESSED_DIR / split_dir_name / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    acc = PCKAccumulator(num_keypoints=num_keypoints)
    for img_path in images:
        gt_boxes, gt_kpts, gt_vis = read_yolo_pose_label(
            lbl_dir / f"{img_path.stem}.txt", num_keypoints
        )
        result = model.predict(source=str(img_path), conf=conf, verbose=False)[0]

        if result.boxes is not None and len(result.boxes):
            pred_boxes = result.boxes.xywhn.cpu().numpy()
            pred_kpts = result.keypoints.xyn.cpu().numpy()
        else:
            pred_boxes = np.zeros((0, 4))
            pred_kpts = np.zeros((0, num_keypoints, 2))

        acc.add_image(pred_boxes, pred_kpts, gt_boxes, gt_kpts, gt_vis, iou_thres=iou_thres)

    return acc.summary(keypoint_names=keypoint_names)


def evaluate_split(model, size, split, imgsz, conf, iou_thres, num_keypoints, keypoint_names):
    # ultralytics data yaml의 키는 train / val / test 이고, 우리 폴더명은 valid 이므로 매핑
    split_key = {"train": "train", "valid": "val", "test": "test"}[split]

    official = run_official_val(model, size, split, split_key, imgsz)
    pck = run_pck(model, split, num_keypoints, keypoint_names, conf, iou_thres)

    result = {
        "size": size,
        "split": split,
        "official": official,
        "pck": pck,
    }

    METRICS_DIR.joinpath(size).mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / size / f"{split}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"[{size}/{split}] pose mAP50={official['pose']['mAP50']:.4f} "
        f"mAP50-95={official['pose']['mAP50-95']:.4f} "
        f"| box mAP50={official['box']['mAP50']:.4f} "
        f"| PCK@0.2={pck['by_threshold']['PCK@0.2']['overall']}"
    )
    return result


def save_summary(size, summary):
    """compare_models.py가 읽는 reports/metrics/<size>/summary.json 을 저장한다.
    evaluate.py를 단독 실행할 때뿐 아니라 train.py가 학습 직후 평가할 때도
    반드시 호출해야 한다 — 안 그러면 compare_models.py가 이전 실행의 stale한
    summary.json을 그대로 읽어버린다."""
    METRICS_DIR.joinpath(size).mkdir(parents=True, exist_ok=True)
    summary_path = METRICS_DIR / size / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[evaluate] 요약 저장: {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="train/valid/test split별 pose 성능 평가")
    parser.add_argument("--size", required=True, choices=list(SIZES))
    parser.add_argument("--weights", type=Path, default=None, help="기본값: models/results/<size>/weights/best.pt")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="PCK 계산용 예측 confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="PCK용 GT-예측 인스턴스 매칭 IoU 임계값")
    args = parser.parse_args()

    weights = args.weights or (RESULTS_DIR / args.size / "weights" / "best.pt")
    if not weights.exists():
        raise SystemExit(f"가중치 파일을 찾을 수 없습니다: {weights} (먼저 scripts/train.py를 실행하세요)")

    spec = load_pose_spec(DATA_YAML)
    model = YOLO(str(weights))

    summary = {}
    for split in args.splits:
        summary[split] = evaluate_split(
            model, args.size, split, args.imgsz, args.conf, args.iou_thres,
            spec["num_keypoints"], spec["keypoint_names"],
        )

    save_summary(args.size, summary)


if __name__ == "__main__":
    main()
