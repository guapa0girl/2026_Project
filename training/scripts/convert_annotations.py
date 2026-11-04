"""COCO-Keypoints 형식(JSON)으로 내보낸 원본 라벨을 YOLO-Pose txt 형식으로
변환해서 data/raw/label/ 에 저장한다.

CVAT, Label Studio, Roboflow 등 대부분의 라벨링 툴은 COCO keypoints JSON을
내보내기(export) 옵션으로 지원한다. 라벨링 툴이 YOLO-Pose txt를 직접
내보낼 수 있다면 이 스크립트 없이 data/raw/label/ 에 바로 넣으면 된다.

전제 조건:
  - COCO json의 categories[0].keypoints 순서가 configs/pig_pose.yaml 의
    keypoint_names 순서와 동일해야 한다. 순서가 다르면 --keypoint-order 로
    COCO 순서를 그대로 적어서 자동 재정렬시켜야 한다.

사용법:
    python scripts/convert_annotations.py --coco data/raw/annotations/result.json
"""
import argparse
import json
from pathlib import Path

from utils.paths import DATA_YAML, RAW_ANNOTATIONS_DIR, RAW_IMAGES_DIR, RAW_LABELS_DIR
from utils.skeleton import load_pose_spec


def build_reorder_index(coco_kpt_names, target_kpt_names):
    """COCO json의 키포인트 순서를 configs/pig_pose.yaml 순서로 맞추기 위한 인덱스."""
    if coco_kpt_names == target_kpt_names:
        return list(range(len(target_kpt_names)))
    missing = set(target_kpt_names) - set(coco_kpt_names)
    if missing:
        raise ValueError(
            f"COCO json에 없는 키포인트 이름: {missing}. "
            "--keypoint-order 로 COCO 원본 순서를 명시하거나 이름을 맞춰주세요."
        )
    return [coco_kpt_names.index(name) for name in target_kpt_names]


def main():
    parser = argparse.ArgumentParser(description="COCO-Keypoints JSON -> YOLO-Pose txt 변환")
    parser.add_argument("--coco", type=Path, required=True, help="COCO keypoints json 경로")
    parser.add_argument(
        "--keypoint-order",
        nargs="+",
        default=None,
        help="COCO categories[0].keypoints 순서를 강제 지정 (지정 안 하면 json의 값을 그대로 사용)",
    )
    args = parser.parse_args()

    spec = load_pose_spec(DATA_YAML)
    target_names = spec["keypoint_names"]
    num_kpts = spec["num_keypoints"]

    with open(args.coco, "r", encoding="utf-8") as f:
        coco = json.load(f)

    coco_kpt_names = args.keypoint_order or coco["categories"][0]["keypoints"]
    if len(coco_kpt_names) != num_kpts:
        raise ValueError(
            f"COCO 키포인트 개수({len(coco_kpt_names)})가 kpt_shape({num_kpts})와 다릅니다."
        )
    reorder = build_reorder_index(coco_kpt_names, target_names)

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    RAW_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    written, missing_images = 0, 0

    for image_id, anns in anns_by_image.items():
        img_info = images_by_id[image_id]
        file_name = Path(img_info["file_name"]).name
        img_w, img_h = img_info["width"], img_info["height"]

        if not (RAW_IMAGES_DIR / file_name).exists():
            missing_images += 1

        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw, nh = w / img_w, h / img_h

            kp = ann["keypoints"]  # flat [x1,y1,v1, x2,y2,v2, ...] in COCO order
            kp = [kp[i * 3:i * 3 + 3] for i in range(len(kp) // 3)]
            kp_reordered = [kp[i] for i in reorder]

            parts = [0, cx, cy, nw, nh]  # class_id 0 = pig
            for kx, ky, kv in kp_reordered:
                parts += [kx / img_w, ky / img_h, int(kv)]
            lines.append(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in parts))

        label_path = RAW_LABELS_DIR / f"{Path(file_name).stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        written += 1

    print(f"[convert] {written}개 라벨 파일을 {RAW_LABELS_DIR} 에 저장했습니다.")
    if missing_images:
        print(f"[convert] 경고: 대응하는 원본 이미지를 data/raw/img 에서 찾지 못한 항목 {missing_images}건")


if __name__ == "__main__":
    main()
