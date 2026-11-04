"""data/processed/{train,valid,test}의 (사람이 라벨링한) 키포인트로 급식
자세 특징을 뽑고, 규칙 기반 분류(scripts/utils/posture.py)로 1차 라벨을
붙여서 reports/posture/rule_based_dataset.csv 에 저장한다.

모델 추론이 아니라 **정답(ground-truth) 키포인트**를 쓴다 — 이미 라벨링된
데이터라 GPU 없이 빠르고, 나중에 진짜 정답(사람이 "섭식/비섭식"을 눈으로
보고 붙인 라벨)이 생기면 이 CSV의 rule_label 컬럼과 바로 비교해서 규칙
threshold를 튜닝할 수 있다.

이 CSV가 scripts/train_posture_classifier.py의 입력이 된다 — 규칙 기반
라벨을 "약한 정답"(weak label)으로 삼아 RandomForest를 학습시키는 방식으로
1번(규칙)과 2번(ML) 방법을 결합한다.

사용법:
    python scripts/label_posture_rule_based.py
    python scripts/label_posture_rule_based.py --dip-thresh 60 --dist-thresh 0.4
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from utils.label_io import read_yolo_pose_label
from utils.paths import CONFIGS_DIR, DATA_YAML, PROCESSED_DIR, REPORTS_DIR, SPLITS
from utils.posture import FEATURE_NAMES, extract_features, load_zones, rule_based_classify
from utils.skeleton import load_pose_spec

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
OUT_CSV = REPORTS_DIR / "posture" / "rule_based_dataset.csv"


def imread_unicode(path):
    """cv2.imread는 Windows에서 한글 등 비-ASCII 경로를 못 읽는 경우가 있어
    바이트로 직접 읽어서 디코드한다 (data/processed의 파일명 다수가 한글 포함)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    parser = argparse.ArgumentParser(description="규칙 기반 급식 자세 라벨링으로 부트스트랩 데이터셋 생성")
    parser.add_argument("--zones", type=Path, default=CONFIGS_DIR / "feeder_zones.yaml")
    parser.add_argument("--dip-thresh", type=float, default=55.0, help="head_dip_deg 임계값(도)")
    parser.add_argument("--dist-thresh", type=float, default=0.35, help="nose_zone_dist_norm 임계값")
    args = parser.parse_args()

    spec = load_pose_spec(DATA_YAML)
    num_kpts = spec["num_keypoints"]
    zones = load_zones(args.zones)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {"eating": 0, "not_eating": 0}
    for split in SPLITS:
        img_dir = PROCESSED_DIR / split / "images"
        lbl_dir = PROCESSED_DIR / split / "labels"
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            boxes, kpts, vis = read_yolo_pose_label(label_path, num_kpts)
            if len(boxes) == 0:
                continue

            h, w = imread_unicode(img_path).shape[:2]
            for i in range(len(boxes)):
                cx, cy, bw, bh = boxes[i]
                bbox_px = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                kpts_px = kpts[i] * [w, h]

                feat = extract_features(kpts_px, vis[i], bbox_px, zones)
                label = rule_based_classify(feat, dip_thresh_deg=args.dip_thresh, dist_thresh=args.dist_thresh)
                counts["eating" if label else "not_eating"] += 1

                row = {"split": split, "image": img_path.name, "instance": i}
                row.update(feat)
                row["rule_label"] = int(label)
                rows.append(row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "instance"] + FEATURE_NAMES + ["rule_label"])
        writer.writeheader()
        writer.writerows(rows)

    total = counts["eating"] + counts["not_eating"]
    print(f"[posture] {total}개 인스턴스 처리 -> {OUT_CSV}")
    print(f"[posture] 규칙 기반 분류: 섭식 {counts['eating']}건 ({counts['eating']/total*100:.1f}%), "
          f"비섭식 {counts['not_eating']}건 ({counts['not_eating']/total*100:.1f}%)")
    print("[posture] 다음 단계: python scripts/train_posture_classifier.py")


if __name__ == "__main__":
    main()
