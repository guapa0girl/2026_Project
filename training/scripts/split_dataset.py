"""data/raw/{images,labels} 의 전체 누적 데이터셋을 train/valid/test 로 분할해서
data/processed/ 에 (다시) 채워 넣는다.

라벨링된 데이터가 추가될 때마다(예: 800장 -> 1200장) 이 스크립트를 다시 실행하면
data/processed/ 가 "현재 시점의 raw 전체"를 기준으로 새로 채워진다. 이 프로젝트는
파인튜닝이 아니라 매번 전체 재학습을 하므로, split도 매번 raw 전체를 기준으로
새로 만드는 것이 맞다(이전 split과 이어질 필요가 없음).

사용법:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --train 0.8 --valid 0.1 --test 0.1 --seed 42
"""
import argparse
import random
import shutil

from utils.label_io import clip_yolo_pose_label_file
from utils.paths import DATA_YAML, PROCESSED_DIR, RAW_IMAGES_DIR, RAW_LABELS_DIR, SPLITS
from utils.skeleton import load_pose_spec

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_pairs():
    """raw/img 의 각 이미지에 대응하는 raw/label/*.txt 가 있는지 확인.
    라벨이 없는 이미지는 건너뛰고 경고를 출력한다(아직 라벨링 안 된 원본)."""
    pairs = []
    skipped = []
    for img_path in sorted(RAW_IMAGES_DIR.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = RAW_LABELS_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            skipped.append(img_path.name)
    return pairs, skipped


def clear_processed():
    for split in SPLITS:
        for sub in ("images", "labels"):
            d = PROCESSED_DIR / split / sub
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.name != ".gitkeep":
                    f.unlink()


def place(pairs, split_name, use_hardlink, num_keypoints):
    """이미지는 (가능하면) 하드링크/복사로, 라벨은 항상 좌표를 [0,1]로 클립해서
    새로 써서 배치한다 (CVAT export 좌표가 이미지 경계를 벗어나는 인스턴스가
    많아 그대로 두면 Ultralytics가 학습에서 통째로 제외하기 때문)."""
    img_dir = PROCESSED_DIR / split_name / "images"
    lbl_dir = PROCESSED_DIR / split_name / "labels"
    for img_path, label_path in pairs:
        img_dst = img_dir / img_path.name
        lbl_dst = lbl_dir / label_path.name
        if use_hardlink:
            try:
                img_dst.hardlink_to(img_path)
            except (OSError, NotImplementedError):
                shutil.copy2(img_path, img_dst)
        else:
            shutil.copy2(img_path, img_dst)
        clip_yolo_pose_label_file(label_path, lbl_dst, num_keypoints)


def main():
    parser = argparse.ArgumentParser(description="raw 데이터셋을 train/valid/test 로 분할")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--valid", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardlink", action="store_true", help="복사 대신 하드링크로 배치(같은 드라이브에서 용량 절약)")
    args = parser.parse_args()

    ratio_sum = args.train + args.valid + args.test
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train+valid+test 비율의 합은 1.0 이어야 합니다 (현재 {ratio_sum})")

    pairs, skipped = find_pairs()
    if not pairs:
        raise SystemExit(
            "data/raw/img 와 data/raw/label 에서 짝이 맞는 파일을 찾지 못했습니다. "
            "README의 '원본 데이터 업로드 위치'를 확인하세요."
        )

    rng = random.Random(args.seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = round(n_total * args.train)
    n_valid = round(n_total * args.valid)

    train_pairs = shuffled[:n_train]
    valid_pairs = shuffled[n_train:n_train + n_valid]
    test_pairs = shuffled[n_train + n_valid:]

    num_keypoints = load_pose_spec(DATA_YAML)["num_keypoints"]

    clear_processed()
    place(train_pairs, "train", args.hardlink, num_keypoints)
    place(valid_pairs, "valid", args.hardlink, num_keypoints)
    place(test_pairs, "test", args.hardlink, num_keypoints)

    print(f"[split] 전체 {n_total}장 -> train {len(train_pairs)} / valid {len(valid_pairs)} / test {len(test_pairs)}")
    print("[split] 라벨 좌표를 [0,1] 범위로 클립해서 저장했습니다 (원본 data/raw/label은 그대로 유지).")
    if skipped:
        print(f"[split] 라벨이 없어 건너뛴 이미지 {len(skipped)}장 (예: {skipped[:5]})")


if __name__ == "__main__":
    main()
