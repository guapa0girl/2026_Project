"""yolo_det 공통 경로·상수.

폴더 구조 (레포 루트 기준):
    yolo_det/
    ├── configs/pig_det.yaml        클래스 이름 (0 other / 1 visiting)
    ├── scripts/                    prepare_data.py → train.py → evaluate.py
    ├── data/                       (자동 생성, git 제외) train/valid/test 이미지·라벨, gt_counts.csv
    ├── models/<run>/<size>/        (자동 생성, git 제외) 학습 결과, weights/best.pt
    ├── outputs/<run>/<size>/       (자동 생성, git 제외) 평가 상세 csv
    └── reports/<run>/<size>/       평가 요약 json·md (git에 올림)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # yolo_det/
REPO = ROOT.parent                                  # 2026_Project/

CONFIG_YAML = ROOT / "configs" / "pig_det.yaml"
DATA_DIR = ROOT / "data"
RESOLVED_YAML = DATA_DIR / "_pig_det.resolved.yaml"  # 절대경로가 들어가서 머신마다 다름 → data/ 안에 둠(git 제외)
GT_COUNTS_CSV = DATA_DIR / "gt_counts.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"

# 원본: 서인 레포 규칙 그대로 counting/data/labels/source/<task>_sorted.zip (CVAT for images 1.1)
DEFAULT_SOURCE_DIR = REPO / "counting" / "data" / "labels" / "source"

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 클래스 번호: counting/scripts/evaluate_methods.py 의 STATE_MAP(visiting=1, other=0)과 같은 규칙
CLS_OTHER, CLS_VISITING = 0, 1
CLASS_NAMES = {CLS_OTHER: "other", CLS_VISITING: "visiting"}

# counting/scripts/evaluate_methods.py 의 STATE_MAP 복사본 (CH010 task의 오타 "visitig" 포함).
# 그쪽에 값이 추가되면 여기도 같이 고칠 것.
STATE_MAP = {
    "visitig(feeding)": CLS_VISITING,
    "visiting(feeding)": CLS_VISITING,
    "visiting": CLS_VISITING,
    "other": CLS_OTHER,
    "unlabeling": None,
    "unlabeled": None,
}

# 사이즈별 batch: training/configs/train_*.yaml (pose 실험)과 같게
SIZE_BATCH = {"n": 16, "s": 16, "m": 8, "l": 8}
