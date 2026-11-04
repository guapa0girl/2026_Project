"""이 프로젝트 전체에서 공유하는 경로 상수 모음."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DIR / "img"
RAW_LABELS_DIR = RAW_DIR / "label"
RAW_ANNOTATIONS_DIR = RAW_DIR / "annotations"
PROCESSED_DIR = DATA_DIR / "processed"
DATA_PAST_TRAIN_DIR = DATA_DIR / "past_train"

CONFIGS_DIR = ROOT / "configs"
DATA_YAML = CONFIGS_DIR / "pig_pose.yaml"
# Ultralytics는 data yaml의 'path'가 상대경로면 실행 시점의 CWD 기준으로 존재
# 여부를 확인하고, 없으면 전역 설정(datasets_dir)을 기준으로 재해석한다. 이
# 컴퓨터의 전역 datasets_dir/runs_dir가 이 프로젝트가 아닌 다른 프로젝트를
# 가리키고 있어서, CWD가 scripts/ 가 아닐 때 실행하면 엉뚱한 폴더를 읽거나
# 그쪽에 결과물을 쓰는 사고가 났다. 그래서 학습/평가 호출에는 이 절대경로
# resolved yaml(런타임에 생성, git에는 올리지 않음)을 항상 사용한다.
RESOLVED_DATA_YAML = CONFIGS_DIR / "_pig_pose.resolved.yaml"

MODELS_DIR = ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
RESULTS_DIR = MODELS_DIR / "results"

PAST_TRAIN_DIR = ROOT / "past_train"

REPORTS_DIR = ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
COMPARISON_DIR = REPORTS_DIR / "comparison"

VIDEO_RESULT_DIR = ROOT / "video_result"

SIZES = ("n", "s", "m", "l", "x")
SIZE_LABELS = {"n": "nano", "s": "small", "m": "medium", "l": "large", "x": "xlarge"}
PRETRAINED_WEIGHTS = {
    "n": "yolov8n-pose.pt",
    "s": "yolov8s-pose.pt",
    "m": "yolov8m-pose.pt",
    "l": "yolov8l-pose.pt",
    "x": "yolov8x-pose.pt",
}
SPLITS = ("train", "valid", "test")
