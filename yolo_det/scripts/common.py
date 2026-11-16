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
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # yolo_det/
REPO = ROOT.parent                                  # 2026_Project/

CONFIG_YAML = ROOT / "configs" / "pig_det.yaml"
# 데이터 폴더: 기본 data/. fold를 동시에 돌릴 때는 환경변수로 따로 줌 (예: YOLO_DET_DATA=data_k5f0)
_DATA_NAME = os.environ.get("YOLO_DET_DATA", "data")
if not _DATA_NAME.startswith("data") or "/" in _DATA_NAME or "\\" in _DATA_NAME:
    raise SystemExit(f"YOLO_DET_DATA={_DATA_NAME!r} — 'data'로 시작하는 폴더 이름만 (git 제외 규칙 yolo_det/data*/)")
DATA_DIR = ROOT / _DATA_NAME
RESOLVED_YAML = DATA_DIR / "_pig_det.resolved.yaml"  # 절대경로가 들어가서 머신마다 다름 → data/ 안에 둠(git 제외)
GT_COUNTS_CSV = DATA_DIR / "gt_counts.csv"
SPLIT_JSON = DATA_DIR / "split.json"                  # 분할 방식·valid 목록 (prepare_data.py가 씀)


def sha256_16(path) -> str:
    """파일 sha256 앞 16자리 (가중치 파일이 같은지 확인용)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def env_info() -> dict:
    """재현용 실행 환경: 라이브러리 버전, GPU, 레포 git 커밋."""
    import platform
    import subprocess
    info = {"python": platform.python_version()}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"확인 실패: {e}"
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception as e:  # noqa: BLE001
        info["ultralytics"] = f"확인 실패: {e}"
    try:
        run = lambda *c: subprocess.run(["git", *c], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
        info["git_commit"] = run("rev-parse", "--short", "HEAD") or "알 수 없음"
        info["git_uncommitted_yolo_det"] = bool(run("status", "--porcelain", "--", "yolo_det/scripts"))
    except Exception:  # noqa: BLE001
        info["git_commit"] = "알 수 없음"
    return info
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"

# 원본: 레포 공통 규칙 그대로 counting/data/labels/source/<task>_sorted.zip (CVAT for images 1.1)
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
