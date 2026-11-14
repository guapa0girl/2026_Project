"""YOLOv8 detect (2클래스: other / visiting) 학습.

pose 실험(training/scripts/train.py, PROGRESS.md 2장)과 조건을 맞춤:
    COCO 사전학습 yolov8{n,s,m,l}.pt 에서 매번 처음부터 / imgsz 640 / epoch 200 / 조기 종료 끔(patience 0)
    / optimizer auto / batch n·s 16, m·l 8 / best.pt = valid 성능 최고 epoch

결과: yolo_det/models/<run>/<size>/weights/best.pt  (run 이름으로 설정별 결과를 나란히 보관)

사용법 (레포 루트에서, prepare_data.py 먼저):
    python yolo_det/scripts/train.py --sizes m                    # medium만
    python yolo_det/scripts/train.py --sizes n s m l --run ep200  # 전부
    python yolo_det/scripts/train.py --sizes m --resume           # 끊긴 학습 이어서
    python yolo_det/scripts/train.py --sizes n --epochs 1 --run smoke --device cpu   # 동작 확인용

    # 서인 pose 가중치에서 시작 (backbone·neck만 옮기고 detect head는 새로 학습). 크기가 같아야 함.
    python yolo_det/scripts/train.py --sizes m --run ep200_poseinit --init yolo_det/pig_pose_best.pt --device 0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS_DIR, RESOLVED_YAML, SIZE_BATCH  # noqa: E402


def check_init_size(init: str, size: str):
    """--init 가중치의 모델 크기(n/s/m/l)가 --sizes 와 같은지 확인. 다르면 층 모양이 안 맞아 거의 안 옮겨짐."""
    from ultralytics import YOLO
    src = YOLO(init)
    scale = (src.model.yaml or {}).get("scale")
    print(f"[init] {init}: task={src.task}, 크기={scale or '알 수 없음'}")
    if scale and scale != size:
        raise SystemExit(f"--init 가중치 크기({scale})와 --sizes({size})가 다름")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", default=["m"], choices=list(SIZE_BATCH))
    ap.add_argument("--run", default="ep200", help="결과 폴더 이름 (models/<run>/<size>)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=0, help="0 = 조기 종료 끔")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=None, help="기본: 사이즈별 pose 실험 값")
    ap.add_argument("--device", default=None, help="예: 0 / cpu (기본: Ultralytics 자동)")
    ap.add_argument("--workers", type=int, default=8, help="Windows에서 데이터로더 오류가 나면 0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="models/<run>/<size>/weights/last.pt 에서 이어서")
    ap.add_argument("--init", default=None,
                    help="시작 가중치 (예: 서인 pose best.pt). 없으면 COCO yolov8<size>.pt. "
                         "모양이 맞는 층(backbone·neck)만 옮기고 head는 새로 학습")
    a = ap.parse_args()

    if a.init:
        if not Path(a.init).exists():
            raise SystemExit(f"--init {a.init} 파일이 없음")
        if len(a.sizes) != 1:
            raise SystemExit("--init 은 사이즈 하나만 (가중치와 같은 크기): 예) --sizes m")
        if a.run == "ep200":
            raise SystemExit("--init 결과가 COCO 결과(models/ep200)를 덮어쓰지 않게 --run 을 따로 주세요: 예) --run ep200_poseinit")

    if not RESOLVED_YAML.exists():
        raise SystemExit(f"{RESOLVED_YAML} 가 없음 — 먼저 python yolo_det/scripts/prepare_data.py 실행")

    from ultralytics import YOLO

    for size in a.sizes:
        last = MODELS_DIR / a.run / size / "weights" / "last.pt"
        if a.resume and last.exists():
            print(f"\n===== yolov8{size} ({a.run}) 이어서 학습: {last} =====")
            YOLO(str(last)).train(resume=True)
            continue

        if a.init:
            check_init_size(a.init, size)
            print(f"\n===== yolov8{size} ({a.run}) {a.init} 에서 시작해 학습 =====")
            # detect 구조를 새로 만든 뒤 모양이 맞는 층만 옮김 (YOLO(a.init)로 열면 pose 모델로 학습돼 버림)
            model = YOLO(f"yolov8{size}.yaml").load(a.init)
        else:
            print(f"\n===== yolov8{size} ({a.run}) COCO 사전학습에서 처음부터 학습 =====")
            model = YOLO(f"yolov8{size}.pt")       # 없으면 Ultralytics가 자동 다운로드
        model.train(
            data=str(RESOLVED_YAML),
            epochs=a.epochs,
            patience=a.patience,
            imgsz=a.imgsz,
            batch=a.batch or SIZE_BATCH[size],
            optimizer="auto",
            seed=a.seed,
            workers=a.workers,
            device=a.device,
            project=str(MODELS_DIR / a.run),
            name=size,
            exist_ok=True,
        )
        print(f"[train] 완료 → {MODELS_DIR / a.run / size / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
