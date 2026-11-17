"""추론 시간 측정 — 사진 읽기부터 visiting 마릿수까지 end-to-end (11/16 랩미팅 통일 조건).

조건: 같은 장비 한 대에서 모든 방법 측정 / batch 1 / imgsz 640 / 예열 5장 제외 / 000 86장 × 3회 /
      사진 1장 시간의 중앙값 · p95 (ms) / GPU와 CPU 각각.
측정 구간 (사진 1장):
    Geo · RF 계열 : 사진 읽기(cv2.imread) → 포즈 모델 추론(conf 0.4, imgsz 640) → 검출 파싱
                    → 특징 계산(featurize_det, Geo 판정 포함) → 판정(Geo 규칙 / RF predict) → 마릿수
    YOLO          : 사진 읽기 → YOLO detect 추론(agnostic NMS, iou 0.7, 그 fold conf) → visiting 박스 수
    특징 계산은 방법과 상관없이 모든 특징을 한 번에 구하므로(Geo 규칙 포함) 네 방법이 같은 값이고,
    방법별로 다른 것은 마지막 판정 시간뿐임. RF는 사진마다 그 사진의 돼지 전부를 한 번에 predict (n_jobs=1:
    한 장씩 처리할 때 스레드 생성 비용을 빼기 위함, 예측 결과는 같음).
모델: 포즈 = training/models/results/ep200_<split>/m/weights/best.pt, RF = 그 분할 train 사진으로 학습 (평가와 동일)
      YOLO = --yolo-weights (없으면 YOLO는 건너뜀. 수아 서버에서 같은 스크립트로 같이 재면 같은 GPU 조건)
실시간 기준 (p95): 1초 1프레임 운용 ≤ 1000 ms, 30 fps 전체 프레임 ≤ 33.3 ms
출력: reports/speed/<장비 이름>/ (speed_raw.csv, speed_summary.csv, speed.md)

사용법 (counting/ 에서):
    python scripts/benchmark_speed.py                                   # GPU 0 + CPU, 포즈 = ep200_k5_f0
    python scripts/benchmark_speed.py --devices 0 --yolo-weights <yolo best.pt> --yolo-conf 0.65
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_methods as em  # noqa: E402
from feeding_monitor.method_features import METHOD_FEATURES, METHODS, featurize_det  # noqa: E402

CLS_VISITING = 1          # yolo_det/scripts/common.py와 같음
LABEL = {"geo": "Geo", "rf": "RF", "rf_geo": "RF+Geo", "rf_geo_leg": "RF+Geo+Leg", "yolo": "YOLO"}
RT = [("1 fps (1000 ms)", 1000.0), ("30 fps (33.3 ms)", 1000 / 30)]


def sync(device):
    if device != "cpu":
        import torch
        torch.cuda.synchronize()


def device_name(device):
    if device == "cpu":
        try:                                                   # Windows: 실제 모델명 (platform.processor()는 계열 번호만 줌)
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            return platform.processor() or "CPU"
    import torch
    return torch.cuda.get_device_name(int(device))


def train_rfs(split):
    """평가와 똑같이: 001+002 사람 keypoint 개체 중 그 분할 train 사진으로 RF 학습."""
    cams = em.Cameras(ROOT / "configs" / "eval_cameras.yaml")
    rows = [r for x in ("state_001.xml", "state_002.xml") for r in em.parse_cvat_xml(ROOT / "data" / "labels" / x)]
    rows = [r for r in rows if cams.has(r["camera"])]
    s1 = em.featurize(rows, cams)
    lists = json.loads((REPO / "training" / "data" / "splits" / f"{split}.json").read_text(encoding="utf-8"))
    keep = {s.split("__", 1)[1] for s in lists["train"]}
    s1 = s1[s1["image"].isin(keep)]
    models = {}
    for m, feats in METHOD_FEATURES.items():
        clf = em.new_rf(42).fit(s1[feats], s1["label"])
        clf.n_jobs = 1
        models[m] = (clf, feats)
    return cams, models


def bench_pose(images, device, weights, cams, models, warmup, runs):
    import cv2
    from ultralytics import YOLO
    from feeding_monitor.pose_tracker import _parse_detections
    cfg, zones = cams.cfg["SAMPLE"], cams.zones["SAMPLE"]
    model = YOLO(str(weights))
    rows = []
    seq = images[:warmup] + images * runs
    for k, p in enumerate(seq):
        t0 = time.perf_counter()
        frame = cv2.imread(str(p))
        t1 = time.perf_counter()
        res = model.predict(frame, conf=cfg.model.conf, imgsz=cfg.model.imgsz, iou=cfg.model.iou,
                            device=device, verbose=False)[0]
        preds = _parse_detections(res, 0)
        sync(device)
        t2 = time.perf_counter()
        feats = pd.DataFrame([featurize_det(d, cfg, zones) for d in preds])
        t3 = time.perf_counter()
        cls_t, counts = {"geo": 0.0}, {"geo": int(feats["pred_geo"].sum()) if len(feats) else 0}
        for m, (clf, fc) in models.items():
            a = time.perf_counter()
            counts[m] = int(clf.predict(feats[fc]).sum()) if len(feats) else 0
            cls_t[m] = time.perf_counter() - a
        if k < warmup:
            continue
        for m in METHODS:
            rows.append({"method": m, "image": p.stem, "run": (k - warmup) // len(images),
                         "n_det": len(preds), "count": counts[m],
                         "read_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3, "feature_ms": (t3 - t2) * 1e3,
                         "classify_ms": cls_t[m] * 1e3,
                         "total_ms": (t3 - t0 + cls_t[m]) * 1e3})
    return rows


def bench_yolo(images, device, weights, conf, iou, warmup, runs):
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(weights))
    rows = []
    seq = images[:warmup] + images * runs
    for k, p in enumerate(seq):
        t0 = time.perf_counter()
        frame = cv2.imread(str(p))
        t1 = time.perf_counter()
        r = model.predict(frame, conf=conf, iou=iou, agnostic_nms=True, imgsz=640, device=device, verbose=False)[0]
        n = int((r.boxes.cls.cpu().numpy() == CLS_VISITING).sum())
        sync(device)
        t2 = time.perf_counter()
        if k < warmup:
            continue
        rows.append({"method": "yolo", "image": p.stem, "run": (k - warmup) // len(images), "n_det": len(r.boxes),
                     "count": n, "read_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3, "feature_ms": 0.0,
                     "classify_ms": 0.0, "total_ms": (t2 - t0) * 1e3})
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="k5_f0", help="포즈 모델 · RF 학습 분할 (속도는 분할과 무관, 기본 k5_f0)")
    ap.add_argument("--devices", nargs="+", default=["0", "cpu"])
    ap.add_argument("--images", default=str(ROOT / "data" / "raw" / "000_img"))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--yolo-conf", type=float, default=0.5)
    ap.add_argument("--yolo-iou", type=float, default=0.7)
    ap.add_argument("--out", default=str(ROOT / "reports" / "speed"))
    a = ap.parse_args()

    import torch
    import ultralytics
    images = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    weights = REPO / "training" / "models" / "results" / f"ep200_{a.split}" / "m" / "weights" / "best.pt"
    cams, models = train_rfs(a.split)
    print(f"[speed] 사진 {len(images)}장 × {a.runs}회, 예열 {a.warmup}장 제외, 포즈 {weights.relative_to(REPO)}")

    rows = []
    for dev in a.devices:
        dev = dev if dev == "cpu" else int(dev)
        name = device_name(dev)
        print(f"[speed] {name} — 포즈 + Geo · RF 계열")
        for r in bench_pose(images, dev, weights, cams, models, a.warmup, a.runs):
            rows.append({"device": "GPU" if dev != "cpu" else "CPU", "device_name": name, **r})
        if a.yolo_weights:
            print(f"[speed] {name} — YOLO")
            for r in bench_yolo(images, dev, a.yolo_weights, a.yolo_conf, a.yolo_iou, a.warmup, a.runs):
                rows.append({"device": "GPU" if dev != "cpu" else "CPU", "device_name": name, **r})
    raw = pd.DataFrame(rows)

    gpu = next((n for n in raw.device_name.unique() if raw[raw.device_name == n].device.iloc[0] == "GPU"), "nogpu")
    out = Path(a.out) / "".join(c if c.isalnum() else "_" for c in gpu).strip("_")
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "speed_raw.csv", index=False, encoding="utf-8-sig")

    summ = []
    for (dev, m), g in raw.groupby(["device", "method"], sort=False):
        rec = {"device": dev, "device_name": g.device_name.iloc[0], "method": m, "n": len(g),
               "median_ms": g.total_ms.median(), "p95_ms": g.total_ms.quantile(0.95), "mean_ms": g.total_ms.mean(),
               **{f"{s}_median_ms": g[f"{s}_ms"].median() for s in ("read", "infer", "feature", "classify")}}
        for lab, lim in RT:
            rec[f"realtime_{lab.split()[0]}{lab.split()[1]}"] = rec["p95_ms"] <= lim
        summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "speed_summary.csv", index=False, encoding="utf-8-sig")

    env = (f"Python {platform.python_version()} / torch {torch.__version__} / ultralytics {ultralytics.__version__} / "
           f"CPU {device_name('cpu')}" + (f" / GPU {gpu}" if gpu != "nogpu" else ""))
    md = ["# 추론 시간 (end-to-end, 사진 1장)", "",
          f"환경: {env}", "",
          f"조건: batch 1, imgsz 640, 예열 {a.warmup}장 제외, 000 {len(images)}장 × {a.runs}회, "
          f"포즈 = `ep200_{a.split}` best.pt (conf 0.4), RF = 같은 분할 train 사진으로 학습. "
          "사진 읽기부터 마릿수까지.", "",
          "| 장비 | 방법 | 중앙값 (ms) | p95 (ms) | 읽기 | 모델 추론 | 특징 계산 | 판정 | 1 fps (≤1000 ms) | 30 fps (≤33.3 ms) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    order = {m: i for i, m in enumerate(list(LABEL))}
    for _, r in summ.sort_values(["device", "method"], key=lambda s: s.map(order) if s.name == "method" else s,
                                 ascending=[False, True]).iterrows():
        ok = ["✅" if r.p95_ms <= lim else "❌" for _, lim in RT]
        md.append(f"| {r.device} ({r.device_name}) | {LABEL[r.method]} | {r.median_ms:.1f} | {r.p95_ms:.1f} | "
                  f"{r.read_median_ms:.1f} | {r.infer_median_ms:.1f} | {r.feature_median_ms:.1f} | "
                  f"{r.classify_median_ms:.2f} | {ok[0]} | {ok[1]} |")
    md += ["", "단계별 칸은 중앙값(ms). '모델 추론' = 포즈 또는 YOLO predict(전처리 · NMS · 후처리 포함). "
           "'특징 계산' = 검출된 돼지 전부의 특징 + Geo 판정. '판정' = RF predict (Geo는 특징 계산에 포함, 0).",
           "실시간 판정은 p95 기준.", ""]
    if not a.yolo_weights:
        md += ["YOLO는 이 장비에서 재지 않음 (`--yolo-weights` 없음). 같은 장비 비교를 위해 YOLO 가중치로 다시 재거나, "
               "수아 서버에서 이 스크립트로 Geo · RF 계열을 같이 재야 함.", ""]
    (out / "speed.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("saved", out)


if __name__ == "__main__":
    main()
