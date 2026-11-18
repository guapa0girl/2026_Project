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
모델: fold마다 그 fold의 모델로 잼 (기본: 랜덤 5-fold rk5_f0~f4 + 시간대 5-fold k5_f0~f4, 모델 10개)
      포즈 = training/models/results/ep200_<split>/m/weights/best.pt, RF = 그 fold train 사진으로 학습 (평가와 동일)
      YOLO = --yolo-weights 경로 틀 ('{split}'이 fold 이름으로 바뀜), conf = 그 fold conf
             (reports/fold_study/inputs/yolo_fold_metrics.csv의 test 행). 없으면 YOLO는 건너뜀.
집계: fold마다 사진 1장 시간의 중앙값 · p95를 구하고, 분할 방식(랜덤 5-fold / 시간대 5-fold)마다
      fold 5개의 평균 ± 표본 표준편차로 보고. 실시간 판정은 fold 5개 p95 평균 기준(최대 p95도 같이 표기).
실시간 기준 (p95): 1초 1프레임 운용 ≤ 1000 ms, 30 fps 전체 프레임 ≤ 33.3 ms
출력: reports/speed/<장비 이름>/ (speed_raw.csv, speed_folds.csv, speed_summary.csv, speed.md)

사용법 (저장소 루트 또는 counting/ 에서):
    python counting/scripts/benchmark_speed.py                       # GPU 0 + CPU, 모델 10개
    python counting/scripts/benchmark_speed.py --devices 0 cpu --yolo-weights "yolo_det/models/ep200_{split}/m/weights/best.pt"
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
    ap.add_argument("--splits", nargs="+", default=[f"rk5_f{i}" for i in range(5)] + [f"k5_f{i}" for i in range(5)],
                    help="잴 fold (기본: 랜덤 5-fold + 시간대 5-fold, 모델 10개)")
    ap.add_argument("--devices", nargs="+", default=["0", "cpu"])
    ap.add_argument("--images", default=str(ROOT / "data" / "raw" / "000_img"))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--yolo-weights", default=None, help="YOLO best.pt 경로 틀, '{split}' 자리에 fold 이름")
    ap.add_argument("--yolo-iou", type=float, default=0.7)
    ap.add_argument("--out", default=str(ROOT / "reports" / "speed"))
    a = ap.parse_args()

    import torch
    import ultralytics
    images = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    yconf = {}
    if a.yolo_weights:
        t = pd.read_csv(ROOT / "reports" / "fold_study" / "inputs" / "yolo_fold_metrics.csv", encoding="utf-8-sig")
        t = t[t.set == "test"]
        yconf = {("rk5" if s == "random_5fold" else "k5") + f"_{f}": c for s, f, c in zip(t.scheme, t.fold, t.conf)}
    print(f"[speed] 사진 {len(images)}장 × {a.runs}회, 예열 {a.warmup}장 제외, fold {len(a.splits)}개: {' '.join(a.splits)}")

    rows = []
    for split in a.splits:
        weights = REPO / "training" / "models" / "results" / f"ep200_{split}" / "m" / "weights" / "best.pt"
        cams, models = train_rfs(split)
        for dev in a.devices:
            dev = dev if dev == "cpu" else int(dev)
            name = device_name(dev)
            base = {"split": split, "scheme": "random_5fold" if split.startswith("rk5_") else "time_5fold",
                    "fold": split[-2:], "device": "GPU" if dev != "cpu" else "CPU", "device_name": name}
            print(f"[speed] {split} · {name} — 포즈 + Geo · RF 계열")
            rows += [{**base, **r} for r in bench_pose(images, dev, weights, cams, models, a.warmup, a.runs)]
            if a.yolo_weights:
                print(f"[speed] {split} · {name} — YOLO (conf {yconf[split]})")
                rows += [{**base, **r} for r in bench_yolo(images, dev, a.yolo_weights.format(split=split),
                                                             yconf[split], a.yolo_iou, a.warmup, a.runs)]
    raw = pd.DataFrame(rows)

    gpu = next((n for n in raw.device_name.unique() if raw[raw.device_name == n].device.iloc[0] == "GPU"), "nogpu")
    out = Path(a.out) / "".join(c if c.isalnum() else "_" for c in gpu).strip("_")
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "speed_raw.csv", index=False, encoding="utf-8-sig")

    stages = ("read", "infer", "feature", "classify")
    folds = []                                                   # fold(모델) 하나당 중앙값 · p95
    for (sch, fold, split, dev, m), g in raw.groupby(["scheme", "fold", "split", "device", "method"], sort=False):
        folds.append({"scheme": sch, "fold": fold, "split": split, "device": dev, "device_name": g.device_name.iloc[0],
                      "method": m, "n": len(g), "median_ms": g.total_ms.median(), "p95_ms": g.total_ms.quantile(0.95),
                      **{f"{s}_median_ms": g[f"{s}_ms"].median() for s in stages}})
    folds = pd.DataFrame(folds)
    folds.to_csv(out / "speed_folds.csv", index=False, encoding="utf-8-sig")

    summ = []                                                    # 분할 방식마다 fold 5개 평균 ± 표본 표준편차
    for (sch, dev, m), g in folds.groupby(["scheme", "device", "method"], sort=False):
        rec = {"scheme": sch, "device": dev, "device_name": g.device_name.iloc[0], "method": m, "n_folds": len(g),
               "median_ms": g.median_ms.mean(), "median_ms_sd": g.median_ms.std(ddof=1),
               "p95_ms": g.p95_ms.mean(), "p95_ms_sd": g.p95_ms.std(ddof=1), "p95_ms_max": g.p95_ms.max(),
               **{f"{s}_median_ms": g[f"{s}_median_ms"].mean() for s in stages}}
        for lab, lim in RT:
            rec[f"realtime_{lab.split()[0]}{lab.split()[1]}"] = rec["p95_ms"] <= lim
        summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "speed_summary.csv", index=False, encoding="utf-8-sig")

    env = (f"Python {platform.python_version()} / torch {torch.__version__} / ultralytics {ultralytics.__version__} / "
           f"CPU {device_name('cpu')}" + (f" / GPU {gpu}" if gpu != "nogpu" else ""))
    md = ["# 추론 시간 (end-to-end, 사진 1장) — Random 5-fold / Time 5-fold", "",
          f"환경: {env}", "",
          f"조건: batch 1, imgsz 640, 예열 {a.warmup}장 제외, 000 {len(images)}장 × {a.runs}회, fold마다 그 fold의 모델 "
          f"(포즈 `ep200_<fold>` best.pt conf 0.4, RF = 그 fold train 사진으로 학습). 사진 읽기부터 마릿수까지.", "",
          "값 = fold 5개(모델 5개) 각각의 중앙값 · p95를 평균 ± 표본 표준편차. 단계별 칸은 fold 중앙값의 평균.", "",
          "| 분할 | 장비 | 방법 | 중앙값 (ms) | p95 (ms) | p95 최대 (fold) | 읽기 | 모델 추론 | 특징 계산 | 판정 | 1 fps (≤1000 ms) | 30 fps (≤33.3 ms) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {m: i for i, m in enumerate(list(LABEL))}
    slab = {"random_5fold": "Random 5-fold", "time_5fold": "Time 5-fold"}
    summ["_o"] = summ.method.map(order)
    for _, r in summ.sort_values(["scheme", "device", "_o"], ascending=[True, False, True]).iterrows():
        ok = ["✅" if r.p95_ms <= lim else "❌" for _, lim in RT]
        md.append(f"| {slab[r.scheme]} | {r.device} ({r.device_name}) | {LABEL[r.method]} | "
                  f"{r.median_ms:.1f} ± {r.median_ms_sd:.1f} | {r.p95_ms:.1f} ± {r.p95_ms_sd:.1f} | {r.p95_ms_max:.1f} | "
                  f"{r.read_median_ms:.1f} | {r.infer_median_ms:.1f} | {r.feature_median_ms:.1f} | "
                  f"{r.classify_median_ms:.2f} | {ok[0]} | {ok[1]} |")
    md += ["", "'모델 추론' = 포즈 또는 YOLO predict(전처리 · NMS · 후처리 포함). "
           "'특징 계산' = 검출된 돼지 전부의 특징 + Geo 판정. '판정' = RF predict (Geo는 특징 계산에 포함, 0).",
           "실시간 판정은 fold 5개 p95 평균 기준. fold별 값은 `speed_folds.csv`.", ""]
    if not a.yolo_weights:
        md += ["YOLO는 이 장비에서 재지 않음 (`--yolo-weights` 없음). 같은 장비 비교를 위해 YOLO 가중치로 다시 재거나, "
               "수아 서버에서 이 스크립트로 Geo · RF 계열을 같이 재야 함.", ""]
    (out / "speed.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("saved", out)


if __name__ == "__main__":
    main()
