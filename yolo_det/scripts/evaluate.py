"""YOLO detect의 visiting 판정을 Geo / RF 계열과 같은 기준(Stage 2, test = 000)으로 평가.

사진당 visiting 마릿수 = conf ≥ 임계값인 visiting(cls 1) 박스 수
    * agnostic NMS: 한 마리에 visiting 박스와 other 박스가 겹쳐 남지 않게 클래스 구분 없이 NMS
    * 임계값은 valid(28장)의 사진당 마릿수 MAE가 가장 작은 값으로 고르고 test에 고정 적용
      (test를 보고 고르지 않음). --conf 로 직접 지정 가능.
정답 = 사람 라벨(data/gt_counts.csv, CVAT XML의 State 기준 visiting 마릿수)

지표 = counting/scripts/evaluate_methods.py 의 count_metrics() 를 그대로 import
    사진당 MAE / bias / 정확 일치율 / ±1 이내 / 총 점유량 오차(%) + 구간(프레임 번호 6등분 ≈ 10초) 집계 오차
진단 = 사람 박스와 IoU ≥ 0.5로 짝지어
    missed_visiting_undetected(못 찾음) / missed_visiting_judged(찾았지만 other) /
    wrong_visiting(other를 visiting) / extra_visiting(사람 라벨에 없는 박스를 visiting으로 셈)
참고 = Ultralytics box mAP50 / mAP50-95 (클래스별)

사용법 (레포 루트에서):
    python yolo_det/scripts/evaluate.py --run ep200 --size m
    python yolo_det/scripts/evaluate.py --run ep200 --size m --conf 0.4      # 임계값 직접 지정
결과: outputs/<run>/<size>/ (상세 csv, git 제외), reports/<run>/<size>/ (summary.json·md, git에 올림)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASS_NAMES, CLS_VISITING, DATA_DIR, GT_COUNTS_CSV, IMAGE_EXTS, MODELS_DIR,  # noqa: E402
                    OUTPUTS_DIR, REPO, REPORTS_DIR, RESOLVED_YAML)


def load_count_metrics():
    """서인 평가 코드의 count_metrics를 그대로 사용 (지표 정의를 한 곳에서만 관리)."""
    path = REPO / "counting" / "scripts" / "evaluate_methods.py"
    spec = importlib.util.spec_from_file_location("evaluate_methods", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.count_metrics


def frame_index(stem: str):
    """evaluate_methods.frame_index 와 같은 규칙 (원래 이미지 이름 끝의 frame_<번호>)."""
    m = re.search(r"frame_(\d+)$", stem)
    return int(m.group(1)) if m else None


def predict_split(model, split: str, imgsz: int, iou: float, device):
    """split의 모든 이미지 → {stem: (cls[N], conf[N], xyxyn[N,4])}. 낮은 conf까지 받아 두고 임계값은 나중에 적용."""
    imgs = sorted(p for p in (DATA_DIR / split / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    out = {}
    for p in imgs:
        r = model.predict(str(p), conf=0.01, iou=iou, agnostic_nms=True, imgsz=imgsz,
                          device=device, verbose=False)[0]
        b = r.boxes
        out[p.stem] = (b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy(), b.xyxyn.cpu().numpy())
    return out


def visiting_counts(preds: dict, thr: float) -> dict:
    return {s: int(((c == CLS_VISITING) & (f >= thr)).sum()) for s, (c, f, _) in preds.items()}


def choose_threshold(preds_valid: dict, gt: pd.DataFrame, count_metrics) -> tuple[float, pd.DataFrame]:
    rows = []
    for thr in np.round(np.arange(0.05, 0.951, 0.05), 2):
        cnt = visiting_counts(preds_valid, thr)
        g = gt.set_index("image").loc[list(cnt)]
        rows.append({"conf": thr, **count_metrics(g.n_visiting.values, list(cnt.values()))})
    scan = pd.DataFrame(rows)
    # MAE 최소 → 같으면 |bias| 작은 쪽 → 그래도 같으면 Ultralytics 기본값 0.25에 가까운 쪽
    best = scan.assign(abs_bias=scan.count_bias.abs(), d=(scan.conf - 0.25).abs()) \
               .sort_values(["count_MAE", "abs_bias", "d"]).iloc[0]
    return float(best.conf), scan


def read_gt_boxes(split: str, stem: str):
    """사람 박스 (cls, xyxyn) — prepare_data.py가 쓴 라벨."""
    f = DATA_DIR / split / "labels" / f"{stem}.txt"
    cls, boxes = [], []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c, cx, cy, w, h = map(float, line.split())
            cls.append(int(c))
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(cls, int), np.array(boxes, float).reshape(-1, 4)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def diagnose(preds: dict, split: str, thr: float, iou_thr: float = 0.5) -> dict:
    from scipy.optimize import linear_sum_assignment
    d = {"missed_visiting_undetected": 0, "missed_visiting_judged": 0, "wrong_visiting": 0, "extra_visiting": 0}
    for stem, (c, f, xy) in preds.items():
        keep = f >= thr
        pc, pxy = c[keep], xy[keep]
        gc, gxy = read_gt_boxes(split, stem)
        iou = iou_matrix(gxy, pxy)
        pairs = []
        if iou.size:
            r, k = linear_sum_assignment(-iou)
            pairs = [(i, j) for i, j in zip(r, k) if iou[i, j] >= iou_thr]
        mg, mp = {i for i, _ in pairs}, {j for _, j in pairs}
        d["missed_visiting_undetected"] += int(sum(gc[i] == CLS_VISITING for i in range(len(gc)) if i not in mg))
        for i, j in pairs:
            d["missed_visiting_judged"] += int(gc[i] == CLS_VISITING and pc[j] != CLS_VISITING)
            d["wrong_visiting"] += int(gc[i] != CLS_VISITING and pc[j] == CLS_VISITING)
        d["extra_visiting"] += int(sum(pc[j] == CLS_VISITING for j in range(len(pc)) if j not in mp))
    return d


def bin_metrics(frames: pd.DataFrame, n_bins: int) -> tuple[dict, pd.DataFrame | None]:
    """evaluate_methods.run_stage2 의 구간 집계와 같은 계산 (프레임 번호 범위를 n_bins 등분)."""
    idx = frames.orig_image.map(frame_index)
    if idx.isna().any() or n_bins <= 1:
        print("[evaluate] 이미지 이름에 frame_<번호>가 없어 구간 집계는 건너뜀")
        return {}, None
    lo, hi = idx.min(), idx.max() + 1
    frames = frames.assign(bin=((idx - lo) * n_bins // (hi - lo)).astype(int))
    agg = frames.groupby("bin").agg(n_frames=("image", "size"), human_mean=("human_visiting", "mean"),
                                    yolo_mean=("yolo_visiting", "mean"))
    e = agg.yolo_mean - agg.human_mean
    return {"bin_MAE": float(e.abs().mean()), "bin_bias": float(e.mean()),
            "bin_max_abs_err": float(e.abs().max())}, agg


def box_map(model, imgsz, device) -> dict:
    """참고용: Ultralytics 공식 box mAP (test split)."""
    try:
        m = model.val(data=str(RESOLVED_YAML), split="test", imgsz=imgsz, device=device, plots=False, verbose=False)
        per_cls = {CLASS_NAMES[i]: float(m.box.maps[i]) for i in CLASS_NAMES}
        return {"box_mAP50": float(m.box.map50), "box_mAP50-95": float(m.box.map),
                **{f"box_mAP50-95_{k}": v for k, v in per_cls.items()}}
    except Exception as e:  # mAP는 참고값이라 실패해도 본 평가는 진행
        print(f"[evaluate][경고] box mAP 계산 실패: {e}")
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="ep200")
    ap.add_argument("--size", default="m")
    ap.add_argument("--weights", default=None, help="기본: models/<run>/<size>/weights/best.pt")
    ap.add_argument("--conf", type=float, default=None, help="visiting 박스 conf 임계값 (기본: valid에서 선택)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU (Ultralytics predict 기본값)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--bins", type=int, default=6, help="구간 집계 수 (evaluate_methods.py와 같게 6 ≈ 10초)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    weights = Path(a.weights) if a.weights else MODELS_DIR / a.run / a.size / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"{weights} 가 없음 — 먼저 train.py 실행")
    gt = pd.read_csv(GT_COUNTS_CSV, dtype={"task": str})
    count_metrics = load_count_metrics()

    from ultralytics import YOLO
    model = YOLO(str(weights))

    out = OUTPUTS_DIR / a.run / a.size
    rep = REPORTS_DIR / a.run / a.size
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)

    # 1) 임계값: valid에서 선택
    if a.conf is None:
        thr, scan = choose_threshold(predict_split(model, "valid", a.imgsz, a.iou, a.device), gt, count_metrics)
        scan.to_csv(out / "valid_conf_scan.csv", index=False, encoding="utf-8-sig")
        thr_note = f"valid {len(gt[gt.split == 'valid'])}장에서 사진당 MAE 최소"
    else:
        thr, thr_note = a.conf, "직접 지정(--conf)"
    print(f"[evaluate] visiting conf 임계값 = {thr:.2f} ({thr_note})")

    # 2) test(000) 평가
    preds = predict_split(model, "test", a.imgsz, a.iou, a.device)
    cnt = visiting_counts(preds, thr)
    frames = gt[gt.split == "test"].set_index("image").loc[list(cnt)].reset_index()
    frames = frames.rename(columns={"n_visiting": "human_visiting"})
    frames["yolo_visiting"] = frames.image.map(cnt)
    frames["yolo_pigs"] = frames.image.map({s: int((f >= thr).sum()) for s, (_, f, _) in preds.items()})

    metrics = count_metrics(frames.human_visiting, frames.yolo_visiting)
    bins, agg = bin_metrics(frames, a.bins)
    diag = diagnose(preds, "test", thr)
    mp = box_map(model, a.imgsz, a.device)

    frames.to_csv(out / "test_frames.csv", index=False, encoding="utf-8-sig")
    if agg is not None:
        agg.to_csv(out / "test_bins.csv", encoding="utf-8-sig")

    summary = {"method": "YOLO", "run": a.run, "size": a.size, "weights": weights.name,
               "conf_threshold": thr, "conf_threshold_from": thr_note, "nms": f"agnostic, iou {a.iou}",
               "test": {**metrics, **bins, **diag}, "box_map_test": mp}
    (rep / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    m = summary["test"]
    md = [f"# YOLO 직접 판정 — {a.run} / {a.size} (test = 000, 사진 {m['n_frames']}장)", "",
          f"- 가중치: `models/{a.run}/{a.size}/weights/{weights.name}`, visiting conf ≥ {thr:.2f} ({thr_note}), "
          f"NMS {summary['nms']}", "",
          "| 방법 | 사진당 MAE (bias) | 정확 일치 | ±1 이내 | 총 점유량 오차 | 구간 평균 오차 |",
          "|---|---|---|---|---|---|",
          f"| YOLO ({a.size}) | {m['count_MAE']:.3f} ({m['count_bias']:+.3f}) | {m['exact_rate']:.1%} | "
          f"{m['within1_rate']:.1%} | {m['total_error_pct']:+.1f}% | {m.get('bin_MAE', float('nan')):.3f} |", "",
          f"- 총 visiting: YOLO {m['total_model']} / 사람 {m['total_human']}, "
          f"많이 센 사진 {m['over_frames']} / 적게 센 사진 {m['under_frames']}",
          f"- 진단(IoU ≥ 0.5): 못 찾은 visiting {m['missed_visiting_undetected']}, "
          f"찾았지만 other로 판정 {m['missed_visiting_judged']}, other를 visiting으로 {m['wrong_visiting']}, "
          f"사람 라벨에 없는 박스를 visiting으로 {m['extra_visiting']}"]
    if mp:
        md.append("- 참고 box mAP (test): " + ", ".join(f"{k} {v:.3f}" for k, v in mp.items()))
    (rep / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n[saved] {rep}  (상세: {out})")


if __name__ == "__main__":
    main()
