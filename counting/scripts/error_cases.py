"""가림 · 밀집 오차 사례 그림 (논문 한계점용).

1) test(000) 10개 분할(k5_f0~f4, rk5_f0~f4) 결과에서 사람 visiting 개체 중 포즈 모델이 못 찾은(검출 안 됨) 횟수를
   사진별로 집계 → missed_visiting_by_image.csv
2) 고른 사진마다: 왼쪽 = 사람 라벨(visiting 검정 점선 박스, other 흰 박스), 오른쪽 = 포즈 모델 검출 + RF+Geo+Leg 판정
   (visiting 진한 파랑, other 회색), 포즈 모델이 못 찾은 사람 visiting은 검정 점선 + "missed". 원 = 사료통 ROI.
   보여 주는 포즈 모델 = 그 사진에서 visiting을 놓친 첫 분할(시간대 우선).
입력: outputs/eval/split_<split>/stage2_instances.csv (run_split_eval.ps1), data/labels/state_000.xml, data/raw/000_img
3) 고른 사진마다 10개 분할 모델 결과를 나란히 (all_models_<no>.png, 사료통 주변 확대) + all_models_summary.csv
출력: reports/fold_study/error_cases/ (missed_visiting_by_image.csv, error_cases.png, all_models_<no>.png,
      all_models_summary.csv, error_cases.md)

사용법 (counting/ 에서):  python scripts/error_cases.py [--images 57 76]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_methods as em  # noqa: E402
from feeding_monitor.method_features import featurize_det  # noqa: E402

SPLITS = [f"k5_f{i}" for i in range(5)] + [f"rk5_f{i}" for i in range(5)]
OUT = ROOT / "reports" / "fold_study" / "error_cases"
C_VIS, C_OTHER = "#08519c", "#8c8c8c"


def missed_table():
    rows = []
    for s in SPLITS:
        d = pd.read_csv(ROOT / "outputs" / "eval" / f"split_{s}" / "stage2_instances.csv", encoding="utf-8-sig")
        d["split"] = s
        rows.append(d)
    d = pd.concat(rows)
    v = d[d.label == 1]
    g = v.groupby("image").agg(gt_visiting=("label", "size"), missed_undetected=("matched", lambda s: int((~s).sum())),
                               judged_other_rf_geo_leg=("pred_rf_geo_leg", lambda s: 0))
    g["gt_visiting"] //= len(SPLITS)
    g["judged_other_rf_geo_leg"] = v[v.matched].groupby("image")["pred_rf_geo_leg"].apply(lambda s: int((s == 0).sum()))
    g = g.fillna(0).astype(int)
    g["n_models"] = len(SPLITS)
    g.insert(0, "no", [int(i.split("_")[0]) for i in g.index])
    first = {}
    for img in g.index:
        hit = [s for s in SPLITS if not d[(d.split == s) & (d.image == img) & (d.label == 1)].matched.all()]
        first[img] = hit[0] if hit else ""
    g["first_split_missed"] = pd.Series(first)
    return g.sort_values(["missed_undetected", "judged_other_rf_geo_leg"], ascending=False)


def draw(ax_gt, ax_pr, img_path, gt_rows, split, cams):
    import cv2
    from ultralytics import YOLO
    from feeding_monitor.pose_tracker import _parse_detections
    cfg, zones = cams.cfg["SAMPLE"], cams.zones["SAMPLE"]
    frame = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    w = REPO / "training" / "models" / "results" / f"ep200_{split}" / "m" / "weights" / "best.pt"
    res = YOLO(str(w)).predict(frame, conf=cfg.model.conf, imgsz=cfg.model.imgsz, iou=cfg.model.iou, verbose=False)[0]
    preds = _parse_detections(res, 0)
    rf = rf_model(split)
    feats = pd.DataFrame([featurize_det(p, cfg, zones) for p in preds])
    vis_pred = rf[0].predict(feats[rf[1]]) if len(feats) else []
    pairs = em.match([r["det"] for r in gt_rows], preds, 0.25)
    matched_gt = {i for i, _ in pairs}

    for ax in (ax_gt, ax_pr):
        ax.imshow(rgb)
        ax.add_patch(Circle((cfg.feeder.cx, cfg.feeder.cy), cfg.feeder.r, fill=False, ec="#ffffff", lw=1.2, ls=":"))
        ax.set_xticks([])
        ax.set_yticks([])
    for r in gt_rows:
        x1, y1, x2, y2 = r["det"].box
        vis = r["label"] == 1
        ax_gt.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#000000" if vis else "#ffffff",
                                  lw=2.4 if vis else 1.0, ls="--" if vis else "-"))
    kinds = []
    for i, r in enumerate(gt_rows):
        if r["label"] == 1 and i not in matched_gt:
            x1, y1, x2, y2 = r["det"].box
            best = max((iou(r["det"].box, p.box) for p in preds), default=0.0)
            kind = "keypoints off" if best >= 0.5 else "no detection"   # 박스는 맞는데 keypoint 매칭 실패 vs 아예 못 찾음
            kinds.append(kind)
            ax_pr.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#000000", lw=2.4, ls="--"))
            ax_pr.text(x1 + 3, y1 + 3, f"missed: {kind}", va="top", fontsize=9, color="#000000",
                       bbox=dict(fc="#ffffff", ec="none", pad=1.5))
    for p, v in zip(preds, vis_pred):
        x1, y1, x2, y2 = p.box
        ax_pr.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=C_VIS if v else C_OTHER,
                                  lw=2.2 if v else 1.0))
    n_vis = sum(r["label"] for r in gt_rows)
    return len(gt_rows), n_vis, len(preds), int(np.sum(vis_pred)), kinds


_POSE = {}


def model_result(frame, split, cams):
    """그 분할 포즈 모델 검출 + RF+Geo+Leg 판정 (평가와 같은 설정)."""
    from ultralytics import YOLO
    from feeding_monitor.pose_tracker import _parse_detections
    cfg, zones = cams.cfg["SAMPLE"], cams.zones["SAMPLE"]
    if split not in _POSE:
        _POSE[split] = YOLO(str(REPO / "training" / "models" / "results" / f"ep200_{split}" / "m" / "weights" / "best.pt"))
    res = _POSE[split].predict(frame, conf=cfg.model.conf, imgsz=cfg.model.imgsz, iou=cfg.model.iou, verbose=False)[0]
    preds = _parse_detections(res, 0)
    clf, fc = rf_model(split)
    feats = pd.DataFrame([featurize_det(p, cfg, zones) for p in preds])
    vis = clf.predict(feats[fc]) if len(feats) else np.array([])
    return preds, vis


def plot_all_models(no, stem, img_path, gt_rows, cams, out_path):
    """한 사진 × 10개 분할 모델 나란히. 1열 = 사람 라벨, 2~6열 = f0~f4, 위 = 시간대, 아래 = 랜덤.
    사료통 주변을 잘라서 보여 줌 (visiting 판정이 일어나는 곳)."""
    import cv2
    cfg = cams.cfg["SAMPLE"]
    frame = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    vis_boxes = [r["det"].box for r in gt_rows if r["label"] == 1]
    xs = [cfg.feeder.cx - cfg.feeder.r, cfg.feeder.cx + cfg.feeder.r] + [b[0] for b in vis_boxes] + [b[2] for b in vis_boxes]
    ys = [cfg.feeder.cy - cfg.feeder.r, cfg.feeder.cy + cfg.feeder.r] + [b[1] for b in vis_boxes] + [b[3] for b in vis_boxes]
    m = 30
    x0, x1 = max(0, min(xs) - m), min(W, max(xs) + m)
    y0, y1 = max(0, min(ys) - m), min(H, max(ys) + m)
    n_vis = sum(r["label"] for r in gt_rows)

    def base(ax):
        ax.imshow(rgb)
        ax.add_patch(Circle((cfg.feeder.cx, cfg.feeder.cy), cfg.feeder.r, fill=False, ec="#ffffff", lw=1.0, ls=":"))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_xticks([])
        ax.set_yticks([])

    fig, axes = plt.subplots(2, 6, figsize=(19, 7.2))
    rows = [("Time 5-fold", [f"k5_f{i}" for i in range(5)]), ("Random 5-fold", [f"rk5_f{i}" for i in range(5)])]
    summary = []
    for r_i, (lab, splits) in enumerate(rows):
        for c_i, split in enumerate(splits):
            ax = axes[r_i, c_i + 1]
            base(ax)
            preds, vis = model_result(frame, split, cams)
            pairs = em.match([g["det"] for g in gt_rows], preds, 0.25)
            mg = {i for i, _ in pairs}
            kinds = []
            for i, g in enumerate(gt_rows):
                if g["label"] == 1 and i not in mg:
                    b = g["det"].box
                    kind = "kpts off" if max((iou(b, p.box) for p in preds), default=0) >= 0.5 else "no det."
                    kinds.append(kind)
                    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#000000",
                                           lw=2.0, ls="--"))
                    ax.text(b[0] + 2, b[1] + 2, kind, va="top", fontsize=7, bbox=dict(fc="#ffffff", ec="none", pad=1))
            for p, v in zip(preds, vis):
                b = p.box
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                       ec=C_VIS if v else C_OTHER, lw=2.0 if v else 0.9))
            cnt = int(np.sum(vis))
            ok = cnt == n_vis                     # 빨강 = 센 마릿수가 틀림 (매칭 실패가 있어도 마릿수가 맞으면 검정)
            note = "" if not kinds else (f"\nmissed {len(kinds)}" if not ok else
                                         f"\nmissed {len(kinds)}, count right by chance")
            ax.set_title(f"{lab} f{c_i}: counted {cnt} / GT {n_vis}" + note,
                         fontsize=9, loc="left", color="#000000" if ok else "#b2182b",
                         fontweight="normal" if ok else "bold")
            summary.append({"image_no": no, "scheme": lab.lower().replace(" 5-fold", "_5fold"), "fold": f"f{c_i}", "split": split,
                            "gt_visiting": n_vis, "counted_visiting": cnt, "detected_pigs": len(preds),
                            "missed_no_detection": kinds.count("no det."), "missed_kpts_off": kinds.count("kpts off")})
    for r_i in range(2):
        ax = axes[r_i, 0]
        base(ax)
        for g in gt_rows:
            b = g["det"].box
            v = g["label"] == 1
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#000000" if v else "#ffffff",
                                   lw=2.0 if v else 0.8, ls="--" if v else "-"))
        ax.set_title(f"Human labels: {n_vis} visiting", fontsize=9, loc="left")
        ax.set_ylabel(rows[r_i][0], fontsize=11, fontweight="bold")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#000000", lw=2.0, ls="--", label="Human visiting (missed by model if drawn on a model panel)"),
         Line2D([], [], color=C_VIS, lw=2.0, label="Model: counted visiting (RF+Geo+Leg)"),
         Line2D([], [], color=C_OTHER, lw=0.9, label="Model: other"),
         Line2D([], [], color="#bbbbbb", lw=1.0, ls=":", label="Feeder ROI")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9)
    n_wrong = sum(s["counted_visiting"] != n_vis for s in summary)
    fig.suptitle(f"Test image #{no}: all 10 pose models (+ RF+Geo+Leg), cropped around the feeder — "
                 f"count wrong in {n_wrong} of 10 (red titles)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return summary


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


_RF = {}


def rf_model(split):
    if split not in _RF:
        import json
        cams = em.Cameras(ROOT / "configs" / "eval_cameras.yaml")
        rows = [r for x in ("state_001.xml", "state_002.xml") for r in em.parse_cvat_xml(ROOT / "data" / "labels" / x)]
        s1 = em.featurize([r for r in rows if cams.has(r["camera"])], cams)
        keep = {s.split("__", 1)[1] for s in
                json.loads((REPO / "training" / "data" / "splits" / f"{split}.json").read_text(encoding="utf-8"))["train"]}
        s1 = s1[s1["image"].isin(keep)]
        feats = em.METHOD_FEATURES["rf_geo_leg"]
        _RF[split] = (em.new_rf(42).fit(s1[feats], s1["label"]), feats)
    return _RF[split]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", type=int, default=[57, 76], help="000 사진 번호 (1~86)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    g = missed_table()
    g.to_csv(OUT / "missed_visiting_by_image.csv", encoding="utf-8-sig")
    cams = em.Cameras(ROOT / "configs" / "eval_cameras.yaml")
    gt_all = em.parse_cvat_xml(ROOT / "data" / "labels" / "state_000.xml")
    imgs = {p.stem: p for p in (ROOT / "data" / "raw" / "000_img").iterdir()}

    fig, axes = plt.subplots(len(a.images), 2, figsize=(12, 4.6 * len(a.images)))
    axes = np.atleast_2d(axes)
    md = ["# 가림 · 밀집 오차 사례 (test 000)", "",
          f"사람 visiting 개체를 포즈 모델이 못 찾은 횟수 (10개 분할 모델 합계, 사진별): `missed_visiting_by_image.csv`", "",
          "| # | 사람 visiting | 못 찾음 (10개 모델 합) | 찾았지만 RF+Geo+Leg가 other로 판정 |", "|---|---|---|---|"]
    for _, r in g[g.missed_undetected > 0].head(10).iterrows():
        md.append(f"| {r.no} | {r.gt_visiting} | {r.missed_undetected} / {r.gt_visiting * r.n_models} | "
                  f"{r.judged_other_rf_geo_leg} |")
    md += ["", "'못 찾음' = 평가 코드(keypoint 평균거리 / 박스 대각선 ≤ 0.25로 매칭)에서 매칭되지 않은 사람 visiting. "
           "그림에서는 박스 IoU ≥ 0.5인 검출이 있으면 'keypoints off'(검출은 했지만 keypoint가 어긋나 매칭 실패, "
           "그 검출은 '라벨 없는 검출'로 따로 셈), 없으면 'no detection'(아예 못 찾음)으로 나눔.", "", "## 그림", ""]
    for k, no in enumerate(a.images):
        stem = next(s for s in g.index.union(pd.Index([i for i in imgs])) if s.startswith(f"{no:04d}_"))
        split = g.loc[stem, "first_split_missed"] if stem in g.index and g.loc[stem, "first_split_missed"] else "k5_f0"
        rows = [r for r in gt_all if r["image"] == stem]
        n_gt, n_vis, n_pred, n_pvis, kinds = draw(axes[k, 0], axes[k, 1], imgs[stem], rows, split, cams)
        miss = g.loc[stem, "missed_undetected"] if stem in g.index else 0
        axes[k, 0].set_title(f"#{no} human labels: {n_gt} pigs, {n_vis} visiting (black dashed)", fontsize=10, loc="left")
        axes[k, 1].set_title(f"pose model ep200_{split} + RF+Geo+Leg: {n_pred} detected, {n_pvis} counted visiting\n"
                             f"visiting pigs missed by pose model in {miss} of {n_vis * len(SPLITS)} (10 models)",
                             fontsize=10, loc="left")
        md.append(f"- #{no} ({stem}): 사람 {n_gt}마리 중 visiting {n_vis}, 포즈 모델(ep200_{split}) 검출 {n_pred}, "
                  f"visiting으로 셈 {n_pvis}. 10개 분할 모델에서 visiting을 못 찾은 횟수 {miss} / {n_vis * len(SPLITS)}. "
                  f"그림의 못 찾은 visiting: {', '.join(kinds) or '없음'}.")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#000000", lw=2.4, ls="--", label="Human: visiting"),
         Line2D([], [], color="#bbbbbb", lw=1.0, label="Human: other (white box)"),
         Line2D([], [], color=C_VIS, lw=2.2, label="Model: counted visiting"),
         Line2D([], [], color=C_OTHER, lw=1.0, label="Model: other"),
         Line2D([], [], color="#bbbbbb", lw=1.2, ls=":", label="Feeder ROI")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Error cases on test (000): visiting pigs missed by the pose model in crowded / occluded feeder scenes",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(OUT / "error_cases.png", dpi=150)
    plt.close(fig)
    md += ["", "![error cases](error_cases.png)", "",
           "## 10개 분할 모델 나란히 (사료통 주변 확대)", "",
           "1열 = 사람 라벨, 위 줄 = Time 5-fold f0~f4, 아래 줄 = Random 5-fold f0~f4. 빨간 제목 = 센 마릿수가 GT와 다름. "
           "모델 칸의 검정 점선 = 그 모델이 매칭하지 못한 사람 visiting (no det. = 검출 없음, kpts off = 검출은 됐지만 "
           "keypoint가 어긋남).", "",
           "| # | 분할 | f0 | f1 | f2 | f3 | f4 |", "|---|---|---|---|---|---|---|"]
    allsum = []
    for no in a.images:
        stem = next(s for s in imgs if s.startswith(f"{no:04d}_"))
        rows = [r for r in gt_all if r["image"] == stem]
        s = plot_all_models(no, stem, imgs[stem], rows, cams, OUT / f"all_models_{no:02d}.png")
        allsum += s
        for sch in ("time_5fold", "random_5fold"):
            cells = [f"{x['counted_visiting']}/{x['gt_visiting']}" +
                     (f" (no det. {x['missed_no_detection']}, kpts off {x['missed_kpts_off']})"
                      if x["missed_no_detection"] + x["missed_kpts_off"] else "")
                     for x in s if x["scheme"] == sch]
            md.append(f"| {no} | {'Time 5-fold' if sch == 'time_5fold' else 'Random 5-fold'} | " + " | ".join(cells) + " |")
    pd.DataFrame(allsum).to_csv(OUT / "all_models_summary.csv", index=False, encoding="utf-8-sig")
    md += ["", "칸 = 센 visiting / 사람 visiting (괄호 = 매칭 못 한 사람 visiting 종류별 수).", ""]
    md += [f"![#{no} all models](all_models_{no:02d}.png)" for no in a.images] + [""]
    (OUT / "error_cases.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
