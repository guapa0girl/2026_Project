"""RF+Geo → RF+Geo+Leg에서 무엇이 바뀌었나 — 특징 중요도 비교와 test(000) 개체별 특징 값.

평가(evaluate_methods.py --stage2-train-split, run_split_eval.ps1)와 똑같이 fold마다
  - RF / RF+Geo / RF+Geo+Leg를 그 fold train 사진의 사람 관절로 학습 (나무 100개, balanced, seed 42)
  - 그 fold 포즈 모델(ep200_<fold>/m best.pt, conf 0.4, NMS 0.5)로 test(000)를 추론해 특징 계산 · 판정
  - 사람 개체와 짝짓기(관절 평균 거리 ÷ 사람 박스 대각선 ≤ 0.25, 헝가리안)
를 다시 돌려, 평가 때 저장하지 않은 '개체별 특징 값'과 '모델별 Gini 중요도'를 남긴다.
재현 확인: 여기서 센 visiting→other 오류 수가 fold_metrics.csv의 missed_visiting_judged와 같아야 함.

출력: reports/fold_study/paper/
    rf_importance_by_method.csv    fold × 방법 × 특징 Gini 중요도
    test_instances_features.csv    fold × 사람 개체(짝지어진 것) × 특징 값 · 방법별 판정
    leg_effect.md                  요약 (중요도 비교, 판정이 바뀐 개체, 학습 vs test 특징 분포)
사용법 (counting/ 에서, GPU 권장, 사진 · 라벨 · 가중치 필요):  python scripts/analyze_leg_effect.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import evaluate_methods as em  # noqa: E402
from feeding_monitor.method_features import GEO_FEATURES, LEG_FEATURES, METHOD_FEATURES, RF_FEATURES  # noqa: E402

OUT = ROOT / "reports" / "fold_study" / "paper"
SPLITS = [f"rk5_f{i}" for i in range(5)] + [f"k5_f{i}" for i in range(5)]
CONF_SIGNALS = ["geo_limb_visibility", "geo_aspect_ratio", "geo_lateral_asymmetry"]   # 신뢰도 기반 누움 신호 3개
GROUP = {**{f: "RF" for f in RF_FEATURES}, **{f: "Geo" for f in GEO_FEATURES}, **{f: "Leg" for f in LEG_FEATURES}}


def main():
    import cv2
    from feeding_monitor.pose_tracker import PoseTracker, _parse_detections
    sys.stdout.reconfigure(encoding="utf-8")
    labels = ROOT / "data" / "labels"
    cams = em.Cameras(ROOT / "configs" / "eval_cameras.yaml")
    hum = [r for x in ("state_001.xml", "state_002.xml") for r in em.parse_cvat_xml(labels / x)]
    s1 = em.featurize([r for r in hum if cams.has(r["camera"])], cams)
    gt_rows = [r for r in em.parse_cvat_xml(labels / "state_000.xml") if cams.has(r["camera"])]
    images = em.find_images(ROOT / "data" / "raw" / "000_img")
    by_image = {}
    for r in gt_rows:
        by_image.setdefault(r["image"], []).append(r)

    imp_rows, inst_rows = [], []
    for split in SPLITS:
        keep = {s.split("__", 1)[1] for s in json.loads(
            (REPO / "training" / "data" / "splits" / f"{split}.json").read_text(encoding="utf-8"))["train"]}
        tr_df = s1[s1["image"].isin(keep)]
        models = {}
        for m, feats in METHOD_FEATURES.items():
            clf = em.new_rf(42).fit(tr_df[feats], tr_df["label"])
            models[m] = (clf, feats)
            imp_rows += [{"split": split, "method": m, "feature": f, "group": GROUP[f], "importance": v}
                         for f, v in zip(feats, clf.feature_importances_)]
        weights = REPO / "training" / "models" / "results" / f"ep200_{split}" / "m" / "weights" / "best.pt"
        for c in cams.cfg.values():
            c.model.weights = str(weights)
        trackers = {}
        for stem, rows in sorted(by_image.items()):
            cam = rows[0]["camera"]
            cfg, zones = cams.cfg[cam], cams.zones[cam]
            if cam not in trackers:
                trackers[cam] = PoseTracker(cfg.model)
            tr = trackers[cam]
            res = tr.model.predict(cv2.imread(str(images[stem])), conf=cfg.model.conf, imgsz=cfg.model.imgsz,
                                   iou=cfg.model.iou, device=tr.device, verbose=False)[0]
            preds = _parse_detections(res, 0)
            if not preds:
                continue
            pf = em.featurize([{"image": stem, "camera": cam, "clip": rows[0]["clip"], "label": -1, "det": d,
                                "source": "model"} for d in preds], cams)
            for m, (clf, feats) in models.items():
                pf[f"pred_{m}"] = clf.predict(pf[feats])
                pf[f"prob_{m}"] = clf.predict_proba(pf[feats])[:, 1]
            for i, j in em.match([r["det"] for r in rows], preds, 0.25):
                rec = {"split": split, "image": stem, "gt_label": rows[i]["label"]}
                rec.update({c: pf.iloc[j][c] for c in RF_FEATURES + GEO_FEATURES + LEG_FEATURES})
                rec.update({c: pf.iloc[j][c] for c in pf.columns if c.startswith(("pred_", "prob_"))})
                inst_rows.append(rec)
        print(f"[{split}] done", flush=True)

    imp = pd.DataFrame(imp_rows)
    inst = pd.DataFrame(inst_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    imp.to_csv(OUT / "rf_importance_by_method.csv", index=False, encoding="utf-8-sig")
    inst.to_csv(OUT / "test_instances_features.csv", index=False, encoding="utf-8-sig")

    # ---- 재현 확인: visiting→other(짝지어진 사람 visiting을 other로 판정) fold 평균
    fm = pd.read_csv(ROOT / "reports" / "fold_study" / "fold_metrics.csv", encoding="utf-8-sig")
    md = ["# RF+Geo → RF+Geo+Leg: 무엇이 바뀌었나", "",
          "평가와 같은 학습 · 추론 · 짝짓기를 다시 돌려 개체별 특징 값과 Gini 중요도를 저장함 (`scripts/analyze_leg_effect.py`).", "",
          "## 재현 확인 (test(000), visiting → other, fold 평균 마릿수)", "",
          "| 분할 | 방법 | 이번 계산 | fold_metrics.csv |", "|---|---|---|---|"]
    vis = inst[inst.gt_label == 1]
    for pre, lab in [("rk5", "Random 5-fold"), ("k5", "Time 5-fold")]:
        for m in ["geo", "rf", "rf_geo", "rf_geo_leg"]:
            mine = vis[vis.split.str.startswith(pre + "_")].groupby("split")[f"pred_{m}"].apply(lambda s: (s == 0).sum())
            ref = fm[(fm.split.str.startswith(pre + "_")) & (fm.set == "test") & (fm.method == m)].missed_visiting_judged
            md.append(f"| {lab} | {m} | {mine.mean():.1f} | {ref.mean():.1f} |")

    # ---- 중요도 비교 (fold 10개 평균)
    g = imp.groupby(["method", "feature"]).importance.mean().unstack(0)
    md += ["", "## Gini 중요도 (fold 10개 평균)", "", "| 특징 | 그룹 | RF | RF+Geo | RF+Geo+Leg |", "|---|---|---|---|---|"]
    for f in RF_FEATURES + GEO_FEATURES + LEG_FEATURES:
        cell = lambda m: "–" if pd.isna(g.loc[f].get(m, np.nan)) else f"{g.loc[f][m]:.3f}"  # noqa: E731
        md.append(f"| {f} | {GROUP[f]} | {cell('rf')} | {cell('rf_geo')} | {cell('rf_geo_leg')} |")
    grp = imp.groupby(["method", "group"]).importance.sum().unstack(1) / len(SPLITS)
    md += ["", "그룹 합계: " + "; ".join(f"{m}: " + ", ".join(f"{k} {v:.3f}" for k, v in grp.loc[m].dropna().items())
                                     for m in ["rf", "rf_geo", "rf_geo_leg"]), ""]
    conf_sum = {m: g.loc[CONF_SIGNALS, m].sum() for m in ["rf_geo", "rf_geo_leg"]}
    md.append(f"신뢰도 기반 누움 신호 3개 합계: RF+Geo {conf_sum['rf_geo']:.3f} → RF+Geo+Leg {conf_sum['rf_geo_leg']:.3f}")

    # ---- 판정이 바뀐 사람 visiting 개체
    flip_fix = vis[(vis.pred_rf_geo == 0) & (vis.pred_rf_geo_leg == 1)]      # RF+Geo가 놓친 것을 RF+Geo+Leg가 맞힘
    flip_break = vis[(vis.pred_rf_geo == 1) & (vis.pred_rf_geo_leg == 0)]
    ok_both = vis[(vis.pred_rf_geo == 1) & (vis.pred_rf_geo_leg == 1)]
    md += ["", "## 사람 visiting 개체 중 두 방법의 판정이 갈린 경우 (fold 10개 합)", "",
           f"- RF+Geo는 other, RF+Geo+Leg는 visiting(고침): {len(flip_fix)}개",
           f"- RF+Geo는 visiting, RF+Geo+Leg는 other(망침): {len(flip_break)}개",
           f"- 둘 다 visiting: {len(ok_both)}개", "",
           "### 특징 값 비교 (중앙값)", "",
           "| 특징 | 학습(사람 관절) visiting | test 둘 다 맞힘 | test RF+Geo만 틀림 |", "|---|---|---|---|"]
    tr_vis = s1[s1.label == 1]
    for f in CONF_SIGNALS + ["geo_nose_dist_over_r", "nose_zone_dist_norm", "leg_angle_deg", "paw_offset"]:
        md.append(f"| {f} | {tr_vis[f].median():.3f} | {ok_both[f].median():.3f} | "
                  f"{flip_fix[f].median() if len(flip_fix) else float('nan'):.3f} |")
    md += ["", "학습은 사람 관절이라 보이는 관절의 신뢰도가 모두 1.0이고, test는 포즈 모델이 낸 신뢰도임.", ""]

    # ---- 학습 데이터의 visiting / other 특징 분포 (사료통 근처 other = 코-사료통 거리 ÷ 반지름 < 1.2)
    near_other = s1[(s1.label == 0) & (s1.geo_nose_dist_over_r < 1.2)]
    md += ["## 학습 데이터(사람 관절)의 특징 분포 (중앙값)", "",
           "| 특징 | visiting | other 전체 | 사료통 근처 other (거리 ÷ r < 1.2) | test에서 고친 visiting |", "|---|---|---|---|---|"]
    for f in ["geo_nose_dist_over_r", "geo_aspect_ratio", "leg_angle_deg", "paw_offset", "leg_side_bias", "leg_extension"]:
        md.append(f"| {f} | {s1[s1.label == 1][f].median():.3f} | {s1[s1.label == 0][f].median():.3f} | "
                  f"{near_other[f].median():.3f} | {flip_fix[f].median() if len(flip_fix) else float('nan'):.3f} |")

    # ---- 코 위치별 정확도 (test(000), fold 10개 합): 가장자리 = 거리 ÷ r 0.85~1.1
    edge_v = vis[vis.geo_nose_dist_over_r.between(0.85, 1.1)]
    inner_v = vis[vis.geo_nose_dist_over_r < 0.85]
    oth = inst[inst.gt_label == 0]
    edge_o = oth[oth.geo_nose_dist_over_r.between(0.85, 1.1)]
    md += ["", "## test(000) 코 위치별 판정 (fold 10개 합, 사람 개체와 짝지어진 검출)", "",
           "| 구간 | 개체 수 | 지표 | RF+Geo | RF+Geo+Leg |", "|---|---|---|---|---|",
           f"| visiting, 코가 사료통 안쪽 (거리 ÷ r < 0.85) | {len(inner_v)} | visiting으로 맞힌 비율 | "
           f"{inner_v.pred_rf_geo.mean():.3f} | {inner_v.pred_rf_geo_leg.mean():.3f} |",
           f"| visiting, 코가 사료통 가장자리 (0.85~1.1) | {len(edge_v)} | visiting으로 맞힌 비율 | "
           f"{edge_v.pred_rf_geo.mean():.3f} | {edge_v.pred_rf_geo_leg.mean():.3f} |",
           f"| other, 코가 사료통 가장자리 (0.85~1.1) | {len(edge_o)} | visiting으로 잘못 센 비율 | "
           f"{edge_o.pred_rf_geo.mean():.3f} | {edge_o.pred_rf_geo_leg.mean():.3f} |", "",
           f"고친 {len(flip_fix)}건의 분할별 수: " + ", ".join(f"{k} {v}" for k, v in
                                                       flip_fix.split.str.replace(r'_f\d', '', regex=True).value_counts().items())
           + f" (fold당 Random 5-fold {(flip_fix.split.str.startswith('rk5')).sum() / 5:.1f}, "
             f"Time 5-fold {(flip_fix.split.str.startswith('k5')).sum() / 5:.1f}마리).",
           f"고친 개체의 visiting 확률 중앙값: RF+Geo {flip_fix.prob_rf_geo.median():.2f} → "
           f"RF+Geo+Leg {flip_fix.prob_rf_geo_leg.median():.2f} (판정 기준 0.5).", ""]
    (OUT / "leg_effect.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
