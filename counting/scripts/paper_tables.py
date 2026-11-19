"""논문용 지표 모음 — compare_folds.py 결과(reports/fold_study/)를 읽어 표로 정리.

입력 (compare_folds.py가 먼저 돌아 있어야 함):
    reports/fold_study/fold_metrics.csv          분할 × fold × val/test × 방법 지표
    reports/fold_study/per_frame_000/<m>.csv      000 사진별 카운트 (gt, k5_f0..f4, rk5_f0..f4)
    reports/fold_study/pose_keypoints.csv         keypoint별 PCK@0.1
    ../training/reports/metrics/ep200_<split>/m/summary.json   포즈 모델 지표
출력: reports/fold_study/paper/
    ../speed/<GPU>/speed_summary.csv              추론 시간 (benchmark_speed.py, 표 8)
    paper_metrics.md        표 1~11 (아래)
    count_metrics.csv       표 1·2 (방법 × 분할 × val/test, 평균·표본 표준편차)
    bootstrap_ci.csv        표 3 (test MAE 95% CI)
    wilcoxon.csv            표 4 (사진 단위 짝비교)
    pose_metrics.csv        표 6
지표 정의:
    사진 단위 — MAE, RMSE, ME(bias = 평균(pred − gt)), Acc(정확 일치율), Acc±1, TCE(총 마릿수 오차 %),
                10초 구간 MAE (test만)
    개체 단위 visiting (오류 분류에서 계산. 사람 개체와 짝짓기: Geo · RF 계열 = 관절 평균 거리 ÷ 사람 박스 대각선
        ≤ 0.25, YOLO = 박스 IoU ≥ 0.5, 둘 다 헝가리안)
        TP = 사람 visiting − 못 찾음 − other로 판정,  FP = other→visiting + 라벨 없는 검출→visiting,
        FN = 못 찾음 + other로 판정  →  Precision, Recall, F1
    95% CI — test 86장에서 사진을 복원추출(10,000회, seed 0), 사진별 |오차|는 fold 5개 평균
    Wilcoxon — 같은 사진 86장의 |오차|(fold 5개 평균)를 짝지어 비교 (zero_method="zsplit")

사용법 (counting/ 에서):  python scripts/paper_tables.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
FS = ROOT / "reports" / "fold_study"
OUT = FS / "paper"
K = 5
METHODS = [("geo", "Geo"), ("rf", "RF"), ("rf_geo", "RF+Geo"), ("rf_geo_leg", "RF+Geo+Leg"), ("yolo", "YOLO")]
LABEL = dict(METHODS)
SPEED_GPU, SPEED_CPU = "RTX 6000 Ada", "AMD EPYC 7513"   # 추론 시간 측정 서버 (speed_summary.csv의 CPU 이름은 x86_64로만 찍힘)
SCHEMES = [("random_5fold", "Random 5-fold", "rk5"), ("time_5fold", "Time 5-fold", "k5")]
SETS = [("test", "test(000)"), ("val", "Validation")]
RF_BEST = "rf_geo_leg"   # 두 분할 모두 5-fold 평균 val MAE 최소 (fold_study.md 1장)


def ms(v, fmt=".3f", scale=1.0):
    v = pd.Series(v, dtype=float).dropna() * scale
    if v.empty:
        return "–"
    if len(v) == 1:
        return format(v.iloc[0], fmt)
    return f"{format(v.mean(), fmt)} ± {format(v.std(ddof=1), fmt.lstrip('+'))}"


def add_prf(df):
    tp = df.total_human - df.missed_visiting_undetected - df.missed_visiting_judged
    fp = df.wrong_visiting + df.extra_visiting
    fn = df.missed_visiting_undetected + df.missed_visiting_judged
    df["vis_precision"] = tp / (tp + fp)
    df["vis_recall"] = tp / (tp + fn)
    df["vis_F1"] = 2 * df.vis_precision * df.vis_recall / (df.vis_precision + df.vis_recall)
    return df


def per_image_abs_err(method, prefix):
    f = FS / "per_frame_000" / f"{method}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, encoding="utf-8-sig").sort_values("frame_idx")
    cols = [f"{prefix}_f{i}" for i in range(K) if f"{prefix}_f{i}" in d]
    if len(cols) < K:
        return None
    return (d[cols].sub(d["gt"], axis=0)).abs().mean(axis=1).to_numpy()


def load_pose(split):
    f = REPO / "training" / "reports" / "metrics" / f"ep200_{split}" / "m" / "summary.json"
    s = json.loads(f.read_text(encoding="utf-8"))
    rec = {"split": split}
    for part, key in (("val", "valid"), ("test", "test")):
        t = s[key]
        rec[f"{part}_box_mAP50"] = t["official"]["box"]["mAP50"]
        rec[f"{part}_box_mAP50-95"] = t["official"]["box"]["mAP50-95"]
        rec[f"{part}_pose_mAP50"] = t["official"]["pose"]["mAP50"]
        rec[f"{part}_pose_mAP50-95"] = t["official"]["pose"]["mAP50-95"]
        for th in ("0.05", "0.1", "0.2"):
            rec[f"{part}_PCK@{th}"] = t["pck"]["by_threshold"][f"PCK@{th}"]["overall"]
    return rec


RF_FEATS, GEO_FEATS, LEG_FEATS = [], [], []


def rf_importance():
    """RF+Geo+Leg를 fold 10개(Random 5-fold · Time 5-fold) train 사진으로 평가와 똑같이 학습해 특징 중요도를 모음.
    라벨 XML(counting/data/labels, git 제외)이 없으면 None."""
    labels = ROOT / "data" / "labels"
    if not (labels / "state_001.xml").exists():
        return None
    sys.path.insert(0, str(ROOT / "scripts"))
    import evaluate_methods as em
    from feeding_monitor.method_features import GEO_FEATURES, LEG_FEATURES, METHOD_FEATURES, RF_FEATURES
    RF_FEATS[:], GEO_FEATS[:], LEG_FEATS[:] = RF_FEATURES, GEO_FEATURES, LEG_FEATURES
    cams = em.Cameras(ROOT / "configs" / "eval_cameras.yaml")
    rows = [r for x in ("state_001.xml", "state_002.xml") for r in em.parse_cvat_xml(labels / x)]
    s1 = em.featurize([r for r in rows if cams.has(r["camera"])], cams)
    feats = METHOD_FEATURES["rf_geo_leg"]
    out = []
    for _, _, p in SCHEMES:
        for i in range(K):
            split = f"{p}_f{i}"
            keep = {s.split("__", 1)[1] for s in json.loads(
                (REPO / "training" / "data" / "splits" / f"{split}.json").read_text(encoding="utf-8"))["train"]}
            d = s1[s1["image"].isin(keep)]
            clf = em.new_rf(42).fit(d[feats], d["label"])
            out += [{"split": split, "feature": f, "importance": v} for f, v in zip(feats, clf.feature_importances_)]
    return pd.DataFrame(out)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_prf(pd.read_csv(FS / "fold_metrics.csv", encoding="utf-8-sig"))

    cols = [("count_MAE", "MAE", ".3f", 1), ("count_RMSE", "RMSE", ".3f", 1), ("count_bias", "ME", "+.3f", 1),
            ("exact_rate", "Acc (%)", ".1f", 100), ("within1_rate", "Acc±1 (%)", ".1f", 100),
            ("total_error_pct", "TCE (%)", "+.1f", 1), ("bin_MAE", "10-s MAE", ".3f", 1),
            ("vis_precision", "Precision", ".3f", 1), ("vis_recall", "Recall", ".3f", 1), ("vis_F1", "F1", ".3f", 1)]
    summ = []
    for key, _, _ in SCHEMES:
        for part, _ in SETS:
            for m, _ in METHODS:
                g = df[(df.scheme == key) & (df.set == part) & (df.method == m)]
                rec = {"scheme": key, "set": part, "method": m, "n_folds": g.count_MAE.notna().sum()}
                for c, _, _, sc in cols:
                    v = g[c].dropna() * sc
                    rec[c] = v.mean() if len(v) else np.nan
                    rec[f"{c}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
                summ.append(rec)
    pd.DataFrame(summ).to_csv(OUT / "count_metrics.csv", index=False, encoding="utf-8-sig")

    md = ["# Paper metrics — Random 5-fold vs Time 5-fold, 5 methods", "",
          "fold 5개 평균 ± 표본 표준편차(ddof=1). test = 000 86장(visiting 133마리), val = 그 fold의 valid 사진(001+002). "
          "'–' = 값 없음. 지표 정의와 계산 방법은 `counting/scripts/paper_tables.py` 맨 위 설명.", ""]

    # 표 1·2: 카운트 지표 (test · val 같은 열. val의 10-s MAE는 정의되지 않아 '–')
    for t_no, (part, plab) in zip((1, 2), SETS):
        use = cols
        md += [f"## Table {t_no}. Counting performance — {plab}", "",
               "| Split | Method | " + " | ".join(lab for _, lab, _, _ in use) + " |",
               "|---|---|" + "---|" * len(use)]
        for key, slab, _ in SCHEMES:
            for m, mlab in METHODS:
                g = df[(df.scheme == key) & (df.set == part) & (df.method == m)]
                if g.count_MAE.notna().sum() == 0:
                    md.append(f"| {slab} | {mlab} | " + " | ".join("–" for _ in use) + " |")
                    continue
                star = " †" if m == RF_BEST else ""
                md.append(f"| {slab} | {mlab}{star} | " + " | ".join(ms(g[c], f, sc) for c, _, f, sc in use) + " |")
        md += ["", "† RF best (5-fold 평균 val MAE 최소). ME = 평균(예측 − 정답), TCE = 총 마릿수 오차 / 정답 총 마릿수. "
               "Precision · Recall · F1 = visiting 개체 단위(사람 개체와 짝짓기: Geo · RF 계열 = 관절 평균 거리 ÷ "
               "사람 박스 대각선 ≤ 0.25, YOLO = 박스 IoU ≥ 0.5)." +
               (" val은 모든 방법이 조금 낙관적임: 포즈 best.pt(Geo · RF 계열)와 YOLO best.pt · conf를 같은 valid로 골랐음."
                " 10-s MAE는 1분 연속 영상(test(000))을 프레임 번호로 6등분해 구하는 지표라, 여러 클립에서 뽑은 val 사진에는 "
                "정의되지 않음(–). YOLO val Precision · Recall · F1은 YOLO 재평가(valid 개체 진단 추가) 후 채워짐."
                if part == "val" else ""), ""]

    # 표 3: bootstrap 95% CI (test MAE)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 86, size=(10000, 86))
    ci_rows = []
    for key, slab, p in SCHEMES:
        for m, mlab in METHODS:
            e = per_image_abs_err(m, p)
            if e is None:
                continue
            boot = e[idx].mean(axis=1)
            ci_rows.append({"scheme": key, "method": m, "MAE": e.mean(),
                            "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    md += ["## Table 3. Test MAE with 95% bootstrap CI (images resampled, 10,000×)", "",
           "| Method | " + " | ".join(slab for _, slab, _ in SCHEMES) + " |", "|---|" + "---|" * len(SCHEMES)]
    for m, mlab in METHODS:
        cells = []
        for key, _, _ in SCHEMES:
            r = ci[(ci.scheme == key) & (ci.method == m)]
            cells.append("–" if r.empty else f"{r.MAE.iloc[0]:.3f} [{r.ci_low.iloc[0]:.3f}, {r.ci_high.iloc[0]:.3f}]")
        md.append(f"| {mlab} | " + " | ".join(cells) + " |")
    md += ["", "사진별 |오차|는 fold 5개 평균. 86장이 1분 영상 하나에서 나와 사진끼리 독립이 아니므로 CI는 낙관적임.", ""]

    # 표 4: Wilcoxon (test)
    w_rows = []
    for key, slab, p in SCHEMES:
        base = per_image_abs_err(RF_BEST, p)
        for m, mlab in METHODS:
            if m == RF_BEST:
                continue
            e = per_image_abs_err(m, p)
            if e is None or base is None:
                continue
            d = e - base
            pv = wilcoxon(e, base, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
            w_rows.append({"comparison": f"{LABEL[RF_BEST]} vs {mlab}", "scheme": key, "mean_diff": d.mean(),
                           "better": int((d > 0).sum()), "same": int((d == 0).sum()), "worse": int((d < 0).sum()),
                           "p": pv})
    for m, mlab in METHODS:
        a, b = per_image_abs_err(m, "rk5"), per_image_abs_err(m, "k5")
        if a is None or b is None:
            continue
        d = a - b
        pv = wilcoxon(a, b, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
        w_rows.append({"comparison": f"{mlab}: Random 5-fold vs Time 5-fold", "scheme": "both", "mean_diff": d.mean(),
                       "better": int((d > 0).sum()), "same": int((d == 0).sum()), "worse": int((d < 0).sum()), "p": pv})
    w = pd.DataFrame(w_rows)
    w.to_csv(OUT / "wilcoxon.csv", index=False, encoding="utf-8-sig")
    md += ["## Table 4. Paired comparison on test(000) images (Wilcoxon signed-rank, n = 86)", "",
           "| Comparison (A vs B) | Split | ΔMAE (B − A) | A better / same / B better (images) | p |",
           "|---|---|---|---|---|"]
    for _, r in w.iterrows():
        sl = {"random_5fold": "Random 5-fold", "time_5fold": "Time 5-fold", "both": "–"}[r.scheme]
        md.append(f"| {r.comparison} | {sl} | {r.mean_diff:+.3f} | {r.better} / {r.same} / {r.worse} | "
                  f"{'<0.001' if r.p < 0.001 else f'{r.p:.3f}'} |")
    md += ["", "사진별 |오차|(fold 5개 평균)로 비교. Random 5-fold vs Time 5-fold 줄은 A = Random 5-fold, B = Time 5-fold (Δ > 0이면 Random 5-fold가 더 나쁨). "
           "사진끼리 독립이 아니라 p값은 참고용.", ""]

    # 표 5: 오류 분해 (test, 개체 수)
    md += ["## Table 5. Error decomposition on test(000) (visiting pigs per fold)", "",
           "| Split | Method | Counted / GT | Missed (undetected) | Missed (judged other) | Other → visiting | "
           "Unlabeled → visiting |", "|---|---|---|---|---|---|---|"]
    for key, slab, _ in SCHEMES:
        for m, mlab in METHODS:
            g = df[(df.scheme == key) & (df.set == "test") & (df.method == m)]
            if g.empty:
                continue
            md.append(f"| {slab} | {mlab} | {ms(g.total_model, '.1f')} / 133 | {ms(g.missed_visiting_undetected, '.1f')} | "
                      f"{ms(g.missed_visiting_judged, '.1f')} | {ms(g.wrong_visiting, '.1f')} | {ms(g.extra_visiting, '.1f')} |")
    md += ["", "Geo · RF 계열은 같은 포즈 모델을 써서 'Missed (undetected)'가 같음.", ""]

    # 표 6: 포즈 모델
    pose = pd.DataFrame([load_pose(f"{p}_f{i}") for _, _, p in SCHEMES for i in range(K)])
    pose.to_csv(OUT / "pose_metrics.csv", index=False, encoding="utf-8-sig")
    pcols = ["box_mAP50", "box_mAP50-95", "pose_mAP50", "pose_mAP50-95", "PCK@0.05", "PCK@0.1", "PCK@0.2"]
    md += ["## Table 6. Pose model (YOLOv8m-pose, 18 keypoints)", "",
           "| Split | Set | " + " | ".join(pcols) + " |", "|---|---|" + "---|" * len(pcols)]
    for key, slab, p in SCHEMES:
        sub = pose[pose.split.str.startswith(p + "_")]
        for part, plab in SETS:
            md.append(f"| {slab} | {plab} | " + " | ".join(ms(sub[f"{part}_{c}"]) for c in pcols) + " |")
    kp = pd.read_csv(FS / "pose_keypoints.csv", encoding="utf-8-sig")
    md += ["", "PCK@α = 정답과 예측 keypoint 거리가 α × 정답 박스 긴 변(max(w, h)) 미만인 비율 "
           "(`training/scripts/utils/pck.py`; 예측 conf 0.25, 사람 박스와 IoU ≥ 0.5로 짝지은 돼지만). "
           "mAP는 Ultralytics val. best.pt = val pose mAP50-95 최고 epoch. keypoint별 PCK@0.1:", "",
           "| Keypoint | " + " | ".join(f"{s} {pl}" for _, s, _ in SCHEMES for _, pl in SETS) + " |",
           "|---|" + "---|" * 4]
    for name in kp.keypoint.drop_duplicates():
        md.append(f"| {name} | " + " | ".join(
            ms(kp[(kp.scheme == key) & (kp.set == part) & (kp.keypoint == name)]["PCK@0.1"])
            for key, _, _ in SCHEMES for part, _ in SETS) + " |")
    md.append("")

    # 표 7: YOLO 검출
    y = df[df.method == "yolo"]
    ycols = ["box_P", "box_R", "box_mAP50", "box_mAP50-95", "box_mAP50-95_visiting", "box_mAP50-95_other"]
    md += ["## Table 7. YOLO detector (visiting / other boxes)", "",
           "| Split | Set | conf | P | R | mAP50 | mAP50-95 | mAP50-95 visiting | mAP50-95 other |",
           "|---|---|---|---|---|---|---|---|---|"]
    for key, slab, _ in SCHEMES:
        for part, plab in SETS:
            g = y[(y.scheme == key) & (y.set == part)]
            md.append(f"| {slab} | {plab} | {ms(g.conf, '.2f')} | " +
                      " | ".join(ms(g[c]) if c in g else "–" for c in ycols) + " |")
    md += ["", "conf = 그 fold valid에서 사진당 MAE 최소. 값 출처: `yolo_det/reports/ep200_<fold>/m/summary.json`.", ""]

    # 표 8: 추론 시간 (benchmark_speed.py 결과, 5가지 방법 같은 장비)
    md += ["## Table 8. Inference time (end-to-end per image)", ""]
    sp_files = sorted((ROOT / "reports" / "speed").glob("*/speed_summary.csv"))
    if sp_files:
        sp = pd.read_csv(sp_files[0], encoding="utf-8-sig")
        sp_dev = {"GPU": f"GPU ({SPEED_GPU})", "CPU": f"CPU ({SPEED_CPU})"}
        md += ["| Split | Device | Method | Median (ms) | p95 (ms) | Read | Model | Features | RF | 1 fps (≤1000 ms) | 30 fps (≤33.3 ms) |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, slab, _ in SCHEMES:
            for dev in ["GPU", "CPU"]:
                for m, mlab in METHODS:
                    r = sp[(sp.scheme == key) & (sp.device == dev) & (sp.method == m)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    md.append(f"| {slab} | {sp_dev[dev]} | {mlab} | {r.median_ms:.1f} ± {r.median_ms_sd:.1f} | "
                              f"{r.p95_ms:.1f} ± {r.p95_ms_sd:.1f} | {r.read_median_ms:.1f} | {r.infer_median_ms:.1f} | "
                              f"{r.feature_median_ms:.1f} | {r.classify_median_ms:.1f} | "
                              f"{'✅' if r.p95_ms <= 1000 else '❌'} | {'✅' if r.p95_ms <= 1000 / 30 else '❌'} |")
        md += ["", "조건: 5가지 방법 모두 같은 서버 한 대, batch 1, imgsz 640, 예열 5장 제외, test(000) 86장 × 3회, "
               "사진 읽기부터 마릿수까지, fold마다 그 fold 모델(YOLO는 그 fold conf) → fold 5개 중앙값 · p95의 평균 ± 표본 표준편차. "
               "단계별 칸(Read · Model · Features · RF)은 fold 중앙값의 평균(ms). 실시간 판정은 p95 평균 기준. "
               f"원자료: `counting/reports/speed/{sp_files[0].parent.name}/` (`benchmark_speed.py`).", ""]
    else:
        md += ["(측정 결과 없음: `counting/scripts/benchmark_speed.py`를 먼저 실행)", ""]

    # 표 9: 사진당 visiting 마릿수 분포 (데이터)
    gt_test = pd.read_csv(FS / "per_frame_000" / "geo.csv", encoding="utf-8-sig")["gt"]
    gt_val = pd.concat([pd.read_csv(FS / "frames" / f"ep200_k5_f{i}" / "geo_val.csv", encoding="utf-8-sig")
                        for i in range(K)], ignore_index=True)
    gt_val["camera"] = gt_val.image.str.extract(r"(CH\d{3})")[0]
    md += ["## Table 9. Visiting pigs per image (human labels)", "",
           "| Set | Images | 0 | 1 | 2 | 3+ | Mean ± SD | Total visiting |", "|---|---|---|---|---|---|---|---|"]
    for lab, g in [("train + valid 001 (CH005)", gt_val[gt_val.camera == "CH005"]["gt"]),
                   ("train + valid 002 (CH010)", gt_val[gt_val.camera == "CH010"]["gt"]),
                   ("train + valid 001 + 002", gt_val["gt"]), ("test(000)", gt_test)]:
        c = [int((g == k).sum()) for k in (0, 1, 2)] + [int((g >= 3).sum())]
        md.append(f"| {lab} | {len(g)} | " + " | ".join(map(str, c)) + f" | {g.mean():.2f} ± {g.std(ddof=1):.2f} | {int(g.sum())} |")
    md += ["", "train + valid 190장은 5-fold valid를 합친 것(각 사진이 한 번씩 valid). 0 / 1 / 2 / 3+ = 사진 수.", ""]

    # 표 10: 카메라별 val MAE (Geo · RF 계열; YOLO는 사진별 val csv가 없어 제외)
    md += ["## Table 10. Validation MAE by camera (fold 5개 평균 ± 표본 표준편차)", "",
           "| Split | Method | CH005 (001, 아침) | CH010 (002, 낮~저녁) |", "|---|---|---|---|"]
    cam_rows = []
    for key, slab, p in SCHEMES:
        for m, mlab in METHODS[:-1]:
            per = {"CH005": [], "CH010": []}
            for i in range(K):
                f = FS / "frames" / f"ep200_{p}_f{i}" / f"{m}_val.csv"
                v = pd.read_csv(f, encoding="utf-8-sig")
                v["camera"] = v.image.str.extract(r"(CH\d{3})")[0]
                for cam in per:
                    e = v[v.camera == cam].error
                    per[cam].append(e.abs().mean() if len(e) else np.nan)
                    cam_rows.append({"scheme": key, "method": m, "fold": f"f{i}", "camera": cam,
                                     "n_images": len(e), "count_MAE": e.abs().mean() if len(e) else np.nan})
            md.append(f"| {slab} | {mlab} | {ms(per['CH005'])} | {ms(per['CH010'])} |")
    pd.DataFrame(cam_rows).to_csv(OUT / "val_mae_by_camera.csv", index=False, encoding="utf-8-sig")
    md += ["", "fold마다 그 fold valid 중 그 카메라 사진만으로 MAE를 계산 후 평균. fold별 장수는 `val_mae_by_camera.csv`.", ""]

    # 표 11: RF+Geo+Leg 특징 중요도 (fold 10개 평균)
    imp = rf_importance()
    if imp is not None:
        imp.to_csv(OUT / "rf_feature_importance.csv", index=False, encoding="utf-8-sig")
        s = imp.groupby("feature")["importance"].agg(["mean", "std"]).sort_values("mean", ascending=False)
        grp = {**{f: "RF" for f in RF_FEATS}, **{f: "Geo" for f in GEO_FEATS}, **{f: "Leg" for f in LEG_FEATS}}
        md += ["## Table 11. RF+Geo+Leg feature importance (Gini, mean ± SD over 10 fold models)", "",
               "| Rank | Feature | Group | Importance |", "|---|---|---|---|"]
        for r, (f, row) in enumerate(s.iterrows(), 1):
            md.append(f"| {r} | {f} | {grp.get(f, '')} | {row['mean']:.3f} ± {row['std']:.3f} |")
        gs = imp.assign(group=imp.feature.map(grp)).groupby(["split", "group"]).importance.sum().groupby("group")
        md += ["", "그룹 합계: " + ", ".join(f"{g} {v.mean():.3f} ± {v.std(ddof=1):.3f}" for g, v in gs) +
               ". RF는 그 fold train 사진의 사람 관절로 학습(평가와 동일), 나무 100개 · balanced · seed 42.", ""]

    md += ["## Figures", "",
           "| Figure | File | Caption |", "|---|---|---|",
           "| A | `../graph_A_frame_counts_000.png` | Per-image visiting count on test(000) (GT, Geo, RF best, YOLO; mean over 5 folds, band = min–max) |",
           "| B | `../graph_B_mae_random_vs_time.png` | Per-image count MAE by method, Random 5-fold vs Time 5-fold (mean ± SD) |",
           "| C | `../graph_C_total_counts.png` | Total visiting pigs counted per fold vs human GT |",
           "| D | `../graph_D_keypoint_pck.png` | Per-keypoint PCK@0.1, Random 5-fold vs Time 5-fold |",
           "| E | `../per_image/graph_E_per_image_counts.png` | Visiting pigs counted per test(000) image, every method vs GT (mean over 5 folds, band = min–max) |",
           "| F | `../per_image/graph_F_per_image_error_heatmap.png` | Per-image counting error of all 50 models on test(000) |",
           "| G | `../error_cases/all_models_57.png`, `all_models_76.png` | Occlusion / crowding error cases: all 10 pose models side by side |",
           "| J | `../graph_J_random_5fold_frames.png`, `../graph_J_time_5fold_frames.png` | Visiting pigs per test(000) image (frame 0–85), human GT vs Geo, RF+Geo+Leg, YOLO, one figure per split (mean of 5 fold models; lines offset by −0.06 / 0 / +0.06) |",
           "| – | `../per_image/by_model/<method>.png` | Per-method counts on test(000): GT row + 10 fold models |", ""]
    (OUT / "paper_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
