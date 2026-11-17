"""랜덤 5-fold vs 시간대 5-fold — Geo / RF / RF+Geo / RF+Geo+Leg / YOLO를 val·test(000)에서 비교 (11/16 랩미팅 통일 기준).

분할 (training/data/splits/<split>.json, make_study_splits.py — YOLO prepare_data.py와 같은 함수):
    시간대 k5_f0~f4 : 001·002를 시간순 5구간, fold마다 한 구간이 valid
    랜덤   rk5_f0~f4 : task마다 seed 42로 섞고, fold마다 train·valid 장수를 k5_f<i>와 task별로 똑같이 맞춤
입력:
    Geo·RF 계열 : outputs/eval/split_<split>/      (test = 000)  ← run_split_eval.ps1
                  outputs/eval/split_<split>_val/  (val = 그 fold의 valid 사진)  ← run_split_eval.ps1 -Valid
                  포즈 = training/models/results/ep200_<split>/m (seed 42, best.pt = val pose mAP50-95 최고)
                  RF  = 그 fold train 사진으로 학습 (나무 100개, balanced, seed 42, 기준 0.5)
    YOLO        : ../yolo_det/reports/ep200_<split>/m/summary.json  ("test", 있으면 "valid")
                  ../yolo_det/reports/ep200_<split>/m/test_frames.csv (있으면 그래프 A·RMSE에 사용)
    포즈 지표   : ../training/reports/metrics/ep200_<split>/m/summary.json
선택 기준 (val만 사용, test를 보고 고르지 않음):
    RF best = RF / RF+Geo / RF+Geo+Leg 중 5-fold 평균 val MAE 최소 (분할 방식마다 따로)
출력: reports/fold_study/ (git에 올림)
    fold_metrics.csv     분할 방식 × fold × val/test × 방법, 모든 지표
    fold_summary.csv     분할 방식 × val/test × 방법, 평균·표본 표준편차
    pose_folds.csv       fold별 포즈 모델 val·test 지표
    fold_composition.csv fold별 train/valid 장수 (task별)
    frames/ep200_<split>/<method>_<val|test>.csv   사진별 카운트 (image, frame_idx, gt, pred, error = pred − gt)
    fold_study.md        표
    graph_A_frame_counts_000.png   000 프레임별 카운트 (GT + Geo + RF best + YOLO, fold 평균선 + 최소~최대 띠)
    graph_B_mae_random_vs_time.png 방법별 MAE (랜덤 vs 시간대, error bar = fold 간 표본 표준편차)

사용법 (counting/ 에서):  python scripts/compare_folds.py
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "yolo_det" / "scripts"))
from prepare_data import time_key  # noqa: E402

K = 5
METHODS = [("geo", "Geo"), ("rf", "RF"), ("rf_geo", "RF+Geo"), ("rf_geo_leg", "RF+Geo+Leg"), ("yolo", "YOLO")]
RF_FAMILY = ["rf", "rf_geo", "rf_geo_leg"]
LABEL = dict(METHODS)
SCHEMES = [("random", "랜덤 5-fold", "rk5"), ("time", "시간대 5-fold", "k5")]
SETS = [("val", "val"), ("test", "test (000)")]
COLS = ["count_MAE", "count_RMSE", "count_bias", "exact_rate", "within1_rate", "total_model", "total_human",
        "total_error_pct", "over_frames", "under_frames", "bin_MAE",
        "missed_visiting_undetected", "missed_visiting_judged", "wrong_visiting", "extra_visiting"]
# 그래프 색 (통일): GT 검정, Geo 회색, RF 계열 파랑(진할수록 특징 많음), YOLO 주황
COLOR = {"gt": "#000000", "geo": "#8c8c8c", "rf": "#9ecae1", "rf_geo": "#4292c6", "rf_geo_leg": "#08519c",
         "yolo": "#f28e2b"}
NAME_TO_KST = timedelta(hours=9)


def splits_of(prefix):
    return [f"{prefix}_f{i}" for i in range(K)]


def frames_csv(fr: pd.DataFrame, method: str) -> pd.DataFrame:
    """stage2_frames.csv → 통일 형식 (image, frame_idx, gt, pred, error)."""
    out = pd.DataFrame({"image": fr["image"],
                        "frame_idx": fr["frame_idx"] if "frame_idx" in fr else np.nan,
                        "gt": fr["human_visiting"].astype(int), "pred": fr[f"count_{method}"].astype(int)})
    out["error"] = out["pred"] - out["gt"]
    return out.sort_values(["frame_idx", "image"] if out["frame_idx"].notna().all() else ["image"])


def load_geo_rf(split, part, eval_root, frames_out):
    d = eval_root / (f"split_{split}" if part == "test" else f"split_{split}_val")
    if not (d / "stage2_metrics.csv").exists():
        return []
    met = pd.read_csv(d / "stage2_metrics.csv", encoding="utf-8-sig").set_index("method")
    fr = pd.read_csv(d / "stage2_frames.csv", encoding="utf-8-sig")
    rows = []
    for m, _ in METHODS[:-1]:
        if m not in met.index:
            continue
        rows.append({"split": split, "set": part, "method": m, **{c: met.loc[m].get(c, np.nan) for c in COLS}})
        f = frames_csv(fr, m)
        f.to_csv(frames_out / f"ep200_{split}" / f"{m}_{part}.csv", index=False, encoding="utf-8-sig")
    return rows


def load_yolo(split, part):
    d = REPO / "yolo_det" / "reports" / f"ep200_{split}" / "m"
    if not (d / "summary.json").exists():
        return []
    y = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    key = "test" if part == "test" else next((k for k in ("valid", "val") if k in y), None)
    if key is None:
        return []
    rec = {"split": split, "set": part, "method": "yolo", **{c: y[key].get(c, np.nan) for c in COLS},
           "yolo_conf": y.get("conf_threshold")}
    fcsv = d / f"{part}_frames.csv"
    if pd.isna(rec["count_RMSE"]) and fcsv.exists():                 # 수아 재평가 전 summary에는 RMSE가 없음
        e = pd.read_csv(fcsv, encoding="utf-8-sig")["error"]
        rec["count_RMSE"] = float(np.sqrt((e ** 2).mean()))
    return [rec]


def load_pose(split):
    f = REPO / "training" / "reports" / "metrics" / f"ep200_{split}" / "m" / "summary.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text(encoding="utf-8"))
    rec = {"split": split}
    for part, key in (("val", "valid"), ("test", "test")):
        t = s[key]
        rec[f"{part}_pose_mAP50-95"] = t["official"]["pose"]["mAP50-95"]
        rec[f"{part}_PCK@0.1"] = t["pck"]["by_threshold"]["PCK@0.1"]["overall"]
        rec[f"{part}_PCK@0.2"] = t["pck"]["by_threshold"]["PCK@0.2"]["overall"]
    return rec


def composition(split):
    d = json.loads((REPO / "training" / "data" / "splits" / f"{split}.json").read_text(encoding="utf-8"))
    rec = {"split": split}
    for part in ("train", "valid", "test"):
        rec[part] = len(d[part])
        for task in ("001", "002"):
            if part != "test":
                rec[f"{part}_{task}"] = sum(s.startswith(f"{task}__") for s in d[part])
    for task in ("001", "002"):                                       # valid 촬영 시각 범위 (KST = 이름 + 9h)
        ts = sorted(datetime.strptime(time_key(s.split("__", 1)[1])[0], "%Y%m%d%H%M%S") + NAME_TO_KST
                    for s in d["valid"] if s.startswith(f"{task}__"))
        clips = len(set(ts))                                          # time_key[0] = 클립 시작 시각
        rec[f"valid_{task}_clips"] = clips
        rec[f"valid_{task}_kst"] = (f"{ts[0]:%H:%M}" if clips == 1 else f"{ts[0]:%H:%M}~{ts[-1]:%H:%M}") + \
            f" ({clips}클립)" if ts else ""
    return rec


def ms(v, fmt=".3f", pct=False):
    v = pd.Series(v, dtype=float).dropna() * (100 if pct else 1)
    if v.empty:
        return "–"
    if len(v) == 1:
        return format(v.iloc[0], fmt)
    return f"{format(v.mean(), fmt)} ± {format(v.std(ddof=1), fmt.replace('+', ''))}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=str(ROOT / "outputs" / "eval"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "fold_study"))
    a = ap.parse_args()
    eval_root, out = Path(a.eval_root), Path(a.out)
    frames_out = out / "frames"
    for _, _, p in SCHEMES:
        for s in splits_of(p):
            (frames_out / f"ep200_{s}").mkdir(parents=True, exist_ok=True)

    rows = []
    for key, _, p in SCHEMES:
        for i, s in enumerate(splits_of(p)):
            for part, _ in SETS:
                for r in load_geo_rf(s, part, eval_root, frames_out) + load_yolo(s, part):
                    rows.append({"scheme": key, "fold": f"f{i}", **r})
    df = pd.DataFrame(rows)
    df.to_csv(out / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    summ = []
    for (key, part, m), g in df.groupby(["scheme", "set", "method"]):
        rec = {"scheme": key, "set": part, "method": m, "n_folds": len(g)}
        for c in COLS:
            v = g[c].dropna()
            rec[c] = v.mean() if len(v) else np.nan
            rec[f"{c}_std"] = v.std(ddof=1) if len(v) > 1 else np.nan
        summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "fold_summary.csv", index=False, encoding="utf-8-sig")

    pose = pd.DataFrame([r for _, _, p in SCHEMES for s in splits_of(p) if (r := load_pose(s))])
    pose.to_csv(out / "pose_folds.csv", index=False, encoding="utf-8-sig")
    comp = pd.DataFrame([composition(s) for _, _, p in SCHEMES for s in splits_of(p)])
    comp.to_csv(out / "fold_composition.csv", index=False, encoding="utf-8-sig")

    # RF best: 5-fold 평균 val MAE 최소 (val만 사용)
    rf_best = {}
    for key, _, _ in SCHEMES:
        v = summ[(summ.scheme == key) & (summ.set == "val") & summ.method.isin(RF_FAMILY)].set_index("method")
        rf_best[key] = v["count_MAE"].idxmin() if len(v) and v["count_MAE"].notna().any() else None

    def vals(key, part, m, c):
        g = df[(df.scheme == key) & (df.set == part) & (df.method == m)].set_index("fold")
        return [g.loc[f"f{i}", c] if f"f{i}" in g.index else np.nan for i in range(K)]

    # ---------------- markdown ----------------
    md = ["# 랜덤 5-fold vs 시간대 5-fold — 5가지 방법 비교", "",
          "값은 fold 5개의 평균 ± 표본 표준편차(ddof=1). 사진당 visiting 마릿수 기준. val = 그 fold의 valid 사진, "
          "test = 000 86장(사람 visiting 133마리). '–' = 아직 값 없음.", "",
          "## 1. 통합 결과표 (MAE / RMSE)", "",
          "| 방법 | 랜덤 val MAE | 랜덤 val RMSE | 랜덤 test MAE | 랜덤 test RMSE | 시간대 val MAE | 시간대 val RMSE | 시간대 test MAE | 시간대 test RMSE |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m, lab in METHODS:
        cells = [ms(vals(k, part, m, c)) for k, _, _ in SCHEMES for part, _ in SETS for c in ("count_MAE", "count_RMSE")]
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
    md += ["", "RF best (5-fold 평균 val MAE 최소, test는 보지 않음): " +
           ", ".join(f"{lab} → **{LABEL.get(rf_best[k], '–')}**" for k, lab, _ in SCHEMES), ""]

    md += ["## 2. fold별 값 (f0~f4)", ""]
    for key, slab, _ in SCHEMES:
        for part, plab in SETS:
            for c, clab in (("count_MAE", "MAE"), ("count_RMSE", "RMSE")):
                md += [f"**{slab} · {plab} · {clab}**", "",
                       "| 방법 | f0 | f1 | f2 | f3 | f4 | 평균 ± 표준편차 |", "|---|---|---|---|---|---|---|"]
                for m, lab in METHODS:
                    v = vals(key, part, m, c)
                    md.append(f"| {lab} | " + " | ".join("–" if pd.isna(x) else f"{x:.3f}" for x in v) +
                              f" | {ms(v)} |")
                md.append("")

    md += ["## 3. test(000) 기타 지표", "",
           "| 분할 | 방법 | bias | 정확 일치 | ±1 이내 | 모델이 센 총 마릿수 (사람 133) | 1분 총 점유량 오차 | 10초 구간 오차 | 못 찾음 / other로 판정 / other→visiting / 추가 검출 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for key, slab, _ in SCHEMES:
        for m, lab in METHODS:
            def v(c):
                return vals(key, "test", m, c)
            if all(pd.isna(x) for x in v("count_MAE")):
                continue
            errs = " / ".join(ms(v(c), ".1f") for c in COLS[11:])
            md.append(f"| {slab} | {lab} | {ms(v('count_bias'), '+.3f')} | {ms(v('exact_rate'), '.1f', True)}% | "
                      f"{ms(v('within1_rate'), '.1f', True)}% | {ms(v('total_model'), '.1f')} | "
                      f"{ms(v('total_error_pct'), '+.1f')}% | {ms(v('bin_MAE'))} | {errs} |")
    md.append("")

    md += ["## 4. fold 구성 (train / valid 장수, valid 사진이 속한 클립의 시작 시각 KST = 파일 이름 + 9시간)", "",
           "001 = 7/8 아침(클립 4개), 002 = 7/16 낮~저녁(클립 17개). 랜덤과 시간대는 fold마다 train·valid 장수가 task별로 같음.", "",
           "| fold | train (001 / 002) | valid (001 / 002) | 랜덤 valid 시각 범위 001 · 002 | 시간대 valid 시각 범위 001 · 002 |",
           "|---|---|---|---|---|"]
    for i in range(K):
        r = comp[comp.split == f"rk5_f{i}"].iloc[0]
        t = comp[comp.split == f"k5_f{i}"].iloc[0]
        md.append(f"| f{i} | {t.train} ({t.train_001} / {t.train_002}) | {t.valid} ({t.valid_001} / {t.valid_002}) | "
                  f"{r.valid_001_kst} · {r.valid_002_kst} | {t.valid_001_kst} · {t.valid_002_kst} |")
    md.append("")

    if len(pose):
        md += ["## 5. fold별 포즈 모델 (medium ep200, seed 42, best.pt = val pose mAP50-95 최고)", "",
               "| 분할 | fold | val pose mAP50-95 | val PCK@0.1 | test pose mAP50-95 | test PCK@0.1 | test PCK@0.2 |",
               "|---|---|---|---|---|---|---|"]
        for key, slab, p in SCHEMES:
            sub = pose[pose.split.str.startswith(p + "_")]
            for _, r in sub.iterrows():
                md.append(f"| {slab} | {r.split[-2:]} | {r['val_pose_mAP50-95']:.3f} | {r['val_PCK@0.1']:.3f} | "
                          f"{r['test_pose_mAP50-95']:.3f} | {r['test_PCK@0.1']:.3f} | {r['test_PCK@0.2']:.3f} |")
            if len(sub):
                md.append(f"| {slab} | 평균 ± 표준편차 | {ms(sub['val_pose_mAP50-95'])} | {ms(sub['val_PCK@0.1'])} | "
                          f"{ms(sub['test_pose_mAP50-95'])} | {ms(sub['test_PCK@0.1'])} | {ms(sub['test_PCK@0.2'])} |")
        md.append("")
    (out / "fold_study.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    plot_b(df, out)
    plot_a(out, rf_best)
    print("saved", out)


def plot_b(df, out):
    """그래프 B: 방법별 MAE, 랜덤 vs 시간대. 막대 = 5-fold 평균, error bar = fold 간 표본 표준편차."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    w = 0.38
    for ax, (part, plab) in zip(axes, SETS):
        for j, (key, _, _) in enumerate(SCHEMES):
            means, sds = [], []
            for m, _ in METHODS:
                v = df[(df.scheme == key) & (df.set == part) & (df.method == m)]["count_MAE"].dropna()
                means.append(v.mean() if len(v) else np.nan)
                sds.append(v.std(ddof=1) if len(v) > 1 else np.nan)
            xs = np.arange(len(METHODS)) + (j - 0.5) * w
            b = ax.bar(xs, np.nan_to_num(means), w, color=[COLOR[m] for m, _ in METHODS],
                       hatch="" if key == "random" else "///", edgecolor="white" if key == "random" else "#333",
                       linewidth=0.6)
            ax.errorbar(xs, means, yerr=sds, fmt="none", capsize=3, elinewidth=0.9, ecolor="#333")
            for x, mval, s in zip(xs, means, sds):
                ax.text(x, (0 if np.isnan(mval) else mval + (0 if np.isnan(s) else s)) + 0.01,
                        "n/a" if np.isnan(mval) else f"{mval:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(METHODS)), [lab for _, lab in METHODS])
        ax.set_title(f"{'Validation' if part == 'val' else 'Test (000, 86 images)'}")
        ax.grid(axis="y", color="#e5e5e5")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Per-image count MAE (pigs)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#bbbbbb", label="Random 5-fold"),
                        Patch(facecolor="#bbbbbb", hatch="///", edgecolor="#333", label="Time 5-fold")],
               frameon=False, ncol=2, loc="lower center")
    fig.suptitle("Visiting count MAE by method: random vs time 5-fold (mean ± sample SD over folds)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "graph_B_mae_random_vs_time.png", dpi=150)
    plt.close(fig)


def plot_a(out, rf_best):
    """그래프 A: 000 프레임별 visiting 마릿수. GT + Geo + RF best + YOLO, fold 평균선 + 최소~최대 띠."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, (key, slab, p) in zip(axes, SCHEMES):
        gt_drawn = False
        for m in ["geo", rf_best.get(key), "yolo"]:
            if m is None:
                continue
            fr = []
            for s in splits_of(p):
                f = out / "frames" / f"ep200_{s}" / f"{m}_test.csv"
                if m == "yolo":
                    f = REPO / "yolo_det" / "reports" / f"ep200_{s}" / "m" / "test_frames.csv"
                if f.exists():
                    fr.append(pd.read_csv(f, encoding="utf-8-sig").sort_values("frame_idx").reset_index(drop=True))
            if not fr:
                continue
            x = np.arange(len(fr[0]))
            if not gt_drawn:
                ax.step(x, fr[0]["gt"], where="mid", color=COLOR["gt"], lw=1.8, label="GT (human)")
                gt_drawn = True
            p_ = np.vstack([f["pred"].to_numpy() for f in fr])
            lab = {"geo": "Geo", "yolo": "YOLO"}.get(m, f"RF best ({LABEL[m]})")
            ax.plot(x, p_.mean(0), color=COLOR[m], lw=1.3, label=f"{lab}, mean of {len(fr)} folds")
            ax.fill_between(x, p_.min(0), p_.max(0), color=COLOR[m], alpha=0.18, lw=0)
        ax.set_title("Random 5-fold" if key == "random" else "Time 5-fold", loc="left", fontsize=10)
        ax.set_ylabel("Visiting pigs")
        ax.set_yticks([0, 1, 2, 3])
        ax.grid(axis="y", color="#e5e5e5")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left")
    axes[-1].set_xlabel("Image order in 000 (time →)")
    fig.suptitle("Per-image visiting count on 000 (line = mean over folds, band = min–max)")
    fig.tight_layout()
    fig.savefig(out / "graph_A_frame_counts_000.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
