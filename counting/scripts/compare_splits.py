"""분할 방식(랜덤 / 시간대 / 시간대 5-fold)별로 Geo / RF / RF+Geo / RF+Geo+Leg / YOLO를 000에서 비교.

입력 (모두 test = 000, 같은 count_metrics):
    Geo·RF 계열 : outputs/eval/split_<name>/stage2_metrics.csv   (run_split_eval.ps1)
                  포즈 모델 = 그 분할 train으로 학습한 medium ep200, RF = 그 분할 train 사진으로 학습
    YOLO        : ../yolo_det/reports/ep200[_sorted|_k5_f<i>]/m/summary.json  (수아, COCO 시작, epoch 200)
    포즈 지표   : ../training/reports/metrics/ep200[_<name>]/m/summary.json

출력: reports/split_comparison/ (git에 올림)
    split_comparison.csv  분할 × 방법 (5-fold는 평균·표준편차)
    split_folds.csv       5-fold fold별 값
    pose_by_split.csv     분할별 포즈 모델 test 지표
    split_comparison.md   표 (문서에 붙일 형식)
    split_comparison.png  그림

사용법 (counting/ 에서):  python scripts/compare_splits.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
METHODS = [("geo", "Geo"), ("rf", "RF"), ("rf_geo", "RF+Geo"), ("rf_geo_leg", "RF+Geo+Leg"), ("yolo", "YOLO")]
SCHEMES = [("random", "랜덤", ["random"]), ("sorted", "시간대", ["sorted"]),
           ("k5", "시간대 5-fold", [f"k5_f{i}" for i in range(5)])]
COLS = ["count_MAE", "count_bias", "exact_rate", "within1_rate", "total_error_pct", "bin_MAE",
        "missed_visiting_undetected", "missed_visiting_judged", "wrong_visiting", "extra_visiting"]


def run_suffix(split):          # random → ep200, sorted → ep200_sorted, k5_f0 → ep200_k5_f0
    return "ep200" if split == "random" else f"ep200_{split}"


def load_split(split, eval_root):
    df = pd.read_csv(eval_root / f"split_{split}" / "stage2_metrics.csv", encoding="utf-8-sig").set_index("method")
    rows = [{"method": m, **df.loc[m, COLS].to_dict()} for m, _ in METHODS if m in df.index]
    y = json.loads((REPO / "yolo_det" / "reports" / run_suffix(split) / "m" / "summary.json").read_text(encoding="utf-8"))
    rows.append({"method": "yolo", **{c: y["test"][c] for c in COLS}, "yolo_conf": y["conf_threshold"]})
    return pd.DataFrame(rows).assign(split=split)


def load_pose(split):
    f = REPO / "training" / "reports" / "metrics" / run_suffix(split) / "m" / "summary.json"
    t = json.loads(f.read_text(encoding="utf-8"))["test"]
    p = t["pck"]
    return {"split": split, "pose_mAP50-95": t["official"]["pose"]["mAP50-95"],
            "box_mAP50-95": t["official"]["box"]["mAP50-95"],
            "PCK@0.1": p["by_threshold"]["PCK@0.1"]["overall"], "PCK@0.2": p["by_threshold"]["PCK@0.2"]["overall"],
            "missed": p["missed_instances"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=str(ROOT / "outputs" / "eval"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "split_comparison"))
    a = ap.parse_args()
    eval_root, out = Path(a.eval_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    per_split = pd.concat([load_split(s, eval_root) for _, _, ss in SCHEMES for s in ss], ignore_index=True)
    folds = per_split[per_split.split.str.startswith("k5")]
    folds.to_csv(out / "split_folds.csv", index=False, encoding="utf-8-sig")

    rows = []
    for key, label, ss in SCHEMES:
        sub = per_split[per_split.split.isin(ss)]
        for m, mlabel in METHODS:
            g = sub[sub.method == m]
            rec = {"scheme": key, "scheme_label": label, "method": m, "method_label": mlabel, "n_runs": len(g)}
            for c in COLS:
                rec[c] = g[c].mean()
                rec[f"{c}_std"] = g[c].std(ddof=1) if len(g) > 1 else float("nan")
            rows.append(rec)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "split_comparison.csv", index=False, encoding="utf-8-sig")

    pose = pd.DataFrame([load_pose(s) for _, _, ss in SCHEMES for s in ss])
    pose.to_csv(out / "pose_by_split.csv", index=False, encoding="utf-8-sig")

    # ---------------- markdown ----------------
    def cell(r, c, fmt, pct=False):
        v, s = r[c] * (100 if pct else 1), r[f"{c}_std"] * (100 if pct else 1)
        txt = format(v, fmt) + ("%" if pct else "")
        return txt if pd.isna(s) else f"{txt} ± {format(s, fmt)}"

    md = ["| 분할 | 방법 | 사진당 MAE | bias | 정확 일치 | 1분 총 점유량 오차 | 10초 구간 오차 | 못 찾음 / other로 판정 / other→visiting / 추가 검출 |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        tot = f"{r.total_error_pct:+.1f}%" + ("" if pd.isna(r.total_error_pct_std) else f" ± {r.total_error_pct_std:.1f}")
        err = " / ".join(f"{r[c]:.1f}".rstrip("0").rstrip(".") for c in COLS[6:])
        md.append(f"| {r.scheme_label} | {r.method_label} | {cell(r, 'count_MAE', '.3f')} | {r.count_bias:+.3f} | "
                  f"{cell(r, 'exact_rate', '.1f', pct=True)} | {tot} | {cell(r, 'bin_MAE', '.3f')} | {err} |")
    md += ["", "| fold | " + " | ".join(l for _, l in METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for s in [f"k5_f{i}" for i in range(5)]:
        f = folds[folds.split == s].set_index("method")
        md.append(f"| {s[-1]} | " + " | ".join(f"{f.loc[m, 'count_MAE']:.3f}" for m, _ in METHODS) + " |")
    md += ["", "| 분할 | 포즈 test pose mAP50-95 | PCK@0.1 | PCK@0.2 | box mAP50-95 | 놓친 돼지 (박스 IoU 0.5) |",
           "|---|---|---|---|---|---|"]
    for _, r in pose.iterrows():
        md.append(f"| {r.split} | {r['pose_mAP50-95']:.3f} | {r['PCK@0.1']:.3f} | {r['PCK@0.2']:.3f} | "
                  f"{r['box_mAP50-95']:.3f} | {int(r.missed)} |")
    (out / "split_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ---------------- figure ----------------
    colors = {"geo": "#b9b8b0", "rf": "#b9d3f2", "rf_geo": "#5b9be0", "rf_geo_leg": "#174a8c", "yolo": "#e8833a"}
    xlabels = {"random": "random\n(162/28)", "sorted": "time\n(159/31)", "k5": "time 5-fold\n(mean ± sd)"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    w = 0.8 / len(METHODS)
    for ax, col, title in [(axes[0], "count_MAE", "count error per frame, pigs (lower = better)"),
                           (axes[1], "total_error_pct", "total visiting count error, % (closer to 0 = better)")]:
        for i, (m, mlabel) in enumerate(METHODS):
            g = summ[summ.method == m].set_index("scheme").loc[[k for k, _, _ in SCHEMES]]
            xs = [j + (i - (len(METHODS) - 1) / 2) * w for j in range(len(SCHEMES))]
            err = g[f"{col}_std"].to_numpy()
            b = ax.bar(xs, g[col], w, color=colors[m], label=mlabel)
            ok = ~pd.isna(err)                               # 오차 막대는 5-fold(여러 run)에만
            if ok.any():
                ax.errorbar([x for x, k in zip(xs, ok) if k], g[col].to_numpy()[ok], yerr=err[ok], fmt="none",
                            capsize=2, elinewidth=0.8, ecolor="#555")
            ax.bar_label(b, fmt="%.2f" if col == "count_MAE" else "%+.1f", padding=2, fontsize=6.5)
        ax.set_xticks(range(len(SCHEMES)), [xlabels[k] for k, _, _ in SCHEMES])
        ax.set_title(title)
        ax.grid(axis="y", color="#e1e0d9")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].axhline(0, color="#888780", lw=0.8)
    lo, hi = axes[1].get_ylim()
    axes[1].set_ylim(lo, max(hi, 0) + (hi - lo) * 0.08)
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.15)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, ncol=5, loc="lower center")
    fig.suptitle("Visiting count on 000 (1-min sample video, 86 imgs) by train/valid split — "
                 "pose medium ep200 + RF trained on each split's train / YOLO detect ep200")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "split_comparison.png", dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
