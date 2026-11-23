"""헝가리안 짝짓기 기준 통일: 자세 추정 기반 방법도 객체 검출과 같은 '상자 IoU ≥ 0.5'로 짝지어 오류 분해 · P/R/F1을 다시 계산.

기존 평가: 자세 추정 기반 = 관절 평균 거리 ÷ 정답 상자 대각선 ≤ 0.25, 객체 검출 = 상자 IoU ≥ 0.5.
여기서는 자세 추정 모델이 내는 돼지 상자와 정답 상자(스켈레톤 점의 최소·최대 좌표)를 IoU로 짝지음.
사진당 마릿수(MAE 등)는 짝짓기와 무관하므로 바뀌지 않음.

입력: paper_figures.py의 추론 캐시(fold별, 없으면 만듦), fold_metrics.csv (객체 검출 값)
출력: reports/fold_study/paper/iou_matching.md, iou_matching.csv
사용법 (counting/ 에서):  python scripts/analyze_iou_matching.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figures as pf  # noqa: E402

OUT = pf.FS / "paper"
CACHE = Path(r"C:/ETRI/share/hci2027_figs")
SPLITS = [f"rk5_f{i}" for i in range(5)] + [f"k5_f{i}" for i in range(5)]
METHODS = ["geo", "rf", "rf_geo", "rf_geo_leg"]


def iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    _, _, test = pf.cams_rows()
    gt_by = pf.by_image(test)
    rows, inst = [], []
    for split in SPLITS:
        cache = pf.build_cache(split, CACHE)
        for m in METHODS:
            d = dict(undet=0, judged=0, wrong=0, extra=0, human_vis=0)
            for stem, gts in gt_by.items():
                gl = np.array([g["label"] for g in gts])
                gb = [g["det"].box for g in gts]
                rec = cache[stem]
                pb, pv = rec["boxes"], np.asarray(rec["pred"].get(m, []))
                d["human_vis"] += int(gl.sum())
                pairs = []
                if len(pb):
                    M = iou(gb, pb)
                    r, c = linear_sum_assignment(-M)
                    pairs = [(i, j) for i, j in zip(r, c) if M[i, j] >= 0.5]
                mg, mp = {i for i, _ in pairs}, {j for _, j in pairs}
                d["undet"] += int(sum(gl[i] == 1 for i in range(len(gl)) if i not in mg))
                for i, j in pairs:
                    d["judged"] += int(gl[i] == 1 and pv[j] == 0)
                    d["wrong"] += int(gl[i] == 0 and pv[j] == 1)
                d["extra"] += int(sum(pv[j] == 1 for j in range(len(pv)) if j not in mp))
            rows.append({"split": split, "method": m, **d})
        # 다리 특징 효과 (RF+Geo vs RF+Geo+Leg), 같은 IoU 짝짓기로: 짝지어진 사람 visiting · other 개체
        for stem, gts in gt_by.items():
            rec = cache[stem]
            if not rec["boxes"]:
                continue
            gb = [g["det"].box for g in gts]
            M = iou(gb, rec["boxes"])
            r, c = linear_sum_assignment(-M)
            for i, j in zip(r, c):
                if M[i, j] >= 0.5:
                    inst.append({"split": split, "gt_label": gts[i]["label"], "dist_r": rec["dist_r"][j],
                                 "rf_geo": rec["pred"]["rf_geo"][j], "rf_geo_leg": rec["pred"]["rf_geo_leg"][j]})
    df = pd.DataFrame(rows)
    fm = pd.read_csv(pf.FS / "fold_metrics.csv", encoding="utf-8-sig")
    y = fm[(fm.set == "test") & (fm.method == "yolo")][["split", "total_human", "missed_visiting_undetected",
                                                       "missed_visiting_judged", "wrong_visiting", "extra_visiting"]]
    y.columns = ["split", "human_vis", "undet", "judged", "wrong", "extra"]
    df = pd.concat([df, y.assign(method="yolo")], ignore_index=True)
    tp = df.human_vis - df.undet - df.judged
    df["precision"] = tp / (tp + df.wrong + df.extra)
    df["recall"] = tp / df.human_vis
    df["f1"] = 2 * df.precision * df.recall / (df.precision + df.recall)
    df.to_csv(OUT / "iou_matching.csv", index=False, encoding="utf-8-sig")

    old = fm[(fm.set == "test")].copy()
    t_old = old.total_human - old.missed_visiting_undetected - old.missed_visiting_judged
    old["f1_kpt"] = 2 * t_old / (2 * t_old + old.wrong_visiting + old.extra_visiting + old.missed_visiting_undetected
                                 + old.missed_visiting_judged)
    md = ["# 짝짓기 기준 통일 (모든 방법 상자 IoU ≥ 0.5)", "",
          "자세 추정 기반 방법을 객체 검출과 같은 상자 IoU ≥ 0.5로 다시 짝지은 결과. 사진당 마릿수 지표는 변하지 않음.", ""]
    for pre, lab in [("rk5", "Random 5-fold"), ("k5", "Time 5-fold")]:
        md += [f"## {lab} (Test, fold당 마릿수, fold 5개 평균)", "",
               "| 방법 | 미검출 | visiting → other | other → visiting | 라벨 없는 검출 → visiting | Precision | Recall | F1 | (기존 관절 기준 F1) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for m in METHODS + ["yolo"]:
            g = df[(df.split.str.startswith(pre + "_")) & (df.method == m)]
            o = old[(old.split.str.startswith(pre + "_")) & (old.method == m)]
            md.append(f"| {pf.NAME[m]} | {g.undet.mean():.1f} | {g.judged.mean():.1f} | {g.wrong.mean():.1f} | "
                      f"{g.extra.mean():.1f} | {g.precision.mean():.3f} | {g.recall.mean():.3f} | {g.f1.mean():.3f} | "
                      f"{o.f1_kpt.mean():.3f} |")
        md.append("")
    it = pd.DataFrame(inst)
    vis, oth = it[it["gt_label"] == 1], it[it["gt_label"] == 0]
    edge = lambda x: x[x.dist_r.between(0.85, 1.1)]  # noqa: E731
    fix = vis[(vis.rf_geo == 0) & (vis.rf_geo_leg == 1)]
    brk = vis[(vis.rf_geo == 1) & (vis.rf_geo_leg == 0)]
    md += ["## 다리 특징 효과 (같은 IoU 짝짓기, fold 10개 합)", "",
           "| 항목 | RF(자세+기하) | RF(자세+기하+다리) |", "|---|---|---|",
           f"| 가장자리(거리비 0.85~1.1) visiting을 visiting으로 판정한 비율 (n={len(edge(vis))}) | "
           f"{edge(vis).rf_geo.mean():.3f} | {edge(vis).rf_geo_leg.mean():.3f} |",
           f"| 안쪽(거리비 0.85 미만) visiting을 visiting으로 판정한 비율 (n={len(vis[vis.dist_r < 0.85])}) | "
           f"{vis[vis.dist_r < 0.85].rf_geo.mean():.3f} | {vis[vis.dist_r < 0.85].rf_geo_leg.mean():.3f} |",
           f"| 가장자리 other를 visiting으로 판정한 비율 (n={len(edge(oth))}) | "
           f"{edge(oth).rf_geo.mean():.3f} | {edge(oth).rf_geo_leg.mean():.3f} |", "",
           f"RF(자세+기하)가 other로 판정한 visiting을 RF(자세+기하+다리)가 바로잡은 경우 {len(fix)}건 "
           f"(Random {fix.split.str.startswith('rk5').sum()}, Time {fix.split.str.startswith('k5').sum()}), 반대 경우 {len(brk)}건.", ""]
    (OUT / "iou_matching.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
