"""test(000) 사진별 카운트: 사람 정답 vs Geo / RF+Geo+Leg(RF 계열 대표) / YOLO — 논문용으로 방법마다 칸을 나눈 그림.

입력: reports/fold_study/per_frame_000/<method>.csv  (compare_folds.py 결과; gt, k5_f0..f4, rk5_f0..f4)
출력: reports/fold_study/
    graph_H_counts_3methods.png       3행(방법) × 2열(Random 5-fold · Time 5-fold).
                                      위 = 사진별 visiting 마릿수(회색 면 = 사람 정답, 선 = fold 5개 평균, 띠 = 최소~최대),
                                      아래 = 사진별 평균 오차(예측 − 정답). 칸마다 MAE · 총 마릿수 표시
    graph_I_cumulative_3methods.png   사진 순서대로 누적한 visiting 총 마릿수 (사람 133 vs 방법별 fold 평균)
사용법 (counting/ 에서):  python scripts/plot_counts_3methods.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "reports" / "fold_study"
K = 5
METHODS = [("geo", "Geo"), ("rf_geo_leg", "RF+Geo+Leg"), ("yolo", "YOLO")]
SCHEMES = [("rk5", "Random 5-fold"), ("k5", "Time 5-fold")]
COLOR = {"gt": "#000000", "geo": "#5a5a5a", "rf_geo_leg": "#08519c", "yolo": "#e8710a"}
GT_FILL = "#ededed"
FPS = 24          # 000 = 1분 샘플 영상, 프레임 번호 0~1440 → 1440 / 60 s = 24 fps로 환산 (초 = 프레임 번호 / 24)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})


def load(method, prefix):
    d = pd.read_csv(FS / "per_frame_000" / f"{method}.csv", encoding="utf-8-sig").sort_values("frame_idx")
    pred = d[[f"{prefix}_f{i}" for i in range(K)]].to_numpy(float).T     # (5, 86)
    return d["gt"].to_numpy(float), pred, d["frame_idx"].to_numpy(float) / FPS


def fold_stats(gt, pred):
    mae = np.abs(pred - gt).mean(1)
    tot = pred.sum(1)
    return mae.mean(), mae.std(ddof=1), tot.mean(), tot.std(ddof=1)


def plot_h():
    fig = plt.figure(figsize=(14, 10.5))
    outer = fig.add_gridspec(len(METHODS), len(SCHEMES), hspace=0.42, wspace=0.08)
    for r, (m, mlab) in enumerate(METHODS):
        for c, (p, slab) in enumerate(SCHEMES):
            gt, pred, x = load(m, p)
            mean = pred.mean(0)
            inner = outer[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
            ax, ae = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])

            ax.fill_between(x, 0, gt, step="mid", color=GT_FILL, lw=0, label="Human GT")
            ax.step(x, gt, where="mid", color=COLOR["gt"], lw=1.0)
            ax.fill_between(x, pred.min(0), pred.max(0), step="mid", color=COLOR[m], alpha=0.18, lw=0,
                            label="min–max of 5 folds")
            ax.step(x, mean, where="mid", color=COLOR[m], lw=2.0, label=f"{mlab}, mean of 5 folds")
            mae, mae_sd, tot, tot_sd = fold_stats(gt, pred)
            ax.text(0.99, 0.96, f"MAE {mae:.3f} ± {mae_sd:.3f}\ncounted {tot:.1f} / {gt.sum():.0f} "
                    f"({(tot - gt.sum()) / gt.sum() * 100:+.1f}%)", transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color="#333333", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.6))
            ax.set_ylim(0, 3.2)
            ax.set_yticks([0, 1, 2, 3])
            ax.set_xlim(-0.5, 60.5)
            ax.tick_params(labelbottom=False)
            ax.grid(axis="y", color="#ececec", lw=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{mlab} — {slab}", loc="left", fontweight="bold" if m == "rf_geo_leg" else "normal")
            if c == 0:
                ax.set_ylabel("Count")
            else:
                ax.tick_params(labelleft=False)

            err = mean - gt
            ae.bar(x, err, width=0.3, color=COLOR[m], lw=0)
            ae.axhline(0, color="#999999", lw=0.8)
            ae.set_ylim(-2.1, 2.1)
            ae.set_yticks([-2, 0, 2])
            ae.set_xlim(-0.5, 60.5)
            ae.grid(axis="y", color="#ececec", lw=0.8)
            ae.set_axisbelow(True)
            ae.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ae.set_ylabel("Error", fontsize=9)
            else:
                ae.tick_params(labelleft=False)
            if r == len(METHODS) - 1:
                ae.set_xlabel("Time in test(000) video (s)")
                ae.set_xticks(range(0, 61, 10))
            else:
                ae.tick_params(labelbottom=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(fc=GT_FILL, ec=COLOR["gt"], lw=1.0, label="Human GT"),
               *[Line2D([], [], color=COLOR[m], lw=2.0, label=f"{lab} (mean of 5 folds)") for m, lab in METHODS],
               Patch(fc="#9e9e9e", alpha=0.35, lw=0, label="min–max of 5 folds")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Visiting pigs per image on test(000): human GT vs Geo, RF+Geo+Leg, YOLO\n"
                 "(upper: count per image; lower: mean error = predicted − GT; 86 images over 60 s)", fontsize=12, y=0.99)
    fig.subplots_adjust(top=0.89)
    fig.savefig(FS / "graph_H_counts_3methods.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_i():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (p, slab) in zip(axes, SCHEMES):
        gt = None
        ends = []
        for m, mlab in METHODS:
            g, pred, x = load(m, p)
            gt = g
            cum = np.cumsum(pred, axis=1)
            ax.fill_between(x, cum.min(0), cum.max(0), color=COLOR[m], alpha=0.15, lw=0)
            ax.plot(x, cum.mean(0), color=COLOR[m], lw=2.0, label=f"{mlab} (mean of 5 folds)")
            ends.append((cum.mean(0)[-1], mlab, COLOR[m]))
        ax.plot(x, np.cumsum(gt), color=COLOR["gt"], lw=2.2, ls="--", label="Human GT")
        ends.append((gt.sum(), "Human GT", COLOR["gt"]))
        # 선 끝 직접 라벨 (겹치지 않게 최소 간격 확보)
        ends.sort()
        ys, last = [], -1e9
        for v, _, _ in ends:
            y = max(v, last + 7.5)
            ys.append(y)
            last = y
        for (v, lab, col), y in zip(ends, ys):
            ax.annotate(f"{lab} {v:.1f}" if lab != "Human GT" else f"{lab} {v:.0f}", xy=(60, v),
                        xytext=(61.5, y), fontsize=9, color="#222222", va="center",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
        ax.set_xlim(0, 60.5)
        ax.set_xticks(range(0, 61, 10))
        ax.set_title(slab, loc="left")
        ax.set_xlabel("Time in test(000) video (s)")
        ax.grid(color="#ececec", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Cumulative count")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Cumulative visiting count over test(000): human GT vs Geo, RF+Geo+Leg, YOLO "
                 "(line = mean of 5 fold models, band = min–max)", fontsize=11.5)
    fig.tight_layout()
    fig.subplots_adjust(right=0.9, wspace=0.28)
    fig.savefig(FS / "graph_I_cumulative_3methods.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_h()
    plot_i()
    print(f"[saved] {FS / 'graph_H_counts_3methods.png'}, {FS / 'graph_I_cumulative_3methods.png'}")
