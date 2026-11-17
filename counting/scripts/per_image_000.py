"""test(000) 86장 — 사진마다 각 모델이 센 visiting 마릿수를 사람 정답(GT)과 함께 표·그래프로.

입력: reports/fold_study/per_frame_000/<geo|rf|rf_geo|rf_geo_leg|yolo>.csv  (compare_folds.py 출력)
      열 = frame_idx, image, gt, k5_f0..k5_f4, rk5_f0..rk5_f4
출력: reports/fold_study/per_image/
    per_image_000_long.csv   한 줄 = 사진 × 분할 × fold × 방법 (image, frame_idx, gt, scheme, fold, method, pred, error)
    per_image_000_wide.csv   한 줄 = 사진, 열 = gt + <scheme>_<method>_f<i> 50개 + 방법·분할별 fold 평균
    per_image_000.md         사진별 표 (분할마다 한 표, 칸 = f0~f4 카운트) + 사진 단위 요약
    graph_E_per_image_counts.png   방법별 small multiples: GT 계단선 + fold 평균선 + fold 최소~최대 띠
    graph_F_per_image_error_heatmap.png  모델 50개 × 사진 86장 오차(예측 − 정답) heatmap, 맨 위 GT 띠
    by_model/<method>.csv    모델(방법)별 표: no, frame_idx, image, gt,
                             random_f0..f4, random_mean, random_n_correct, time_f0..f4, time_mean, time_n_correct
                             (YOLO 포함, n_correct = fold 5개 중 GT와 같은 개수)
    by_model/<method>.png    모델별 그래프: 행 = GT · 랜덤 f0~f4 · 시간대 f0~f4, 칸 숫자 = 센 마릿수, 칸 색 = 오차

사용법 (counting/ 에서):  python scripts/per_image_000.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "reports" / "fold_study"
OUT = FS / "per_image"
K = 5
METHODS = [("geo", "Geo"), ("rf", "RF"), ("rf_geo", "RF+Geo"), ("rf_geo_leg", "RF+Geo+Leg"), ("yolo", "YOLO")]
SCHEMES = [("random", "Random 5-fold", "랜덤 5-fold", "rk5"), ("time", "Time 5-fold", "시간대 5-fold", "k5")]
COLOR = {"gt": "#000000", "geo": "#8c8c8c", "rf": "#9ecae1", "rf_geo": "#4292c6", "rf_geo_leg": "#08519c",
         "yolo": "#f28e2b"}
LINE = {**COLOR, "rf": "#6baed6"}            # 연한 RF 파랑은 선으로 그리면 잘 안 보여 한 단계 진하게
# 오차 diverging: 덜 셈 = 파랑 계열, 더 셈 = 빨강 계열, 0 = 중립 회색 (−3 이하 / +3 이상은 끝 색)
ERR_COLORS = ["#08306b", "#2166ac", "#92c5de", "#eeeeee", "#f4a582", "#d6604d", "#67001f"]
ERR_LEVELS = [-3, -2, -1, 0, 1, 2, 3]


def load():
    frames = {}
    for m, _ in METHODS:
        f = FS / "per_frame_000" / f"{m}.csv"
        if f.exists():
            frames[m] = pd.read_csv(f, encoding="utf-8-sig").sort_values("frame_idx").reset_index(drop=True)
    base = next(iter(frames.values()))[["frame_idx", "image", "gt"]]
    for m, d in frames.items():
        assert (d["image"].to_numpy() == base["image"].to_numpy()).all() and (d["gt"] == base["gt"]).all(), m
    return base, frames


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    base, frames = load()
    n = len(base)

    long, wide = [], base.copy()
    wide.insert(0, "no", np.arange(1, n + 1))
    for key, _, _, p in SCHEMES:
        for m, _ in METHODS:
            if m not in frames:
                continue
            cols = [f"{p}_f{i}" for i in range(K)]
            for i, c in enumerate(cols):
                wide[f"{key}_{m}_f{i}"] = frames[m][c]
                long.append(pd.DataFrame({"image": base.image, "frame_idx": base.frame_idx, "gt": base["gt"],
                                          "scheme": key, "fold": f"f{i}", "method": m, "pred": frames[m][c],
                                          "error": frames[m][c] - base["gt"]}))
            wide[f"{key}_{m}_mean"] = frames[m][cols].mean(axis=1).round(2)
    long = pd.concat(long, ignore_index=True)
    long.to_csv(OUT / "per_image_000_long.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(OUT / "per_image_000_wide.csv", index=False, encoding="utf-8-sig")

    bm = OUT / "by_model"
    bm.mkdir(exist_ok=True)
    for m, lab in METHODS:
        if m not in frames:
            continue
        t = base.copy()
        t.insert(0, "no", np.arange(1, n + 1))
        for key, _, _, p in SCHEMES:
            cols = [f"{p}_f{i}" for i in range(K)]
            for i, c in enumerate(cols):
                t[f"{key}_f{i}"] = frames[m][c]
            P = frames[m][cols]
            t[f"{key}_mean"] = P.mean(axis=1).round(2)
            t[f"{key}_n_correct"] = P.eq(base["gt"], axis=0).sum(axis=1)
        t.to_csv(bm / f"{m}.csv", index=False, encoding="utf-8-sig")
        plot_model(base, frames[m], m, lab, bm / f"{m}.png")

    # ---------------- markdown ----------------
    md = ["# test(000) 사진별 카운트 — 사람 정답(GT)과 모델 비교", "",
          "000 86장(1분 영상, 시간순). 칸 = 그 방법의 fold 모델 5개(f0 f1 f2 f3 f4)가 센 visiting 마릿수. "
          "**굵게** = 5개 중 하나라도 GT와 다름. ✓ = 5개 모두 GT와 같음. 맨 오른쪽 = 모델 25개 중 GT와 같은 개수.", ""]
    for key, _, slab, p in SCHEMES:
        md += [f"## {slab}", "",
               "| # | frame | GT | " + " | ".join(lab for m, lab in METHODS if m in frames) + " | 정답 모델 수 |",
               "|---|---|---|" + "---|" * (len(frames) + 1)]
        for r in range(n):
            g = int(base["gt"][r])
            cells, ok = [], 0
            for m, _ in METHODS:
                if m not in frames:
                    continue
                v = [int(frames[m].loc[r, f"{p}_f{i}"]) for i in range(K)]
                ok += sum(x == g for x in v)
                s = " ".join(str(x) for x in v)
                cells.append("✓" if all(x == g for x in v) else f"**{s}**")
            md.append(f"| {r + 1} | {int(base.frame_idx[r])} | {g} | " + " | ".join(cells) + f" | {ok}/{K * len(frames)} |")
        md.append("")

    md += ["## 사진 단위 요약 (test 86장)", "",
           "| 분할 | 방법 | 5개 fold 모두 정답인 사진 | 1개 이상 틀린 사진 | 더 센 사진 (fold 합) | 덜 센 사진 (fold 합) | 가장 많이 틀린 사진 (# : 틀린 fold 수) |",
           "|---|---|---|---|---|---|---|"]
    for key, _, slab, p in SCHEMES:
        for m, lab in METHODS:
            if m not in frames:
                continue
            P = frames[m][[f"{p}_f{i}" for i in range(K)]].to_numpy()
            E = P - base["gt"].to_numpy()[:, None]
            wrong = (E != 0).sum(axis=1)
            top = np.argsort(-wrong, kind="stable")[:5]
            md.append(f"| {slab} | {lab} | {(wrong == 0).sum()} | {(wrong > 0).sum()} | {(E > 0).sum()} | "
                      f"{(E < 0).sum()} | " + ", ".join(f"{t + 1}:{wrong[t]}" for t in top if wrong[t] > 0) + " |")
    g = base["gt"].to_numpy()
    starts = [0] + [i for i in range(1, n) if g[i] != g[i - 1]]
    runs = [(s, (starts[k + 1] if k + 1 < len(starts) else n) - 1) for k, s in enumerate(starts)]
    md += ["", "GT 구간: " + ", ".join(f"{a + 1}\\~{b + 1}번 {g[a]}마리" if a != b else f"{a + 1}번 {g[a]}마리"
                                       for a, b in runs) + ".", "",
           "그래프: `graph_E_per_image_counts.png`(방법별 카운트), `graph_F_per_image_error_heatmap.png`(모델 50개 오차).",
           "",
           "모델별 표 · 그래프: `by_model/<geo|rf|rf_geo|rf_geo_leg|yolo>.csv` · `.png` "
           "(열 = no, frame_idx, image, gt, random_f0~f4, random_mean, random_n_correct, time_f0~f4, time_mean, time_n_correct).",
           ""]
    (OUT / "per_image_000.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    plot_e(base, frames)
    plot_f(base, frames)
    print("saved", OUT)


def plot_e(base, frames):
    """그래프 E: 행 = 방법, 열 = 분할. GT 계단선 + fold 평균선 + fold 최소~최대 띠."""
    ms = [(m, lab) for m, lab in METHODS if m in frames]
    fig, axes = plt.subplots(len(ms), 2, figsize=(15, 2.0 * len(ms) + 1.2), sharex=True, sharey=True)
    x = np.arange(len(base))
    for c, (key, elab, _, p) in enumerate(SCHEMES):
        for r, (m, lab) in enumerate(ms):
            ax = axes[r, c]
            P = frames[m][[f"{p}_f{i}" for i in range(K)]].to_numpy(float)
            ax.fill_between(x, P.min(1), P.max(1), step="mid", color=COLOR[m], alpha=0.25, lw=0,
                            label="fold min–max")
            ax.step(x, P.mean(1), where="mid", color=LINE[m], lw=1.6, label=f"{lab}, mean of 5 folds")
            ax.step(x, base["gt"], where="mid", color=COLOR["gt"], lw=1.4, ls="--", label="GT (human)")
            mae = np.abs(P - base["gt"].to_numpy()[:, None]).mean()
            ax.text(0.995, 0.96, f"MAE {mae:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color="#333")
            if r == 0:
                ax.set_title(elab, fontsize=11)
            if c == 0:
                ax.set_ylabel(lab, fontsize=10)
            ax.set_yticks([0, 1, 2, 3])
            ax.set_ylim(-0.25, 3.25)
            ax.grid(axis="y", color="#e5e5e5", lw=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Test image (000, time order, 1–86)")
        ax.set_xticks(np.arange(0, 86, 10), [str(i + 1) for i in range(0, 86, 10)])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color="#000", ls="--", lw=1.4, label="GT (human)"),
         Line2D([], [], color="#555", lw=1.6, label="Model, mean of 5 folds (method color)"),
         Patch(color="#555", alpha=0.25, label="Range over 5 folds (min–max)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Visiting pigs counted per test image: every method vs human GT", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "graph_E_per_image_counts.png", dpi=150)
    plt.close(fig)


def plot_model(base, fr, m, lab, path):
    """모델별 그래프: 행 = GT + 랜덤 f0~f4 + 시간대 f0~f4, 열 = 사진 86장. 칸 숫자 = 센 마릿수, 칸 색 = 오차."""
    gt = base["gt"].to_numpy()
    rows = [("GT (human)", gt, None)]
    for key, elab, _, p in SCHEMES:
        for i in range(K):
            v = fr[f"{p}_f{i}"].to_numpy()
            rows.append((f"{elab.split()[0]} f{i}", v, v - gt))
    n, R = len(gt), len(rows)
    E = np.vstack([np.zeros(n) if e is None else np.clip(e, -3, 3) for _, _, e in rows])
    cmap = ListedColormap(ERR_COLORS)
    norm = BoundaryNorm(np.arange(-3.5, 4.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(22, 0.42 * R + 1.6))
    im = ax.imshow(E, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.imshow(np.ma.masked_where(np.broadcast_to(np.arange(R)[:, None] != 0, (R, n)), np.tile(gt, (R, 1))), aspect="auto",
              cmap=ListedColormap(["#ffffff", "#9e9e9e", "#000000"]), vmin=-0.5, vmax=2.5, interpolation="nearest")
    for r, (_, v, e) in enumerate(rows):
        for j in range(n):
            dark = (r == 0 and v[j] == 2) or (e is not None and abs(e[j]) >= 2)
            ax.text(j, r, str(int(v[j])), ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if dark else "#222222",
                    fontweight="bold" if (e is not None and e[j] != 0) else "normal")
    ax.set_yticks(np.arange(R), [r[0] for r in rows], fontsize=9)
    ax.set_xticks(np.arange(0, n, 5), [str(i + 1) for i in range(0, n, 5)], fontsize=8)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, R, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", lw=0.8)
    ax.tick_params(which="minor", length=0)
    ax.axhline(0.5, color="#333", lw=1.6)
    ax.axhline(K + 0.5, color="#333", lw=1.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Test image (000, time order, 1–86)")
    cb = fig.colorbar(im, ax=ax, ticks=ERR_LEVELS, fraction=0.015, pad=0.01)
    cb.ax.set_yticklabels(["≤ −3", "−2", "−1", "0", "+1", "+2", "≥ +3"], fontsize=8)
    cb.set_label("Error = model − GT")
    acc = [(rows[r][2] == 0).mean() for r in range(1, R)]
    mae = [np.abs(rows[r][2]).mean() for r in range(1, R)]
    ax.set_title(f"{lab}: pigs counted per test image (numbers) vs human GT (top row); "
                 f"MAE random {np.mean(mae[:K]):.3f}, time {np.mean(mae[K:]):.3f}; "
                 f"exact {np.mean(acc[:K]) * 100:.1f}% / {np.mean(acc[K:]) * 100:.1f}%", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_f(base, frames):
    """그래프 F: 모델 50개(방법 × 분할 × fold) × 사진 86장의 오차(예측 − 정답). 맨 위 = GT 마릿수 띠."""
    ms = [(m, lab) for m, lab in METHODS if m in frames]
    rows, labels, groups = [], [], []
    for key, elab, _, p in SCHEMES:
        for m, lab in ms:
            for i in range(K):
                rows.append(frames[m][f"{p}_f{i}"].to_numpy() - base["gt"].to_numpy())
                labels.append(f"{lab} f{i}")
            groups.append((elab, lab))
    E = np.clip(np.vstack(rows), -3, 3)
    n_models = len(rows)
    cmap = ListedColormap(ERR_COLORS)
    norm = BoundaryNorm(np.arange(-3.5, 4.5, 1), cmap.N)

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, n_models * 0.9], width_ratios=[1, 0.012], hspace=0.04, wspace=0.02)
    ax_gt = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_gt)
    cax = fig.add_subplot(gs[1, 1])

    gt_cmap = ListedColormap(["#ffffff", "#9e9e9e", "#000000"])
    ax_gt.imshow(base["gt"].to_numpy()[None, :], aspect="auto", cmap=gt_cmap, vmin=-0.5, vmax=2.5,
                 interpolation="nearest")
    for j, g in enumerate(base["gt"]):
        ax_gt.text(j, 0, str(int(g)), ha="center", va="center", fontsize=6, color="#fff" if g == 2 else "#000")
    ax_gt.set_yticks([0], ["GT (human)"])
    ax_gt.tick_params(axis="x", bottom=False, labelbottom=False)
    for s in ax_gt.spines.values():
        s.set_visible(False)

    im = ax.imshow(E, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(np.arange(n_models), labels, fontsize=6.5)
    ax.set_xticks(np.arange(0, 86, 5), [str(i + 1) for i in range(0, 86, 5)], fontsize=8)
    ax.set_xlabel("Test image (000, time order, 1–86)")
    ax.set_xticks(np.arange(-0.5, 86, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_models, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", lw=0.6)
    ax.tick_params(which="minor", length=0)
    for b in range(K, n_models, K):                                   # 방법 사이 구분선
        ax.axhline(b - 0.5, color="#555", lw=0.8 if b % (K * len(ms)) else 2.2)
    for gi, (elab, lab) in enumerate(groups):                          # 왼쪽 분할 표시
        if gi % len(ms) == 0:
            mid = gi * K + K * len(ms) / 2 - 0.5
            ax.text(-0.085, mid, elab, rotation=90, va="center", ha="center", fontsize=11, fontweight="bold",
                    transform=ax.get_yaxis_transform())
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, cax=cax, ticks=ERR_LEVELS)
    cb.ax.set_yticklabels(["≤ −3", "−2", "−1", "0 (correct)", "+1", "+2", "≥ +3"], fontsize=8)
    cb.set_label("Error = model − GT (pigs)")
    fig.suptitle("Per-image counting error of all 50 models on test (000): blue = undercount, red = overcount",
                 fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.11, right=0.93, top=0.965, bottom=0.05)
    fig.savefig(OUT / "graph_F_per_image_error_heatmap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
