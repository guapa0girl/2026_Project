"""Split the 50 held-out eval images (pose-training never saw them) 40:10 and compare
Geo / RF / RF+Geo / RF+Geo+Leg on the 10, with the RFs trained on the other 40.

Both training and test use MODEL keypoints (the pose model is run on every image), so
this is the operational setting: RF learns from the same kind of keypoints it is
applied to. Training rows = model detections matched to a human skeleton (label from
the human State); unmatched model detections have no label and are not used for
training. Test is scored exactly like evaluate_methods.py Stage 2 (human pigs the model
missed count as "other", per-image visiting count uses ALL model detections).

Split is by CLIP, not by image: frames from one clip are ~2 s apart (near-duplicates),
so an image-level split would leak the test scenes into training.
  main split : the clip combination with exactly --test-images images whose visiting
               ratio is closest to the overall ratio (ties -> more clips -> name order)
  robustness : every clip combination with exactly --test-images images (sampled up to
               --max-splits), mean +- std over those splits

Usage (from counting/):
    python scripts/eval_split50.py --cameras configs/eval_cameras.yaml \
        --xml data/labels/ch010_eval50.xml --images data/labels/eval50_img \
        --only-list ../training/eval_frames.txt --out outputs/eval_split50
"""
import argparse
import itertools
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_methods as em  # noqa: E402
from feeding_monitor.method_features import METHOD_FEATURES, METHODS  # noqa: E402
from feeding_monitor.pose_tracker import PoseTracker, _parse_detections  # noqa: E402


def detect_all(gt_rows, images, cams, thresh):
    """Run the pose model once per image; return per-detection features + match info."""
    by_image = {}
    for r in gt_rows:
        by_image.setdefault(r["image"], []).append(r)
    trackers, det_rows, gt_out = {}, [], []
    for stem, rows in sorted(by_image.items()):
        cam = rows[0]["camera"]
        cfg = cams.cfg[cam]
        if cam not in trackers:
            trackers[cam] = PoseTracker(cfg.model)
        tr = trackers[cam]
        res = tr.model.predict(cv2.imread(str(images[stem])), conf=cfg.model.conf, imgsz=cfg.model.imgsz,
                               iou=cfg.model.iou, device=tr.device, verbose=False)[0]
        preds = _parse_detections(res, 0)
        pairs = dict((j, i) for i, j in em.match([r["det"] for r in rows], preds, thresh))
        pf = em.featurize([{"image": stem, "camera": cam, "clip": rows[0]["clip"], "label": -1, "det": d,
                            "source": "model"} for d in preds], cams) if preds else pd.DataFrame()
        for j in range(len(preds)):
            rec = pf.iloc[j].to_dict()
            rec["gt_label"] = rows[pairs[j]]["label"] if j in pairs else -1
            rec["gt_index"] = pairs.get(j, -1)
            det_rows.append(rec)
        for i, r in enumerate(rows):
            gt_out.append({"image": stem, "clip": r["clip"], "gt_index": i, "label": r["label"],
                           "matched": i in pairs.values()})
    return pd.DataFrame(det_rows), pd.DataFrame(gt_out)


def score(dets, gts, test_images, train_images, seed):
    """Train RFs on matched detections of train_images, score all methods on test_images."""
    tr = dets[dets.image.isin(train_images) & (dets.gt_label >= 0)]
    te = dets[dets.image.isin(test_images)].copy()
    for m, feats in METHOD_FEATURES.items():
        clf = em.new_rf(seed).fit(tr[feats], tr["gt_label"].astype(int))
        te[f"pred_{m}"] = clf.predict(te[feats]) if len(te) else []
    te["pred_geo"] = te["pred_geo"].astype(int)
    g = gts[gts.image.isin(test_images)]
    rows = []
    for m in METHODS:
        col = f"pred_{m}"
        # instance level: each human pig -> prediction of its matched detection, else "other"
        hit = te[te.gt_index >= 0].set_index(["image", "gt_index"])[col]
        p = [int(hit.get((r.image, r.gt_index), 0)) for r in g.itertuples()]
        met = em.metrics(g["label"].values, p)
        cnt = te.groupby("image")[col].sum().reindex(sorted(set(test_images)), fill_value=0)
        hum = g.groupby("image")["label"].sum().reindex(cnt.index, fill_value=0)
        d = cnt - hum
        met.update({"count_MAE": d.abs().mean(), "count_bias": d.mean(), "exact_count_rate": (d == 0).mean()})
        rows.append({"method": m, **met})
    return pd.DataFrame(rows).set_index("method"), len(tr), int(tr.gt_label.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--only-list", default=None)
    ap.add_argument("--test-images", type=int, default=10)
    ap.add_argument("--max-splits", type=int, default=50)
    ap.add_argument("--match-thresh", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cams = em.Cameras(a.cameras)
    keep = em.read_list(a.only_list)
    gt_rows = [r for r in em.parse_cvat_xml(a.xml) if (not keep or r["image"] in keep) and cams.has(r["camera"])]
    images = em.find_images(Path(a.images))
    dets, gts = detect_all(gt_rows, images, cams, a.match_thresh)
    print(f"[detect] 이미지 {gts.image.nunique()}장, 사람 개체 {len(gts)} (visiting {int(gts.label.sum())}), "
          f"모델 검출 {len(dets)}개 중 매칭 {int((dets.gt_index >= 0).sum())}")

    clip_img = gts.groupby("clip").image.nunique()
    clip_vis = gts.groupby("clip").label.sum()
    clip_n = gts.groupby("clip").size()
    overall = gts.label.mean()
    combos = []
    clips = sorted(clip_img.index)
    for k in range(1, len(clips) + 1):
        for c in itertools.combinations(clips, k):
            if clip_img[list(c)].sum() == a.test_images and clip_vis[list(c)].sum() > 0:
                ratio = clip_vis[list(c)].sum() / clip_n[list(c)].sum()
                combos.append((abs(ratio - overall), -len(c), c))
    if not combos:
        raise SystemExit(f"정확히 {a.test_images}장이 되는 클립 조합이 없습니다")
    combos.sort()
    main_clips = combos[0][2]

    def images_of(cs):
        return sorted(gts[gts["clip"].isin(cs)].image.unique())

    test_imgs = images_of(main_clips)
    train_imgs = sorted(set(gts.image) - set(test_imgs))
    table, n_tr, n_tr_vis = score(dets, gts, test_imgs, train_imgs, a.seed)
    g_te = gts[gts.image.isin(test_imgs)]
    table.to_csv(out / "split_main_metrics.csv", encoding="utf-8-sig")
    pd.DataFrame({"image": test_imgs}).to_csv(out / "split_main_test_images.csv", index=False, encoding="utf-8-sig")

    rnd = random.Random(a.seed)
    pool = [c for _, _, c in combos]
    sample = pool if len(pool) <= a.max_splits else rnd.sample(pool, a.max_splits)
    rep = []
    for c in sample:
        ti = images_of(c)
        t, _, _ = score(dets, gts, ti, sorted(set(gts.image) - set(ti)), a.seed)
        rep.append(t.assign(split="+".join(x[-6:] for x in c)))
    rep = pd.concat(rep).reset_index()
    rep.to_csv(out / "split_repeats.csv", index=False, encoding="utf-8-sig")
    summ = rep.groupby("method")[["visiting_F1", "precision", "recall", "count_MAE", "count_bias"]].agg(["mean", "std"])
    summ = summ.reindex(METHODS)

    cols = ["n", "n_visiting", "visiting_F1", "precision", "recall", "TP", "FP", "FN", "TN",
            "count_MAE", "count_bias", "exact_count_rate"]
    lines = [
        "# 평가 50장 40:10 분할 — Geo / RF / RF+Geo / RF+Geo+Leg",
        "",
        f"- 학습 40장: 모델 검출 중 사람 라벨과 짝지어진 {n_tr}개 (visiting {n_tr_vis}) 로 RF 학습 (모델 keypoint)",
        f"- 시험 10장: 클립 {len(main_clips)}개 {list(main_clips)}, 사람 개체 {len(g_te)} (visiting {int(g_te.label.sum())}, "
        f"{g_te.label.mean():.1%}; 전체 {overall:.1%})",
        "- 분할은 클립 단위 (같은 클립 사진은 2초 간격 비슷한 장면이라 섞으면 누수)",
        "",
        "## 시험 10장 결과",
        "```", em.fmt_table(table[cols]), "```",
        "",
        f"## 다른 40:10 클립 조합 {len(sample)}개 평균 ± 표준편차 (정확히 10장이 되는 조합 {len(pool)}개 중)",
        "```", em.fmt_table(summ), "```",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    plot(table, summ, len(sample), out / "split50.png")


def plot(table, summ, n_rep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"geo": "Geo\n(rule)", "rf": "RF", "rf_geo": "RF+Geo", "rf_geo_leg": "RF+Geo\n+Leg"}
    x = np.arange(len(METHODS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, col, title, high in [(axes[0], "visiting_F1", "visiting F1 (higher = better)", True),
                                 (axes[1], "count_MAE", "count error per image, pigs (lower = better)", False)]:
        main = table.loc[METHODS, col].values
        mean, std = summ[(col, "mean")].values, summ[(col, "std")].values
        best = int(np.argmax(main) if high else np.argmin(main))
        bars = ax.bar(x - 0.18, main, 0.36, color=["#2a78d6" if i == best else "#b9b8b0" for i in range(len(x))],
                      label="test 10 images (main split)")
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
        ax.errorbar(x + 0.18, mean, yerr=std, fmt="o", color="#e07b39", capsize=4,
                    label=f"mean ± std over {n_rep} other 40:10 splits")
        ax.set_xticks(x, [names[m] for m in METHODS])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.1 if col == "visiting_F1" else max(main.max(), (mean + std).max()) * 1.3)
        ax.grid(axis="y", color="#e1e0d9")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Eval 50 images split 40:10 by clip (CH010, model keypoints): RF trained on 40, tested on 10")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    main()
