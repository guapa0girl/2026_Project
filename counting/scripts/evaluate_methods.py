"""Compare three ways of deciding "is this pig visiting the feeder?" against
human State labels (CVAT), in two stages.

  Stage 1 -- method comparison on HUMAN-CORRECTED keypoints
      Data : CVAT for images 1.1 XML(s) with skeletons + State attribute
             (final study: tasks 001 CH005 + 002 CH010, the pose-training frames).
      Why  : the pose model never enters, so it doesn't matter that these
             frames were used to train it. Isolates "which classifier is
             better given good keypoints".
      How  : clip-level StratifiedGroupKFold. Geo rule needs no training;
             RF / RF+Geo are trained on the other folds (out-of-fold preds).

  Stage 2 -- end-to-end on HELD-OUT frames with MODEL keypoints
      Data : held-out labeled frames the pose model never saw + their images
             (final study: task 000, the 1-minute sample video, 86 imgs).
      How  : run the pose model, match each detection to a human skeleton
             (Hungarian on keypoint distance), then apply Geo rule and the
             RF / RF+Geo models trained on ALL stage-1 rows.
             Also reports per-frame visiting COUNT error -- that is what the
             10-min occupancy graph actually uses.

Methods
  geo     : feeding_classifier.classify()  (ring + neck->nose direction
            [+ posture if enabled in the camera's config])
  rf      : RandomForest on the 6 posture_features (README "RF 단독")
  rf_geo  : rf features + Geo rule values (nose distance / r, head angle to
            feeder center, 3 posture signals)
  rf_geo_leg : rf_geo + leg-geometry lying features (leg angle / extension /
            side bias -- the "ㄷ" sleeping shape; src/feeding_monitor/leg_features.py)

Usage (from counting/):
  ..\\.venv\\Scripts\\python scripts\\evaluate_methods.py ^
      --cameras configs\\eval_cameras.yaml ^
      --stage1-xml data\\labels\\state_001.xml data\\labels\\state_002.xml ^
      --stage2-xml data\\labels\\state_000.xml ^
      --stage2-images "<000 이미지 폴더>" ^
      --out outputs\\eval
  Either stage can be run alone (omit the other stage's args).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feeding_monitor.config import load_config                      # noqa: E402
from feeding_monitor.method_features import (LEG_FEATURES, METHOD_FEATURES, METHODS,  # noqa: E402
                                             featurize_det)
from feeding_monitor.pose_tracker import Detection                    # noqa: E402
from feeding_monitor.posture_features import load_zones               # noqa: E402

# CVAT skeleton node order == pose model keypoint order (verified 2026-11-04/05)
KPT_ORDER = ["tail_tip", "tail", "l_hip", "r_hip", "l_foot", "r_foot", "r_flank", "l_flank",
             "l_shoulder", "r_shoulder", "l_hand", "neck", "nose", "l_ear", "l_ear_tip",
             "r_ear", "r_ear_tip", "r_hand"]

STATE_MAP = {
    "visitig(feeding)": 1,    # CH010 task typo, stored as-is
    "visiting(feeding)": 1,
    "visiting": 1,
    "other": 0,
    "unlabeling": None,       # not labeled -> dropped
    "unlabeled": None,
}



# --------------------------------------------------------------------------
# filenames -> camera / clip
# --------------------------------------------------------------------------

def camera_of(name: str) -> str:
    m = re.search(r"CH\d{3}", name)
    return m.group(0) if m else "SAMPLE"


def clip_of(name: str) -> str:
    """Group id for StratifiedGroupKFold: camera + clip start time."""
    cam = camera_of(name)
    m = re.search(r"(\d{14})-(\d{14})", name)
    return f"{cam}_{m.group(1)}" if m else f"{cam}_single_clip"


# --------------------------------------------------------------------------
# CVAT XML
# --------------------------------------------------------------------------

def read_list(path) -> set[str]:
    if not path:
        return set()
    return {Path(l.strip()).stem for l in open(path, encoding="utf-8") if l.strip()}


def parse_cvat_xml(path: str | Path) -> list[dict]:
    """One row per human skeleton that has a usable State label."""
    rows, unknown = [], set()
    root = ET.parse(path).getroot()
    for im in root.iter("image"):
        stem = Path(im.get("name")).stem
        for sk in im.findall("skeleton"):
            raw = None
            for a in sk.findall("attribute"):
                if a.get("name", "").lower() == "state":
                    raw = (a.text or "").strip()
            if raw is None:
                continue
            key = raw.lower()
            if key not in STATE_MAP:
                unknown.add(raw)
                continue
            label = STATE_MAP[key]
            if label is None:
                continue
            pts = {p.get("label").lower(): p for p in sk.findall("points")}
            kpts, xs, ys = [], [], []
            for nm in KPT_ORDER:
                p = pts.get(nm)
                if p is None or p.get("outside") == "1":
                    kpts.append((0.0, 0.0, 0.0))
                    continue
                x, y = map(float, p.get("points").split(";")[0].split(","))
                kpts.append((x, y, 1.0))   # human point -> full confidence; occluded flag deliberately ignored
                xs.append(x)
                ys.append(y)
            if not xs:
                continue
            det = Detection(frame_idx=0, track_id=None, det_conf=1.0,
                            box=(min(xs), min(ys), max(xs), max(ys)), kpts=kpts)
            rows.append({"image": stem, "camera": camera_of(stem), "clip": clip_of(stem),
                         "label": label, "det": det, "source": Path(path).name})
    if unknown:
        raise ValueError(f"{path}: 알 수 없는 State 값 {sorted(unknown)} -> STATE_MAP에 추가하세요")
    return rows


# --------------------------------------------------------------------------
# cameras (feeder ROI + rule + zones per camera)
# --------------------------------------------------------------------------

class Cameras:
    def __init__(self, path: str | Path):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["cameras"]
        self.cfg, self.zones = {}, {}
        for cam, v in raw.items():
            self.cfg[cam] = load_config(ROOT / v["config"])
            self.zones[cam] = load_zones(ROOT / v["zones"])

    def has(self, cam: str) -> bool:
        return cam in self.cfg


# --------------------------------------------------------------------------
# features + geo rule
# --------------------------------------------------------------------------

def featurize(rows: list[dict], cams: Cameras) -> pd.DataFrame:
    out = []
    for r in rows:
        rec = {k: v for k, v in r.items() if k != "det"}
        rec.update(featurize_det(r["det"], cams.cfg[r["camera"]], cams.zones[r["camera"]]))
        out.append(rec)
    return pd.DataFrame(out)


def new_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"n": len(y), "n_visiting": int(y.sum()),
            "visiting_F1": f1_score(y, p, zero_division=0),
            "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0),
            "macro_F1": f1_score(y, p, average="macro", zero_division=0),
            "accuracy": float((y == p).mean()),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def fmt_table(df: pd.DataFrame) -> str:
    return df.to_string(float_format=lambda v: f"{v:.3f}")


# --------------------------------------------------------------------------
# stage 1
# --------------------------------------------------------------------------

def run_stage1(df: pd.DataFrame, out: Path, n_splits: int, seed: int) -> tuple[list[str], str]:
    lines = []
    groups = df["clip"].values
    y = df["label"].values
    n_groups = len(set(groups))
    k = min(n_splits, n_groups)
    pos_groups = df.loc[df.label == 1, "clip"].nunique()
    if k < 2 or pos_groups < 2:
        msg = f"[stage1] CV 불가: 클립 {n_groups}개, visiting이 있는 클립 {pos_groups}개 (각각 2개 이상 필요)"
        print(msg)
        return [msg], ""
    k = min(k, pos_groups)

    for m in METHOD_FEATURES:
        df[f"pred_{m}"] = -1
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_rows = []
    for f, (tr, te) in enumerate(sgkf.split(df, y, groups)):
        df.iloc[te, df.columns.get_loc("fold")] = f
        for m, feats in METHOD_FEATURES.items():
            clf = new_rf(seed).fit(df.iloc[tr][feats], y[tr])
            df.iloc[te, df.columns.get_loc(f"pred_{m}")] = clf.predict(df.iloc[te][feats])
        for m in METHODS:
            fold_rows.append({"fold": f, "method": m, "n_test": len(te),
                              "visiting_F1": f1_score(y[te], df.iloc[te][f"pred_{m}"], zero_division=0)})

    overall = pd.DataFrame([{"method": m, **metrics(y, df[f"pred_{m}"])} for m in METHODS]).set_index("method")
    folds = pd.DataFrame(fold_rows)
    fold_summary = folds.groupby("method")["visiting_F1"].agg(["mean", "std"]).rename(
        columns={"mean": "fold_F1_mean", "std": "fold_F1_std"})
    overall = overall.join(fold_summary)

    per_cam = pd.DataFrame([{"camera": c, "method": m, **metrics(g["label"], g[f"pred_{m}"])}
                            for c, g in df.groupby("camera") for m in METHODS])

    allf = METHOD_FEATURES["rf_geo_leg"]
    imp = pd.Series(new_rf(seed).fit(df[allf], y).feature_importances_, index=allf).sort_values(ascending=False)
    # leg features: how visiting vs other differ (helps decide a leg-based lying rule later)
    leg = df[LEG_FEATURES].where(df[LEG_FEATURES] != -1.0)
    leg_summary = leg.groupby(df["label"].map({1: "visiting", 0: "other"})).median().T

    df.to_csv(out / "stage1_predictions.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(out / "stage1_metrics.csv", encoding="utf-8-sig")
    folds.to_csv(out / "stage1_folds.csv", index=False, encoding="utf-8-sig")
    per_cam.to_csv(out / "stage1_per_camera.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(out / "stage1_rf_geo_leg_importance.csv", header=["importance"], encoding="utf-8-sig")
    leg_summary.to_csv(out / "stage1_leg_feature_medians.csv", encoding="utf-8-sig")

    lines += [f"## Stage 1 — 사람 keypoint, 클립 단위 {k}-fold",
              f"개체 {len(df)} (visiting {int(y.sum())}), 이미지 {df.image.nunique()}, 클립 {n_groups}, "
              f"카메라 {', '.join(sorted(df.camera.unique()))}", "",
              "```", fmt_table(overall[["n", "visiting_F1", "precision", "recall", "macro_F1", "accuracy",
                                         "fold_F1_mean", "fold_F1_std", "TP", "FP", "FN", "TN"]]), "```", "",
              "geo 규칙이 놓친 이유(visiting인데 geo=0):",
              "```", df[(df.label == 1) & (df.pred_geo == 0)]["geo_reason"].value_counts().to_string(), "```", "",
              "RF+Geo+Leg 특징 중요도 (전체 데이터로 학습):", "```", imp.to_string(float_format=lambda v: f"{v:.3f}"), "```", "",
              "다리 기하 특징 중앙값 (visiting vs other):", "```", leg_summary.to_string(float_format=lambda v: f"{v:.3f}"), "```"]
    return lines, "ok"


# --------------------------------------------------------------------------
# stage 2
# --------------------------------------------------------------------------

def find_images(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}


def match(gt: list[Detection], pred: list[Detection], thresh: float) -> list[tuple[int, int]]:
    """Hungarian on mean distance of keypoints visible in the human skeleton,
    normalised by the human box diagonal. Pairs above `thresh` are rejected."""
    if not gt or not pred:
        return []
    cost = np.full((len(gt), len(pred)), 9.0)
    for i, g in enumerate(gt):
        x1, y1, x2, y2 = g.box
        diag = math.hypot(x2 - x1, y2 - y1) + 1e-6
        vis = [k for k, (_, _, c) in enumerate(g.kpts) if c > 0]
        for j, p in enumerate(pred):
            if len(p.kpts) != len(g.kpts):
                continue
            cost[i, j] = np.mean([math.hypot(g.kpts[k][0] - p.kpts[k][0], g.kpts[k][1] - p.kpts[k][1])
                                  for k in vis]) / diag
    r, c = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(r, c) if cost[i, j] <= thresh]


def run_stage2(gt_rows: list[dict], images: dict[str, Path], cams: Cameras, stage1_df: pd.DataFrame | None,
               out: Path, match_thresh: float, seed: int, exclude_shared_clips: bool = False) -> list[str]:
    import cv2
    from feeding_monitor.pose_tracker import PoseTracker, _parse_detections

    models = {}
    train_note = ""
    if stage1_df is not None and exclude_shared_clips:
        eval_clips = {r["clip"] for r in gt_rows}
        n_before = len(stage1_df)
        stage1_df = stage1_df[~stage1_df["clip"].isin(eval_clips)]
        train_note = (f"RF 학습 데이터에서 평가 50장과 같은 클립의 개체 {n_before - len(stage1_df)}개 제외 "
                      f"(남은 {len(stage1_df)}개) -- 2초 간격 비슷한 장면이 시험지로 새는 것 방지")
        print(f"[stage2] {train_note}")
    if stage1_df is not None and len(stage1_df) and stage1_df.label.nunique() == 2:
        for m, feats in METHOD_FEATURES.items():
            models[m] = (new_rf(seed).fit(stage1_df[feats], stage1_df["label"]), feats)
    else:
        print("[stage2] stage1 데이터가 없어 RF / RF+Geo는 건너뜀 (geo 규칙만 평가)")

    trackers = {}
    by_image: dict[str, list[dict]] = {}
    for r in gt_rows:
        by_image.setdefault(r["image"], []).append(r)

    inst, frames, missing_img = [], [], []
    for stem, rows in sorted(by_image.items()):
        if stem not in images:
            missing_img.append(stem)
            continue
        cam = rows[0]["camera"]
        cfg, zones = cams.cfg[cam], cams.zones[cam]
        if cam not in trackers:
            trackers[cam] = PoseTracker(cfg.model)
        tr = trackers[cam]
        frame = cv2.imread(str(images[stem]))
        res = tr.model.predict(frame, conf=cfg.model.conf, imgsz=cfg.model.imgsz, iou=cfg.model.iou,
                               device=tr.device, verbose=False)[0]
        preds = _parse_detections(res, 0)

        pf = featurize([{"image": stem, "camera": cam, "clip": rows[0]["clip"], "label": -1, "det": d,
                         "source": "model"} for d in preds], cams) if preds else pd.DataFrame()
        for m, (clf, feats) in models.items():
            pf[f"pred_{m}"] = clf.predict(pf[feats]) if len(pf) else []

        pairs = match([r["det"] for r in rows], preds, match_thresh)
        matched_gt = {i for i, _ in pairs}
        matched_pred = {j for _, j in pairs}
        for i, r in enumerate(rows):
            rec = {"image": stem, "clip": r["clip"], "label": r["label"]}
            j = next((jj for ii, jj in pairs if ii == i), None)
            rec["matched"] = j is not None
            for m in METHODS:
                col = f"pred_{m}"
                rec[col] = int(pf.iloc[j][col]) if (j is not None and col in pf) else (0 if j is None else -1)
            inst.append(rec)

        fr = {"image": stem, "clip": rows[0]["clip"], "n_gt": len(rows), "n_pred": len(preds),
              "n_matched": len(pairs), "human_visiting": sum(r["label"] for r in rows)}
        for m in METHODS:
            col = f"pred_{m}"
            if not preds:
                fr[f"count_{m}"], fr[f"extra_visiting_{m}"] = 0, 0
                continue
            fr[f"count_{m}"] = int(pf[col].sum()) if col in pf else np.nan   # ALL detections, matched or not
            fr[f"extra_visiting_{m}"] = int(sum(pf.iloc[j][col] for j in range(len(preds))
                                                if j not in matched_pred)) if col in pf else np.nan
        frames.append(fr)

    if missing_img:
        print(f"[stage2][경고] 이미지 없음 {len(missing_img)}장: {missing_img[:5]} ...")
    inst_df, fr_df = pd.DataFrame(inst), pd.DataFrame(frames)
    if inst_df.empty:
        return ["## Stage 2 — 이미지를 하나도 찾지 못해 건너뜀"]

    avail = [m for m in METHODS if (inst_df[f"pred_{m}"] >= 0).all()]
    table = pd.DataFrame([{"method": m, **metrics(inst_df["label"], inst_df[f"pred_{m}"])} for m in avail]).set_index("method")
    cnt = pd.DataFrame([{"method": m,
                         "count_MAE": (fr_df[f"count_{m}"] - fr_df.human_visiting).abs().mean(),
                         "count_bias": (fr_df[f"count_{m}"] - fr_df.human_visiting).mean(),
                         "exact_count_rate": (fr_df[f"count_{m}"] == fr_df.human_visiting).mean(),
                         "extra_visiting_total": fr_df[f"extra_visiting_{m}"].sum()} for m in avail]).set_index("method")

    inst_df.to_csv(out / "stage2_instances.csv", index=False, encoding="utf-8-sig")
    fr_df.to_csv(out / "stage2_frames.csv", index=False, encoding="utf-8-sig")
    table.join(cnt).to_csv(out / "stage2_metrics.csv", encoding="utf-8-sig")

    n_gt, n_m = len(inst_df), int(inst_df.matched.sum())
    return [f"## Stage 2 — 평가용 {fr_df.image.nunique()}장, 모델 keypoint (학습 미사용 프레임)",
            f"사람 개체 {n_gt} 중 모델 검출과 매칭 {n_m} ({n_m / n_gt:.1%}), "
            f"매칭 안 된 모델 검출 {int((fr_df.n_pred - fr_df.n_matched).sum())}개",
            "매칭 실패한 사람 개체는 '모델이 못 찾음 = other로 예측'으로 계산 (실제 운용과 동일)",
            (train_note or "RF 학습: Stage 1 전체"), "",
            "개체 단위:", "```", fmt_table(table[["n", "visiting_F1", "precision", "recall", "macro_F1",
                                                 "TP", "FP", "FN", "TN"]]), "```", "",
            "프레임 단위 visiting 마릿수 (그래프에 실제로 쓰이는 값):",
            "```", fmt_table(cnt), "```",
            "count_bias > 0 이면 과대 집계, extra_visiting = 사람 라벨에 없는 검출(중복·오검출)을 visiting으로 센 수"]


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", required=True, help="카메라별 config/zones yaml (configs/eval_cameras.yaml)")
    ap.add_argument("--stage1-xml", nargs="*", default=[], help="CVAT for images 1.1 XML (사람 keypoint + State)")
    ap.add_argument("--stage1-exclude", default=None, help="stage1에서 뺄 이미지 목록 (txt, 한 줄에 하나)")
    ap.add_argument("--stage2-xml", nargs="*", default=[], help="평가용 프레임 XML (State 라벨)")
    ap.add_argument("--stage2-images", default=None, help="평가용 프레임 원본 이미지 폴더")
    ap.add_argument("--stage2-only-list", default=None, help="stage2에 쓸 이미지 목록만 남김 (txt, 한 줄에 하나)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--match-thresh", type=float, default=0.25, help="keypoint 평균거리/박스대각선 허용치")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage2-exclude-shared-clips", action="store_true",
                    help="Stage 2용 RF 학습에서 평가 프레임과 같은 영상 클립의 Stage 1 개체를 뺀다 (엄격한 평가)")
    ap.add_argument("--out", default="outputs/eval")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = Cameras(a.cameras)
    report = ["# 판정 방법 비교 (Geo 규칙 / RF / RF+Geo / RF+Geo+Leg)", ""]

    s1 = None
    if a.stage1_xml:
        excl = read_list(a.stage1_exclude)
        rows = [r for x in a.stage1_xml for r in parse_cvat_xml(x)]
        n0 = len(rows)
        rows = [r for r in rows if r["image"] not in excl]
        n1 = len(rows)
        skipped = sorted({r["camera"] for r in rows if not cams.has(r["camera"])})
        rows = [r for r in rows if cams.has(r["camera"])]
        print(f"[stage1] 라벨 개체 {n0} -> 제외 목록 {n0 - n1} -> ROI 미설정 카메라 제외 {n1 - len(rows)}"
              f" -> 사용 {len(rows)}" + (f"  (미설정: {skipped})" if skipped else ""))
        if rows:
            s1 = featurize(rows, cams)
            lines, _ = run_stage1(s1, out, a.n_splits, a.seed)
            report += lines + [""]
            if skipped:
                report += [f"(ROI 미설정으로 제외된 카메라: {', '.join(skipped)})", ""]

    if a.stage2_xml:
        if not a.stage2_images:
            raise SystemExit("--stage2-images 가 필요합니다")
        only = read_list(a.stage2_only_list)
        rows = [r for x in a.stage2_xml for r in parse_cvat_xml(x)]
        if only:
            rows = [r for r in rows if r["image"] in only]
        rows = [r for r in rows if cams.has(r["camera"])]
        print(f"[stage2] 평가 개체 {len(rows)} (이미지 {len({r['image'] for r in rows})})")
        report += run_stage2(rows, find_images(Path(a.stage2_images)), cams, s1, out, a.match_thresh, a.seed,
                             exclude_shared_clips=a.stage2_exclude_shared_clips)

    (out / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
