"""Count visiting pigs per second with every method on a count_day.py output
folder (<clip>_dets.csv + <clip>_1s.csv), then bin on clock boundaries and
plot the methods side by side. No YOLO re-run: everything comes from the saved
keypoints, so a new rule setting or a retrained RF takes minutes.

  geo variants : the rule with one or more configs (e.g. direction 60 vs 90)
  rf / rf_geo / rf_geo_leg : RandomForest trained on Stage-1 labels
                 (human keypoints + CVAT State; same features as evaluate_methods.py)

Usage (from counting/):
  ..\\.venv\\Scripts\\python scripts\\apply_methods.py outputs\\ch010_20260716_day ^
      --config configs\\farm_ch010.yaml --zones configs\\zones\\feeder_zones_ch010.yaml ^
      --geo dir60=configs\\ablation\\farm_ch010_dir60.yaml --geo dir90=configs\\ablation\\farm_ch010_dir90.yaml ^
      --cameras configs\\eval_cameras.yaml --stage1-xml data\\labels\\state_000.xml data\\labels\\state_001.xml data\\labels\\state_002.xml ^
      --stage1-exclude ..\\training\\eval_frames.txt
  (no --stage1-xml -> geo variants only)

Outputs in the same folder:
  <clip>_methods_1s.csv          per second, one column per method
  day_methods_1s.csv             all clips joined
  day_methods_<N>min.csv         clock-aligned bins: n_samples, mean and max per method
  day_methods_<N>min.png         mean per bin, one line per method
  rf_models.joblib               the trained RFs (reuse with --load-models)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from feeding_monitor.config import load_config                      # noqa: E402
from feeding_monitor.method_features import METHOD_FEATURES, featurize_det  # noqa: E402
from feeding_monitor.pose_tracker import Detection                    # noqa: E402
from feeding_monitor.posture_features import load_zones               # noqa: E402

N_KPT = 18


def dets_from_csv(path: Path) -> tuple[pd.DataFrame, list[Detection]]:
    df = pd.read_csv(path)
    dets = []
    kx = [df[f"k{j}_x"].to_numpy() for j in range(N_KPT)]
    ky = [df[f"k{j}_y"].to_numpy() for j in range(N_KPT)]
    kc = [df[f"k{j}_c"].to_numpy() for j in range(N_KPT)]
    box = df[["x1", "y1", "x2", "y2"]].to_numpy()
    for i in range(len(df)):
        dets.append(Detection(frame_idx=int(df.elapsed_sec.iat[i]), track_id=None, det_conf=1.0,
                              box=tuple(box[i]), kpts=[(kx[j][i], ky[j][i], kc[j][i]) for j in range(N_KPT)]))
    return df, dets


def train_models(a) -> dict:
    import evaluate_methods as em
    from sklearn.ensemble import RandomForestClassifier

    cams = em.Cameras(a.cameras)
    excl = em.read_list(a.stage1_exclude)
    rows = [r for x in a.stage1_xml for r in em.parse_cvat_xml(x)]
    rows = [r for r in rows if r["image"] not in excl and cams.has(r["camera"])]
    if not rows:
        raise SystemExit("Stage 1 라벨이 없습니다 (카메라 ROI 설정·제외 목록 확인)")
    df = em.featurize(rows, cams)
    print(f"[train] Stage 1 개체 {len(df)} (visiting {int(df.label.sum())}), "
          f"카메라 {', '.join(sorted(df.camera.unique()))}")
    models = {}
    for m, feats in METHOD_FEATURES.items():
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=a.seed, n_jobs=-1)
        models[m] = (clf.fit(df[feats], df["label"]), feats)
    return models


def bin_clock(day: pd.DataFrame, cols: list[str], minutes: int) -> pd.DataFrame:
    t = pd.to_datetime(day["wall_time"])
    key = t.dt.floor(f"{minutes}min")
    g = day.groupby(key)
    out = pd.DataFrame({"n_samples": g.size()})
    for c in cols:
        out[f"mean_{c}"] = g[c].mean().round(3)
        out[f"max_{c}"] = g[c].max()
    out.index.name = "bin_start"
    return out.reset_index()


def plot(bins: pd.DataFrame, cols: list[str], minutes: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    full = minutes * 60
    fig, ax = plt.subplots(figsize=(max(8, len(bins) * 0.9), 4.2))
    x = bins["bin_start"] + pd.Timedelta(minutes=minutes / 2)
    for c in cols:
        ax.plot(x, bins[f"mean_{c}"], marker="o", lw=2, label=c)
    partial = bins["n_samples"] < 0.9 * full
    for t0 in bins.loc[partial, "bin_start"]:
        ax.axvspan(t0, t0 + pd.Timedelta(minutes=minutes), color="0.9", zorder=0)
    ax.set_ylabel(f"visiting pigs ({minutes}-min mean)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=len(cols), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_title("Visiting pigs by method (grey = bin not fully recorded)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="count_day.py 출력 폴더 (*_dets.csv, *_1s.csv)")
    ap.add_argument("--config", required=True, help="이 영상 카메라의 config (ROI, RF 특징 계산용)")
    ap.add_argument("--zones", required=True)
    ap.add_argument("--geo", action="append", default=[], metavar="NAME=CONFIG",
                    help="Geo 규칙 설정 (여러 번 가능). 없으면 --config 로 'geo' 하나")
    ap.add_argument("--cameras", default=None, help="Stage 1 라벨의 카메라별 ROI (eval_cameras.yaml)")
    ap.add_argument("--stage1-xml", nargs="*", default=[])
    ap.add_argument("--stage1-exclude", default=None)
    ap.add_argument("--load-models", default=None, help="저장해 둔 rf_models.joblib 재사용")
    ap.add_argument("--bin-min", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    folder = Path(a.folder)
    files = sorted(folder.glob("*_dets.csv"))
    if not files:
        raise SystemExit(f"{folder}: *_dets.csv 없음 (count_day.py --save-dets 로 만든 폴더인지 확인)")

    cfg = load_config(a.config)
    zones = load_zones(a.zones)
    geos = {}
    for g in a.geo or [f"geo={a.config}"]:
        name, path = g.split("=", 1)
        geos[name] = load_config(path)

    models = {}
    if a.load_models:
        import joblib
        models = joblib.load(a.load_models)
    elif a.stage1_xml:
        if not a.cameras:
            raise SystemExit("--stage1-xml 을 쓰려면 --cameras 도 필요합니다")
        models = train_models(a)
        import joblib
        joblib.dump(models, folder / "rf_models.joblib")
    cols = list(geos) + list(models)
    print(f"[methods] {', '.join(cols)}")

    from feeding_monitor.feeding_classifier import classify
    days = []
    for f in files:
        clip = f.name[: -len("_dets.csv")]
        base_path = folder / f"{clip}_1s.csv"
        if not base_path.exists():
            print(f"[skip] {clip}: _1s.csv 없음")
            continue
        base = pd.read_csv(base_path)[["elapsed_sec", "wall_time"]]
        raw, dets = dets_from_csv(f)
        if dets:
            for name, gcfg in geos.items():
                raw[name] = [int(classify(d, gcfg.feeder, gcfg.feeding_rule).is_feeding) for d in dets]
            if models:
                feat = pd.DataFrame([featurize_det(d, cfg, zones) for d in dets])
                for m, (clf, feats) in models.items():
                    raw[m] = clf.predict(feat[feats])
            per_sec = raw.groupby("elapsed_sec")[cols].sum()
        else:
            per_sec = pd.DataFrame(columns=cols)
        sec = base.set_index("elapsed_sec").join(per_sec, how="left").fillna({c: 0 for c in cols})
        sec[cols] = sec[cols].astype(int)
        sec = sec.reset_index()
        sec.insert(0, "clip", clip)
        sec.to_csv(folder / f"{clip}_methods_1s.csv", index=False, encoding="utf-8-sig")
        days.append(sec)
        print(f"[clip] {clip}: {len(sec)}초, 검출 {len(dets)}개, 평균 " +
              ", ".join(f"{c}={sec[c].mean():.2f}" for c in cols))

    day = pd.concat(days).sort_values("wall_time")
    day.to_csv(folder / "day_methods_1s.csv", index=False, encoding="utf-8-sig")
    bins = bin_clock(day, cols, a.bin_min)
    stem = folder / f"day_methods_{a.bin_min}min"
    bins.to_csv(str(stem) + ".csv", index=False, encoding="utf-8-sig")
    plot(bins, cols, a.bin_min, Path(str(stem) + ".png"))

    print(f"\n===== {day.wall_time.min()} ~ {day.wall_time.max()}  ({len(day)}초) =====")
    summ = pd.DataFrame({c: {"1초 평균": day[c].mean(), "최대": day[c].max(),
                             f"{a.bin_min}분 평균의 최대": bins[f"mean_{c}"].max()} for c in cols}).T
    print(summ.round(2).to_string())
    if len(cols) > 1:
        print("\n방법 간 1초 마릿수 일치율 (같은 숫자를 낸 초의 비율):")
        agree = pd.DataFrame({c1: {c2: (day[c1] == day[c2]).mean() for c2 in cols} for c1 in cols})
        print(agree.round(2).to_string())
    print(f"\n[saved] {stem}.csv / .png, day_methods_1s.csv, <clip>_methods_1s.csv")


if __name__ == "__main__":
    main()
