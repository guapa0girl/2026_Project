"""zip(task)별 시간대 분포: 영상 클립(촬영 시작~끝 시각)마다 사진 수·visiting 수와 분할별 배정(train/valid/test).

사진 이름 예: "0001_frame_0001__[CH005] 20260707211528-20260707213722_min001"
    → 카메라 CH005, 클립 이름 시각 2026-07-07 21:15:28 ~ 21:37:22, 클립 안 위치 min001
    ※ 이름의 시각은 실제 촬영 시각보다 9시간 느림(UTC로 기록된 것으로 보임) → 실제 시각(KST) = 이름 시각 + 9시간
      예: 002 "20260716012647" → 실제 2026-07-16 10:26:47. 이 스크립트의 표·그림은 모두 실제 시각(KST)으로 표시.
000(1분 샘플 영상)은 이름에 시각이 없어 클립 1개로 표시.

출력: reports/data/
    time_distribution_clips.csv   클립 단위 표 (분할별 배정 포함)
    time_distribution_hours.csv   task × 시(hour) 사진 수
    time_distribution.md          표
    time_distribution.png         task별 타임라인 (사진 하나 = 점, 분할별 valid 위치 표시)

사용법 (training/ 에서, make_study_splits.py 다음):  python scripts/time_distribution.py
"""
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "yolo_det" / "scripts"))
from prepare_data import read_task  # noqa: E402

SOURCE = REPO / "counting" / "data" / "labels" / "source"
SPLITS = ["random", "sorted", "k5_f0", "k5_f1", "k5_f2", "k5_f3", "k5_f4"]
_RE = re.compile(r"\[(CH\d{3})\] (\d{14})-(\d{14})_(?:min(\d+)|f(\d+))")
NAME_TO_KST = timedelta(hours=9)        # 파일 이름 시각 + 9시간 = 실제 촬영 시각(KST)


def parse(orig):
    m = _RE.search(orig)
    if not m:
        return None
    ts = lambda s: datetime.strptime(s, "%Y%m%d%H%M%S") + NAME_TO_KST  # noqa: E731
    return {"camera": m.group(1), "start": ts(m.group(2)), "end": ts(m.group(3)),
            "pos": int(m.group(4) or m.group(5)), "pos_unit": "min" if m.group(4) else "frame"}


def main():
    out = ROOT / "reports" / "data"
    out.mkdir(parents=True, exist_ok=True)
    split_of = {}                                          # split -> {stem: train/valid/test}
    for s in SPLITS:
        f = ROOT / "data" / "splits" / f"{s}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            split_of[s] = {stem: k for k in ("train", "valid", "test") for stem in d[k]}

    imgs = []
    for task in ["001", "002", "000"]:
        for it in read_task(SOURCE / f"{task}_sorted.zip", task):
            p = parse(it["orig"]) or {"camera": "SAMPLE", "start": None, "end": None, "pos": None, "pos_unit": None}
            stem = Path(it["name"]).stem
            imgs.append({"task": task, "image": stem, **p, "n_pigs": it["n_pigs"], "n_visiting": it["n_visiting"],
                         **{s: split_of[s].get(stem, "-") for s in split_of}})
    df = pd.DataFrame(imgs)

    # ---- clip table ----
    rows = []
    for (task, start), g in df.groupby(["task", df.start.fillna(pd.Timestamp(0))], sort=True):
        st, en = g.start.iloc[0], g.end.iloc[0]
        rec = {"task": task, "camera": g.camera.iloc[0],
               "date": st.strftime("%Y-%m-%d") if pd.notna(st) else "-",
               "start": st.strftime("%H:%M:%S") if pd.notna(st) else "-",
               "end": en.strftime("%H:%M:%S") if pd.notna(en) else "-",
               "minutes": round((en - st).total_seconds() / 60, 1) if pd.notna(st) else 1.0,
               "n_images": len(g), "n_pigs": int(g.n_pigs.sum()), "n_visiting": int(g.n_visiting.sum())}
        for s in split_of:
            c = Counter(g[s])
            tag = {"train": "T", "valid": "V", "test": "test ", "-": "-"}
            rec[s] = "/".join(f"{tag[k]}{c[k]}" for k in tag if c[k])
        rows.append(rec)
    clips = pd.DataFrame(rows)
    clips.to_csv(out / "time_distribution_clips.csv", index=False, encoding="utf-8-sig")

    timed = df[df.start.notna()].copy()
    timed["hour"] = timed.start.dt.hour
    hours = timed.pivot_table(index="task", columns="hour", values="image", aggfunc="count", fill_value=0)
    hours.to_csv(out / "time_distribution_hours.csv", encoding="utf-8-sig")

    # ---- markdown ----
    md = ["시각은 모두 실제 촬영 시각(KST) = 파일 이름 시각 + 9시간.", "",
          "| task | 카메라 | 날짜 | 클립 수 | 사진 | visiting | 촬영 시간대 (클립 시작 시각) |", "|---|---|---|---|---|---|---|"]
    for task, g in clips.groupby("task", sort=False):
        md.append(f"| {task} | {g.camera.iloc[0]} | {', '.join(sorted(set(g.date)))} | {len(g)} | {g.n_images.sum()} | "
                  f"{g.n_visiting.sum()} | {', '.join(g.start)} |")
    md += ["", "클립별 (분할 칸: T = train, V = valid, test = test, - = 5-fold 경계라 뺀 사진, 숫자 = 사진 수):", "",
           "| task | 날짜 | 시작 \\~ 끝 | 분 | 사진 | visiting | " + " | ".join(split_of) + " |",
           "|---|---|---|---|---|---|" + "---|" * len(split_of)]
    for _, r in clips.iterrows():
        md.append(f"| {r.task} | {r.date} | {r.start} \\~ {r.end} | {r.minutes:g} | {r.n_images} | {r.n_visiting} | "
                  + " | ".join(r[s] for s in split_of) + " |")
    (out / "time_distribution.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # ---- figure: one row per task, x = clock time of each image (clip start + position) ----
    def img_time(r):
        return r.start + pd.Timedelta(minutes=r.pos) if r.pos_unit == "min" else \
            r.start + (r.end - r.start) * (r.pos / max(1, timed[timed.start == r.start].pos.max()))
    timed["t"] = timed.apply(img_time, axis=1)
    tasks = sorted(timed.task.unique())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(14, 2.6 * len(tasks)), squeeze=False)
    rowsy = ["random", "sorted"] + [s for s in SPLITS if s.startswith("k5")]
    col = {"train": "#b9d3f2", "valid": "#e8833a", "-": "#d0cfc8"}
    for ax, task in zip(axes[:, 0], tasks):
        g = timed[timed.task == task]
        for y, s in enumerate(rowsy):
            if s not in g:
                continue
            for k, c in col.items():
                h = g[g[s] == k]
                ax.scatter(h.t, [y] * len(h), s=10, color=c, label=k if (y == 0 and task == tasks[0]) else None)
        ax.set_yticks(range(len(rowsy)), rowsy, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"task {task} ({g.camera.iloc[0]}) — {len(g)} imgs, "
                     f"{g.start.dt.date.min()} ~ {g.start.dt.date.max()}", fontsize=10)
        ax.grid(axis="x", color="#e1e0d9")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.35))
    fig.suptitle("Capture time (KST = filename time + 9 h) of each labeled image and its split "
                 "(000 = 1-min sample video, always test)")
    fig.tight_layout()
    fig.savefig(out / "time_distribution.png", dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
