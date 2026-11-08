"""Extract one frame per second for a DM ground-truth section (e.g. 10 min = 600 images)
using EXACTLY the frames count_day.py / count_full_video.py ran the model on, so the
human count and the model count can be compared second by second.

Frame rule (same as count_full_video.py):
    clip KST start = filename UTC start + 9h
    sec   = target KST time - clip KST start   (integer seconds)
    frame = min(total_frames - 1, round(sec * 200))

Usage (from counting/):
    python scripts/extract_dm_frames.py --videos data/raw/CH010_20260716 \
        --start "2026-07-16 10:30:00" --minutes 10 --name A --out <folder> \
        --labeled ../training/eval_frames.txt ../training/data/raw/data/002.zip

Output (<out>/<name>/):
    images/HH_MM_SS.jpg      one per second (missing if the frame could not be read)
    <name>_labeling_sheet.csv (UTF-8 BOM, opens in Excel)
        file, kst_time, visiting_count(blank), ambiguous(blank), memo, status, overlap_both, assignee
        status: ok | 누락 (read failure) | 학습사진_비교제외 (same second as a pose-training / eval frame)
"""
import argparse
import csv
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import cv2

FPS = 200.0
NAME_RE = re.compile(r"\[?(CH\d{3})\]?[ _](\d{14})-(\d{14})")


def clip_kst(name: str):
    m = NAME_RE.search(name)
    if not m:
        return None
    s = datetime.strptime(m.group(2), "%Y%m%d%H%M%S") + timedelta(hours=9)
    e = datetime.strptime(m.group(3), "%Y%m%d%H%M%S") + timedelta(hours=9)
    return m.group(1), s, e


def labeled_seconds(paths) -> set[datetime]:
    """KST seconds of frames already used for pose training / eval (names like
    '...[CH010] 20260716012647-20260716015128_f0203999_novinf')."""
    names = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".zip":
            names += [Path(n).name for n in zipfile.ZipFile(p).namelist()]
        else:
            names += [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    out = set()
    for n in names:
        c = clip_kst(n)
        f = re.search(r"_f(\d+)", n)
        if c and f:
            out.add(c[1] + timedelta(seconds=round(int(f.group(1)) / FPS)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--start", required=True, help='KST, e.g. "2026-07-16 10:30:00"')
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--name", required=True, help="section name, e.g. A")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labeled", nargs="*", default=[], help="eval_frames.txt / CVAT export zips to flag")
    ap.add_argument("--overlap-min", type=int, default=1, help="first N minutes: both labelers count")
    ap.add_argument("--assignees", default="진홍,경태", help="two names; 1st half / 2nd half")
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()

    start = datetime.strptime(a.start, "%Y-%m-%d %H:%M:%S")
    n = a.minutes * 60
    clips = []
    for v in sorted(Path(a.videos).glob("*.avi")):
        c = clip_kst(v.name)
        if c:
            clips.append((c[1], c[2], v))
    if not clips:
        raise SystemExit(f"{a.videos}: 영상 없음")
    used = labeled_seconds(a.labeled)
    names = a.assignees.split(",")

    out = Path(a.out) / a.name
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    caps, totals = {}, {}
    rows, n_ok, n_fail, n_used = [], 0, 0, 0
    for i in range(n):
        t = start + timedelta(seconds=i)
        fname = t.strftime("%H_%M_%S") + ".jpg"
        status = "누락"
        clip = None                      # latest clip that started at or before t
        for s, e, v in clips:
            if s <= t:
                clip = (s, v)
        if clip is not None:
            s, v = clip
            sec = int((t - s).total_seconds())
            if v not in caps:
                caps[v] = cv2.VideoCapture(str(v))
                totals[v] = int(caps[v].get(cv2.CAP_PROP_FRAME_COUNT))
                # The first seek right after opening can land on a different frame in these
                # DVR files; count_full_video.py always seeks from the previous second, so warm
                # up the same way (seek to sec-1 and read) before the first real frame.
                if sec > 0:
                    caps[v].set(cv2.CAP_PROP_POS_FRAMES, min(totals[v] - 1, round((sec - 1) * FPS)))
                    caps[v].read()
            if sec < int(totals[v] / FPS) + 1:
                caps[v].set(cv2.CAP_PROP_POS_FRAMES, min(totals[v] - 1, round(sec * FPS)))
                ok, frame = caps[v].read()
                if ok:
                    cv2.imwrite(str(img_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                    status = "ok"
        if status == "ok" and t in used:
            status = "학습사진_비교제외"
            n_used += 1
        n_ok += status != "누락"
        n_fail += status == "누락"
        overlap = "Y" if i < a.overlap_min * 60 else ""
        assignee = "둘 다" if overlap else (names[0] if i < n // 2 else names[-1])
        rows.append([fname if status != "누락" else "", t.strftime("%Y-%m-%d %H:%M:%S"), "", "", "",
                     status, overlap, assignee])
    for c in caps.values():
        c.release()

    sheet = out / f"{a.name}_labeling_sheet.csv"
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "kst_time", "visiting_count", "ambiguous", "memo", "status", "overlap_both", "assignee"])
        w.writerows(rows)
    print(f"[{a.name}] {start:%H:%M:%S}~{start + timedelta(seconds=n - 1):%H:%M:%S}  "
          f"이미지 {n_ok}장, 누락 {n_fail}초, 학습사진 겹침 {n_used}초 -> {out}")


if __name__ == "__main__":
    main()
