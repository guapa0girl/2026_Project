"""Run count_full_video.py over every clip of a day (e.g. CH010 24h download).

DVR clip filenames look like "[CH010] 20260716012647-20260716015128.avi" and
the timestamps in them are UTC -- KST = filename + 9h (confirmed 2026-11-04 by
matching a known 10:00 KST clip). Each clip gets its own --start-time in KST,
so every output row's wall_time is real Korean local time.

Resumable: a clip whose <out-dir>/<clip>_1s.csv already exists is skipped,
so if the run stops (sleep, crash) just run the same command again.

Usage (Mac -- caffeinate keeps the laptop awake all night):
    caffeinate -i python scripts/count_day.py --config configs/farm_ch010.yaml \
        --videos "data/raw/CH010_20260716" --out-dir outputs/ch010_20260716 \
        --bin-sec 600 --save-dets
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def kst_start(name: str) -> datetime:
    m = re.search(r"(\d{14})-(\d{14})", name)
    if not m:
        raise ValueError(f"파일명에서 시각을 못 찾음: {name}")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") + timedelta(hours=9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--videos", required=True, help="folder with the day's .avi clips")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--zones", default=None, help="(unused, kept so older commands still run)")
    p.add_argument("--fps", type=float, default=200.0)
    p.add_argument("--bin-sec", type=float, default=600.0)
    p.add_argument("--save-dets", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="test: only the first N clips")
    a = p.parse_args()

    clips = sorted(Path(a.videos).glob("*.avi"), key=lambda c: kst_start(c.name))
    if a.limit:
        clips = clips[: a.limit]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"클립 {len(clips)}개: {kst_start(clips[0].name):%m-%d %H:%M} ~ {kst_start(clips[-1].name):%m-%d %H:%M} (KST)")

    failed = []
    for i, clip in enumerate(clips, 1):
        prefix = out_dir / clip.stem.replace("[", "").replace("]", "").replace(" ", "_")
        if Path(str(prefix) + "_1s.csv").exists():
            print(f"[{i}/{len(clips)}] 건너뜀(완료): {clip.name}")
            continue
        start = kst_start(clip.name)
        print(f"\n[{i}/{len(clips)}] {clip.name}  → 시작 {start:%Y-%m-%d %H:%M:%S} KST")
        cmd = [sys.executable, str(ROOT / "scripts" / "count_full_video.py"),
               "--config", a.config, "--source", str(clip), "--fps", str(a.fps),
               "--bin-sec", str(a.bin_sec), "--start-time", start.isoformat(),
               "--out-prefix", str(prefix)]
        if a.save_dets:
            cmd.append("--save-dets")
        if subprocess.run(cmd).returncode != 0:
            failed.append(clip.name)

    print("\n완료." + (f" 실패 {len(failed)}개: {failed}" if failed else " 실패 없음."))


if __name__ == "__main__":
    main()
