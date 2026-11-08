"""Labeling-guide example images for the DM ground truth, drawn from HUMAN State labels
(CVAT XML with skeletons): green = visiting (count), red = other whose nose is inside
the feeder circle (e.g. lying with the head on the lid -> do NOT count).

Usage (from counting/):
    python scripts/make_dm_examples.py --xml data/labels/ch010_eval50.xml \
        --images data/labels/eval50_img --config configs/farm_ch010.yaml --out <folder> --n 4
"""
import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from feeding_monitor.config import load_config  # noqa: E402

NOSE = "nose"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()

    f = load_config(a.config).feeder
    imgs = {p.stem: p for p in Path(a.images).glob("*.jpg")}
    cands = []
    for im in ET.parse(a.xml).getroot().iter("image"):
        stem = Path(im.get("name")).stem
        if stem not in imgs:
            continue
        pigs = []
        for sk in im.iter("skeleton"):
            state = next((x.text or "" for x in sk.iter("attribute") if x.get("name", "").lower() == "state"), "")
            pts = {p.get("label").lower(): tuple(map(float, p.get("points").split(","))) for p in sk.iter("points")}
            xs = [p[0] for p in pts.values()]
            ys = [p[1] for p in pts.values()]
            nose = pts.get(NOSE)
            inside = nose is not None and math.hypot(nose[0] - f.cx, nose[1] - f.cy) <= f.r
            vis = state.lower().startswith("visit")
            if vis or (inside and state.lower() == "other"):
                pigs.append((vis, (min(xs), min(ys), max(xs), max(ys)), nose))
        n_vis = sum(p[0] for p in pigs)
        n_no = len(pigs) - n_vis
        if n_vis and n_no:
            cands.append((min(n_vis, n_no), n_vis + n_no, stem, pigs))
    cands.sort(reverse=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, (_, _, stem, pigs) in enumerate(cands[: a.n], 1):
        img = cv2.imread(str(imgs[stem]))
        cv2.circle(img, (int(f.cx), int(f.cy)), int(f.r), (0, 220, 255), 2)
        n_vis = 0
        for vis, (x1, y1, x2, y2), nose in pigs:
            col = (0, 200, 0) if vis else (0, 0, 255)
            n_vis += vis
            cv2.rectangle(img, (int(x1) - 8, int(y1) - 8), (int(x2) + 8, int(y2) + 8), col, 4)
            cv2.putText(img, "COUNT" if vis else "NO", (int(x1), max(30, int(y1) - 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
            if nose:
                cv2.circle(img, (int(nose[0]), int(nose[1])), 9, col, -1)
        cv2.putText(img, f"answer = {n_vis}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)
        p = out / f"example_{k}_answer_{n_vis}.jpg"
        cv2.imwrite(str(p), cv2.resize(img, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(p.name, stem)


if __name__ == "__main__":
    main()
