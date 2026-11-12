"""CVAT "for images 1.1" export (images + annotations.xml with pig skeletons) → the zip
layout ingest_raw_zips.py reads:  <task>/img/*  +  <task>/label/labels/Train/*.txt + data.yaml

YOLO-Pose label per skeleton (same as CVAT's own "YOLO Pose" export — checked on 299 pigs
of task 001: identical boxes and visibility flags):
    0 cx cy w h  x1 y1 v1 ... x18 y18 v18        (all normalized)
    box  = min/max of the skeleton's non-outside points
    v    = 2 visible, 1 occluded, 0 outside (x = y = 0)
    keypoint order = CVAT skeleton node order (data-node-id ascending), same as configs/pig_pose.yaml

Only images listed in annotations.xml are used (the *_sorted zips also carry a second,
byte-identical copy of each image under an older name; those are ignored). macOS
"__MACOSX/" entries are ignored.

Usage (from training/):
    python scripts/cvat_xml_to_raw_zip.py --src ../001_sorted.zip --task 001 --out data/raw/data/001.zip
"""
import argparse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

KPT_ORDER = ["tail_tip", "tail", "l_hip", "r_hip", "l_foot", "r_foot", "r_flank", "l_flank", "l_shoulder",
             "r_shoulder", "l_hand", "neck", "nose", "l_ear", "l_ear_tip", "r_ear", "r_ear_tip", "r_hand"]
DATA_YAML = "kpt_shape:\n- 18\n- 3\nnames:\n  0: pig\npath: .\ntrain: train.txt\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="CVAT export zip (<name>/annotations.xml + <name>/images/*)")
    ap.add_argument("--task", required=True, help="task id = top folder in the output zip (e.g. 001)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = zipfile.ZipFile(a.src)
    names = [n for n in src.namelist() if not n.startswith("__MACOSX") and not n.endswith("/")]
    xml_name = next(n for n in names if n.endswith("annotations.xml"))
    by_base = {Path(n).name: n for n in names if Path(n).suffix.lower() in {".jpg", ".jpeg", ".png"}}
    root = ET.fromstring(src.read(xml_name))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_img = n_pig = n_occ = n_out = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        z.writestr(f"{a.task}/label/data.yaml", DATA_YAML)
        for im in root.findall("image"):
            base = Path(im.get("name")).name
            if base not in by_base:
                raise SystemExit(f"{a.src}: XML의 이미지 {base} 가 zip에 없음")
            W, H = float(im.get("width")), float(im.get("height"))
            rows = []
            for sk in im.findall("skeleton"):
                pts = {p.get("label").lower(): p for p in sk.findall("points")}
                kp, xs, ys = [], [], []
                for nm in KPT_ORDER:
                    p = pts.get(nm)
                    if p is None or p.get("outside") == "1":
                        kp += ["0", "0", "0"]
                        n_out += 1
                        continue
                    x, y = map(float, p.get("points").split(";")[0].split(","))
                    xs.append(x)
                    ys.append(y)
                    v = 1 if p.get("occluded") == "1" else 2
                    n_occ += v == 1
                    kp += [f"{x / W:.6f}", f"{y / H:.6f}", str(v)]
                if not xs:
                    continue
                box = [(min(xs) + max(xs)) / 2 / W, (min(ys) + max(ys)) / 2 / H,
                       (max(xs) - min(xs)) / W, (max(ys) - min(ys)) / H]
                rows.append(" ".join(["0"] + [f"{b:.6f}" for b in box] + kp))
                n_pig += 1
            stem = Path(base).stem
            z.writestr(f"{a.task}/img/{base}", src.read(by_base[base]))
            z.writestr(f"{a.task}/label/labels/Train/{stem}.txt", "\n".join(rows) + ("\n" if rows else ""))
            n_img += 1
    print(f"[{a.task}] {a.src} -> {out}: 이미지 {n_img}장, 돼지 {n_pig}마리 "
          f"(가림 관절 {n_occ}, 화면 밖 관절 {n_out})")


if __name__ == "__main__":
    main()
