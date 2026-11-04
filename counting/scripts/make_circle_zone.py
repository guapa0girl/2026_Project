"""Write a polygon zone file (for the RandomForest path) from the circle
found by calibrate_feeder.py, so geo and RF use the same feeder area.

    python scripts/make_circle_zone.py --cx 925 --cy 520 --r 335 --out configs/zones/feeder_zones_ch010.yaml
"""
import argparse, math
import yaml

p = argparse.ArgumentParser()
p.add_argument("--cx", type=float, required=True)
p.add_argument("--cy", type=float, required=True)
p.add_argument("--r", type=float, required=True)
p.add_argument("--n", type=int, default=48, help="polygon vertices")
p.add_argument("--out", required=True)
a = p.parse_args()

pts = [[round(a.cx + a.r * math.cos(2 * math.pi * i / a.n)), round(a.cy + a.r * math.sin(2 * math.pi * i / a.n))]
       for i in range(a.n)]
with open(a.out, "w", encoding="utf-8") as f:
    yaml.safe_dump({"zones": [pts]}, f)
print(f"[saved] {a.out} ({a.n}각형, 중심=({a.cx},{a.cy}) r={a.r})")
