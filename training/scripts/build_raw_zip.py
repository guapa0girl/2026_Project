"""CVAT 'YOLO Pose 1.1' export(라벨만) + 로컬 원본 이미지 → training/data/raw/data/ 에
넣을 zip(<task>/img + <task>/label) 으로 묶는다.

CVAT 무료 플랜은 이미지 포함 export가 막혀 있어서, 라벨 zip과 PC에 있는 원본 이미지를
짝지어 ingest_raw_zips.py 가 요구하는 구조를 직접 만든다.
평가용 프레임(자세 판정 라벨 50장, eval_frames.txt)은 pose 학습에서 반드시 제외한다.

제외 매칭은 (1) 파일명 완전 일치, 또는 (2) 파일명 속 "[CH010] <영상>_f<프레임>" 부분
일치로 판단한다. task마다 "frame_0001__" 같은 접두어가 다르게 붙어도 같은 프레임이면
걸러지도록 하기 위함.

사용법 (training 폴더에서):
  python scripts/build_raw_zip.py --labels-zip "CVAT_export.zip" --images "원본이미지폴더" \
      --exclude eval_frames.txt --task CH010_431 --out data/raw/data/CH010_431.zip

CVAT task가 여러 개면 task마다 한 번씩 실행 (--task 이름은 서로 다르게).
"""
import argparse
import re
import zipfile
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# "[CH010] 20260716012647-20260716015128_f0203999" 같은 영상+프레임 식별자
FRAME_ID_RE = re.compile(r"\[[^\]]+\]\s*\d{14}-\d{14}_f\d+")


def frame_id(name):
    m = FRAME_ID_RE.search(name)
    return m.group(0) if m else None


p = argparse.ArgumentParser()
p.add_argument("--labels-zip", required=True, help="CVAT YOLO Pose 1.1 export zip (labels/Train/*.txt, data.yaml)")
p.add_argument("--images", required=True, help="원본 이미지 폴더 (하위 폴더까지 검색)")
p.add_argument("--exclude", default=None, help="학습에서 뺄 이미지 이름 목록 txt (한 줄에 하나, 확장자 없이)")
p.add_argument("--task", required=True, help="zip 안 최상위 폴더 이름 (task마다 다르게)")
p.add_argument("--out", required=True)
a = p.parse_args()

exclude = set()
if a.exclude:
    exclude = {l.strip() for l in open(a.exclude, encoding="utf-8") if l.strip()}
exclude_ids = {frame_id(e) for e in exclude} - {None}

images = {}
for f in Path(a.images).rglob("*"):
    if f.suffix.lower() in IMAGE_EXTS:
        images[f.stem] = f

src = zipfile.ZipFile(a.labels_zip)
names = src.namelist()
label_files = [n for n in names if n.lower().endswith(".txt") and "/labels/" in "/" + n]
yaml_name = next((n for n in names if n.endswith("data.yaml")), None)
if yaml_name is None or not label_files:
    raise SystemExit("라벨 zip에서 data.yaml 또는 labels/*.txt 를 찾지 못했습니다. YOLO Pose 1.1 export가 맞는지 확인하세요.")

n_ok = n_excl = n_missing = n_empty = 0
missing, train_list, excluded = [], [], []
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr(f"{a.task}/label/data.yaml", src.read(yaml_name))
    for n in sorted(label_files):
        stem = Path(n).stem
        if stem in exclude or frame_id(stem) in exclude_ids:
            n_excl += 1
            excluded.append(stem)
            continue
        data = src.read(n)
        if not data.strip():
            n_empty += 1          # 라벨이 하나도 없는 프레임 (돼지 0마리로 학습됨) -- 개수만 보고
        img = images.get(stem)
        if img is None:
            n_missing += 1
            missing.append(stem)
            continue
        z.writestr(f"{a.task}/label/labels/Train/{stem}.txt", data)
        z.write(img, f"{a.task}/img/{img.name}")
        train_list.append(f"data/images/Train/{img.name}")
        n_ok += 1
    z.writestr(f"{a.task}/label/Train.txt", "\n".join(train_list) + "\n")

print(f"[done] {out}")
print(f"  포함: {n_ok}장 / 평가용 제외: {n_excl}장 / 이미지 없음(제외): {n_missing}장 / 빈 라벨: {n_empty}장")
if n_excl != len(exclude) and a.exclude:
    print(f"  [참고] 제외 목록 {len(exclude)}장 중 이 export에서 찾은 것은 {n_excl}장 (다른 task에 있으면 정상)")
if missing:
    print("  [경고] 이미지를 못 찾은 라벨 (앞 10개):")
    for m in missing[:10]:
        print("   -", m)
