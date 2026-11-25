"""그림 6 (논의 및 한계의 밀집 장면 예시): 학습 190장 중 visiting이 가장 많은 사진을, 그 사진이 validation에 들어간
Time 5-fold 모델(학습에 이 사진을 안 쓴 모델)로 판정해 정답과 RF(자세+기하+다리) 판정 상자를 나란히 그림.
현재 결과: 0069번 사진(Train1, CH005), k5_f3 — 정답 visiting 7마리, RF(자세+기하+다리) 7마리.
           급이 파이프 아래 돼지 1마리는 검출 상자가 정답 상자와 IoU 0.49로 기준(0.5)에 조금 못 미쳐,
           개체 단위로는 '미검출 1 + 라벨 없는 검출 1'로 집계됨 (실제 다른 돼지를 잘못 센 것은 아님).
출력: <out>/fig_dense_case.png (300 dpi), 콘솔에 정답 · 판정 마릿수, 오류 분해, 짝이 안 된 상자와 IoU

다른 PC에서 다시 출력하는 방법 (counting/ 에서)
  python scripts/dense_case.py
  - paper_figures.py 와 같은 준비가 필요함 (글꼴 · 패키지는 paper_figures.py 맨 위 설명 참고, 글꼴 폴더는 HCI_FONT_DIR 로 지정).
  - 필요한 파일 (GitHub에 없는 자료, 레포 기준 경로에 둠):
      counting/data/labels/state_001.xml, state_002.xml        학습 사진 라벨
      counting/data/raw/train_img/                              학습 원본 사진
      training/data/splits/k5_f*.json                           fold 목록 (이 사진이 어느 fold의 validation인지 찾음)
      training/models/results/ep200_k5_f*/m/weights/best.pt     해당 fold 자세 추정 가중치
  - 출력 폴더를 바꾸려면 아래 OUT 값을 고침 (농장 사진이 들어가므로 레포 밖 권장).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figures as pf  # noqa: E402
from analyze_iou_matching import iou  # noqa: E402

OUT = Path(r"C:/ETRI/share/hci2027_figs")


def main():
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    from scipy.optimize import linear_sum_assignment
    from feeding_monitor.pose_tracker import PoseTracker, _parse_detections
    sys.stdout.reconfigure(encoding="utf-8")
    cams, train, _ = pf.cams_rows()
    tr_by = pf.by_image(train)
    stem = max(tr_by.items(), key=lambda kv: (sum(r["label"] for r in kv[1]), len(kv[1])))[0]
    split = next(f"k5_f{i}" for i in range(5) if any(s.split("__", 1)[1] == stem for s in json.loads(
        (pf.REPO / "training" / "data" / "splits" / f"k5_f{i}.json").read_text(encoding="utf-8"))["valid"]))
    s1 = pf.em.featurize(train, cams)
    keep = {s.split("__", 1)[1] for s in json.loads(
        (pf.REPO / "training" / "data" / "splits" / f"{split}.json").read_text(encoding="utf-8"))["train"]}
    feats = pf.METHOD_FEATURES["rf_geo_leg"]
    tr = s1[s1["image"].isin(keep)]
    clf = pf.em.new_rf(42).fit(tr[feats], tr["label"])
    rows = tr_by[stem]
    cam = rows[0]["camera"]
    cfg = cams.cfg[cam]
    cfg.model.weights = str(pf.REPO / "training" / "models" / "results" / f"ep200_{split}" / "m" / "weights" / "best.pt")
    img_path = pf.em.find_images(pf.ROOT / "data" / "raw" / "train_img")[stem]
    t = PoseTracker(cfg.model)
    im = cv2.imread(str(img_path))
    res = t.model.predict(im, conf=cfg.model.conf, imgsz=cfg.model.imgsz, iou=cfg.model.iou, device=t.device, verbose=False)[0]
    preds = _parse_detections(res, 0)
    df = pf.em.featurize([{"image": stem, "camera": cam, "clip": rows[0]["clip"], "label": -1, "det": d, "source": "model"}
                          for d in preds], cams)
    pv = clf.predict(df[feats])
    gl = np.array([r["label"] for r in rows])
    M = iou([r["det"].box for r in rows], [d.box for d in preds])
    r_, c_ = linear_sum_assignment(-M)
    pairs = [(i, j) for i, j in zip(r_, c_) if M[i, j] >= 0.5]
    mg, mp = {i for i, _ in pairs}, {j for _, j in pairs}
    undet = int(sum(gl[i] == 1 for i in range(len(gl)) if i not in mg))
    judged = sum(int(gl[i] == 1 and pv[j] == 0) for i, j in pairs)
    wrong = sum(int(gl[i] == 0 and pv[j] == 1) for i, j in pairs)
    extra = int(sum(pv[j] == 1 for j in range(len(pv)) if j not in mp))
    print(f"stem={stem}\nsplit={split} (이 사진이 validation)\n정답 visiting={gl.sum()} / 돼지 {len(gl)}마리")
    print(f"자세 추정 검출 {len(preds)}마리, RF(자세+기하+다리) visiting={int(pv.sum())}")
    print(f"미검출 visiting={undet}, visiting→other={judged}, other→visiting={wrong}, 라벨 없는 검출→visiting={extra}")
    # 짝이 안 된 정답 visiting / visiting 판정 검출의 상자와, 가장 가까운 상대의 IoU (IoU 0.5 미만이라 짝이 안 된 것인지 확인)
    for i in range(len(gl)):
        if i not in mg and gl[i] == 1:
            print(f"  짝 없는 정답 visiting {np.round(rows[i]['det'].box).astype(int).tolist()}  최대 IoU={M[i].max():.2f}")
    for j in range(len(pv)):
        if j not in mp and pv[j] == 1:
            print(f"  짝 없는 visiting 판정 {np.round(preds[j].box).astype(int).tolist()}  최대 IoU={M[:, j].max():.2f}")

    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    feeder = cfg.feeder
    crop = pf.crop_box(feeder, rgb.shape, scale=1.75, aspect=1.5)
    sub = rgb[crop[1]:crop[3], crop[0]:crop[2]]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75))
    for ax, which in zip(axes, ("gt", "rf")):
        ax.imshow(sub)
        ax.add_patch(Circle((feeder.cx - crop[0], feeder.cy - crop[1]), feeder.r, fill=False, ec="white", ls="--", lw=0.9))
        if which == "gt":
            for r in rows:
                if r["label"] == 1:
                    x1, y1, x2, y2 = r["det"].box
                    ax.add_patch(Rectangle((x1 - crop[0], y1 - crop[1]), x2 - x1, y2 - y1, fill=False, ec=pf.VIS_C, lw=1.6))
            ax.set_title(f"정답 visiting {gl.sum()}마리", fontsize=9)
        else:
            for d, v in zip(preds, pv):
                if v == 1:
                    x1, y1, x2, y2 = d.box
                    ax.add_patch(Rectangle((x1 - crop[0], y1 - crop[1]), x2 - x1, y2 - y1, fill=False,
                                           ec=pf.COLOR["rf_geo_leg"], lw=1.6))
            ax.set_title(f"RF(자세+기하+다리): {int(pv.sum())}마리", fontsize=9,
                         color="#000000" if int(pv.sum()) == gl.sum() else "#c0392b")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "fig_dense_case.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
