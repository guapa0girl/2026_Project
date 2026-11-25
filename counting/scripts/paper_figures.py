"""HCI 2027 논문 그림 (교수님 첨삭 · 검토 메모 반영) — 한국어 라벨, 나눔고딕, 모두 300 dpi.

출력 파일 (--out 폴더, 기본 C:/ETRI/share/hci2027_figs)
  fig_data_examples.png   그림 1: 세트별 예시 사진(Train1 · Train2 · Test, 1×3), 사료통 원 · 정답 상자
  fig_pipeline.png        그림 2: 처리 흐름(무채색 순서도) (a) 자세 추정 기반 방법, (b) 객체 검출 기반 방법
  fig_counts_time.png     그림 3: Test 사진별 visiting 수 (Time 5-fold), 그림 5 사례 사진 번호 빨간 표시
  fig_importance.png      그림 4: RF(자세+기하+다리) 특징 중요도 (RF(자세+기하) 비교)
  fig5a_cases_good.png    그림 5(a): RF(자세+기하+다리)가 잘 찾은 사례 (72 · 81번)
  fig5b_cases_error.png   그림 5(b): 오차 사례 (57번 가림 · 65번 과대 집계), 1×4 = 57 정답 · 57 RF · 65 정답 · 65 RF
  fig_cases.png / fig_cases_rf.png   이전 판 사례 그림 (방법 4개 비교 / RF 한 장)
  fig_dashboard.png       (원고에서 삭제됨) 대시보드 활용 예시
  ※ 그림 6(밀집 장면)은 scripts/dense_case.py 가 따로 만듦

==================================================================================================
다른 PC에서 그림 · 그래프를 다시 출력하는 방법
==================================================================================================
0) 공통 준비
   - 레포를 받고 counting/ 폴더에서 실행함 (모든 경로는 이 파일 위치 기준으로 자동 계산됨).
   - 파이썬 패키지: numpy, pandas, matplotlib, scikit-learn, opencv-python, ultralytics, torch
     (기존 PC: C:/ETRI/Piggy_Feeding_Counting/.venv). 예) pip install -r counting/requirements.txt
   - 한글 글꼴: 나눔고딕(NanumGothic-Regular.ttf, NanumGothic-Bold.ttf)이 든 폴더를 아래 중 하나로 알려 줌.
       a. 실행 옵션  --font-dir <폴더>
       b. 환경 변수  HCI_FONT_DIR=<폴더>
       c. 기본 위치  C:/ETRI/share/hci2027_overleaf/fonts  (Overleaf 프로젝트의 fonts/ 폴더와 같음)
     셋 다 없으면 시스템 글꼴(맑은 고딕 / AppleGothic)로 그림. 글꼴이 바뀌면 글자 폭이 달라져 배치가 약간 달라질 수 있음.

1) 레포만 있어도 다시 그릴 수 있는 그림 (원본 사진 · 라벨 · 가중치 불필요)
   - 그림 2 (pipeline), 그림 3 (counts), 그림 4 (importance)
   - 입력: reports/fold_study/per_frame_000/*.csv, reports/fold_study/paper/rf_importance_by_method.csv (레포에 있음)
   - 명령:  python scripts/paper_figures.py --only pipeline counts importance --out <출력 폴더>

2) 농장 원본 데이터가 있어야 하는 그림 (GitHub에 올리지 않는 자료이므로 공유 드라이브 등에서 복사)
   - 그림 1 (examples), 그림 5 (cases), 대시보드 (dashboard)
   - 필요한 파일 (레포 기준 경로에 그대로 둠)
       counting/data/labels/state_000.xml, state_001.xml, state_002.xml   CVAT 라벨 (Test, Train1, Train2)
       counting/data/raw/000_img/, counting/data/raw/train_img/            원본 사진
       training/data/splits/k5_f0~4.json, rk5_f0~4.json                     fold 목록
       training/models/results/ep200_<split>/m/weights/best.pt             자세 추정 가중치 (그림 5의 대표 fold = k5_f3)
       대시보드만: --day-dir 의 *_methods_1s_all.csv (CH010 7/16 1초 판정 결과)
   - 그림 5는 처음 한 번 자세 추정 + RF 추론 결과를 <출력 폴더>/cache_<split>.pkl 로 저장해 두고, 다음부터는 캐시를 씀
     (GPU 권장, CPU도 가능하지만 느림). 다른 fold로 바꾸면 --split k5_f0 처럼 지정.
     기존 PC의 cache_*.pkl 을 같이 복사해 오면 가중치 · 원본 사진 없이도 그림 5의 판정 상자는 다시 그릴 수 있으나,
     배경 사진이 필요하므로 원본 사진(000_img)은 있어야 함.
   - 명령:  python scripts/paper_figures.py                              (전부)
            python scripts/paper_figures.py --only examples cases       (일부만)

3) 출력은 농장 사진이 들어가므로 기본적으로 레포 밖(--out)에 저장함. 논문 · PAPER.md 용 고해상도 사본은
   reports/fold_study/paper_figures/ 에 복사해 커밋함 (원본 데이터 · 라벨 · 가중치는 커밋 금지).
4) 숫자 확인: 그림 5는 사진별 판정 마릿수가 per_frame_000 CSV와 같은지 assert로 확인함 (다르면 중단).
==================================================================================================
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import evaluate_methods as em  # noqa: E402
from feeding_monitor.method_features import METHOD_FEATURES  # noqa: E402

FS = ROOT / "reports" / "fold_study"
DEFAULT_FONT_DIR = Path(r"C:/ETRI/share/hci2027_overleaf/fonts")


def setup_fonts(font_dir=None):
    """한글 글꼴 등록. 우선순위: 인자(--font-dir) > 환경 변수 HCI_FONT_DIR > 기본 위치 > 시스템 글꼴."""
    import os
    for d in (font_dir, os.environ.get("HCI_FONT_DIR"), DEFAULT_FONT_DIR):
        if d and Path(d).is_dir():
            for f in Path(d).glob("NanumGothic*.ttf"):
                font_manager.fontManager.addfont(str(f))
    have = {f.name for f in font_manager.fontManager.ttflist}
    fams = [f for f in ("NanumGothic", "Malgun Gothic", "AppleGothic") if f in have] + ["DejaVu Sans"]   # 설치된 것만
    plt.rcParams.update({"font.family": fams, "axes.unicode_minus": False, "font.size": 9})


setup_fonts()        # 다른 스크립트(dense_case.py)에서 import 할 때도 기본 글꼴이 잡히도록

NAME = {"geo": "기하 규칙", "rf": "RF(자세)", "rf_geo": "RF(자세+기하)", "rf_geo_leg": "RF(자세+기하+다리)", "yolo": "객체 검출"}
COLOR = {"gt": "#000000", "geo": "#5a5a5a", "rf_geo_leg": "#08519c", "yolo": "#e8710a"}
VIS_C, OTH_C, MISS_C = "#ff8c00", "#bdbdbd", "#e31a1c"
FEAT_KO = {
    "nose_in_zone": "코의 구역 포함 여부", "nose_zone_dist_norm": "코-구역 거리", "head_dip_deg": "머리 숙임 각도",
    "spine_dip_deg": "척추 숙임 각도", "neck_rel_height": "목 상대 높이", "key_kpt_conf": "주요 관절 검출 비율",
    "geo_nose_dist_over_r": "코-사료통 거리비", "geo_head_angle_deg": "머리 방향 각도", "geo_limb_visibility": "다리 가시성",
    "geo_aspect_ratio": "몸통 종횡비", "geo_lateral_asymmetry": "좌우 비대칭", "leg_extension": "다리 길이",
    "leg_angle_deg": "다리-척추 각도", "paw_offset": "발과 척추선 거리", "leg_side_bias": "발이 한쪽에 몰린 정도",
    "body_bend_deg": "몸통 굽힘"}
GROUP_KO = {"RF": "자세", "Geo": "기하", "Leg": "다리"}
GROUP_C = {"RF": "#6baed6", "Geo": "#3182bd", "Leg": "#08519c"}


def cams_rows():
    cams = em.Cameras(ROOT / "configs" / "eval_cameras.yaml")
    lab = ROOT / "data" / "labels"
    train = [r for x in ("state_001.xml", "state_002.xml") for r in em.parse_cvat_xml(lab / x) if cams.has(r["camera"])]
    test = [r for r in em.parse_cvat_xml(lab / "state_000.xml") if cams.has(r["camera"])]
    return cams, train, test


def by_image(rows):
    out = {}
    for r in rows:
        out.setdefault(r["image"], []).append(r)
    return out


def test_order(test_rows):
    """Test 사진 번호(1~86, 시간순) → stem."""
    d = pd.read_csv(FS / "per_frame_000" / "geo.csv", encoding="utf-8-sig").sort_values("frame_idx").reset_index(drop=True)
    return {i + 1: s for i, s in enumerate(d["image"])}, d


# ------------------------------------------------------------------ 추론 캐시
def build_cache(split, out):
    path = out / f"cache_{split}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    import cv2
    from feeding_monitor.pose_tracker import PoseTracker, _parse_detections
    cams, train, test = cams_rows()
    s1 = em.featurize(train, cams)
    keep = {s.split("__", 1)[1] for s in json.loads(
        (REPO / "training" / "data" / "splits" / f"{split}.json").read_text(encoding="utf-8"))["train"]}
    tr = s1[s1["image"].isin(keep)]
    models = {m: (em.new_rf(42).fit(tr[f], tr["label"]), f) for m, f in METHOD_FEATURES.items()}
    w = REPO / "training" / "models" / "results" / f"ep200_{split}" / "m" / "weights" / "best.pt"
    for c in cams.cfg.values():
        c.model.weights = str(w)
    images = em.find_images(ROOT / "data" / "raw" / "000_img")
    cache, trackers = {}, {}
    for stem, rows in by_image(test).items():
        cam = rows[0]["camera"]
        cfg = cams.cfg[cam]
        if cam not in trackers:
            trackers[cam] = PoseTracker(cfg.model)
        t = trackers[cam]
        res = t.model.predict(cv2.imread(str(images[stem])), conf=cfg.model.conf, imgsz=cfg.model.imgsz,
                              iou=cfg.model.iou, device=t.device, verbose=False)[0]
        preds = _parse_detections(res, 0)
        rec = {"boxes": [p.box for p in preds], "pred": {}}
        if preds:
            pf = em.featurize([{"image": stem, "camera": cam, "clip": rows[0]["clip"], "label": -1, "det": d,
                                "source": "model"} for d in preds], cams)
            rec["pred"]["geo"] = pf["pred_geo"].to_numpy()
            rec["dist_r"] = pf["geo_nose_dist_over_r"].to_numpy()          # 코-사료통 거리 ÷ 반지름 (가장자리 분석용)
            for m, (clf, f) in models.items():
                rec["pred"][m] = clf.predict(pf[f])
        cache[stem] = rec
    path.write_bytes(pickle.dumps(cache))
    return cache


# ------------------------------------------------------------------ 그림: 데이터 예시
def crop_box(feeder, img_shape, scale=1.9, aspect=1.4):
    cx, cy, r = feeder.cx, feeder.cy, feeder.r
    h = 2 * r * scale / 1.0
    w = h * aspect
    x0, y0 = max(0, cx - w / 2), max(0, cy - h / 2)
    x1, y1 = min(img_shape[1], x0 + w), min(img_shape[0], y0 + h)
    return int(x0), int(y0), int(x1), int(y1)


def draw_gt(ax, rows, feeder, crop=None, show_other=True):
    ox, oy = (crop[0], crop[1]) if crop else (0, 0)
    ax.add_patch(Circle((feeder.cx - ox, feeder.cy - oy), feeder.r, fill=False, ec="white", ls="--", lw=1.0))
    for r in rows:
        x1, y1, x2, y2 = r["det"].box
        if r["label"] == 1:
            ax.add_patch(Rectangle((x1 - ox, y1 - oy), x2 - x1, y2 - y1, fill=False, ec=VIS_C, lw=1.6))
        elif show_other:
            ax.add_patch(Rectangle((x1 - ox, y1 - oy), x2 - x1, y2 - y1, fill=False, ec=OTH_C, lw=0.6, alpha=0.8))


def fig_data_examples(out):
    import cv2
    cams, train, test = cams_rows()
    tr_img = em.find_images(ROOT / "data" / "raw" / "train_img")
    te_img = em.find_images(ROOT / "data" / "raw" / "000_img")
    bi = {**by_image(train), **by_image(test)}

    def pick(rows_by, cam, n_vis):
        cand = [(s, rs) for s, rs in rows_by.items() if rs[0]["camera"] == cam and sum(r["label"] for r in rs) == n_vis]
        cand.sort(key=lambda x: (-len(x[1]), x[0]))
        return cand[len(cand) // 2][0]

    tr_by, te_by = by_image(train), by_image(test)
    sel = [("Train1", pick(tr_by, "CH005", 2), tr_img), ("Train2", pick(tr_by, "CH010", 2), tr_img),
           ("Test", pick(te_by, "SAMPLE", 2), te_img)]
    # 밀집 장면은 논의 · 한계 절의 그림(scripts/dense_case.py)으로 옮김
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.15))
    axes = axes.ravel()
    for ax, (lab, stem, imgs) in zip(axes, sel):
        im = cv2.cvtColor(cv2.imread(str(imgs[stem])), cv2.COLOR_BGR2RGB)
        rows = bi[stem]
        feeder = cams.cfg[rows[0]["camera"]].feeder
        ax.imshow(im)
        draw_gt(ax, rows, feeder)
        n_v = sum(r["label"] for r in rows)
        ax.set_title(f"{lab}: 돼지 {len(rows)}마리 (visiting {n_v})", fontsize=8)
        ax.axis("off")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=VIS_C, lw=2, label="visiting (정답)"),
                        Line2D([], [], color=OTH_C, lw=1, label="other (정답)"),
                        Line2D([], [], color="#777", ls="--", lw=1, label="사료통 원")],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig_data_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [s for _, s, _ in sel]


# ------------------------------------------------------------------ 그림: 처리 흐름
def fig_pipeline(out):
    """무채색 흐름도: 처리 상자(사각형) + 판정(마름모) + 예/아니오 분기."""
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")
    K, FC, FC2 = "#333333", "#f2f2f2", "#d9d9d9"

    def box(cx, cy, w, h, t, fc=FC, bold=False):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                    fc=fc, ec=K, lw=0.8))
        ax.text(cx, cy, t, ha="center", va="center", fontsize=7.2, linespacing=1.25,
                fontweight="bold" if bold else "normal", color="#111")

    def diamond(cx, cy, w, h, t):
        ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                             closed=True, fc="#ffffff", ec=K, lw=0.8))
        ax.text(cx, cy, t, ha="center", va="center", fontsize=6.8, linespacing=1.15)

    def arr(x1, y1, x2, y2, label=None, lx=0, ly=0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=K))
        if label:
            ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=6.5, color="#333", ha="center", va="center")

    ya, yb = 35, 11
    ax.text(13, 46.5, "(a) 자세 추정 기반 방법", fontsize=8.3, fontweight="bold", va="center")
    ax.text(13, 22.5, "(b) 객체 검출 기반 방법", fontsize=8.3, fontweight="bold", va="center")
    box(5.5, 23, 9.5, 10, "CCTV\n사진", FC2)
    box(20, ya, 14, 11, "자세 추정\nYOLOv8m-pose\n(keypoint 18종)")
    box(37, ya, 15, 11, "특징 계산\n자세 · 기하 · 다리\n(사료통 원 기준)")
    box(54, ya, 14, 11, "돼지별 판정\n기하 규칙 또는\n랜덤 포레스트")
    diamond(70, ya, 11, 11, "visiting?")
    box(20, yb, 14, 11, "객체 검출\nYOLOv8m\n(visiting / other)")
    box(37, yb, 15, 11, "겹친 상자 정리\n(NMS)")
    diamond(54, yb, 13, 12, "visiting\n신뢰도 ≥\n임계값?")
    box(91, 23, 15, 10, "사진당\nvisiting 수", FC2, bold=True)
    arr(10.3, 25, 13, ya - 2)
    arr(10.3, 21, 13, yb + 2)
    arr(27, ya, 29.5, ya)
    arr(44.5, ya, 47, ya)
    arr(61, ya, 64.5, ya)
    arr(27, yb, 29.5, yb)
    arr(44.5, yb, 47.5, yb)
    arr(75.5, ya, 83.5, 26, "예", 0, 2.2)
    arr(60.5, yb, 83.5, 20, "예", 0, -2.2)
    arr(70, ya - 5.5, 70, ya - 10, "아니오: 제외", 6.5, 0)
    arr(54, yb - 6, 54, yb - 10, "아니오: 제외", 6.5, 0)
    fig.savefig(out / "fig_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 그림: 사진별 마릿수
def fig_counts(out, cases):
    from matplotlib.ticker import MultipleLocator
    off = {"geo": -0.06, "rf_geo_leg": 0.0, "yolo": 0.06}
    fig, ax = plt.subplots(figsize=(12, 3.0))
    gt = None
    for m in ["geo", "rf_geo_leg", "yolo"]:
        d = pd.read_csv(FS / "per_frame_000" / f"{m}.csv", encoding="utf-8-sig").sort_values("frame_idx")
        gt = d["gt"].to_numpy(float)
        pred = d[[f"k5_f{i}" for i in range(5)]].to_numpy(float).T
        x = np.arange(1, len(gt) + 1)
        if m == "geo":
            ax.step(x, gt, where="mid", color="#c8c8c8", lw=7, solid_capstyle="butt", label="정답", zorder=1)
        mae = np.abs(pred - gt).mean()
        ax.step(x, pred.mean(0) + off[m], where="mid", color=COLOR[m], lw=1.8, zorder=3,
                label=f"{NAME[m]} (MAE {mae:.3f})")
    ax.set_ylim(-0.25, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_xlim(0, len(gt) + 0.5)
    nums = sorted(set([0] + list(range(5, 86, 5)) + list(cases)))
    ax.set_xticks(nums, ["" if n in cases else str(n) for n in nums])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    for n in cases:
        ax.axvline(n, color="#c0392b", lw=0.9, ls=":", alpha=0.8, zorder=0.6)
        ax.annotate(str(n), xy=(n, 0), xycoords=("data", "axes fraction"), xytext=(0, -20), textcoords="offset points",
                    ha="center", va="top", color="#c0392b", fontweight="bold", fontsize=9, annotation_clip=False)
    ax.xaxis.labelpad = 20
    ax.set_xlabel("Test 사진 번호 (시간순, 1~86; 빨간 숫자 = 그림 5의 사례 사진)")
    ax.set_ylabel("visiting 수")
    ax.grid(axis="y", color="#ececec", lw=0.8)
    ax.grid(axis="x", which="major", color="#eeeeee", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    fig.tight_layout()
    fig.savefig(out / "fig_counts_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 그림: 특징 중요도
def fig_importance(out):
    imp = pd.read_csv(FS / "paper" / "rf_importance_by_method.csv", encoding="utf-8-sig")
    leg = imp[imp.method == "rf_geo_leg"].groupby(["feature", "group"]).importance.agg(["mean", "std"]).reset_index()
    geo = imp[imp.method == "rf_geo"].groupby("feature").importance.mean()
    leg = leg.sort_values("mean")
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    y = np.arange(len(leg))
    ax.barh(y, leg["mean"], xerr=leg["std"], color=[GROUP_C[g] for g in leg["group"]], height=0.7,
            error_kw=dict(lw=0.7, capsize=2, ecolor="#555"))
    gv = [geo.get(f, np.nan) for f in leg["feature"]]
    ax.scatter(gv, y, marker="D", s=22, facecolor="white", edgecolor="#e8710a", lw=1.2, zorder=3,
               label="RF(자세+기하)에서의 중요도")
    ax.set_yticks(y, [FEAT_KO[f] for f in leg["feature"]], fontsize=8.5)
    ax.set_xlabel("특징 중요도 (Gini, fold 10개 평균 ± SD)")
    ax.grid(axis="x", color="#ececec", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    tot = imp[imp.method == "rf_geo_leg"].groupby("group").importance.sum() / imp.split.nunique()
    h = [Patch(color=GROUP_C[g], label=f"{GROUP_KO[g]} 특징 (합계 {tot[g]:.2f})") for g in ["RF", "Geo", "Leg"]]
    h.append(ax.collections[-1])
    ax.legend(handles=h, loc="lower right", fontsize=8, frameon=True, edgecolor="#dddddd")
    fig.tight_layout()
    fig.savefig(out / "fig_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 그림: 정성 사례
def fig_cases(out, split, cases, case_titles):
    import cv2
    cams, _, test = cams_rows()
    cache = build_cache(split, out)
    order, _ = test_order(test)
    te_img = em.find_images(ROOT / "data" / "raw" / "000_img")
    bi = by_image(test)
    counts = {m: pd.read_csv(FS / "per_frame_000" / f"{m}.csv", encoding="utf-8-sig").set_index("image")[split]
              for m in ["geo", "rf_geo_leg", "yolo"]}
    cols = [("gt", "정답"), ("geo", NAME["geo"]), ("rf_geo_leg", NAME["rf_geo_leg"]), ("yolo", NAME["yolo"])]
    fig, axes = plt.subplots(len(cases), len(cols), figsize=(11, 2.55 * len(cases)))
    for r_i, no in enumerate(cases):
        stem = order[no]
        rows = bi[stem]
        feeder = cams.cfg[rows[0]["camera"]].feeder
        im = cv2.cvtColor(cv2.imread(str(te_img[stem])), cv2.COLOR_BGR2RGB)
        crop = crop_box(feeder, im.shape)
        sub = im[crop[1]:crop[3], crop[0]:crop[2]]
        n_gt = sum(r["label"] for r in rows)
        for c_i, (m, lab) in enumerate(cols):
            ax = axes[r_i, c_i]
            ax.imshow(sub)
            ax.add_patch(Circle((feeder.cx - crop[0], feeder.cy - crop[1]), feeder.r, fill=False, ec="white", ls="--", lw=0.9))
            if m == "gt":
                for r in rows:
                    if r["label"] == 1:
                        x1, y1, x2, y2 = r["det"].box
                        ax.add_patch(Rectangle((x1 - crop[0], y1 - crop[1]), x2 - x1, y2 - y1, fill=False, ec=VIS_C, lw=1.8))
                title, col = f"정답 visiting {n_gt}마리", "#000000"
            elif m == "yolo":
                n = int(counts[m][stem])
                ax.text(0.5, 0.06, "검출 상자 미표시", transform=ax.transAxes, ha="center", fontsize=7, color="white",
                        bbox=dict(fc="#00000088", ec="none", pad=1.5))
                title, col = f"{lab}: {n}마리", ("#000000" if n == n_gt else "#c0392b")
            else:
                rec = cache[stem]
                p = rec["pred"].get(m, [])
                for b, v in zip(rec["boxes"], p):
                    if v == 1:
                        x1, y1, x2, y2 = b
                        ax.add_patch(Rectangle((x1 - crop[0], y1 - crop[1]), x2 - x1, y2 - y1, fill=False,
                                               ec=COLOR[m] if m != "geo" else "#ffffff", lw=1.8))
                n = int(np.sum(p))
                assert n == int(counts[m][stem]), (m, no, n, counts[m][stem])
                title, col = f"{lab}: {n}마리", ("#000000" if n == n_gt else "#c0392b")
            ax.set_title(title, fontsize=8.5, color=col)
            ax.set_xticks([])
            ax.set_yticks([])
            if c_i == 0:
                ax.set_ylabel(f"{no}번\n{case_titles.get(no, '')}", fontsize=8.5, rotation=0, ha="right", va="center", labelpad=6)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=VIS_C, lw=2, label="정답 visiting"),
                        Line2D([], [], color="#ffffff", lw=2, label="기하 규칙이 센 visiting", markeredgecolor="#333"),
                        Line2D([], [], color=COLOR["rf_geo_leg"], lw=2, label="RF(자세+기하+다리)가 센 visiting"),
                        Line2D([], [], color="#777", ls="--", lw=1, label="사료통 원")],
               loc="lower center", ncol=4, frameon=True, facecolor="#9e9e9e", edgecolor="none", fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "fig_cases.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_cases_rf(out, split, cases, case_titles, method="rf_geo_leg"):
    """그림 5 (RF best만): 열 = 사례(순서대로), 행 = 정답 / RF best. 대표 fold 1개."""
    import cv2
    cams, _, test = cams_rows()
    cache = build_cache(split, out)
    order, _ = test_order(test)
    te_img = em.find_images(ROOT / "data" / "raw" / "000_img")
    bi = by_image(test)
    counts = pd.read_csv(FS / "per_frame_000" / f"{method}.csv", encoding="utf-8-sig").set_index("image")[split]
    fig, axes = plt.subplots(2, len(cases), figsize=(2.75 * len(cases), 4.9))
    for c_i, no in enumerate(cases):
        stem = order[no]
        rows = bi[stem]
        feeder = cams.cfg[rows[0]["camera"]].feeder
        im = cv2.cvtColor(cv2.imread(str(te_img[stem])), cv2.COLOR_BGR2RGB)
        crop = crop_box(feeder, im.shape)
        sub = im[crop[1]:crop[3], crop[0]:crop[2]]
        n_gt = sum(r["label"] for r in rows)
        rec = cache[stem]
        p = rec["pred"].get(method, [])
        n = int(np.sum(p))
        assert n == int(counts[stem]), (no, n, counts[stem])
        for r_i in range(2):
            ax = axes[r_i, c_i]
            ax.imshow(sub)
            ax.add_patch(Circle((feeder.cx - crop[0], feeder.cy - crop[1]), feeder.r, fill=False, ec="white", ls="--", lw=0.9))
            if r_i == 0:
                for r in rows:
                    if r["label"] == 1:
                        x1, y1, x2, y2 = r["det"].box
                        ax.add_patch(Rectangle((x1 - crop[0], y1 - crop[1]), x2 - x1, y2 - y1, fill=False, ec=VIS_C, lw=1.8))
                ax.set_title(f"{no}번 · {case_titles[no]}\n정답 visiting {n_gt}마리", fontsize=8.5)
            else:
                for b, v in zip(rec["boxes"], p):
                    if v == 1:
                        x1, y1, x2, y2 = b
                        ax.add_patch(Rectangle((x1 - crop[0], y1 - crop[1]), x2 - x1, y2 - y1, fill=False,
                                               ec=COLOR[method], lw=1.8))
                ax.set_title(f"{NAME[method]}: {n}마리", fontsize=8.5, color="#000000" if n == n_gt else "#c0392b")
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("정답", fontsize=9)
    axes[1, 0].set_ylabel(NAME[method], fontsize=9)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=VIS_C, lw=2, label="정답 visiting"),
                        Line2D([], [], color=COLOR[method], lw=2, label=f"{NAME[method]}가 센 visiting"),
                        Line2D([], [], color="#777", ls="--", lw=1, label="사료통 원")],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.04, right=0.995, top=0.9, bottom=0.08, wspace=0.06, hspace=0.32)
    fig.savefig(out / "fig_cases_rf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_cases_split(out, split, method="rf_geo_leg"):
    """그림 5를 두 장으로: (a) 잘 찾은 사례 72 · 81번 = RF best 판정만 1×2,
    (b) 오차 사례 57(가림) · 65(과대 집계)번 = [57 정답, 57 RF, 65 정답, 65 RF] 1×4.
    두 장의 칸 크기가 같으므로 원고에는 (a)를 (b)의 절반 폭으로 넣으면 사진 크기가 맞음 (분량 절약용 한 줄 배치)."""
    import cv2
    from matplotlib.lines import Line2D
    cams, _, test = cams_rows()
    cache = build_cache(split, out)
    order, _ = test_order(test)
    te_img = em.find_images(ROOT / "data" / "raw" / "000_img")
    bi = by_image(test)
    counts = pd.read_csv(FS / "per_frame_000" / f"{method}.csv", encoding="utf-8-sig").set_index("image")[split]

    def panel(ax, no, which):
        stem = order[no]
        rows = bi[stem]
        feeder = cams.cfg[rows[0]["camera"]].feeder
        im = cv2.cvtColor(cv2.imread(str(te_img[stem])), cv2.COLOR_BGR2RGB)
        crop = crop_box(feeder, im.shape)
        ax.imshow(im[crop[1]:crop[3], crop[0]:crop[2]])
        ax.add_patch(Circle((feeder.cx - crop[0], feeder.cy - crop[1]), feeder.r, fill=False, ec="white", ls="--", lw=0.9))
        n_gt = sum(r["label"] for r in rows)
        if which == "gt":
            boxes, col = [r["det"].box for r in rows if r["label"] == 1], VIS_C
            title, tc = f"{no}번 정답\nvisiting {n_gt}마리", "#000000"
        else:
            rec = cache[stem]
            p = rec["pred"].get(method, [])
            boxes, col = [b for b, v in zip(rec["boxes"], p) if v == 1], COLOR[method]
            n = int(np.sum(p))
            assert n == int(counts[stem]), (no, n, counts[stem])
            title, tc = f"{no}번 {NAME[method]}\n{n}마리 (정답 {n_gt})", ("#000000" if n == n_gt else "#c0392b")
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(Rectangle((x1 - crop[0], y1 - crop[1]), x2 - x1, y2 - y1, fill=False, ec=col, lw=1.8))
        ax.set_title(title, fontsize=8.5, color=tc)
        ax.set_xticks([])
        ax.set_yticks([])

    leg = [Line2D([], [], color=VIS_C, lw=2, label="정답 visiting"),
           Line2D([], [], color=COLOR[method], lw=2, label=f"{NAME[method]}가 센 visiting"),
           Line2D([], [], color="#777", ls="--", lw=1, label="사료통 원")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, no in zip(axes, [72, 81]):
        panel(ax, no, "rf")
    fig.legend(handles=leg[1:], loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig5a_cases_good.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 2.9))
    for c, no in enumerate([57, 65]):
        panel(axes[2 * c], no, "gt")
        panel(axes[2 * c + 1], no, "rf")
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig5b_cases_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 그림: 대시보드
def fig_dashboard(out, day_dir):
    files = sorted(Path(day_dir).glob("*_methods_1s_all.csv"))
    d = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    d["t"] = pd.to_datetime(d["wall_time"])
    s = d.set_index("t")["rf_geo_leg"].sort_index()
    m1 = s.resample("1min").mean()
    m10 = s.resample("10min").mean()
    cum = (s.cumsum() / 60.0)                      # 점유 돼지-분 (1초 샘플)
    base = m10.median()
    low = m10[m10 < 0.5 * base]
    fig = plt.figure(figsize=(6.4, 6.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.55, 1.25, 0.9], hspace=0.75, wspace=0.12)
    # 요약 타일
    tiles = [("관측 시간", f"{(s.index[-1] - s.index[0]).total_seconds() / 3600:.1f}시간"),
             ("평균 점유 마릿수", f"{s.mean():.2f}마리"),
             ("누적 점유량", f"{cum.iloc[-1]:.0f} 돼지·분")]
    for i, (k, v) in enumerate(tiles):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    transform=ax.transAxes, fc="#f5f7fa", ec="#d0d7e0"))
        ax.text(0.5, 0.66, k, transform=ax.transAxes, ha="center", fontsize=8.5, color="#555")
        ax.text(0.5, 0.25, v, transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold", color="#08519c")
    ax = fig.add_subplot(gs[1, :])
    ax.bar(m10.index, m10.values, width=pd.Timedelta(minutes=9), align="edge", color="#c6dbef", label="10분 평균")
    ax.plot(m1.index, m1.values, color="#08519c", lw=1.0, label="1분 평균")
    for t in low.index:
        ax.axvspan(t, t + pd.Timedelta(minutes=10), color="#e31a1c", alpha=0.12, lw=0)
    if len(low):
        ax.plot([], [], color="#e31a1c", alpha=0.3, lw=6, label="알림: 10분 평균 < 중앙값의 50%")
    ax.set_ylabel("점유 마릿수")
    ax.set_title("시간대별 사료통 점유 마릿수", fontsize=10, loc="left")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.legend(fontsize=8, loc="upper center", frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.16))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#ececec")
    ax2 = fig.add_subplot(gs[2, :])
    ax2.plot(cum.index, cum.values, color="#08519c", lw=1.5)
    ax2.set_title("누적 점유량 (돼지·분)", fontsize=10, loc="left")
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(color="#ececec")
    for a in (ax, ax2):
        a.tick_params(axis="x", labelsize=8)
    fig.savefig(out / "fig_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"start": str(s.index[0]), "end": str(s.index[-1]), "mean": float(s.mean()), "cum": float(cum.iloc[-1]),
            "median10": float(base), "low_bins": [str(t) for t in low.index]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="HCI 2027 논문 그림 생성 (자세한 사용법은 파일 맨 위 설명 참고)")
    ap.add_argument("--out", default=r"C:/ETRI/share/hci2027_figs", help="출력 폴더 (농장 사진이 들어가므로 레포 밖 권장)")
    ap.add_argument("--split", default="k5_f3", help="정성 사례(그림 5)의 대표 fold")
    ap.add_argument("--cases", nargs="+", type=int, default=[57, 65, 72, 81], help="그림 3에 빨간 번호로 표시할 사진 번호")
    ap.add_argument("--day-dir", default=r"C:/ETRI/2026_Project_old/counting_outputs/ch010_20260716_day",
                    help="대시보드용 1초 판정 CSV 폴더")
    ap.add_argument("--font-dir", default=None, help="나눔고딕 ttf 폴더 (없으면 HCI_FONT_DIR · 기본 위치 · 시스템 글꼴)")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["examples", "pipeline", "counts", "importance", "cases", "dashboard"],
                    help="일부 그림만 생성. pipeline · counts · importance 는 레포 CSV만으로 가능")
    a = ap.parse_args()
    setup_fonts(a.font_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = set(a.only or ["examples", "pipeline", "counts", "importance", "cases", "dashboard"])
    if "examples" in todo:
        print("data examples:", fig_data_examples(out))
    if "pipeline" in todo:
        fig_pipeline(out)
    if "counts" in todo:
        fig_counts(out, a.cases)
    if "importance" in todo:
        fig_importance(out)
    if "cases" in todo:
        titles = {57: "오차: 가림", 65: "오차: 과대 집계"}
        titles.update({n: "잘된 사례" for n in a.cases if n not in titles})
        fig_cases(out, a.split, a.cases, titles)
        fig_cases_rf(out, a.split, [72, 81, 57, 65], {72: "잘 찾은 사례", 81: "잘 찾은 사례", 57: "가림", 65: "과대 집계"})
        fig_cases_split(out, a.split)
    if "dashboard" in todo:
        print("dashboard:", fig_dashboard(out, a.day_dir))
    print("[saved]", out)


if __name__ == "__main__":
    main()
