"""HCI 2027 원고(docx)의 표 1~9를 300 dpi PNG로 그림 — PAPER.md 0장 · 발표 자료용.

원고 docx에 들어 있는 표를 그대로 읽어서(병합 셀 · 굵은 글씨 · 캡션 포함) 논문식 표(가로줄만)로 다시 그림.
표 3의 수식 셀(Word 수식)은 글자로 읽을 수 없으므로 아래 MATH_CELLS 의 matplotlib 수식으로 바꿔 그림.

출력 (--out, 기본 counting/reports/fold_study/paper_figures/)
  tab1_data.png  tab2_methods.png  tab3_features.png  tab4_pck.png  tab5_mae.png
  tab6_count_metrics.png  tab7_error_decomp.png  tab8_leg_effect.png  tab9_inference_time.png
  (표에는 농장 사진이 없으므로 레포 안에 바로 저장해도 됨. 원고 docx 자체는 레포에 올리지 않음)

==================================================================================================
다른 PC에서 표를 다시 출력하는 방법
==================================================================================================
1) 준비
   - 파이썬 패키지: matplotlib 만 있으면 됨 (docx는 표준 라이브러리 zipfile · xml로 직접 읽음, python-docx 불필요).
   - 원고 docx 파일 (GitHub에 없음. 공유 드라이브 · 메일 등에서 받아 아무 위치에 둠).
   - 한글 글꼴: 나눔고딕 ttf 폴더를 --font-dir 또는 환경 변수 HCI_FONT_DIR 로 지정
     (없으면 C:/ETRI/share/hci2027_overleaf/fonts → 맑은 고딕 / AppleGothic 순으로 찾음).
2) 실행 (counting/ 에서)
     python scripts/docx_tables_png.py "<원고 docx 경로>"
     python scripts/docx_tables_png.py "<원고 docx 경로>" --out <출력 폴더> --font-dir <나눔고딕 폴더>
3) 원고의 표 수정 후에는 이 명령만 다시 돌리면 PNG가 갱신됨.
   - 표 번호는 표 바로 위 문단의 "표 N." 캡션으로 찾음. 캡션이 없는 표(저자 표 등)는 건너뜀.
   - 표 3의 특징 이름이 바뀌면 MATH_CELLS 의 키(특징 이름)도 같이 바꿔야 수식이 들어감.
   - 새 표가 생기면 NAMES 에 "번호: 파일 이름"을 추가 (없으면 tab<N>.png 로 저장).
==================================================================================================
"""
import argparse
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
NAMES = {1: "data", 2: "methods", 3: "features", 4: "pck", 5: "mae", 6: "count_metrics",
         7: "error_decomp", 8: "leg_effect", 9: "inference_time"}
# 표 3: Word 수식 셀 → matplotlib 수식 (키 = 특징 이름)
MATH_CELLS = {
    "코의 구역 포함 여부": r"$\mathbf{1}[p_{nose} \in Z]$",
    "코-구역 거리": r"$d(p_{nose},\ Z)$",
    "주요 관절 검출 비율": r"$\frac{1}{|K|}\sum_{k \in K}\mathbf{1}[c_k > \tau]$  ($\tau$ = 0)",
    "코-사료통 거리비": r"$\|p_{nose} - c\|\ /\ r$",
    "머리 방향 각도": r"$\theta$  (식 2)",
    "다리 가시성 (누움 신호)": r"$\frac{1}{4}\sum_{k \in F} c_k$",
    "몸통 종횡비 (누움 신호)": r"$\|p_{lflank} - p_{rflank}\|\ /\ \|p_{neck} - p_{tail}\|$",
    "좌우 비대칭 (누움 신호)": r"$\frac{1}{|S|}\sum_{(l,r) \in S}|c_l - c_r|$",
}
CAPTION_FIX = {3: ("(: 관절 k의 신뢰도", "($c_k$: 관절 k의 신뢰도")}     # 캡션 속 Word 수식 보충
FS = 8.5                       # 본문 글자 크기(pt)
PAD_X, PAD_Y = 0.10, 0.05      # 셀 여백(inch)


def setup_fonts(font_dir=None):
    for d in (font_dir, os.environ.get("HCI_FONT_DIR"), r"C:/ETRI/share/hci2027_overleaf/fonts"):
        if d and Path(d).is_dir():
            for f in Path(d).glob("NanumGothic*.ttf"):
                font_manager.fontManager.addfont(str(f))
    have = {f.name for f in font_manager.fontManager.ttflist}
    fams = [f for f in ("NanumGothic", "Malgun Gothic", "AppleGothic") if f in have] + ["DejaVu Sans"]
    plt.rcParams.update({"font.family": fams, "axes.unicode_minus": False, "mathtext.fontset": "dejavusans"})


def text_of(el, with_math=True):
    out = []
    for e in el.iter():
        if e.tag == W + "t" or (with_math and e.tag == M + "t"):
            out.append(e.text or "")
    return "".join(out)


def read_tables(docx):
    """(번호, 캡션, 표 아래 주석, 셀 격자) 목록. 셀 = dict(row, col, span, vspan, text, bold, math)."""
    root = ET.fromstring(zipfile.ZipFile(docx).read("word/document.xml"))
    body = root.find(W + "body")
    items, caption = [], None
    kids = list(body)
    for i, el in enumerate(kids):
        if el.tag == W + "p":
            t = text_of(el, with_math=False).strip()
            if re.match(r"^표 \d+\.", t):
                caption = t
            continue
        if el.tag != W + "tbl" or caption is None:
            continue
        no = int(re.match(r"^표 (\d+)\.", caption).group(1))
        note = ""
        nxt = kids[i + 1] if i + 1 < len(kids) else None
        if nxt is not None and nxt.tag == W + "p":
            t = text_of(nxt, with_math=False).strip()
            if t.startswith(("RF(", "*")) and "p" in t:      # 표 5 아래의 유의성 주석
                note = t
        grid, open_v = [], {}
        for r, tr in enumerate(el.findall(W + "tr")):
            c = 0
            for tc in tr.findall(W + "tc"):
                pr = tc.find(W + "tcPr")
                span = 1
                vm = None
                if pr is not None:
                    gs = pr.find(W + "gridSpan")
                    span = int(gs.get(W + "val")) if gs is not None else 1
                    v = pr.find(W + "vMerge")
                    if v is not None:
                        vm = v.get(W + "val") or "continue"
                if vm == "continue":
                    if c in open_v:
                        open_v[c]["vspan"] += 1
                    c += span
                    continue
                txt = " ".join(text_of(p, with_math=False).strip() for p in tc.findall(W + "p")).strip()
                bold = any(b.get(W + "val", "1") not in ("0", "false") for b in tc.iter(W + "b"))
                cell = dict(row=r, col=c, span=span, vspan=1, text=txt, bold=bold,
                            math=any(True for _ in tc.iter(M + "oMath")))
                grid.append(cell)
                if vm == "restart":
                    open_v[c] = cell
                else:
                    open_v.pop(c, None)
                c += span
        cap = caption
        if no in CAPTION_FIX:
            cap = cap.replace(*CAPTION_FIX[no])
        items.append((no, cap, note, grid))
        caption = None
    return items


def render(no, caption, note, grid, out):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    rend = fig.canvas.get_renderer()

    def size(s, bold=False, fs=FS):
        if not s:
            return 0.0, 0.0
        t = fig.text(0, 0, s, fontsize=fs, fontweight="bold" if bold else "normal")
        bb = t.get_window_extent(rend)
        t.remove()
        return bb.width / fig.dpi, bb.height / fig.dpi

    if no == 3:                                    # 수식 셀 채우기
        feats = {c["row"]: c["text"] for c in grid if c["col"] == 1}
        for c in grid:
            if c["col"] == 2 and feats.get(c["row"]) in MATH_CELLS and (c["math"] or "(식 2)" in c["text"] or "τ" in c["text"]):
                c["text"] = MATH_CELLS[feats[c["row"]]]
    ncol = max(c["col"] + c["span"] for c in grid)
    nrow = max(c["row"] + c["vspan"] for c in grid)
    # 머리글 줄 수: 첫 줄에 여러 칸 셀이나 세로 병합 셀이 있으면 2줄(묶음 머리글), 아니면 1줄
    first_data = 2 if any(c["span"] > 1 or c["vspan"] > 1 for c in grid if c["row"] == 0) else 1
    colw = [0.0] * ncol
    rowh = [0.0] * nrow
    sz = {}
    for c in grid:
        w, h = size(c["text"], c["bold"] or c["row"] < first_data)
        sz[id(c)] = (w, h)
        if c["span"] == 1:
            colw[c["col"]] = max(colw[c["col"]], w + 2 * PAD_X)
        if c["vspan"] == 1:
            rowh[c["row"]] = max(rowh[c["row"]], h + 2 * PAD_Y)
    line_h = size("가g")[1] + 2 * PAD_Y
    rowh = [max(h, line_h) for h in rowh]
    for c in grid:                                 # 여러 칸 셀 폭 맞추기
        if c["span"] > 1:
            need = sz[id(c)][0] + 2 * PAD_X
            have = sum(colw[c["col"]:c["col"] + c["span"]])
            if need > have and c["row"] < first_data:
                for k in range(c["col"], c["col"] + c["span"]):
                    colw[k] += (need - have) / c["span"]
    tw, th = sum(colw), sum(rowh)
    cap_w = max(tw, 3.2)
    plt.close(fig)

    # 캡션 줄바꿈 (표 폭에 맞춤)
    fig = plt.figure(figsize=(10, 10), dpi=300)
    rend = fig.canvas.get_renderer()
    words, lines, cur = caption.split(" "), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        t = fig.text(0, 0, trial, fontsize=FS)
        if t.get_window_extent(rend).width / fig.dpi > cap_w and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
        t.remove()
    lines.append(cur)
    plt.close(fig)
    cap_h = len(lines) * line_h * 0.95 + 0.08
    note_lines = []
    if note:
        fig = plt.figure(figsize=(10, 10), dpi=300)
        rend = fig.canvas.get_renderer()
        cur = ""
        for wd in note.split(" "):
            trial = (cur + " " + wd).strip()
            t = fig.text(0, 0, trial, fontsize=FS - 1)
            if t.get_window_extent(rend).width / fig.dpi > cap_w and cur:
                note_lines.append(cur)
                cur = wd
            else:
                cur = trial
            t.remove()
        note_lines.append(cur)
        plt.close(fig)
    note_h = len(note_lines) * line_h * 0.9 + (0.06 if note_lines else 0)
    W_in, H_in = cap_w + 0.1, cap_h + th + note_h + 0.1
    fig = plt.figure(figsize=(W_in, H_in), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_in)
    ax.set_ylim(H_in, 0)
    ax.axis("off")
    x0 = (W_in - tw) / 2
    y = 0.05
    for ln in lines:
        ax.text(W_in / 2, y + line_h * 0.45, ln, ha="center", va="center", fontsize=FS)
        y += line_h * 0.95
    y0 = y + 0.08
    xs = [x0]
    for w in colw:
        xs.append(xs[-1] + w)
    ys = [y0]
    for h in rowh:
        ys.append(ys[-1] + h)
    for c in grid:
        xa, xb = xs[c["col"]], xs[c["col"] + c["span"]]
        ya, yb = ys[c["row"]], ys[c["row"] + c["vspan"]]
        head = c["row"] < first_data
        left = (c["col"] == 0 or len(c["text"]) > 14) and not (head and c["span"] > 1) and c["span"] < ncol
        if c["span"] == ncol:                       # 표 전체 폭 주석 행 (표 8 마지막 줄)
            left = True
        ax.text(xa + PAD_X if left else (xa + xb) / 2, (ya + yb) / 2, c["text"], ha="left" if left else "center",
                va="center", fontsize=FS, fontweight="bold" if (c["bold"] or head) else "normal")
        if head and c["span"] > 1 and c["row"] + 1 < first_data:      # 묶음 머리글 아래 짧은 줄
            ax.plot([xa + 0.05, xb - 0.05], [yb, yb], color="black", lw=0.5)
    ax.plot([x0, xs[-1]], [ys[0], ys[0]], color="black", lw=1.2)
    ax.plot([x0, xs[-1]], [ys[first_data], ys[first_data]], color="black", lw=0.7)
    ax.plot([x0, xs[-1]], [ys[-1], ys[-1]], color="black", lw=1.2)
    if no == 3:                                     # 특징 그룹 사이 가는 줄
        for c in grid:
            if c["col"] == 0 and c["row"] > first_data:
                ax.plot([x0, xs[-1]], [ys[c["row"]], ys[c["row"]]], color="#999", lw=0.4)
    if no == 8:                                     # 마지막 주석 행 위 가는 줄
        ax.plot([x0, xs[-1]], [ys[-2], ys[-2]], color="#999", lw=0.4)
    yy = ys[-1] + 0.06
    for ln in note_lines:
        ax.text(x0, yy + line_h * 0.4, ln, ha="left", va="center", fontsize=FS - 1)
        yy += line_h * 0.9
    name = f"tab{no}_{NAMES.get(no, '')}".rstrip("_") + ".png"
    fig.savefig(out / name, dpi=300, facecolor="white")
    plt.close(fig)
    return name


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="원고 docx의 표를 300 dpi PNG로 (사용법은 파일 맨 위 설명 참고)")
    ap.add_argument("docx", help="원고 docx 경로 (레포 밖)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "fold_study" / "paper_figures"))
    ap.add_argument("--font-dir", default=None)
    a = ap.parse_args()
    setup_fonts(a.font_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for no, cap, note, grid in read_tables(a.docx):
        print(f"표 {no}: {render(no, cap, note, grid, out)}")


if __name__ == "__main__":
    main()
