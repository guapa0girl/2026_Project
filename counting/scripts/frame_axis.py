"""test(000) 사진 번호 가로축의 공통 눈금 (그래프 A · E · F · J · 모델별 그림).

사진 번호 = 시간순 1~86 (per_image_000.md의 #, 오차 사례 그림 all_models_<번호>.png와 같은 번호).
눈금 = 0, 5, 10, ..., 85 + 가림 · 밀집 오차 사례 사진 #57 · #76.
  - 57 · 76은 빨간 굵은 글씨, 바로 옆 55 · 75와 겹치지 않게 조금 아래에 씀
  - 선 그래프는 57 · 76 위치에 옅은 빨간 점선 (lines=True)
"""
from matplotlib.transforms import ScaledTranslation

N_IMAGES = 86
CASES = (57, 76)               # error_cases.py 기본값 (가림 · 밀집 오차 사례)
CASE_COLOR = "#c0392b"


def frame_ticks(ax, offset=0.0, first=0, lines=False, fontsize=None):
    """ax의 가로축에 사진 번호 눈금을 둠.
    offset: 데이터 x = 사진 번호 + offset  (선 그래프 x = 1~86 → 0, heatmap 칸 인덱스 0~85 → −1)
    first : 첫 눈금 (선 그래프 0, heatmap은 사진 1부터라 1)"""
    nums = sorted(set([first] + list(range(5, N_IMAGES + 1, 5)) + list(CASES)))
    ax.set_xticks([n + offset for n in nums], [str(n) for n in nums], fontsize=fontsize)
    shift = ScaledTranslation(0, -8 / 72, ax.figure.dpi_scale_trans)
    for lab, n in zip(ax.get_xticklabels(), nums):
        if n in CASES:
            lab.set_color(CASE_COLOR)
            lab.set_fontweight("bold")
            lab.set_transform(lab.get_transform() + shift)
    for t, n in zip(ax.xaxis.get_major_ticks(), nums):
        if n in CASES:
            t.tick1line.set_markersize(t.tick1line.get_markersize() + 6)
            t.tick1line.set_color(CASE_COLOR)
    if lines:
        for n in CASES:
            ax.axvline(n + offset, color=CASE_COLOR, lw=0.9, ls=":", alpha=0.75, zorder=0.6)
