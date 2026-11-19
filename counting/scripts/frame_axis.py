"""test(000) 사진 번호 가로축의 공통 눈금 (그래프 A · E · F · J · 모델별 그림).

사진 번호 = 시간순 1~86 (per_image_000.md의 #, 오차 사례 그림 all_models_<번호>.png와 같은 번호).
눈금 = 0, 5, 10, ..., 85 + 가림 · 밀집 오차 사례 사진 #57 · #76.
  - 57 · 76은 빨간 굵은 글씨로, 바로 옆 55 · 75와 겹치지 않게 한 줄 아래에 씀
    (눈금 글자가 아니라 축 아래 글자로 붙여서, 가로축을 공유하는 칸마다 똑같이 보이게 함)
  - 선 그래프는 57 · 76 위치에 옅은 빨간 점선 (lines=True)
"""
N_IMAGES = 86
CASES = (57, 76)               # error_cases.py 기본값 (가림 · 밀집 오차 사례)
CASE_COLOR = "#c0392b"


def frame_ticks(ax, offset=0.0, first=0, lines=False, fontsize=None):
    """ax의 가로축에 사진 번호 눈금을 둠. 가로축 숫자를 보이게 하려면 호출 전후에 labelbottom=True.
    offset: 데이터 x = 사진 번호 + offset  (선 그래프 x = 1~86 → 0, heatmap 칸 인덱스 0~85 → −1)
    first : 첫 눈금 (선 그래프 0, heatmap은 사진 1부터라 1)"""
    nums = sorted(set([first] + list(range(5, N_IMAGES + 1, 5)) + list(CASES)))
    ax.set_xticks([n + offset for n in nums], ["" if n in CASES else str(n) for n in nums], fontsize=fontsize)
    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties
    fs = fontsize or FontProperties(size=rcParams["xtick.labelsize"]).get_size_in_points()
    ax.xaxis.labelpad = fs * 1.25 + 8                     # 가로축 제목을 빨간 57 · 76 줄 아래로
    for n in CASES:
        ax.annotate(str(n), xy=(n + offset, 0), xycoords=("data", "axes fraction"), xytext=(0, -(fs * 1.25 + 6)),
                    textcoords="offset points", ha="center", va="top", fontsize=fs, color=CASE_COLOR,
                    fontweight="bold", annotation_clip=False)
        if lines:
            ax.axvline(n + offset, color=CASE_COLOR, lw=0.9, ls=":", alpha=0.75, zorder=0.6)
