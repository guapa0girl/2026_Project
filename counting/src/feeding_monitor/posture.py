"""옆으로 누운 자세(lateral recumbency) 판별 -- 급이 판정 조건 4의 근거.

문제: 사료통 테두리에 코를 얹은 채 자는 돼지는 조건 1~3(신뢰도/거리/방향)을
모두 통과한다. 코가 링 안에 있고, 목->코 벡터가 사료통 중심을 향하기 때문이다.
실제 geo 영상에서 이 케이스가 카운트되는 것이 확인되었다.

탑뷰에는 높이 정보가 없으므로 z축으로 "서 있음/누움"을 직접 볼 수 없다. 대신
2D 투영에 남는 세 가지 간접 신호를 쓴다:

  A) 사지 가시성 -- 서 있으면 네 다리가 몸통 아래로 들어가 위에서 잘 안 보이고,
     옆으로 누우면 몸통 옆으로 뻗어 나와 뚜렷하게 보인다.
  B) 몸통 종횡비 -- 서 있으면 위에서 본 실루엣이 좁고 길다. 옆으로 누우면 배가
     펴지면서 폭이 넓어진다. 비율이므로 체구에 대해 스케일 불변이다.
  C) 좌우 비대칭 -- 옆으로 누우면 좌/우 대칭 키포인트 중 아래쪽이 몸에 가려져
     confidence가 한쪽으로 쏠린다. 서 있으면 위에서 좌우가 대칭으로 보인다.

설계 원칙 -- 근거가 부족하면 배제하지 않는다(fail-open):
    자세 판별에 쓰는 키포인트(사지/옆구리/어깨/엉덩이)는 코/목보다 가려지기
    쉽다. 밀집 상황에서 근거 부족을 "누움"으로 처리하면 붐빌수록 과소계수하는
    체계적 편향이 생긴다. 점유(occupancy) 측정에서 과소계수는 총 급이 시간을
    통째로 깎아먹으므로, 확신이 없으면 통과시킨다. 이 선택의 대가는 자는 개체
    일부를 놓치는 것인데, 그쪽 오차는 무작위에 가깝다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Optional, Sequence

# 18-keypoint CVAT 스켈레톤 인덱스 (svg data-node-id 오름차순).
# pose_tracker.KEYPOINT_NAMES / config.FeedingRule 과 반드시 동기화할 것.
TAIL = 1
L_HIP, R_HIP = 2, 3
L_FOOT, R_FOOT = 4, 5
R_FLANK, L_FLANK = 6, 7
L_SHOULDER, R_SHOULDER = 8, 9
L_HAND = 10
NECK = 11
R_HAND = 17

LIMB_IDX = (L_FOOT, R_FOOT, L_HAND, R_HAND)
LR_PAIRS = ((L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP), (L_FLANK, R_FLANK))

Keypoints = Sequence[tuple[float, float, float]]  # (x, y, conf) * 18


@dataclass
class PostureScore:
    """조건 4의 계산 결과. `lying`이 True면 섭취에서 배제한다."""

    lying: bool
    score: Optional[float]          # 0~1, 클수록 누운 자세. 근거 부족 시 None
    evidence: float                 # 0~1, 판단 근거의 충분도
    limb_visibility: Optional[float]
    aspect_ratio: Optional[float]
    lateral_asymmetry: Optional[float]
    note: str                       # "ok" | "insufficient_evidence"


def _conf(kpts: Keypoints, i: int) -> float:
    return kpts[i][2] if i < len(kpts) else 0.0


def _dist(kpts: Keypoints, i: int, j: int) -> Optional[float]:
    """두 키포인트가 모두 최소 신뢰도를 넘을 때만 거리를 반환."""
    if i >= len(kpts) or j >= len(kpts):
        return None
    return hypot(kpts[i][0] - kpts[j][0], kpts[i][1] - kpts[j][1])


def _ramp(value: float, low: float, high: float, reverse: bool = False) -> float:
    """low 이하는 0, high 이상은 1로 두고 그 사이를 선형 보간.

    하드컷 대신 완만한 램프를 쓰는 이유: 키포인트 jitter로 특징값이 경계에서
    흔들릴 때 프레임마다 판정이 뒤집히는(flicker) 것을 줄이기 위해서다. 세 신호
    각각이 0/1로 튀지 않고 연속적으로 변하므로, 합산 점수도 부드럽게 움직인다.

    reverse=True면 "낮을수록 누움"인 신호(종횡비, 아래 body_aspect_ratio 참고)에
    쓴다 -- 여전히 low<high로 구간을 지정하고, 결과만 1에서 뺀다.
    """
    if high <= low:
        result = 1.0 if value >= high else 0.0
    else:
        result = max(0.0, min(1.0, (value - low) / (high - low)))
    return 1.0 - result if reverse else result


# --------------------------------------------------------------------------
# 개별 신호
# --------------------------------------------------------------------------

def limb_visibility(kpts: Keypoints, kpt_conf_min: float) -> tuple[Optional[float], float]:
    """사지(앞발2 + 뒷발2)가 얼마나 뚜렷하게 보이는가.

    반환: (평균 confidence, 근거 충분도). 네 점 중 몇 개가 임계 이상인지를
    근거 충분도로 쓴다 -- 네 점이 모두 낮으면 "다리가 안 보인다(서 있음)"인지
    "가려져서 못 찾았다(판단 불가)"인지 구분할 수 없기 때문이다.
    """
    confs = [_conf(kpts, i) for i in LIMB_IDX]
    if not confs:
        return None, 0.0
    n_visible = sum(1 for c in confs if c >= kpt_conf_min)
    return sum(confs) / len(confs), n_visible / len(LIMB_IDX)


def body_aspect_ratio(kpts: Keypoints, kpt_conf_min: float) -> Optional[float]:
    """몸통 폭 / 몸통 길이.

    폭  = l_flank <-> r_flank
    길이 = neck <-> tail
    비율이므로 체구·카메라 높이에 대해 스케일 불변이다. 다만 성장 단계별로
    체형 자체가 다르므로 임계값은 단계별로 강건성을 확인해야 한다.

    애초 가설은 "옆으로 누우면 배가 펴지면서 폭이 넓어진다 -> 클수록 누움"
    이었는데, 야간 영상 34장 사람 라벨로
    검증하니 실제로는 **반대**였다: eating median=0.326, lying median=0.156
    (91% 단일 신호 분리 정확도). 사료통에 코를 박고 몸을 걸쳐 다가서는
    자세에서는 몸통이 카메라를 향해 대각선으로 기울어(foreshorten) neck-tail
    투영 길이가 짧아지는 반면, 사료통 밖 바닥에 완전히 뻗어 누운 자세는 위에서
    볼 때 몸통이 오히려 길게 펴져 보여 비율이 작아지는 것으로 추정된다. 그래서
    `evaluate_posture`는 이 신호를 `_ramp(..., reverse=True)`로 쓴다 -- 낮을수록
    "누움" 쪽 점수가 올라간다.
    """
    for i in (L_FLANK, R_FLANK, NECK, TAIL):
        if _conf(kpts, i) < kpt_conf_min:
            return None
    width = _dist(kpts, L_FLANK, R_FLANK)
    length = _dist(kpts, NECK, TAIL)
    if width is None or length is None or length < 1e-3:
        return None
    return width / length


def lateral_asymmetry(kpts: Keypoints) -> tuple[Optional[float], float]:
    """좌/우 대칭 키포인트 confidence 차이의 평균. 클수록 측와 자세.

    옆으로 누우면 바닥에 닿은 쪽이 몸에 가려져 한쪽 confidence만 떨어진다.
    어깨/엉덩이/옆구리 세 쌍을 쓴다. 귀는 머리 방향에 따라 자연스럽게
    비대칭해지므로 제외했다 -- 급이 중 고개를 옆으로 돌린 개체를 누움으로
    오판할 수 있다.
    """
    diffs = []
    for li, ri in LR_PAIRS:
        lc, rc = _conf(kpts, li), _conf(kpts, ri)
        if max(lc, rc) < 0.1:      # 양쪽 다 없음 -> 이 쌍은 근거가 아님
            continue
        diffs.append(abs(lc - rc))
    if not diffs:
        return None, 0.0
    return sum(diffs) / len(diffs), len(diffs) / len(LR_PAIRS)


# --------------------------------------------------------------------------
# 통합 판정
# --------------------------------------------------------------------------

def evaluate_posture(kpts: Keypoints, rule) -> PostureScore:
    """세 신호를 0~1로 정규화해 가중 합산하고 임계값과 비교한다.

    `rule`은 config.FeedingRule (아래 필드를 읽는다):
        posture_enabled            bool
        keypoint_conf_min          float   -- 조건 1과 공유
        posture_limb_low/high      float   -- 사지 가시성 램프 구간
        posture_aspect_low/high    float   -- 종횡비 램프 구간
        posture_asym_low/high      float   -- 비대칭 램프 구간
        posture_weights            (w_limb, w_aspect, w_asym)
        posture_score_max          float   -- 이 값을 넘으면 누움으로 배제
        posture_min_evidence       float   -- 이 미만이면 판단 보류(통과)
    """
    limb, limb_ev = limb_visibility(kpts, rule.keypoint_conf_min)
    aspect = body_aspect_ratio(kpts, rule.keypoint_conf_min)
    asym, asym_ev = lateral_asymmetry(kpts)

    aspect_ev = 1.0 if aspect is not None else 0.0
    evidence = (limb_ev + aspect_ev + asym_ev) / 3.0

    if evidence < rule.posture_min_evidence:
        # fail-open: 근거 부족은 "누움"이 아니라 "판단 보류"다 (모듈 docstring 참고)
        return PostureScore(
            lying=False, score=None, evidence=evidence,
            limb_visibility=limb, aspect_ratio=aspect, lateral_asymmetry=asym,
            note="insufficient_evidence",
        )

    w_limb, w_aspect, w_asym = rule.posture_weights
    terms, weights = [], []
    if limb is not None:
        terms.append(_ramp(limb, rule.posture_limb_low, rule.posture_limb_high))
        weights.append(w_limb)
    if aspect is not None:
        # reverse=True: 종횡비는 낮을수록 누움 (body_aspect_ratio 참고)
        terms.append(_ramp(aspect, rule.posture_aspect_low, rule.posture_aspect_high, reverse=True))
        weights.append(w_aspect)
    if asym is not None:
        terms.append(_ramp(asym, rule.posture_asym_low, rule.posture_asym_high))
        weights.append(w_asym)

    total_w = sum(weights)
    # 사용 가능한 신호만으로 가중평균 -> 일부 신호가 빠져도 점수 스케일이 유지된다
    score = sum(t * w for t, w in zip(terms, weights)) / total_w if total_w > 0 else 0.0

    return PostureScore(
        lying=score > rule.posture_score_max,
        score=score, evidence=evidence,
        limb_visibility=limb, aspect_ratio=aspect, lateral_asymmetry=asym,
        note="ok",
    )