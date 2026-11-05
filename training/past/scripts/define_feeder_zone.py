"""영상의 한 프레임 위에서 마우스로 급식 구역(다각형)을 직접 그려서
configs/feeder_zones.yaml 을 (다시) 만든다. configs/feeder_zones.yaml에는
test_video.avi 기준 추정치가 기본으로 들어있는데, 카메라 앵글이 다르거나
더 정확한 경계가 필요하면 이 스크립트로 다시 잡으면 된다.

화면이 있는 환경(로컬 데스크톱)에서 직접 실행해야 한다 — 원격/헤드리스
환경에서는 cv2.imshow 창이 뜨지 않는다.

조작:
  왼쪽 클릭   : 지금 그리는 구역에 점 추가
  n           : 지금 구역을 마무리하고 새 구역 시작 (급식기가 여러 개일 때)
  z           : 마지막 점 취소 (undo)
  s           : 저장하고 종료
  q / ESC     : 저장하지 않고 종료

사용법:
    python scripts/define_feeder_zone.py --source ../test_video.avi
    python scripts/define_feeder_zone.py --source ../test_video.avi --frame 500
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from utils.paths import CONFIGS_DIR

ZONES_YAML = CONFIGS_DIR / "feeder_zones.yaml"
WIN = "feeder zone | click: add point, n: new zone, z: undo, s: save, q: cancel"


def main():
    parser = argparse.ArgumentParser(description="마우스로 급식 구역 다각형을 정의")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0, help="정의에 쓸 프레임 번호")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"프레임을 읽을 수 없습니다: {args.source} (frame={args.frame})")

    zones = [[]]

    def redraw():
        disp = frame.copy()
        for zi, zone in enumerate(zones):
            for pt in zone:
                cv2.circle(disp, pt, 4, (0, 0, 255), -1)
            if len(zone) > 1:
                closed = zi < len(zones) - 1
                cv2.polylines(disp, [np.array(zone, dtype=np.int32)], closed, (0, 200, 255), 2)
        cv2.imshow(WIN, disp)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            zones[-1].append((x, y))
            redraw()

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_click)
    redraw()

    cancelled = False
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("n"):
            if zones[-1]:
                zones.append([])
        elif key == ord("z") and zones[-1]:
            zones[-1].pop()
            redraw()
        elif key == ord("s"):
            break
        elif key in (ord("q"), 27):
            cancelled = True
            break

    cv2.destroyAllWindows()

    if cancelled:
        print("[zone] 취소했습니다 (저장 안 함).")
        return

    zones = [z for z in zones if len(z) >= 3]
    if not zones:
        raise SystemExit("점 3개 이상인 구역이 없습니다. 다시 시도하세요.")

    data = {"zones": [[[int(x), int(y)] for x, y in zone] for zone in zones]}
    ZONES_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(ZONES_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"[zone] {len(zones)}개 구역을 {ZONES_YAML} 에 저장했습니다.")


if __name__ == "__main__":
    main()
