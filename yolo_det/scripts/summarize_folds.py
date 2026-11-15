"""K-fold 결과 모으기: reports/<prefix>_f<i>/<size>/summary.json → fold별 표 + 평균 ± 표준편차.

사용법 (레포 루트에서):
    python yolo_det/scripts/summarize_folds.py --prefix ep200_k5 --k 5
결과: yolo_det/reports/<prefix>/<size>/folds_summary.md / .json
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORTS_DIR  # noqa: E402

COLS = [("count_MAE", "사진당 MAE", "{:.3f}"), ("count_bias", "bias", "{:+.3f}"), ("exact_rate", "정확 일치", "{:.1%}"),
        ("within1_rate", "±1 이내", "{:.1%}"), ("total_error_pct", "총 점유량 오차(%)", "{:+.1f}"),
        ("bin_MAE", "구간 평균 오차", "{:.3f}"), ("missed_visiting_undetected", "못 찾음", "{:.0f}"),
        ("missed_visiting_judged", "other로 판정", "{:.0f}"), ("wrong_visiting", "other→visiting", "{:.0f}"),
        ("extra_visiting", "라벨 없는 박스", "{:.0f}")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="run 이름 앞부분 (예: ep200_k5 → ep200_k5_f0 ...)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--size", default="m")
    a = ap.parse_args()

    rows, missing = [], []
    for f in range(a.k):
        p = REPORTS_DIR / f"{a.prefix}_f{f}" / a.size / "summary.json"
        if not p.exists():
            missing.append(p)
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"fold": f, "split": s.get("split"), "conf": s["conf_threshold"], **s["test"],
                     "box_mAP50-95": (s.get("box_map_test") or {}).get("box_mAP50-95")})
    if missing:
        print("[경고] 결과가 없는 fold:", *missing, sep="\n  ")
    if not rows:
        raise SystemExit("모을 결과가 없음")

    keys = ["conf"] + [c for c, *_ in COLS] + ["box_mAP50-95"]
    agg = {}
    for c in keys:
        vals = [r[c] for r in rows if r.get(c) is not None]
        if vals:
            agg[c] = {"mean": st.mean(vals), "std": st.stdev(vals) if len(vals) > 1 else 0.0}

    head = ["fold", "conf"] + [n for _, n, _ in COLS] + ["box mAP50-95"]
    fmts = ["{:.2f}"] + [f for *_, f in COLS] + ["{:.3f}"]
    md = [f"# 시간대 {a.k}-fold — {a.prefix} / {a.size} (test = 000, 각 fold의 valid로 임계값 선택)", "",
          f"- 모은 fold: {len(rows)}/{a.k}" + (f" (없음: {len(missing)}개)" if missing else ""), "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        md.append("| " + " | ".join([str(r["fold"])] + [fm.format(r[c]) if r.get(c) is not None else "—"
                                                          for c, fm in zip(keys, fmts)]) + " |")
    md.append("| **평균 ± 표준편차** | " + " | ".join(
        (fm.format(agg[c]["mean"]) + " ± " + fm.format(agg[c]["std"]).lstrip("+")) if c in agg else "—"
        for c, fm in zip(keys, fmts)) + " |")

    out = REPORTS_DIR / a.prefix / a.size
    out.mkdir(parents=True, exist_ok=True)
    (out / "folds_summary.json").write_text(json.dumps({"folds": rows, "mean_std": agg}, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    (out / "folds_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
