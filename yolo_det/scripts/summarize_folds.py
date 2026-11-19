"""K-fold 결과 모으기: reports/<prefix>_f<i>/<size>/summary.json → val 표 + test(000) 표 (fold별 + 평균 ± 표준편차).

팀 통일 규칙: MAE·RMSE 소수 3자리, % 소수 1자리, 표준편차 = 표본 표준편차(n−1).
val 값은 각 fold의 valid에서 conf 임계값을 고른 뒤 같은 valid로 잰 것이라 낙관적임(선택 기준용).

사용법 (레포 루트에서):
    python yolo_det/scripts/summarize_folds.py --prefix ep200_k5 --k 5     # 시간대
    python yolo_det/scripts/summarize_folds.py --prefix ep200_rk5 --k 5    # 랜덤
결과: yolo_det/reports/<prefix>/<size>/folds_summary.md / .json
"""
import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORTS_DIR  # noqa: E402

PCT = "{:.1%}"
# (키, 표 이름, 형식)  — box_* 는 summary.json의 box_map_<split>, 나머지는 count_metrics 결과
VAL_COLS = [("box_P", "P", "{:.3f}"), ("box_R", "R", "{:.3f}"), ("box_mAP50", "mAP50", "{:.3f}"),
            ("box_mAP50-95", "mAP50-95", "{:.3f}"), ("count_MAE", "MAE", "{:.3f}"), ("count_RMSE", "RMSE", "{:.3f}"),
            ("exact_rate", "정확 일치율", PCT)]
TEST_COLS = VAL_COLS + [
    ("count_bias", "bias", "{:+.3f}"), ("within1_rate", "±1 이내", PCT), ("total_error_pct", "총 점유량 오차(%)", "{:+.1f}"),
    ("bin_MAE", "10초 구간 오차", "{:.3f}"), ("missed_visiting_undetected", "못 찾음", "{:.0f}"),
    ("missed_visiting_judged", "other로 판정", "{:.0f}"), ("wrong_visiting", "other→visiting", "{:.0f}"),
    ("extra_visiting", "라벨 없는 박스", "{:.0f}")]


def mean_std(vals):
    return {"mean": st.mean(vals), "std": st.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def table(title, rows, cols, extra_first=None):
    """rows: [{fold, conf, <키>...}] → 마크다운 표 + 평균 ± 표준편차 dict."""
    keys = (["conf"] if extra_first else []) + [c for c, *_ in cols]
    fmts = (["{:.2f}"] if extra_first else []) + [f for *_, f in cols]
    head = ["fold"] + (["conf"] if extra_first else []) + [n for _, n, _ in cols]
    agg = {c: mean_std([r[c] for r in rows if r.get(c) is not None]) for c in keys
           if any(r.get(c) is not None for r in rows)}
    md = [f"## {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        md.append("| " + " | ".join([str(r["fold"])] + [fm.format(r[c]) if r.get(c) is not None else "—"
                                                          for c, fm in zip(keys, fmts)]) + " |")

    def cell(c, fm):
        if c not in agg:
            return "—"
        note = "" if agg[c]["n"] == len(rows) else f" (n={agg[c]['n']})"
        return fm.format(agg[c]["mean"]) + " ± " + fm.format(agg[c]["std"]).lstrip("+") + note
    md.append("| **평균 ± 표준편차** | " + " | ".join(cell(c, fm) for c, fm in zip(keys, fmts)) + " |")
    return md, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="run 이름 앞부분 (예: ep200_k5 → ep200_k5_f0 ...)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--size", default="m")
    a = ap.parse_args()

    val_rows, test_rows, missing, no_valid = [], [], [], []
    for f in range(a.k):
        p = REPORTS_DIR / f"{a.prefix}_f{f}" / a.size / "summary.json"
        if not p.exists():
            missing.append(p)
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        test_rows.append({"fold": f, "conf": s["conf_threshold"], **(s.get("box_map_test") or {}), **s["test"]})
        if "valid" in s:
            val_rows.append({"fold": f, **(s.get("box_map_valid") or {}), **s["valid"]})
        else:
            no_valid.append(f)
    if missing:
        print("[경고] 결과가 없는 fold:", *missing, sep="\n  ")
    if no_valid:
        print(f"[경고] valid 결과가 없는 fold {no_valid} → evaluate.py로 다시 평가 (재학습 필요 없음)")
    if not test_rows:
        raise SystemExit("모을 결과가 없음")

    kind = "랜덤" if re.search(r"_rk\d+$", a.prefix) else "시간대" if re.search(r"_k\d+$", a.prefix) else a.prefix
    md = [f"# {kind} {a.k}-fold — {a.prefix} / {a.size}", "",
          f"- 모은 fold: {len(test_rows)}/{a.k}" + (f" (없음: {len(missing)}개)" if missing else ""),
          "- 평균 ± 표본 표준편차. conf 임계값은 각 fold의 valid에서 사진당 MAE 최소로 고름(000은 보지 않음).", ""]
    md_v, agg_v = table(f"val (fold별 valid, 임계값을 여기서 골라 낙관적)", val_rows, VAL_COLS) if val_rows \
        else ([f"## val", "", "(valid 결과 없음 — evaluate.py로 다시 평가)"], {})
    md_t, agg_t = table("test (000)", test_rows, TEST_COLS, extra_first=True)
    md += md_v + [""] + md_t

    out = REPORTS_DIR / a.prefix / a.size
    out.mkdir(parents=True, exist_ok=True)
    (out / "folds_summary.json").write_text(json.dumps(
        {"kind": kind, "valid": {"folds": val_rows, "mean_std": agg_v}, "test": {"folds": test_rows, "mean_std": agg_t}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "folds_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()