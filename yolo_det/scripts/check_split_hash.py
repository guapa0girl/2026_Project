"""학습 전에 분할 목록이 서인(노션)이 올린 값과 같은지 확인. 학습·데이터 폴더 변경 없음 (분할 함수만 호출).

목록 = prepare_data.py의 split_kfold(시간대) / split_random_kfold(랜덤) 결과, 항목 = "<task>__<원래 이름 stem>"
해시 계산 방식은 정해진 게 없어서 흔한 방식 여러 개를 모두 계산하고, 노션 값과 맞는 방식을 표시함.

사용법 (레포 루트에서):
    python yolo_det/scripts/check_split_hash.py              # 랜덤 rk5 + 시간대 k5 모두
    python yolo_det/scripts/check_split_hash.py --only rk5
"""
import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_SOURCE_DIR  # noqa: E402
from prepare_data import read_task, split_kfold, split_random_kfold  # noqa: E402

# 노션(서인, 11/17)에 올라온 값: fold → (train 장수, valid 장수, train 해시, valid 해시)
EXPECTED = {
    "rk5_f0": (148, 42, "a4ea95beb91e", "f1619ccbdeaa"),
    "rk5_f1": (155, 34, "fcae5d2afec0", "3c0cc9ac12ed"),
    "rk5_f2": (151, 37, "52ffa59391a2", "34101ad2d516"),
    "rk5_f3": (151, 37, "ccfd29eb1fb6", "a82660bf37c3"),
    "rk5_f4": (149, 40, "a157b0286840", "a0c7708a3953"),
    # 시간대(k5) 값도 노션에서 여기에 옮겨 적으면 같이 비교함. 예: "k5_f0": (148, 42, "......", "......"),
}


def variants(names):
    """같은 목록을 여러 방식으로 직렬화 → {방식 이름: 직렬화 bytes}"""
    s = sorted(names)
    v = {
        "줄바꿈 join": "\n".join(s),
        "줄바꿈 join + 끝 줄바꿈": "\n".join(s) + "\n",
        "쉼표 join": ",".join(s),
        "json.dumps": json.dumps(s),
        "json.dumps(ensure_ascii=False)": json.dumps(s, ensure_ascii=False),
        "json.dumps(ensure_ascii=False, indent=1)": json.dumps(s, ensure_ascii=False, indent=1),
        "str(list)": str(s),
    }
    out = {}
    for k, text in v.items():
        out[k] = text.encode("utf-8")
        out[k + " (.jpg 붙임)"] = text.replace("\n", ".jpg\n").encode("utf-8") if "줄바꿈" in k else None
    return {k: b for k, b in out.items() if b is not None}


def hashes(names):
    """{(직렬화, 알고리즘): 앞 12자리}"""
    res = {}
    for vname, b in variants(names).items():
        for algo in ("sha256", "md5", "sha1"):
            res[(vname, algo)] = hashlib.new(algo, b).hexdigest()[:12]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    ap.add_argument("--train-tasks", nargs="+", default=["001", "002"])
    ap.add_argument("--test-task", default="000")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=["rk5", "k5"], default=None)
    a = ap.parse_args()

    items = []
    for task in [*a.train_tasks, a.test_task]:
        items += read_task(Path(a.source_dir) / f"{task}_sorted.zip", task)
    for it in items:
        it.pop("data")

    folds = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for f in range(a.k):
            if a.only != "rk5":
                folds[f"k{a.k}_f{f}"] = split_kfold(items, a.test_task, a.k, f, a.gap)[0]
            if a.only != "k5":
                folds[f"rk{a.k}_f{f}"] = split_random_kfold(items, a.test_task, a.k, f, a.gap, a.seed)[0]

    matched_methods = None          # 모든 비교에서 맞은 (직렬화, 알고리즘)
    rows, all_ok = [], True
    for name in sorted(folds, key=lambda n: (n.startswith("k"), n)):
        sp = folds[name]
        tr = [Path(it["name"]).stem for it in sp["train"]]
        va = [Path(it["name"]).stem for it in sp["valid"]]
        leak = any(it["task"] == a.test_task for it in sp["train"] + sp["valid"]) or set(tr) & set(va)
        h_tr, h_va = hashes(tr), hashes(va)
        exp = EXPECTED.get(name)
        status = "비교값 없음"
        if exp:
            ok_n = (len(tr), len(va)) == exp[:2]
            m = {k for k in h_tr if h_tr[k] == exp[2] and h_va[k] == exp[3]}
            matched_methods = m if matched_methods is None else matched_methods & m
            status = ("✅ 일치" if ok_n and m else "❌ 장수 다름" if not ok_n else "❌ 해시 다름")
            all_ok &= bool(ok_n and m)
        if leak:
            status += " ⚠ 000 섞임/겹침"
            all_ok = False
        rows.append((name, len(tr), len(va), h_tr[("줄바꿈 join", "sha256")], h_va[("줄바꿈 join", "sha256")], status))

    print("| fold | train | valid | train 해시* | valid 해시* | 노션과 비교 |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(map(str, r)) + " |")
    print("* 표의 해시는 sha256('\\n'.join(정렬된 목록)) 앞 12자리 (참고용)")

    if matched_methods:
        print(f"\n해시 방식: {sorted(matched_methods)[0]} 로 노션 값과 일치")
    elif any(n in EXPECTED for n in folds):
        print("\n[주의] ❌ 표시된 fold가 노션 값과 다름 (모든 fold에서 맞는 해시 방식 없음) → 서인에게 해시 계산 코드(무엇을 어떻게 이어 붙였는지) 확인 필요."
              " 장수가 맞으면 분할 자체는 같을 가능성도 있음.")
    print("\n결과:", "모두 일치 → 학습 진행 OK" if all_ok and matched_methods else "확인 필요 (학습 보류)")
    sys.exit(0 if all_ok and matched_methods else 1)


if __name__ == "__main__":
    main()
