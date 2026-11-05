"""scripts/label_posture_rule_based.py가 만든 규칙 기반 라벨(rule_label)을
"약한 정답"(weak label)으로 삼아 RandomForest 급식 자세 분류기를 학습한다.

이렇게 하면 순수 규칙 기반(관계식이 고정)보다 더 유연하게 여러 특징을
조합해서 판단하게 되고, 나중에 실제 사람이 붙인 "섭식/비섭식" 정답이 생기면
--labels-csv 로 그 데이터를 넣어서 규칙 기반 라벨 대신 진짜 정답으로 다시
학습시키면 된다(같은 파이프라인 재사용).

train/valid/test는 scripts/split_dataset.py가 나눈 원본 분할을 그대로
따른다(같은 CSV의 split 컬럼) — 이미지 단위로 나뉘어 있어서 데이터 누수가
없다.

사용법:
    python scripts/train_posture_classifier.py
    python scripts/train_posture_classifier.py --labels-csv path/to/human_labels.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from utils.paths import MODELS_DIR, REPORTS_DIR
from utils.posture import FEATURE_NAMES

DEFAULT_CSV = REPORTS_DIR / "posture" / "rule_based_dataset.csv"
MODEL_PATH = MODELS_DIR / "posture" / "random_forest.joblib"
REPORT_PATH = REPORTS_DIR / "posture" / "classifier_report.json"


def evaluate(clf, df, split_name):
    X = df[FEATURE_NAMES]
    y = df["label"]
    pred = clf.predict(X)

    report = classification_report(y, pred, target_names=["not_eating", "eating"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred).tolist()

    print(f"\n[{split_name}] n={len(df)}  accuracy={report['accuracy']:.3f}  "
          f"eating precision={report['eating']['precision']:.3f} recall={report['eating']['recall']:.3f} "
          f"f1={report['eating']['f1-score']:.3f}")
    print(f"[{split_name}] confusion matrix [[TN,FP],[FN,TP]] = {cm}")
    return {"report": report, "confusion_matrix": cm}


def main():
    parser = argparse.ArgumentParser(description="RandomForest 급식 자세 분류기 학습")
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_CSV,
                         help="split,image,instance,<특징들>,label 컬럼을 가진 CSV. "
                              "기본값은 규칙 기반 부트스트랩 CSV (rule_label을 label로 사용)")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.labels_csv)
    if "label" not in df.columns:
        if "rule_label" not in df.columns:
            raise SystemExit(f"{args.labels_csv} 에 'label' 또는 'rule_label' 컬럼이 없습니다.")
        df["label"] = df["rule_label"]

    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]
    test_df = df[df["split"] == "test"]
    if len(train_df) == 0:
        raise SystemExit("train split에 해당하는 행이 없습니다.")

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",  # 섭식 사례가 적은 클래스 불균형 보정
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(train_df[FEATURE_NAMES], train_df["label"])

    results = {"source_csv": str(args.labels_csv)}
    results["train"] = evaluate(clf, train_df, "train")
    if len(valid_df):
        results["valid"] = evaluate(clf, valid_df, "valid")
    if len(test_df):
        results["test"] = evaluate(clf, test_df, "test")

    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_.tolist()), key=lambda x: -x[1])
    print("\n[feature importance]")
    for name, imp in importances:
        print(f"  {name:<22} {imp:.3f}")
    results["feature_importance"] = importances

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[posture] 모델 저장: {MODEL_PATH}")
    print(f"[posture] 리포트 저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
