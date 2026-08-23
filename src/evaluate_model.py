from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from src.features import FEATURE_COLUMNS, FeatureState, transform_features


TEST_PATH = Path("data/test/test.csv")
FEATURE_STATE_PATH = Path("models/feature_state.joblib")
MODEL_PATH = Path("models/logistic_regression.joblib")
SCALER_PATH = Path("models/standard_scaler.joblib")
RESULTS_PATH = Path("docs/evaluation_results.json")
DEFAULT_THRESHOLD = 0.5


def load_feature_state(path: Path) -> FeatureState:
    raw_state = joblib.load(path)
    if isinstance(raw_state, FeatureState):
        return raw_state
    return FeatureState(**raw_state)


def evaluate(
    test_path: Path = TEST_PATH,
    feature_state_path: Path = FEATURE_STATE_PATH,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    test = pd.read_csv(test_path)
    state = load_feature_state(feature_state_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    test_features = transform_features(test, state)
    x_test = test_features[FEATURE_COLUMNS]
    y_true = test_features["label"].astype(int)

    x_scaled = scaler.transform(x_test)
    probabilities = model.predict_proba(x_scaled)[:, 1]
    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)

    return {
        "threshold": threshold,
        "rows": int(len(test_features)),
        "fraud_rows": int(y_true.sum()),
        "clean_rows": int((y_true == 0).sum()),
        "metrics": {
            "roc_auc": round(float(roc_auc), 6),
            "pr_auc": round(float(pr_auc), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
        },
        "confusion_matrix": {
            "true_positive": int(tp),
            "false_positive": int(fp),
            "true_negative": int(tn),
            "false_negative": int(fn),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SpikeTrail on held-out test data.")
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--feature-state", type=Path, default=FEATURE_STATE_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--scaler", type=Path, default=SCALER_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(
        test_path=args.test,
        feature_state_path=args.feature_state,
        model_path=args.model,
        scaler_path=args.scaler,
        threshold=args.threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Rows: {results['rows']}")
    print(f"Fraud rows: {results['fraud_rows']}")
    print(f"Clean rows: {results['clean_rows']}")
    print(f"Threshold: {results['threshold']}")
    print("Confusion matrix:")
    for key, value in results["confusion_matrix"].items():
        print(f"{key}: {value}")
    print("Metrics:")
    for key, value in results["metrics"].items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
