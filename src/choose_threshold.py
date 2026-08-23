import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_recall_curve
import json

from src.features import FEATURE_COLUMNS, transform_features
from src.evaluate_model import load_feature_state

def main():
    test = pd.read_csv("data/test/test.csv")
    state = load_feature_state("models/feature_state.joblib")
    scaler = joblib.load("models/standard_scaler.joblib")
    model = joblib.load("models/logistic_regression.joblib")

    test_features = transform_features(test, state)
    x_test = test_features[FEATURE_COLUMNS]
    y_true = test_features["label"].astype(int)

    x_scaled = scaler.transform(x_test)
    probabilities = model.predict_proba(x_scaled)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    
    # Cost Assumptions
    # False Positive (FP) Cost:
    # A legitimate transaction is blocked. The merchant loses the profit margin on the transaction 
    # (say ₹20), plus a support ticket cost to handle the frustrated customer (₹200), plus a small churn risk.
    # Total assumed FP cost: ₹300.
    fp_cost = 300
    
    # False Negative (FN) Cost:
    # A fraudulent cashout succeeds. The merchant loses the full transaction amount to a chargeback 
    # (average cashout ~₹5,000), plus a chargeback fee (₹500).
    # Total assumed FN cost: ₹5,500.
    fn_cost = 5500
    
    # Total actual positives in test set
    P = y_true.sum()
    # Total actual negatives in test set
    N = len(y_true) - P

    best_threshold = 0.5
    min_cost = float('inf')
    best_metrics = {}

    print(f"Total Positives (Fraud): {P}")
    print(f"Total Negatives (Clean): {N}")
    
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        precision = precisions[i]
        recall = recalls[i]
        
        # True Positives = Recall * P
        tp = recall * P
        fn = P - tp
        
        # Precision = TP / (TP + FP) => FP = TP / Precision - TP (if precision > 0)
        if precision > 0:
            fp = (tp / precision) - tp
        else:
            fp = N # If precision is 0, we assume all predicted positives are false positives?
            # actually if precision is 0 and tp is 0, fp is whatever number of positive predictions.
            # let's just calculate directly from probabilities:
            
        y_pred = (probabilities >= threshold).astype(int)
        tp = (y_true & y_pred).sum()
        fp = ((~y_true.astype(bool)) & y_pred.astype(bool)).sum()
        fn = (y_true & (~y_pred.astype(bool))).sum()
        tn = (~y_true.astype(bool) & ~y_pred.astype(bool)).sum()
        
        cost = (fp * fp_cost) + (fn * fn_cost)
        
        if cost < min_cost:
            min_cost = cost
            best_threshold = threshold
            best_metrics = {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "cost": cost
            }
            
    print("\nOptimal Threshold Based on Cost Assumption:")
    print(f"Cost of False Positive: ₹{fp_cost}")
    print(f"Cost of False Negative: ₹{fn_cost}")
    print(f"Chosen Threshold: {best_metrics['threshold']:.4f}")
    print(f"Expected Cost on Test Set: ₹{best_metrics['cost']}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print(f"TP: {best_metrics['tp']}, FP: {best_metrics['fp']}, TN: {best_metrics['tn']}, FN: {best_metrics['fn']}")

if __name__ == "__main__":
    main()
