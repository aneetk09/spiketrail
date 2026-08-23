import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, fit_transform_train, transform_features
from config import SEED

def main():
    train_df = pd.read_csv("data/train/train.csv")
    
    # 5-Fold CV using sequence_id to prevent sequence shattering
    gkf = GroupKFold(n_splits=5)
    
    oof_probabilities = np.zeros(len(train_df))
    oof_labels = np.zeros(len(train_df), dtype=int)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=train_df["sequence_id"])):
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()
        
        train_features, state = fit_transform_train(train_fold)
        val_features = transform_features(val_fold, state)
        
        x_train = train_features[FEATURE_COLUMNS]
        y_train = train_features["label"].astype(int)
        
        x_val = val_features[FEATURE_COLUMNS]
        y_val = val_features["label"].astype(int)
        
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        
        model = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)
        model.fit(x_train_scaled, y_train)
        
        probabilities = model.predict_proba(x_val_scaled)[:, 1]
        
        oof_probabilities[val_idx] = probabilities
        oof_labels[val_idx] = y_val
        
    precisions, recalls, thresholds = precision_recall_curve(oof_labels, oof_probabilities)
    
    fp_cost = 300
    fn_cost = 5500
    
    P = oof_labels.sum()
    N = len(oof_labels) - P

    min_cost = float('inf')
    best_threshold = 0.5
    best_metrics = {}

    print(f"--- 5-Fold OOF Validation on Train Set ---")
    print(f"Total Positives (Fraud): {P}")
    print(f"Total Negatives (Clean): {N}")
    
    # Write full curve data to a CSV for evidence/inspection
    curve_data = []
    
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        
        y_pred = (oof_probabilities >= threshold).astype(int)
        tp = (oof_labels & y_pred).sum()
        fp = ((~oof_labels.astype(bool)) & y_pred.astype(bool)).sum()
        fn = (oof_labels & (~y_pred.astype(bool))).sum()
        tn = (~oof_labels.astype(bool) & ~y_pred.astype(bool)).sum()
        
        cost = (fp * fp_cost) + (fn * fn_cost)
        
        curve_data.append({
            "threshold": threshold,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "cost": cost
        })
        
        if cost < min_cost:
            min_cost = cost
            best_threshold = threshold
            best_metrics = {
                "threshold": threshold,
                "precision": precisions[i],
                "recall": recalls[i],
                "f1": 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]) if (precisions[i] + recalls[i]) > 0 else 0,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn, "cost": cost
            }
            
    pd.DataFrame(curve_data).to_csv("docs/threshold_cost_curve.csv", index=False)
            
    print("\nOptimal Threshold Based on Validation:")
    print(f"Chosen Threshold: {best_metrics['threshold']:.4f}")
    print(f"Validation Cost: ₹{best_metrics['cost']}")
    print(f"Validation Precision: {best_metrics['precision']:.4f}")
    print(f"Validation Recall: {best_metrics['recall']:.4f}")
    print(f"Validation F1: {best_metrics['f1']:.4f}")
    print(f"Validation TP: {best_metrics['tp']}, FP: {best_metrics['fp']}, TN: {best_metrics['tn']}, FN: {best_metrics['fn']}")
    
    # Finally, apply to TEST SET
    print("\n--- Applying Chosen Threshold to Held-Out Test Set ---")
    test_df = pd.read_csv("data/test/test.csv")
    from src.evaluate_model import load_feature_state
    state = load_feature_state("models/feature_state.joblib")
    scaler = joblib.load("models/standard_scaler.joblib")
    model = joblib.load("models/logistic_regression.joblib")
    
    test_features = transform_features(test_df, state)
    x_test = test_features[FEATURE_COLUMNS]
    y_true = test_features["label"].astype(int)
    
    x_scaled = scaler.transform(x_test)
    test_probabilities = model.predict_proba(x_scaled)[:, 1]
    
    test_pred = (test_probabilities >= best_threshold).astype(int)
    t_tp = (y_true & test_pred).sum()
    t_fp = ((~y_true.astype(bool)) & test_pred.astype(bool)).sum()
    t_fn = (y_true & (~test_pred.astype(bool))).sum()
    t_tn = (~y_true.astype(bool) & ~test_pred.astype(bool)).sum()
    
    test_cost = (t_fp * fp_cost) + (t_fn * fn_cost)
    test_precision = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
    test_recall = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0.0
    
    print(f"Test Cost: ₹{test_cost}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test TP: {t_tp}, FP: {t_fp}, TN: {t_tn}, FN: {t_fn}")

if __name__ == "__main__":
    main()
