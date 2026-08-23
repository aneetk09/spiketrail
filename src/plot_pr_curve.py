import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

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
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='.', label='Logistic Regression')
    
    # Mark the chosen threshold
    chosen_threshold = 0.5514
    idx = (np.abs(thresholds - chosen_threshold)).argmin()
    plt.scatter(recalls[idx], precisions[idx], color='red', s=100, zorder=5, label=f'Threshold = {chosen_threshold:.2f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Stage 8 Evaluation)')
    plt.legend()
    plt.grid(True)
    plt.savefig('docs/precision_recall_curve.png')
    print("Saved docs/precision_recall_curve.png")

if __name__ == "__main__":
    main()
