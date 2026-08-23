import argparse
import os
import json
import time
import pandas as pd
import joblib
from pathlib import Path
from anthropic import Anthropic

from src.features import FEATURE_COLUMNS, transform_features
from src.evaluate_model import load_feature_state

def get_explanation(client, row, coefficients):
    prompt = f"""You are explaining a fraud decision for a payment transaction to a non-technical risk operator.
The model flagged this transaction because of the following feature values:

"""
    # Only include features that contributed to the fraud score (based on coeff * value > 0)
    for col in FEATURE_COLUMNS:
        val = row[col]
        coef = coefficients[col]
        contribution = val * coef
        prompt += f"- {col}: {val:.3f} (Model weight: {coef:.3f})\n"

    prompt += """
Based ONLY on the feature values above, write a 1-2 sentence plain-language explanation for why this transaction was blocked. 
Do not use generic fraud language, do not mention "model weight" or "coefficients" in the output, and do not reference features the model wasn't given. Keep it direct and factual.
"""
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY environment variable not set. Please set it to run the explanation layer.")
        return
        
    client = Anthropic(api_key=api_key)
    
    test_df = pd.read_csv("data/test/test.csv")
    state = load_feature_state("models/feature_state.joblib")
    scaler = joblib.load("models/standard_scaler.joblib")
    model = joblib.load("models/logistic_regression.joblib")
    
    # Extract coefficients
    coefficients = pd.Series(model.coef_[0], index=FEATURE_COLUMNS)
    
    test_features = transform_features(test_df, state)
    x_test = test_features[FEATURE_COLUMNS]
    
    x_scaled = scaler.transform(x_test)
    probabilities = model.predict_proba(x_scaled)[:, 1]
    
    threshold = 0.4335
    flagged_indices = (probabilities >= threshold).nonzero()[0]
    
    print(f"Found {len(flagged_indices)} flagged transactions. Generating explanations...")
    
    explanations = []
    
    # Process all flagged transactions. We add a small sleep to avoid rate limits.
    for i, idx in enumerate(flagged_indices):
        row = x_test.iloc[idx]
        tx_id = test_features.iloc[idx]["transaction_id"]
        
        explanation = get_explanation(client, row, coefficients)
        
        explanations.append({
            "transaction_id": tx_id,
            "probability": probabilities[idx],
            "explanation": explanation
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(flagged_indices)}...")
        
        time.sleep(0.5) # Rate limit safety
        
    out_path = "docs/explanations.json"
    with open(out_path, "w") as f:
        json.dump(explanations, f, indent=2)
        
    print(f"Saved explanations to {out_path}")

if __name__ == "__main__":
    main()
