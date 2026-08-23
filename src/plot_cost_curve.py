import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("docs/threshold_cost_curve.csv")
    
    plt.figure(figsize=(8, 6))
    plt.plot(df["threshold"], df["cost"], marker='', label='Total Cost (₹)')
    
    # Mark the chosen threshold
    min_idx = df["cost"].idxmin()
    best_threshold = df.loc[min_idx, "threshold"]
    best_cost = df.loc[min_idx, "cost"]
    
    plt.scatter([best_threshold], [best_cost], color='red', s=100, zorder=5, 
                label=f'Min Cost (₹{best_cost:,.0f}) at T={best_threshold:.4f}')
    
    plt.xlabel('Classification Threshold')
    plt.ylabel('Total Expected Cost (₹)')
    plt.title('Cost Curve (OOF Validation on Train Set)')
    plt.legend()
    plt.grid(True)
    plt.savefig('docs/threshold_cost_curve.png')
    print("Saved docs/threshold_cost_curve.png")

if __name__ == "__main__":
    main()
