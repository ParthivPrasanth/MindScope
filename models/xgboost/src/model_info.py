import os
import pandas as pd
from src.config import FEATURE_WEIGHTS_PATH, DETAILED_IMPORTANCE_PATH

def show_model_info():
    weights = pd.read_csv(FEATURE_WEIGHTS_PATH)
    detailed = pd.read_csv(DETAILED_IMPORTANCE_PATH)

    print("\n=== Feature Weights ===")
    print(weights.head(20))

    print("\n=== Detailed Importance ===")
    print(detailed.head(20))

if __name__ == "__main__":
    show_model_info()