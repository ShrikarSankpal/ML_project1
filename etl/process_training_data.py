import pandas as pd
from sklearn.datasets import fetch_california_housing
import os
from paths import TRAINING_DATA_DIR


def save_processed_data(X, y, output_dir=TRAINING_DATA_DIR):
    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, "features.csv")
    target_path = os.path.join(output_dir, "target.csv")

    pd.DataFrame(X).to_csv(features_path, index=False)
    pd.Series(y).to_csv(target_path, index=False)

    print(f"Saved features to {features_path}")
    print(f"Saved target to {target_path}")

def main():
    data = fetch_california_housing()
    X, y = data.data, data.target
    save_processed_data(X, y)

if __name__ == "__main__":
    main()
