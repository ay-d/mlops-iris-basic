# generate_iris_data.py
import os
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

def main():
    data = load_iris(as_frame=True)
    df = data.frame  # berisi fitur + kolom target

    os.makedirs("data", exist_ok=True)
    out_path = Path("data/iris.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved iris dataset to {out_path} with shape {df.shape}")

if __name__ == "__main__":
    main()
