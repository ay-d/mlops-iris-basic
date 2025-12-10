# src/preprocessing.py
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(path: str = "data/iris.csv"):
    """
    1) Load data dari CSV
    2) Pisah fitur & label
    3) Standardisasi fitur
    4) Split train / test
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} tidak ditemukan. Jalankan generate_iris_data.py dulu.")

    df = pd.read_csv(csv_path)

    # asumsi kolom 'target' adalah label
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
