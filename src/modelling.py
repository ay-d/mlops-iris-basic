# src/modelling.py
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.preprocessing import load_and_preprocess


def main():
    # Nama experiment di MLflow
    mlflow.set_experiment("iris-basic-mlops")

    # Aktifkan autolog untuk scikit-learn
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Test accuracy:", acc)

    # contoh metric manual tambahan
    mlflow.log_metric("test_accuracy_manual", acc)


if __name__ == "__main__":
    main()
