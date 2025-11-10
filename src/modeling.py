# src/modeling.py

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error

FEATURES = [
    'season', 'holiday', 'workingday', 'weather',
    'temp', 'atemp', 'humidity', 'windspeed',
    'hour', 'day', 'month', 'year', 'dayofweek'
]

TARGET = 'count'
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def chronological_split(df, features=FEATURES, target=TARGET, split_ratio=0.8):
    X = df[features]
    y = df[target]

    split_index = int(len(df) * split_ratio)

    return (
        X.iloc[:split_index],
        X.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:]
    )


def evaluate_model(model, X_valid, y_valid):
    preds = model.predict(X_valid)
    preds = np.maximum(preds, 0)  # avoid negatives
    rmsle = np.sqrt(mean_squared_log_error(y_valid, preds))
    return rmsle


def train_models(X_train, X_valid, y_train, y_valid):

    models = {
        "linear_regression_model": LinearRegression(),
        "random_forest_model": RandomForestRegressor(n_estimators=200, random_state=42),
        "gradient_boosting_model": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "xgboost_model": XGBRegressor(n_estimators=300, random_state=42, eval_metric='rmse')
    }

    results = {}

    print(">> Training models...")

    for name, model in models.items():
        print(f">> Training {name}...")
        model.fit(X_train, y_train)

        rmsle = evaluate_model(model, X_valid, y_valid)
        print(f"{name} RMSLE: {rmsle:.4f}")

        file_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        joblib.dump(model, file_path)

        results[name] = {
            "model": model,
            "rmsle": rmsle
        }

    print("✅ All models trained and saved")
    return results


if __name__ == "__main__":

    print(">> Loading processed training data...")
    train = pd.read_csv("data/processed/train_processed.csv")

    X_train, X_valid, y_train, y_valid = chronological_split(train)

    trained = train_models(X_train, X_valid, y_train, y_valid)

    # Predict on test (optional)
    test_file = "data/processed/test_processed.csv"
    if os.path.exists(test_file):
        print(">> Loading test data...")
        test = pd.read_csv(test_file)

        X_test = test[FEATURES]

        best_model = trained["xgboost_model"]["model"]

        preds = best_model.predict(X_test)
        preds = np.maximum(preds, 0)

        submission = pd.DataFrame({
            "datetime": test["datetime"],
            "count": preds
        })

        submission.to_csv("submission.csv", index=False)
        print("✅ Test predictions saved to submission.csv")

    print("Done ✅")
