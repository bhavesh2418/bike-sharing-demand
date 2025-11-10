# src/features.py

import numpy as np
import pandas as pd
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

def chronological_split(df, features=FEATURES, target=TARGET, split_ratio=0.8):
    """
    Split train → train/validation chronologically
    """
    X = df[features]
    y = df[target].clip(lower=0)  # Clip negative values to 0

    split_index = int(len(df) * split_ratio)

    X_train = X.iloc[:split_index]
    X_valid = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_valid = y.iloc[split_index:]

    return X_train, X_valid, y_train, y_valid


def evaluate_model(model, X_valid, y_valid):
    preds = model.predict(X_valid)
    preds = np.clip(preds, 0, None)  # Clip predictions to 0 for RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y_valid, preds))
    return rmsle


def train_models(X_train, X_valid, y_train, y_valid):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    scores = {}

    for name, model in models.items():
        print(f">> Training {name}...")
        model.fit(X_train, y_train)
        rmsle = evaluate_model(model, X_valid, y_valid)
        print(f"{name} RMSLE: {rmsle:.4f}")
        scores[name] = rmsle

    return scores


if __name__ == "__main__":
    print(">> Loading processed training data...")
    train = pd.read_csv("data/processed/train_features.csv")

    X_train, X_valid, y_train, y_valid = chronological_split(train)

    print(">> Training models...")
    scores = train_models(X_train, X_valid, y_train, y_valid)

    print("✅ All models trained successfully")
