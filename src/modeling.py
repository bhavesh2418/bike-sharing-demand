# src/modeling.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
    y = df[target]

    split_index = int(len(df) * split_ratio)

    X_train = X.iloc[:split_index]
    X_valid = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_valid = y.iloc[split_index:]

    return X_train, X_valid, y_train, y_valid


def baseline_model(X_train, X_valid, y_train, y_valid):
    """
    Train a Linear Regression baseline model
    """
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    preds = lr.predict(X_valid)
    rmsle = np.sqrt(mean_squared_log_error(y_valid, preds))

    print(f"Baseline Linear Regression RMSLE: {rmsle:.4f}")

    return lr, rmsle


if __name__ == "__main__":
    print(">> Loading processed training data...")
    
    # ✅ FIXED PATH
    train = pd.read_csv("data/processed/train_processed.csv")

    X_train, X_valid, y_train, y_valid = chronological_split(train)

    print(">> Training baseline model...")
    model, score = baseline_model(X_train, X_valid, y_train, y_valid)

    print("Done ✅")
