"""
Feature Engineering for Bike Sharing Demand
-------------------------------------------
Transform raw / processed data into ML-ready features.
"""

import pandas as pd
import numpy as np


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract datetime features from 'datetime' column."""
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add high-level seasonal categories."""
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_workinghour"] = df["hour"].between(8, 18).astype(int)
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean weather & environmental variables."""
    df["temp_feels_diff"] = df["atemp"] - df["temp"]
    df["humidity_temp_ratio"] = df["humidity"] / (df["temp"] + 1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main function combining all feature transformations."""
    df = add_datetime_features(df)
    df = add_seasonal_features(df)
    df = add_weather_features(df)

    # Drop unused columns
    drop_cols = ["datetime"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    return df


# Example usage
if __name__ == "__main__":
    print(">> Loading processed training data...")
    df = pd.read_csv("data/processed/train_processed.csv")

    print(">> Applying feature engineering...")
    df = build_features(df)

    df.to_csv("data/processed/train_features.csv", index=False)
    print("✅ Features saved to: data/processed/train_features.csv")
