# app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# Load trained models
# ==========================
st.title("Bike Sharing Demand Prediction 🚴")

@st.cache_resource
def load_models():
    rf_model  = joblib.load("models/random_forest_model.pkl")
    gb_model  = joblib.load("models/gradient_boosting_model.pkl")
    xgb_model = joblib.load("models/xgboost_model.pkl")
    return rf_model, gb_model, xgb_model


rf_model, gb_model, xgb_model = load_models()

# ==========================
# Sidebar for user input
# ==========================
st.sidebar.header("Input Features")

def user_input_features():
    season = st.sidebar.selectbox("Season (1:spring,2:summer,3:fall,4:winter)", [1,2,3,4])
    holiday = st.sidebar.selectbox("Holiday (0:No, 1:Yes)", [0,1])
    workingday = st.sidebar.selectbox("Working Day (0:No, 1:Yes)", [0,1])
    weather = st.sidebar.selectbox("Weather (1:clear,2:mist,3:light rain,4:heavy rain)", [1,2,3,4])
    temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 20.0)
    atemp = st.sidebar.slider("Feels like Temp (°C)", 0.0, 50.0, 20.0)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
    windspeed = st.sidebar.slider("Windspeed", 0.0, 50.0, 10.0)
    hour = st.sidebar.slider("Hour", 0, 23, 12)
    day = st.sidebar.slider("Day of Month", 1, 31, 15)
    month = st.sidebar.slider("Month", 1, 12, 6)
    year = st.sidebar.slider("Year", 2011, 2012, 2012)
    dayofweek = st.sidebar.slider("Day of Week (0:Sun,6:Sat)", 0, 6, 3)

    data = {
        'season': season,
        'holiday': holiday,
        'workingday': workingday,
        'weather': weather,
        'temp': temp,
        'atemp': atemp,
        'humidity': humidity,
        'windspeed': windspeed,
        'hour': hour,
        'day': day,
        'month': month,
        'year': year,
        'dayofweek': dayofweek
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# ==========================
# Predict
# ==========================
st.subheader("Input Features")
st.write(input_df)

st.subheader("Predictions")

rf_pred = rf_model.predict(input_df)[0]
gb_pred = gb_model.predict(input_df)[0]
xgb_pred = xgb_model.predict(input_df)[0]

st.write(f"RandomForest Prediction: {rf_pred:.0f}")
st.write(f"GradientBoosting Prediction: {gb_pred:.0f}")
st.write(f"XGBoost Prediction: {xgb_pred:.0f}")

best_pred = min(rf_pred, gb_pred, xgb_pred)  # or choose based on RMSLE
st.subheader(f"Suggested Prediction: {best_pred:.0f} bikes")
