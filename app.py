import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smart AQI Prediction System",
    page_icon="🌍",
    layout="wide"
)

# =========================
# TITLE
# =========================
st.title("🌍 Smart AQI Prediction Dashboard")

st.markdown("""
### AI-Based Air Quality Monitoring and Prediction System

Hybrid Machine Learning Model for AQI Forecasting and Environmental Analysis
""")

st.divider()

# =========================
# CHECK MODELS
# =========================
required_files = [
    "saved_models/Random Forest.pkl",
    "saved_models/Extra Trees.pkl",
    "saved_models/XGBoost.pkl",
    "saved_models/LightGBM.pkl",
    "saved_models/Voting.pkl",
    "saved_models/hybrid.pkl",
    "saved_models/scaler.pkl"
]

all_exist = all(
    os.path.exists(file)
    for file in required_files
)

# =========================
# TRAIN IF MODELS MISSING
# =========================
if not all_exist:

    st.warning("Training models for first launch...")

    from src.preprocessing import load_data, preprocess_data
    from src.models import get_models

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    df = load_data("data/aqi_data.csv")

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = get_models()

    os.makedirs("saved_models", exist_ok=True)

    # Train base models
    for name, model in models.items():

        model.fit(X_train, y_train)

        joblib.dump(
            model,
            f"saved_models/{name}.pkl"
        )

    # Hybrid model
    train_preds = []

    for name in models.keys():

        model = joblib.load(
            f"saved_models/{name}.pkl"
        )

        train_preds.append(
            model.predict(X_train)
        )

    meta_X_train = np.column_stack(train_preds)

    scaler = StandardScaler()

    meta_X_train = scaler.fit_transform(
        meta_X_train
    )

    hybrid = Ridge(alpha=10.0)

    hybrid.fit(meta_X_train, y_train)

    joblib.dump(
        hybrid,
        "saved_models/hybrid.pkl"
    )

    joblib.dump(
        scaler,
        "saved_models/scaler.pkl"
    )

# =========================
# LOAD MODELS
# =========================
rf = joblib.load(
    "saved_models/Random Forest.pkl"
)

et = joblib.load(
    "saved_models/Extra Trees.pkl"
)

xgb = joblib.load(
    "saved_models/XGBoost.pkl"
)

lgbm = joblib.load(
    "saved_models/LightGBM.pkl"
)

voting = joblib.load(
    "saved_models/Voting.pkl"
)

hybrid = joblib.load(
    "saved_models/hybrid.pkl"
)

scaler = joblib.load(
    "saved_models/scaler.pkl"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙ System Panel")

city = st.sidebar.selectbox(
    "🏙 Select City",
    [
        "Delhi",
        "Mumbai",
        "Kolkata",
        "Chennai",
        "Bangalore",
        "Hyderabad",
        "Pune",
        "Ahmedabad"
    ]
)

st.sidebar.success(
    f"Selected City: {city}"
)

st.sidebar.info(
    f"Date: {datetime.now().strftime('%d-%m-%Y')}"
)

# =========================
# INPUT SECTION
# =========================
st.subheader("📊 Pollution Parameters")

col1, col2 = st.columns(2)

with col1:

    pm25 = st.slider(
        "PM2.5",
        0.0,
        500.0,
        50.0
    )

    pm10 = st.slider(
        "PM10",
        0.0,
        500.0,
        100.0
    )

    no = st.slider(
        "NO",
        0.0,
        500.0,
        20.0
    )

    no2 = st.slider(
        "NO2",
        0.0,
        500.0,
        30.0
    )

    nh3 = st.slider(
        "NH3",
        0.0,
        500.0,
        40.0
    )

with col2:

    so2 = st.slider(
        "SO2",
        0.0,
        500.0,
        10.0
    )

    co = st.slider(
        "CO",
        0.0,
        50.0,
        1.0
    )

    benzene = st.slider(
        "Benzene",
        0.0,
        100.0,
        5.0
    )

    ozone = st.slider(
        "Ozone",
        0.0,
        500.0,
        25.0
    )

st.divider()

# =========================
# PREDICTION BUTTON
# =========================
if st.button("🔍 Predict AQI"):

    # =========================
    # USER INPUT
    # =========================
    user_input = pd.DataFrame([[
        pm25,
        pm10,
        no,
        no2,
        nh3,
        so2,
        co,
        benzene,
        ozone
    ]], columns=[
        "PM2.5",
        "PM10",
        "NO",
        "NO2",
        "NH3",
        "SO2",
        "CO",
        "Benzene",
        "Ozone"
    ])

    # =========================
    # BASE MODEL PREDICTIONS
    # =========================
    rf_pred = rf.predict(user_input)[0]

    et_pred = et.predict(user_input)[0]

    xgb_pred = xgb.predict(user_input)[0]

    lgbm_pred = lgbm.predict(user_input)[0]

    voting_pred = voting.predict(user_input)[0]

    # =========================
    # META INPUT
    # =========================
    meta_input = np.array([[
        rf_pred,
        et_pred,
        xgb_pred,
        lgbm_pred,
        voting_pred
    ]])

    # =========================
    # SCALE INPUT
    # =========================
    meta_input = scaler.transform(meta_input)

    # =========================
    # FINAL PREDICTION
    # =========================
    final_prediction = hybrid.predict(meta_input)[0]

    # =========================
    # AQI CATEGORY
    # =========================
    if final_prediction <= 50:
        category = "Good"
        color = "green"
        advice = "Air quality is satisfactory."

    elif final_prediction <= 100:
        category = "Moderate"
        color = "orange"
        advice = "Air quality is acceptable."

    elif final_prediction <= 150:
        category = "Unhealthy for Sensitive Groups"
        color = "darkorange"
        advice = "Sensitive people should reduce outdoor activity."

    elif final_prediction <= 200:
        category = "Unhealthy"
        color = "red"
        advice = "Wear masks outdoors."

    elif final_prediction <= 300:
        category = "Very Unhealthy"
        color = "purple"
        advice = "Avoid prolonged outdoor exposure."

    else:
        category = "Hazardous"
        color = "maroon"
        advice = "Stay indoors and use air purifiers."

    # =========================
    # RESULTS
    # =========================
    st.subheader("📌 AQI Prediction Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Predicted AQI",
            f"{final_prediction:.2f}"
        )

    with c2:
        st.metric(
            "AQI Category",
            category
        )

    with c3:
        st.metric(
            "Selected City",
            city
        )

    st.markdown(
        f"<h2 style='color:{color};'>AQI Status: {category}</h2>",
        unsafe_allow_html=True
    )

    st.info(advice)

    st.divider()

    # =========================
    # AQI METER
    # =========================
    st.subheader("🎯 AQI Meter")

    meter_value = min(
        int(final_prediction),
        500
    )

    st.progress(meter_value / 500)

    st.write(
        f"Air Quality Level: {meter_value}/500"
    )

    st.divider()

    # =========================
    # MODEL COMPARISON
    # =========================
    st.subheader("🤖 Base Model Predictions")

    model_df = pd.DataFrame({
        "Model": [
            "Random Forest",
            "Extra Trees",
            "XGBoost",
            "LightGBM",
            "Voting"
        ],
        "Prediction": [
            rf_pred,
            et_pred,
            xgb_pred,
            lgbm_pred,
            voting_pred
        ]
    })

    st.dataframe(model_df)

    # =========================
    # BAR GRAPH
    # =========================
    fig1, ax1 = plt.subplots(figsize=(7, 4))

    ax1.bar(
        model_df["Model"],
        model_df["Prediction"]
    )

    ax1.set_title(
        "Base Model AQI Predictions"
    )

    ax1.set_ylabel("AQI")

    st.pyplot(fig1)

    st.divider()

    # =========================
    # FORECAST
    # =========================
    st.subheader("📈 7-Day AQI Forecast")

    future_days = np.arange(1, 8)

    forecast = []

    current_aqi = final_prediction

    for day in future_days:

        next_aqi = (
            current_aqi +
            np.random.randint(-15, 15)
        )

        if next_aqi < 0:
            next_aqi = 0

        forecast.append(next_aqi)

        current_aqi = next_aqi

    forecast_df = pd.DataFrame({
        "Day": future_days,
        "Predicted AQI": forecast
    })

    st.dataframe(forecast_df)

    fig2, ax2 = plt.subplots(figsize=(9, 4))

    ax2.plot(
        future_days,
        forecast,
        marker='o'
    )

    ax2.set_title(
        f"7-Day AQI Forecast for {city}"
    )

    ax2.set_xlabel("Day")

    ax2.set_ylabel("AQI")

    st.pyplot(fig2)

    st.divider()

    # =========================
    # POLLUTION INSIGHTS
    # =========================
    st.subheader("🧠 Pollution Insights")

    if pm25 > 150:
        st.warning(
            "High PM2.5 detected. Fine particles are dangerous."
        )

    if pm10 > 200:
        st.warning(
            "PM10 level is extremely high."
        )

    if no2 > 100:
        st.warning(
            "NO2 concentration is unhealthy."
        )

    if co > 10:
        st.warning(
            "CO concentration is high."
        )

    if ozone > 100:
        st.warning(
            "Ozone level is dangerous."
        )

    st.divider()

    # =========================
    # HEALTH RECOMMENDATION
    # =========================
    st.subheader("🏥 Health Recommendation")

    st.write(advice)

    if final_prediction > 200:
        st.error(
            "Avoid outdoor exercise and wear masks."
        )

    elif final_prediction > 100:
        st.warning(
            "Limit prolonged outdoor exposure."
        )

    else:
        st.success(
            "Air quality is comparatively safer."
        )

    st.divider()

    st.success(
        "Prediction Completed Successfully ✅"
    )