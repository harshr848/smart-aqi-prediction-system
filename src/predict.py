import joblib
import numpy as np

# =========================
# LOAD SAVED MODELS
# =========================
rf = joblib.load("../saved_models/Random Forest.pkl")
et = joblib.load("../saved_models/Extra Trees.pkl")
xgb = joblib.load("../saved_models/XGBoost.pkl")
lgbm = joblib.load("../saved_models/LightGBM.pkl")

hybrid = joblib.load("../saved_models/hybrid.pkl")
scaler = joblib.load("../saved_models/scaler.pkl")

print("\n==============================")
print(" AQI PREDICTION SYSTEM ")
print("==============================\n")

# =========================
# USER INPUT
# =========================
temp_c = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity: "))
wind_kph = float(input("Enter Wind Speed (kph): "))
pressure_mb = float(input("Enter Pressure (mb): "))
pm2_5 = float(input("Enter PM2.5: "))
pm10 = float(input("Enter PM10: "))
co = float(input("Enter CO: "))
no2 = float(input("Enter NO2: "))

# =========================
# CREATE INPUT ARRAY
# =========================
user_input = np.array([[
    0,              # lat
    0,              # lon
    temp_c,
    humidity,
    wind_kph,
    pressure_mb,
    pm2_5,
    pm10,
    co,
    no2
]])

# =========================
# BASE MODEL PREDICTIONS
# =========================
rf_pred = rf.predict(user_input)[0]
et_pred = et.predict(user_input)[0]
xgb_pred = xgb.predict(user_input)[0]
lgbm_pred = lgbm.predict(user_input)[0]

# =========================
# HYBRID PREDICTION
# =========================
meta_input = np.array([[
    rf_pred,
    et_pred,
    xgb_pred,
    lgbm_pred,
    (rf_pred + et_pred + xgb_pred + lgbm_pred) / 4
]])

# Scale input
meta_input = scaler.transform(meta_input)

# Final AQI prediction
final_prediction = hybrid.predict(meta_input)[0]

# =========================
# AQI CATEGORY
# =========================
if final_prediction <= 50:
    category = "Good"

elif final_prediction <= 100:
    category = "Moderate"

elif final_prediction <= 150:
    category = "Unhealthy for Sensitive Groups"

elif final_prediction <= 200:
    category = "Unhealthy"

elif final_prediction <= 300:
    category = "Very Unhealthy"

else:
    category = "Hazardous"

# =========================
# OUTPUT
# =========================
print("\n==============================")
print(" PREDICTION RESULT ")
print("==============================")

print(f"\nPredicted AQI : {final_prediction:.2f}")
print(f"AQI Category  : {category}")