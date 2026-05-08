print("STARTING AQI TRAINING SYSTEM...")

from preprocessing import load_data, preprocess_data
from models import get_models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# =========================
# CREATE MODEL FOLDER
# =========================
os.makedirs("../saved_models", exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = load_data("../data/aqi_data.csv")

# =========================
# PREPROCESS
# =========================
X, y = preprocess_data(df)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# GET MODELS
# =========================
models = get_models()

results = []

print("\nTRAINING MODELS...\n")

# =========================
# TRAIN BASE MODELS
# =========================
for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    rmse = np.sqrt(mean_squared_error(y_test, pred))

    r2 = r2_score(y_test, pred)

    print(f"{name} R2 Score: {r2}")

    # Save model
    joblib.dump(
        model,
        f"../saved_models/{name}.pkl"
    )

    results.append([
        name,
        mae,
        rmse,
        r2
    ])

# =========================
# HYBRID MODEL
# =========================
print("\nTRAINING HYBRID MODEL...\n")

train_preds = []
test_preds = []

for name in models.keys():

    model = joblib.load(
        f"../saved_models/{name}.pkl"
    )

    train_preds.append(
        model.predict(X_train)
    )

    test_preds.append(
        model.predict(X_test)
    )

# Stack predictions
meta_X_train = np.column_stack(train_preds)

meta_X_test = np.column_stack(test_preds)

# Scaling
scaler = StandardScaler()

meta_X_train = scaler.fit_transform(
    meta_X_train
)

meta_X_test = scaler.transform(
    meta_X_test
)

# Save scaler
joblib.dump(
    scaler,
    "../saved_models/scaler.pkl"
)

# Meta model
hybrid = Ridge(alpha=10.0)

hybrid.fit(meta_X_train, y_train)

# Save hybrid model
joblib.dump(
    hybrid,
    "../saved_models/hybrid.pkl"
)

# Final prediction
final_pred = hybrid.predict(meta_X_test)

# Metrics
mae = mean_absolute_error(y_test, final_pred)

rmse = np.sqrt(mean_squared_error(y_test, final_pred))

r2 = r2_score(y_test, final_pred)

print("\nHYBRID MODEL RESULTS")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

results.append([
    "Hybrid Model",
    mae,
    rmse,
    r2
])

# =========================
# RESULTS TABLE
# =========================
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]
)

print("\nFINAL RESULTS\n")

print(results_df)

# =========================
# SAVE RESULTS
# =========================
results_df.to_csv(
    "../saved_models/results.csv",
    index=False
)

# =========================
# R2 GRAPH
# =========================
plt.figure(figsize=(10, 5))

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.title("Model Comparison")

plt.xlabel("Models")

plt.ylabel("R2 Score")

plt.xticks(rotation=20)

plt.show()

# =========================
# ACTUAL VS PREDICTED
# =========================
plt.figure(figsize=(6, 6))

plt.scatter(y_test, final_pred)

plt.xlabel("Actual AQI")

plt.ylabel("Predicted AQI")

plt.title("Actual vs Predicted AQI")

plt.show()