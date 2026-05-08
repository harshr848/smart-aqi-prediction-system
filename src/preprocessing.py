import pandas as pd

# =========================
# LOAD DATA
# =========================
def load_data(path):

    df = pd.read_csv(path)

    return df


# =========================
# PREPROCESS DATA
# =========================
def preprocess_data(df):

    # Selected Features
    features = [
        "PM2.5",
        "PM10",
        "NO",
        "NO2",
        "NH3",
        "SO2",
        "CO",
        "Benzene",
        "Ozone"
    ]

    target = "AQI"

    # Keep only required columns
    df = df[features + [target]]

    # Remove missing values
    df = df.dropna()

    # Split features and target
    X = df[features]

    y = df[target]

    return X, y