from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    VotingRegressor
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_models():

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Extra Trees
    et = ExtraTreesRegressor(
        n_estimators=100,
        random_state=42
    )

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    # LightGBM
    lgbm = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    # Voting Regressor
    voting = VotingRegressor([
        ("rf", rf),
        ("et", et),
        ("xgb", xgb),
        ("lgbm", lgbm)
    ])

    return {
        "Random Forest": rf,
        "Extra Trees": et,
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "Voting": voting
    }