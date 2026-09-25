"""
Feature engineering shared by training (scripts/2_features.py) and serving (app/app.py).

Keeping this in one place guarantees the model sees identically-built features
in both settings, avoiding training/serving skew.
"""
import pandas as pd

from src.config import (
    CONDITION_MAP, CURRENT_YEAR, DRIVE_MAP, FEATURES, FUEL_MAP, TITLE_MAP, TRANS_MAP,
)


def _mm_key(df: pd.DataFrame) -> pd.Series:
    return df["manufacturer"] + "_" + df["model"]


def fit_encodings(df: pd.DataFrame) -> dict:
    """
    Target-encode make, make+model, and state as median-price ratios relative to
    the global median. This lets the model generalize across hundreds of
    makes/models without one-hot explosion.
    """
    global_median = float(df["price"].median())

    def ratios(keys: pd.Series) -> dict:
        medians = df.groupby(keys)["price"].median()
        return {k: float(v) / global_median for k, v in medians.items()}

    return {
        "global_median": global_median,
        "make_ratios":  ratios(df["manufacturer"]),
        "model_ratios": ratios(_mm_key(df)),
        "state_ratios": ratios(df["state"]),
    }


def add_features(df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    """
    Return a copy of df with every column in FEATURES added.

    Expects the columns year, odometer, manufacturer, model, condition, drive,
    transmission, fuel, title_status, cylinders, state. Unseen categories fall
    back to neutral defaults, so this also works on a single car from the app.
    """
    df = df.copy()
    for col in ["manufacturer", "model", "condition", "drive", "transmission", "fuel", "title_status", "state"]:
        df[col] = df[col].astype(str).str.lower().str.strip()

    df["age"] = CURRENT_YEAR - df["year"]
    df["condition_score"] = df["condition"].map(CONDITION_MAP).fillna(3)
    df["drive_enc"]       = df["drive"].map(DRIVE_MAP).fillna(0)
    df["trans_enc"]       = df["transmission"].map(TRANS_MAP).fillna(1)
    df["fuel_enc"]        = df["fuel"].map(FUEL_MAP).fillna(0)
    df["title_risk"]      = df["title_status"].map(TITLE_MAP).fillna(0)

    # Raw data has strings like "6 cylinders"; the app sends plain numbers.
    df["cylinders_n"] = (
        df["cylinders"].astype(str).str.extract(r"(\d+)")[0].astype(float).fillna(4.0)
    )

    df["make_ratio"]  = df["manufacturer"].map(enc["make_ratios"]).fillna(1.0)
    # Unseen model falls back to its make's ratio.
    df["model_ratio"] = _mm_key(df).map(enc["model_ratios"]).fillna(df["make_ratio"])
    df["state_ratio"] = df["state"].map(enc["state_ratios"]).fillna(1.0)

    df["age_x_miles"] = df["age"] * df["odometer"]
    return df


def feature_matrix(df: pd.DataFrame):
    return df[FEATURES].values
