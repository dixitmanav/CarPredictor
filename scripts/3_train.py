"""Step 3: train the purchase and trade-in models, evaluate them, and save everything the app needs."""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CURRENT_YEAR, DEALER_DISCOUNT, ENCODINGS_PATH, FEATURES, FEATURES_DATA, MODELS_DIR

warnings.filterwarnings("ignore")

df = pd.read_csv(FEATURES_DATA, low_memory=False)
with open(ENCODINGS_PATH) as f:
    enc = json.load(f)

X = df[FEATURES].values
y_purchase = df["price"].values
y_tradein = df["tradein_target"].values

X_tr, X_te, yp_tr, yp_te, yt_tr, yt_te = train_test_split(
    X, y_purchase, y_tradein, test_size=0.2, random_state=42
)
print(f"Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

purchase_model = Pipeline([
    ("scaler", StandardScaler()),
    ("gbr", GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        min_samples_leaf=10, subsample=0.8,
        loss="absolute_error", random_state=42,
    )),
])
print("Training purchase price model (GBR) ...")
purchase_model.fit(X_tr, yp_tr)

tradein_model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=300, max_depth=10,
        min_samples_leaf=10, random_state=42, n_jobs=-1,
    )),
])
print("Training trade-in model (RF) ...")
tradein_model.fit(X_tr, yt_tr)

p_preds = purchase_model.predict(X_te)
t_preds = tradein_model.predict(X_te)

p_mae = mean_absolute_error(yp_te, p_preds)
p_r2  = r2_score(yp_te, p_preds)
t_mae = mean_absolute_error(yt_te, t_preds)
t_r2  = r2_score(yt_te, t_preds)

print(f"\nPurchase model  MAE ${p_mae:,.0f}   R² {p_r2:.3f}")
print(f"Trade-in model  MAE ${t_mae:,.0f}    R² {t_r2:.3f}")

metadata = {
    **enc,
    "features": FEATURES,
    "purchase_std": float(np.std(yp_te - p_preds)),
    "tradein_std": float(np.std(yt_te - t_preds)),
    "dealer_discount": DEALER_DISCOUNT,
    "current_year": CURRENT_YEAR,
    "metrics": {
        "purchase": {"mae": round(p_mae, 2), "r2": round(p_r2, 4)},
        "tradein":  {"mae": round(t_mae, 2), "r2": round(t_r2, 4)},
    },
}

MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(purchase_model, MODELS_DIR / "purchase_model.pkl")
joblib.dump(tradein_model,  MODELS_DIR / "tradein_model.pkl")
with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved models and metadata to {MODELS_DIR}/")
