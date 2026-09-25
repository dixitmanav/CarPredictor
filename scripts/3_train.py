"""Step 3: train the purchase and trade-in models, evaluate them, and save everything the app needs."""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CURRENT_YEAR, DEALER_DISCOUNT, ENCODINGS_PATH, FEATURES, FEATURES_DATA, MODELS_DIR, TRADEIN_QUANTILE,
)

warnings.filterwarnings("ignore")

df = pd.read_csv(FEATURES_DATA, low_memory=False)
with open(ENCODINGS_PATH) as f:
    enc = json.load(f)

X = df[FEATURES].values
y = df["price"].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
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
purchase_model.fit(X_tr, y_tr)

# Trade-in: quantile regression on the same listing prices. Instead of the typical
# price, it predicts the price that only 25% of comparable cars sell below: the
# low end a dealer buying at wholesale would anchor to.
tradein_model = Pipeline([
    ("scaler", StandardScaler()),
    ("gbr", GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        min_samples_leaf=10, subsample=0.8,
        loss="quantile", alpha=TRADEIN_QUANTILE, random_state=42,
    )),
])
print(f"Training trade-in model (GBR, {TRADEIN_QUANTILE:.0%} quantile) ...")
tradein_model.fit(X_tr, y_tr)

p_preds = purchase_model.predict(X_te)
t_preds = tradein_model.predict(X_te)

p_mae = mean_absolute_error(y_te, p_preds)
p_r2  = r2_score(y_te, p_preds)

# A well-calibrated 25% quantile model has ~25% of real prices fall below its prediction.
t_coverage = float(np.mean(y_te < t_preds))
t_pinball  = mean_pinball_loss(y_te, t_preds, alpha=TRADEIN_QUANTILE)
# Baseline: one global 25th-percentile price for every car, ignoring its features.
t_pinball_base = mean_pinball_loss(
    y_te, np.full_like(y_te, np.quantile(y_tr, TRADEIN_QUANTILE), dtype=float), alpha=TRADEIN_QUANTILE
)

print(f"\nPurchase model  MAE ${p_mae:,.0f}   R² {p_r2:.3f}")
print(
    f"Trade-in model  {t_coverage:.1%} of test prices below prediction (target {TRADEIN_QUANTILE:.0%})   "
    f"pinball loss ${t_pinball:,.0f} vs ${t_pinball_base:,.0f} baseline"
)

metadata = {
    **enc,
    "features": FEATURES,
    "purchase_std": float(np.std(y_te - p_preds)),
    # Trade-in errors grow with the car's value, so the app's range is a percentage
    # of the prediction: std of the log residuals (0.29 ≈ ±29%).
    "tradein_log_std": float(np.std(np.log(y_te) - np.log(np.clip(t_preds, 500, None)))),
    "dealer_discount": DEALER_DISCOUNT,
    "current_year": CURRENT_YEAR,
    "metrics": {
        "purchase": {"mae": round(p_mae, 2), "r2": round(p_r2, 4)},
        "tradein":  {
            "quantile": TRADEIN_QUANTILE,
            "coverage": round(t_coverage, 4),
            "pinball_loss": round(t_pinball, 2),
            "pinball_loss_baseline": round(t_pinball_base, 2),
        },
    },
}

MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(purchase_model, MODELS_DIR / "purchase_model.pkl")
joblib.dump(tradein_model,  MODELS_DIR / "tradein_model.pkl")
with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved models and metadata to {MODELS_DIR}/")
