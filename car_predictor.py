import pandas as pd
import numpy as np
import joblib
import json
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

DATA_PATH = Path("vehicles.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CURRENT_YEAR = 2024
PRICE_MIN, PRICE_MAX = 1_500, 80_000
MILEAGE_MAX = 300_000
YEAR_MIN = 2000

print(f"Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  {len(df):,} rows, {df.shape[1]} columns")

str_cols = ["manufacturer", "model", "condition", "drive", "transmission", "fuel", "title_status"]
for col in str_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")

df = df[
    df["price"].between(PRICE_MIN, PRICE_MAX)
    & df["year"].between(YEAR_MIN, CURRENT_YEAR)
    & df["odometer"].between(0, MILEAGE_MAX)
    & df["manufacturer"].notna()
    & (df["manufacturer"] != "nan")
    & df["model"].notna()
    & (df["model"] != "nan")
].copy()

print(f"  {len(df):,} rows after cleaning")

df["age"] = CURRENT_YEAR - df["year"]

CONDITION_MAP = {"salvage": 1, "fair": 2, "good": 3, "excellent": 4, "like new": 5, "new": 6}
DRIVE_MAP     = {"fwd": 0, "rwd": 1, "4wd": 2}
TRANS_MAP     = {"automatic": 1, "manual": 0, "other": 0}
FUEL_MAP      = {"gas": 0, "hybrid": 1, "electric": 2, "diesel": 3, "other": 0}
TITLE_MAP     = {"clean": 0, "rebuilt": 2, "salvage": 4, "lien": 1, "missing": 3, "parts only": 5}

df["condition_score"] = df["condition"].map(CONDITION_MAP).fillna(3)
df["drive_enc"]       = df["drive"].map(DRIVE_MAP).fillna(0)
df["trans_enc"]       = df["transmission"].map(TRANS_MAP).fillna(1)
df["fuel_enc"]        = df["fuel"].map(FUEL_MAP).fillna(0)
df["title_risk"]      = df["title_status"].map(TITLE_MAP).fillna(0)

if "cylinders" in df.columns:
    df["cylinders_n"] = df["cylinders"].astype(str).str.extract(r"(\d+)")[0].astype(float)
df["cylinders_n"] = pd.to_numeric(df.get("cylinders_n", 4.0), errors="coerce").fillna(4.0)

# Target-encode make and make+model as price ratios relative to the global median.
# This lets the model generalize across hundreds of makes/models without one-hot explosion.
global_median = df["price"].median()

make_medians = df.groupby("manufacturer")["price"].median()
df["make_ratio"] = df["manufacturer"].map(make_medians) / global_median

df["mm_key"] = df["manufacturer"] + "_" + df["model"]
mm_medians = df.groupby("mm_key")["price"].median()
df["model_ratio"] = df["mm_key"].map(mm_medians) / global_median

state_medians = df.groupby("state")["price"].median()
df["state_ratio"] = df["state"].map(state_medians) / global_median

df["age_x_miles"] = df["age"] * df["odometer"]

FEATURES = [
    "age", "odometer", "condition_score", "drive_enc", "trans_enc",
    "fuel_enc", "title_risk", "cylinders_n",
    "make_ratio", "model_ratio", "state_ratio", "age_x_miles",
]

df_clean = df.dropna(subset=FEATURES + ["price"]).copy()
X = df_clean[FEATURES].values
y_purchase = df_clean["price"].values

# Trade-in proxy: 25th-percentile price within each (age-bucket, condition) cell.
# Dealers buy at wholesale, so the lower tail of private listings is a reasonable floor.
df_clean["age_bucket"] = pd.cut(df_clean["age"], bins=[0, 3, 6, 9, 15, 25], labels=False)
y_tradein = (
    df_clean.groupby(["age_bucket", "condition_score"])["price"]
    .transform(lambda g: g.quantile(0.25))
    .values
)

X_tr, X_te, yp_tr, yp_te, yt_tr, yt_te = train_test_split(
    X, y_purchase, y_tradein, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_tr):,}  |  Test: {len(X_te):,}")

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

makes_catalog = {
    make: sorted(df_clean[df_clean["manufacturer"] == make]["model"].unique().tolist())
    for make in sorted(df_clean["manufacturer"].unique())
}

metadata = {
    "features": FEATURES,
    "global_median": float(global_median),
    "make_ratios": {k: float(v) for k, v in make_medians.items()},
    "model_ratios": {k: float(v) for k, v in mm_medians.items()},
    "state_ratios": {k: float(v) for k, v in state_medians.items()},
    "makes_catalog": makes_catalog,
    "purchase_std": float(np.std(yp_te - p_preds)),
    "tradein_std": float(np.std(yt_te - t_preds)),
    "dealer_discount": 0.15,
    "current_year": CURRENT_YEAR,
    "metrics": {
        "purchase": {"mae": round(p_mae, 2), "r2": round(p_r2, 4)},
        "tradein":  {"mae": round(t_mae, 2),  "r2": round(t_r2, 4)},
    },
}

joblib.dump(purchase_model, MODELS_DIR / "purchase_model.pkl")
joblib.dump(tradein_model,  MODELS_DIR / "tradein_model.pkl")
with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved models and metadata to {MODELS_DIR}/")
