"""Step 2: fit target encodings, build model features, and derive the trade-in target."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import CLEAN_DATA, ENCODINGS_PATH, FEATURES, FEATURES_DATA
from src.features import add_features, fit_encodings

df = pd.read_csv(CLEAN_DATA, low_memory=False)
print(f"Loaded {len(df):,} clean rows")

enc = fit_encodings(df)
df = add_features(df, enc)
df = df.dropna(subset=FEATURES + ["price"]).copy()

# Trade-in proxy: 25th-percentile price within each (age-bucket, condition) cell.
# Dealers buy at wholesale, so the lower tail of private listings is a reasonable floor.
df["age_bucket"] = pd.cut(df["age"], bins=[0, 3, 6, 9, 15, 25], labels=False)
df["tradein_target"] = (
    df.groupby(["age_bucket", "condition_score"])["price"].transform(lambda g: g.quantile(0.25))
)
df = df.dropna(subset=["tradein_target"])

# makes_catalog feeds the app's make/model dropdowns.
enc["makes_catalog"] = {
    make: sorted(grp["model"].unique().tolist())
    for make, grp in df.groupby("manufacturer")
}

df[["manufacturer", "model", "price", "tradein_target"] + FEATURES].to_csv(FEATURES_DATA, index=False)
with open(ENCODINGS_PATH, "w") as f:
    json.dump(enc, f, indent=2)
print(f"Saved {len(df):,} rows of features and the encodings")
