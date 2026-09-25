"""Step 2: fit target encodings and build model features."""
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

# makes_catalog feeds the app's make/model dropdowns.
enc["makes_catalog"] = {
    make: sorted(grp["model"].unique().tolist())
    for make, grp in df.groupby("manufacturer")
}

df[["manufacturer", "model", "price"] + FEATURES].to_csv(FEATURES_DATA, index=False)
with open(ENCODINGS_PATH, "w") as f:
    json.dump(enc, f, indent=2)
print(f"Saved {len(df):,} rows of features and the encodings")
