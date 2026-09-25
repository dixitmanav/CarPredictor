"""Step 2: split train/test, fit target encodings on the training rows, and build model features."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLEAN_DATA, ENCODINGS_PATH, FEATURES, FEATURES_DATA, MIN_CATALOG_LISTINGS, RANDOM_STATE, TEST_SIZE,
)
from src.features import add_features, fit_encodings

df = pd.read_csv(CLEAN_DATA, low_memory=False)
print(f"Loaded {len(df):,} clean rows")

# Split before fitting encodings so test-set prices never leak into the features.
train_idx, test_idx = train_test_split(df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE)
df["split"] = "train"
df.loc[test_idx, "split"] = "test"

enc = fit_encodings(df.loc[train_idx])
df = add_features(df, enc)
df = df.dropna(subset=FEATURES + ["price"]).copy()

# makes_catalog feeds the app's make/model dropdowns; rare free-text names are left out.
counts = df.groupby(["manufacturer", "model"]).size()
common = counts[counts >= MIN_CATALOG_LISTINGS].reset_index()
enc["makes_catalog"] = {
    make: sorted(grp["model"].tolist())
    for make, grp in common.groupby("manufacturer")
}

df[["manufacturer", "model", "price", "split"] + FEATURES].to_csv(FEATURES_DATA, index=False)
with open(ENCODINGS_PATH, "w") as f:
    json.dump(enc, f, indent=2)
print(f"Saved {len(df):,} rows of features ({(df['split'] == 'train').sum():,} train) and the encodings")
print(f"  {len(enc['model_ratios']):,} make+model encodings, {len(common):,} models in the app catalog")
