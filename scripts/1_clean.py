"""Step 1: load the raw Craigslist dump, filter bad rows, normalize model names."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import (
    CLEAN_DATA, CURRENT_YEAR, MILEAGE_MAX, PRICE_MAX, PRICE_MIN, RAW_COLUMNS, RAW_DATA,
    STR_COLUMNS, YEAR_MIN,
)
from src.normalizer import build_model_map

print(f"Loading {RAW_DATA} ...")
df = pd.read_csv(RAW_DATA, usecols=RAW_COLUMNS, low_memory=False)
print(f"  {len(df):,} rows")

for col in STR_COLUMNS:
    df[col] = df[col].astype(str).str.lower().str.strip()

for col in ["price", "year", "odometer"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[
    df["price"].between(PRICE_MIN, PRICE_MAX)
    & df["year"].between(YEAR_MIN, CURRENT_YEAR)
    & df["odometer"].between(0, MILEAGE_MAX)
    & (df["manufacturer"] != "nan")
    & (df["model"] != "nan")
].copy()
print(f"  {len(df):,} rows after filtering")

# Normalize model names: strip trim levels and fix misspellings within each make.
print("  Normalizing model names...")
for make, grp in df.groupby("manufacturer"):
    mapping = build_model_map(grp["model"])
    df.loc[df["manufacturer"] == make, "model"] = grp["model"].map(mapping)
df = df[df["model"].notna() & (df["model"] != "")].copy()
print(f"  {len(df):,} rows after model normalization")

CLEAN_DATA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_DATA, index=False)
print(f"Saved {CLEAN_DATA.relative_to(CLEAN_DATA.parents[2])}")
