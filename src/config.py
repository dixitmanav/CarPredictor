from pathlib import Path

# All paths are relative to the project root, so scripts work from any directory.
ROOT = Path(__file__).resolve().parents[1]

RAW_DATA       = ROOT / "data" / "raw" / "vehicles.csv"
CLEAN_DATA     = ROOT / "data" / "processed" / "vehicles_clean.csv"
FEATURES_DATA  = ROOT / "data" / "processed" / "features.csv"
ENCODINGS_PATH = ROOT / "data" / "processed" / "encodings.json"
MODELS_DIR     = ROOT / "models"

CURRENT_YEAR = 2024
PRICE_MIN, PRICE_MAX = 1_500, 80_000
MILEAGE_MAX = 300_000
YEAR_MIN = 2000
DEALER_DISCOUNT = 0.15
TRADEIN_QUANTILE = 0.25   # trade-in model predicts this percentile of comparable listing prices

# Columns kept from the raw Craigslist dump; everything else is dropped in step 1.
RAW_COLUMNS = [
    "price", "year", "odometer", "manufacturer", "model", "condition",
    "drive", "transmission", "fuel", "title_status", "cylinders", "state",
]
STR_COLUMNS = ["manufacturer", "model", "condition", "drive", "transmission", "fuel", "title_status", "state"]

CONDITION_MAP = {"salvage": 1, "fair": 2, "good": 3, "excellent": 4, "like new": 5, "new": 6}
DRIVE_MAP     = {"fwd": 0, "rwd": 1, "4wd": 2}
TRANS_MAP     = {"automatic": 1, "manual": 0, "other": 0}
FUEL_MAP      = {"gas": 0, "hybrid": 1, "electric": 2, "diesel": 3, "other": 0}
TITLE_MAP     = {"clean": 0, "rebuilt": 2, "salvage": 4, "lien": 1, "missing": 3, "parts only": 5}

FEATURES = [
    "age", "odometer", "condition_score", "drive_enc", "trans_enc",
    "fuel_enc", "title_risk", "cylinders_n",
    "make_ratio", "model_ratio", "state_ratio", "age_x_miles",
]
