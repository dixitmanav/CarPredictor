# Car Price Advisor

A used-car negotiation tool trained on ~426,000 Craigslist listings. Given a car you want to buy (and optionally one you're trading in), it returns a data-backed **opening offer, target price, and walk-away price**, plus a fair trade-in range, so you walk into a dealership knowing the numbers.

Dealers have more pricing information than buyers do. This project uses real private-party listings to reduce that gap.

## How it works

1. **Clean**: filter 426k raw listings down to realistic daily drivers (2000–2024, $1.5k–$80k, ≤300k miles).
2. **Normalize model names**: Craigslist model names are free text (`"civic ex-l 4dr"`, `"CIVIC"`, `"civic 5-speed"`). A custom normalizer strips trim/body/drivetrain tokens and fuzzy-clusters spelling variants per make with guards so distinct models (`c-class` vs `e-class`, `silverado 1500` vs `2500hd`) never merge.
3. **Engineer features**: age, mileage, condition, drivetrain, fuel, title risk, cylinders, an age × mileage interaction, and target-encoded price ratios for make, make+model, and state (so hundreds of models generalize without one-hot explosion).
4. **Train**: a Gradient Boosting model for purchase price (absolute-error loss, robust to outlier listings) and a Random Forest for trade-in value.
5. **Serve**: a Flask web app turns predictions and model error into negotiation ranges.

Feature engineering lives in one shared module (`src/features.py`) used by both training and the app, so the model sees identically built inputs in both places.

## Results

Purchase-price model on a held-out 20% test set (67,197 listings):

| Metric | Value |
|---|---|
| Mean absolute error | **$3,056** |
| R² | **0.869** |

The trade-in value is currently a rule-based proxy (25th-percentile listing price per age × condition group), so its model is a smoothing layer over that rule rather than an independently validated predictor.

## Project structure

```
CarPredictor/
├── run_all.py              # runs the whole pipeline: clean → features → train
├── requirements.txt
├── data/
│   ├── raw/                # vehicles.csv from Kaggle (not committed, 1.4 GB)
│   └── processed/          # outputs of steps 1–2 (regenerated)
├── src/
│   ├── config.py           # paths, cleaning thresholds, category encodings
│   ├── normalizer.py       # model-name normalization (trim stripping + fuzzy clustering)
│   └── features.py         # feature engineering shared by training and the app
├── scripts/
│   ├── 1_clean.py          # filter raw listings, normalize model names
│   ├── 2_features.py       # target encodings, features, trade-in target
│   └── 3_train.py          # train + evaluate models, save to models/
├── models/                 # trained .pkl files + metadata.json (regenerated)
├── app/
│   ├── app.py              # Flask app
│   └── templates/index.html
├── notebooks/
│   └── car_negotiator.ipynb  # exploratory analysis for a single make/model
└── docs/
    └── analysis_explained.md # every modeling decision, explained
```

## Running it

```bash
pip install -r requirements.txt

# 1. Download vehicles.csv from Kaggle into data/raw/
#    https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

# 2. Run the pipeline (clean → features → train)
python run_all.py

# 3. Start the app, then open http://localhost:5000
python app/app.py
```

Each step can also be run on its own (e.g. `python scripts/3_train.py` to retrain without re-cleaning).

## Data

[Craigslist Cars & Trucks](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) (Austin Reese, Kaggle). Private-party listings reflect market prices rather than dealer sticker prices. See [`docs/analysis_explained.md`](docs/analysis_explained.md) for the full walkthrough of cleaning and modeling decisions.
