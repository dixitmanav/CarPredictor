# Car Price Advisor — Analysis & Modeling Walkthrough

A full account of every decision made in this project, written so it can be explained end-to-end in an interview.

---

## The Problem

When someone buys a used car from a dealer, they face two information gaps:

1. **What should I pay for the car I'm buying?** Dealers post inflated sticker prices; buyers need a data-backed ceiling.
2. **What should I get for my trade-in?** Dealers lowball trade-ins; sellers need a data-backed floor.

The goal is to produce actionable numbers — open offer, target, and walk-away price — for both sides of the deal.

---

## Dataset

**Source:** [Craigslist Used Cars](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) — ~426,000 real listings scraped from Craigslist across the US.

**Why Craigslist?** Private-party listings reflect actual transaction-level prices, not sticker prices. They capture what a car is worth in the real market, not what a dealership wishes it were worth. This is the right data for the problem.

**Key columns used:** `price`, `year`, `odometer`, `manufacturer`, `model`, `condition`, `drive`, `transmission`, `fuel`, `title_status`, `cylinders`, `state`.

---

## Data Cleaning

**Price bounds: $1,500 – $80,000**
Listings below $1,500 are typically junk, salvage parts, or test posts. Above $80,000 is supercar territory that introduces noise for general-use predictions.

**Mileage bounds: 0 – 300,000**
Odometer readings above 300k are rare and unreliable. Entries showing exactly 0 miles on old vehicles are often misreported but left in since a legitimate near-zero reading is possible (e.g., a car off a lot).

**Year range: 2000 – 2024**
Cars older than 2000 are collectibles, not daily drivers. Their pricing follows different logic (rarity, restoration quality) that a regression model won't capture well.

**Manufacturer/model filtering:**
Rows with null or "nan" make/model were dropped — they can't be used for prediction or for populating the UI dropdowns.

**String normalization:**
All categorical columns lowercased and stripped. This prevents "Toyota" and "toyota" from being treated as different values during encoding.

---

## Exploratory Data Analysis (EDA)

### Price Distribution
The price histogram is right-skewed — most listings cluster between $3,000 and $20,000, with a long tail toward $60,000+. This told us the model would need to handle variance well across price ranges, which influenced the choice of MAE-based loss over MSE.

### Odometer Distribution
Also right-skewed. Most cars have between 30,000 and 200,000 miles. The spread confirmed that mileage would be one of the strongest predictors.

### Price vs. Age Scatter
Clear negative correlation — older cars are cheaper — but with significant variance at any given age. This variance is explained by condition, mileage, and make/model, which is exactly what the model learns.

### Median Price by Age
A line chart of median price by vehicle age shows a smooth depreciation curve that flattens after about year 10. Practically, a 3-year-old car loses a lot of value annually; a 15-year-old car doesn't lose much more each year.

### Condition Breakdown
Most listings report "good" condition. Very few report "excellent" or "like new," which is realistic — sellers on Craigslist tend to self-rate conservatively.

### Median Price by State (top 10)
States like California, New York, and Washington have higher median prices than the national average. This confirmed that geographic location should be a feature in the model.

---

## Feature Engineering

### Age
`current_year - year`. More interpretable than raw year because the model learns depreciation curves rather than absolute year effects.

### Condition Score (ordinal encoding)
`{salvage: 1, fair: 2, good: 3, excellent: 4, like new: 5, new: 6}`

This is an ordinal (ordered) encoding, not one-hot. Condition has a clear ranking — excellent is strictly better than good — so treating it as a continuous ordinal number is appropriate. Using one-hot would imply conditions are unordered categories, which is wrong.

### Drive, Transmission, Fuel, Title (ordinal/binary)
Same logic: these have a meaningful order or binary interpretation. Title risk in particular is a risk score — clean=0 implies no risk, salvage=4 implies highest risk. "Parts only" titles also get 4: with only 35 listings (0.01% of the data) the model can't learn a separate effect for them, so they borrow salvage's.

### Cylinders
Extracted as a number from strings like "4 cylinders". 4-cylinder engines are standard commuter cars; 6 and 8 cylinders command a premium.

### Make and Model: Target Encoding
This is the key design decision for generalization.

**The problem:** There are 100+ makes and thousands of make/model combinations. One-hot encoding them produces a massive sparse feature matrix and doesn't generalize to makes seen rarely during training.

**The solution:** Target encoding. For each make, compute the median listing price and divide by the global median. This produces a `make_ratio` (e.g., BMW ≈ 2.1, meaning BMWs sell for about 2.1× the average). The same ratio is computed per `make_model` pair.

This collapses hundreds of categories into a single continuous number that the model can use, and it naturally handles the fact that a 2019 Camry and a 2019 Accord are different price tiers from a 2019 Civic.

**Two safeguards make target encoding honest:**

1. **Fit on training rows only.** The train/test split happens *before* the encodings are computed (`scripts/2_features.py`). If test prices were included, the features would already contain part of the answer the model is graded on. An earlier version made this mistake.
2. **Smoothing.** Two-thirds of make+model names in the data have fewer than 10 listings (many are free-text leftovers). A raw median from one listing is just that car's price: for a single-listing model, the "feature" *is* the target. Each group's median is therefore blended toward its parent's (model → make → overall) as if the parent contributed 10 extra listings:

   `smoothed = (n × group_median + 10 × parent_median) / (n + 10)`

   A model with 5,000 listings keeps essentially its own median. A model with 1 listing is mostly its make's median.

**Why median and not mean?** Price distributions are right-skewed. The mean is pulled up by high-priced outliers. The median is a more stable central tendency for this kind of data.

### State Price Ratio
Same target encoding applied to state. California listings are systematically higher than Mississippi listings. Rather than one-hot encoding 50 states, we encode each state as its median price relative to the national median.

### Age × Mileage Interaction
`age * odometer` captures the combined effect of age and usage. A 10-year-old car with 200k miles is worth much less than a 10-year-old car with 50k miles. This interaction term lets the model learn that relationship directly.

---

## Target Construction

### Purchase Price Target
The raw listing price from the dataset. This represents what private sellers actually ask (and roughly receive) for their cars.

### Trade-In Target (quantile regression)
There is no "trade-in value" column in the dataset — Craigslist doesn't have it. Instead of inventing a target column, the trade-in model learns from the **same listing prices** as the purchase model but is trained to predict a different point: the **25th percentile** price for a car with these exact features. Put simply, it answers "what price do only 1 in 4 comparable cars sell below?"

**Why the 25th percentile?** Dealers buy trade-ins at wholesale, not retail. They need margin to resell. The lower quartile of private listings is the closest publicly available proxy for what dealers pay — it's the price at which sellers are clearly motivated and dealers can make money on resale. We then apply an additional 15% dealer discount on top of this to account for the difference between private-party and wholesale.

**Why not use a separate dataset?** KBB/Edmunds trade-in data isn't publicly available. This approach is transparent, reproducible, and tuned to be slightly conservative on purpose — it's better for the user to walk in with a number that's a slight underestimate of what they should ask than an overestimate.

**What was wrong with the first version:** the original target was the 25th-percentile price of each (age bucket, condition) group, computed ahead of time. Because age and condition were also model inputs, the model could reconstruct the target exactly — it scored R² = 1.000 and MAE ≈ $0 without learning anything about individual cars (a 2015 Civic and a 2015 BMW in "good" condition got the same trade-in value). Quantile regression fixes both problems: the target is real prices, and the prediction depends on every feature, including make and model.

---

## Models

### Purchase Price Model: Gradient Boosting Regressor (GBR)

**Why gradient boosting?** Used cars have non-linear, interaction-heavy price dynamics. A 10-year-old BMW with 30k miles is priced differently than a 10-year-old Corolla with 30k miles — not just additively different. Gradient boosting naturally handles these interaction effects through its tree structure, without requiring you to specify every interaction manually.

**Why not linear regression?** Linear regression would require manually specifying every interaction term and would struggle with the non-linear depreciation curve. It would also be sensitive to the skewed price distribution.

**Why not XGBoost?** The sklearn GBR is well-tested, has no extra dependencies, and the `loss="absolute_error"` setting directly minimizes MAE, which is the metric that matters here — dollars off in prediction. XGBoost would perform similarly but adds a dependency.

**Hyperparameters:**
- `n_estimators=400, learning_rate=0.05`: Slower learning rate with more trees generally beats faster learning with fewer trees. 400 trees at 0.05 lr is a common sweet spot for this data size.
- `max_depth=5`: Enough to capture 5-way feature interactions without overfit.
- `min_samples_leaf=10`: Prevents the model from fitting individual outlier listings.
- `subsample=0.8`: Trains each tree on a random 80% of the data (stochastic gradient boosting), which reduces overfitting and speeds up training.
- `loss="absolute_error"`: Optimizes for MAE directly. The mean-squared-error loss would penalize large errors quadratically, making the model overfit to outlier listings.

**Results on held-out test set (67,197 listings, all makes and models):**
- MAE ≈ $3,099
- R² ≈ 0.863

Interpretation: on average, the model is off by about $3,099 and explains 86.3% of the variation in listing prices. Craigslist prices are noisy (the same car is listed at very different prices by different sellers), so a few thousand dollars of error is expected. The single-model notebook (`notebooks/car_negotiator.ipynb`) scores much higher because it only has to price one model, e.g. Camrys.

### Trade-In Value Model: Gradient Boosting with Quantile Loss

Same algorithm and hyperparameters as the purchase model, with one change: `loss="quantile", alpha=0.25`. A normal regression model aims for the middle of the price distribution. Quantile loss tilts the target by penalizing over-predictions three times as heavily as under-predictions. The result is that the model's best guess settles at the price 25% of comparable cars fall below.

**How it is evaluated.** R² and MAE against actual prices don't make sense here, since the model is *supposed* to predict below the typical price. Instead:
- **Coverage (calibration):** the share of test prices that fall below the prediction. It should be 25%. **Result: 24.8%.**
- **Pinball loss:** the standard error metric for quantile predictions (it's what the model minimizes). Compared with a baseline that predicts the same overall 25th-percentile price for every car: **$1,171 vs $3,595**, a 67% reduction. So the model is genuinely using each car's features.

---

## Model Pipeline: StandardScaler + Model

Both models are wrapped in a `sklearn.Pipeline`:
```
Pipeline([("scaler", StandardScaler()), ("model", ...)])
```

**Why scale?** Gradient boosting and random forests are tree-based and don't require scaling — trees split on thresholds, not distances. The scaler is included anyway so the saved `.pkl` file is completely self-contained: if you ever swap in a distance-based model (like SVR or KNN), you don't need to change the inference code.

**Why Pipeline?** `joblib.dump(pipeline, file)` saves the scaler and model together. When loading for inference, `pipeline.predict(X)` automatically applies the scaler to new inputs. There's no risk of forgetting to scale.

---

## Evaluation

### Metrics
- **MAE (Mean Absolute Error):** Average dollar error. Directly interpretable. "The purchase model is off by about $3,099 on average."
- **R² (coefficient of determination):** Fraction of price variance explained by the model. 0.863 means the purchase model explains 86.3% of why prices differ between listings.
- **Coverage and pinball loss (trade-in model):** see the trade-in model section above.

### Train/Test Split
80/20 split with `random_state=42`, made in step 2 before any target encodings are fit. The test set was held out entirely — its prices are not used for the encodings and no hyperparameter tuning was done on it, so the reported metrics are unbiased estimates of real-world performance.

### Residual Analysis
Plotting `actual - predicted` reveals whether errors are symmetric (good) or systematically biased in a direction (bad). Near-zero mean residual confirms the model isn't systematically under- or over-pricing cars.

---

## Prediction Logic and Negotiation Strategy

### Price Range Construction
The point estimate from the model is converted into a range using the test-set residual standard deviation. We use:
- **Low end:** `max(1500, pred - 0.8 * std)` — conservative lower bound
- **High end:** `pred + 0.6 * std` — slightly asymmetric to account for the right-skew in listing prices

### Opening Offer, Target, Walk-Away
- **Opening offer:** 93% of the low end — starts negotiations below fair value, giving room to move up while staying grounded in data.
- **Target:** midpoint of the fair range — the number you're aiming to close at.
- **Walk-away ceiling:** 102% of the high end — the maximum you'll pay; above this, data says you're overpaying.

### Trade-In Strategy
The model predicts dealer wholesale value. Its fair range is percentage-based rather than a fixed dollar amount, because the model's errors grow with the car's value (about $2.4k on cars under $8k versus $7.1k on cars over $20k). The range is `pred × e^(−0.5σ)` to `pred × e^(0.4σ)`, where σ is the standard deviation of the log residuals on the test set (≈ 0.29). We then:
- **Ask for:** 105% of the predicted high end — starts negotiations above your fair value.
- **Floor:** 90% of the predicted low end — the minimum you'll accept.

### 15% Dealer Discount
Dealers typically pay 15% below private-party lower-quartile prices to build in resale margin. This is a well-documented rule of thumb in automotive retail pricing, and it's reflected in the trade-in prediction pipeline.

---

## Feature Importance (Interview Summary)

For the purchase model, the most important features are:
1. **model_ratio** — which car you're buying is the single biggest price driver
2. **age** — depreciation is the dominant pricing force
3. **odometer** — mileage explains variance within a given age
4. **age_x_miles** — the interaction captures "high miles for an old car" scenarios
5. **condition_score** — dealer-reported condition matters less than age/miles but is significant

For the trade-in model, the ranking is similar since it learns from the same prices; it just aims lower in the distribution.

---

## Limitations

1. **Data is from ~2021.** Prices have changed significantly since then (chip shortage, post-COVID used car boom). The model's absolute predictions may be off by 10–20% for current prices, though the relative relationships (age/miles/condition effects) remain valid.
2. **Regional coverage is uneven.** Some states have many Craigslist listings; others have few. The state ratio encoding may be noisy for low-listing states.
3. **Electric vehicles.** The dataset predates the wide availability of EVs, so electric vehicle predictions have less training data and should be treated cautiously.
4. **Rare makes/models.** For a make/model with few listings, the smoothed encoding leans on the make-level price, so a rare trim of a common make is priced like a typical car of that make. Models with fewer than 10 listings are left out of the app's dropdown.
5. **Mileage of 0.** Listings with exactly 0 miles are kept but likely misreported. They may pull predictions slightly low for very-low-mileage cars.
