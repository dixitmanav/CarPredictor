from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import json
from pathlib import Path

app = Flask(__name__)

MODELS_DIR = Path("models")

purchase_model = joblib.load(MODELS_DIR / "purchase_model.pkl")
tradein_model  = joblib.load(MODELS_DIR / "tradein_model.pkl")
with open(MODELS_DIR / "metadata.json") as f:
    meta = json.load(f)

CONDITION_MAP = {"salvage": 1, "fair": 2, "good": 3, "excellent": 4, "like new": 5, "new": 6}
DRIVE_MAP     = {"fwd": 0, "rwd": 1, "4wd": 2}
TRANS_MAP     = {"automatic": 1, "manual": 0}
FUEL_MAP      = {"gas": 0, "hybrid": 1, "electric": 2, "diesel": 3}
TITLE_MAP     = {"clean": 0, "rebuilt": 2, "salvage": 4, "lien": 1}


def build_features(year, odo, cond, drive, trans, fuel, title, cyls, make, model_name, state):
    age    = meta["current_year"] - year
    make_k = make.lower()
    mm_k   = f"{make_k}_{model_name.lower()}"
    return np.array([[
        age, odo,
        CONDITION_MAP.get(cond, 3),
        DRIVE_MAP.get(drive, 0),
        TRANS_MAP.get(trans, 1),
        FUEL_MAP.get(fuel, 0),
        TITLE_MAP.get(title, 0),
        float(cyls),
        meta["make_ratios"].get(make_k, 1.0),
        meta["model_ratios"].get(mm_k, meta["make_ratios"].get(make_k, 1.0)),
        meta["state_ratios"].get(state.lower(), 1.0),
        age * odo,
    ]])


@app.route("/")
def index():
    makes = sorted(meta["makes_catalog"].keys())
    return render_template("index.html", makes=makes)


@app.route("/models_for_make/<make>")
def models_for_make(make):
    return jsonify(meta["makes_catalog"].get(make.lower(), []))


@app.route("/predict", methods=["POST"])
def predict():
    d = request.json

    buy_X    = build_features(
        int(d["buy_year"]), int(d["buy_odo"]),
        d["buy_cond"], d["buy_drive"], d["buy_trans"],
        d["buy_fuel"], d["buy_title"], int(d["buy_cyls"]),
        d["buy_make"], d["buy_model"], d["state"],
    )
    buy_pred = float(purchase_model.predict(buy_X)[0])
    p_std    = meta["purchase_std"]
    buy_lo   = max(1_500, buy_pred - p_std * 0.8)
    buy_hi   = buy_pred + p_std * 0.6
    target   = (buy_lo + buy_hi) / 2

    result = {
        "buy": {
            "pred":       round(buy_pred),
            "lo":         round(buy_lo),
            "hi":         round(buy_hi),
            "open_offer": round(buy_lo * 0.93),
            "target":     round(target),
            "walkaway":   round(buy_hi * 1.02),
        }
    }

    if d.get("dealer_asking"):
        asking = float(d["dealer_asking"])
        pct    = (asking - target) / target * 100
        result["buy"]["dealer"] = {
            "asking":  round(asking),
            "pct":     round(pct, 1),
            "verdict": "high" if pct > 10 else ("slightly_high" if pct > 3 else "fair"),
        }

    if d.get("trade_make"):
        trade_X   = build_features(
            int(d["trade_year"]), int(d["trade_odo"]),
            d["trade_cond"], d["trade_drive"], d["trade_trans"],
            d["trade_fuel"], d["trade_title"], int(d["trade_cyls"]),
            d["trade_make"], d["trade_model"], d["state"],
        )
        raw       = float(tradein_model.predict(trade_X)[0])
        pred      = raw * (1 - meta["dealer_discount"])
        t_std     = meta["tradein_std"]
        trade_lo  = max(500, pred - t_std * 0.5)
        trade_hi  = pred + t_std * 0.4

        result["trade"] = {
            "pred":     round(pred),
            "lo":       round(trade_lo),
            "hi":       round(trade_hi),
            "ask_for":  round(trade_hi * 1.05),
            "floor":    round(trade_lo * 0.90),
        }
        result["net"] = {
            "fair": round(target - (trade_lo + trade_hi) / 2),
            "best": round(result["buy"]["open_offer"] - result["trade"]["ask_for"]),
        }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
