import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.config import MODELS_DIR
from src.features import add_features, feature_matrix

app = Flask(__name__)

purchase_model = joblib.load(MODELS_DIR / "purchase_model.pkl")
tradein_model  = joblib.load(MODELS_DIR / "tradein_model.pkl")
with open(MODELS_DIR / "metadata.json") as f:
    meta = json.load(f)


def build_features(year, odo, cond, drive, trans, fuel, title, cyls, make, model_name, state):
    car = pd.DataFrame([{
        "year": year, "odometer": odo, "condition": cond, "drive": drive,
        "transmission": trans, "fuel": fuel, "title_status": title, "cylinders": cyls,
        "manufacturer": make, "model": model_name, "state": state,
    }])
    return feature_matrix(add_features(car, meta))


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
        t_log_std = meta["tradein_log_std"]
        trade_lo  = max(500, pred * np.exp(-0.5 * t_log_std))
        trade_hi  = pred * np.exp(0.4 * t_log_std)

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
