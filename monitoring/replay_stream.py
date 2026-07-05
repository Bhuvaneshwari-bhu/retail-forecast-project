"""
Phase 4, Step 3: Replay simulated "future" days through the live API.

IMPORTANT ALIGNMENT FIX:
Your original data/streaming/ files (from simulate_stream.py in Phase 1) were
generated using the first 200 rows of the M5 file as a baseline - NOT
necessarily the same 300 top-selling items the model was actually trained on
in feature_engineering.py. If we replayed those files, most requests would
hit "unseen item" and the demo would look broken through no fault of the
model.

This script fixes that by regenerating a fresh simulated stream scoped
EXACTLY to the model's known items (read from model_config.json's
category_maps), with the same drift-injection design as before. This is the
data your monitoring system will actually watch.

WHAT IT DOES:
1. Loads model_config.json to get the exact list of items/stores the model knows
2. Seeds a rolling history buffer per item-store from the tail of
   features_sample.parquet (so day 1 of replay has real lag/rolling values,
   not zeros)
3. Walks forward day by day, injecting drift at configurable points
4. For each item-store-day, builds the exact feature payload the API expects
   and calls POST /predict - the API's own logging (Phase 3) records it
5. Prints a running comparison of predicted vs actual so you can see
   accuracy degrade when drift hits

RUN (default: 30 days, drift injected on day 15):
    python monitoring/replay_stream.py --days 30 --drift_day 15
"""

import os
import json
import argparse
import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import requests

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "features_sample.parquet")
CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")
DRIFT_LOG_PATH = os.path.join(BASE_DIR, "data", "monitoring", "replay_drift_log.csv")

API_URL = "http://localhost:8000/predict"
REPLAY_START_DATE = "2016-06-20"  # picks up right after real M5 data ends
BASE_DIR_LOGS = os.path.join(BASE_DIR, "data", "logs")
TRAINING_LOG_PATH = os.path.join(BASE_DIR_LOGS, "streamed_training_data.jsonl")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def seed_history(config, n_items, n_stores):
    """Build a per (item, store) rolling deque of the last 28 real days,
    seeded from the tail of features_sample.parquet so day 1 of replay has
    realistic lag/rolling values instead of zeros."""
    df = pd.read_parquet(
        FEATURES_PATH,
        columns=["item_id", "store_id", "date", "lag_1", "sell_price", "is_simulated"],
    )
    real_df = df[df["is_simulated"] == 0].sort_values("date")

    items = config["category_maps"]["item_id"][:n_items]
    stores = config["category_maps"]["store_id"][:n_stores]

    history = {}
    last_price = {}
    for item in items:
        for store in stores:
            sub = real_df[(real_df["item_id"] == item) & (real_df["store_id"] == store)]
            if sub.empty:
                continue
            recent = sub["lag_1"].tail(28).tolist()
            if len(recent) < 28:
                recent = [0.0] * (28 - len(recent)) + recent
            history[(item, store)] = deque(recent, maxlen=28)
            price_series = sub["sell_price"].dropna()
            last_price[(item, store)] = float(price_series.iloc[-1]) if len(price_series) else None

    print(f"Seeded history for {len(history)} item-store pairs")
    return history, last_price, items, stores


def build_features(item, store, state, date, history_deque, price, price_change):
    weekday = date.weekday()
    values = list(history_deque)
    lag_1 = values[-1]
    lag_7 = values[-7]
    lag_28 = values[0]
    rolling_mean_7 = float(np.mean(values[-7:]))
    rolling_mean_28 = float(np.mean(values))
    rolling_std_7 = float(np.std(values[-7:]))

    return {
        "item_id": item,
        "store_id": store,
        "state_id": state,
        "lag_1": float(lag_1),
        "lag_7": float(lag_7),
        "lag_28": float(lag_28),
        "rolling_mean_7": rolling_mean_7,
        "rolling_mean_28": rolling_mean_28,
        "rolling_std_7": rolling_std_7,
        "sell_price": price,
        "price_change": price_change,
        "temp_max_c": None,
        "temp_min_c": None,
        "precipitation_mm": None,
        "is_holiday": 0,
        "days_to_next_holiday": None,
        "weekday": weekday,
        "month": date.month,
        "year": date.year,
        "is_weekend": int(weekday in (5, 6)),
    }


def simulate_actual(base_level, day_index, drift_day, drift_type, magnitude=0.6):
    """Mirrors the drift injection logic from src/simulate_stream.py, scoped
    to the model's known items this time."""
    level = base_level
    extra_std = 0.0
    if drift_day is not None and day_index >= drift_day:
        if drift_type == "level_shift":
            level = base_level * (1 + magnitude)
        elif drift_type == "trend":
            level = base_level * (1 + magnitude * ((day_index - drift_day) / 30))
        elif drift_type == "variance":
            extra_std = base_level * magnitude

    noise_std = max(level * 0.25, 0.3) + extra_std
    return max(0, np.random.normal(level, noise_std))


def _log_training_record(payload, actual, date):
    """Log features + TRUE outcome, so retrain.py has labeled data to learn
    from once drift makes the old model stale. In a real production system
    this would come from waiting for actual sales to be recorded; here we
    have the advantage of knowing the simulated ground truth immediately."""
    record = dict(payload)
    record["actual_units_sold"] = float(actual)
    record["date"] = date.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(TRAINING_LOG_PATH), exist_ok=True)
    with open(TRAINING_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def main(days, n_items, n_stores, drift_day, drift_type):
    config = load_config()
    history, last_price, items, stores = seed_history(config, n_items, n_stores)

    state_map = {"CA_1": "CA", "CA_2": "CA", "CA_3": "CA", "CA_4": "CA",
                 "TX_1": "TX", "TX_2": "TX", "TX_3": "TX",
                 "WI_1": "WI", "WI_2": "WI", "WI_3": "WI"}

    dates = pd.date_range(REPLAY_START_DATE, periods=days, freq="D")
    drift_events = []
    total_calls, total_errors = 0, 0

    for day_index, date in enumerate(dates):
        day_errors = []
        for (item, store), hist in history.items():
            state = state_map.get(store, "CA")
            base_level = max(np.mean(list(hist)[-7:]), 0.5)
            actual = simulate_actual(base_level, day_index, drift_day, drift_type)

            price = last_price.get((item, store))
            price_change = 0.0

            payload = build_features(item, store, state, date, hist, price, price_change)

            try:
                resp = requests.post(API_URL, json=payload, timeout=5)
                resp.raise_for_status()
                pred = resp.json()["predicted_units"]
                day_errors.append(abs(pred - actual))
                total_calls += 1
                _log_training_record(payload, actual, date)
            except requests.exceptions.RequestException as e:
                total_errors += 1
                if total_errors <= 3:
                    print(f"  WARNING: request failed for {item}/{store}: {e}")

            hist.append(actual)  # update rolling history with the new "actual"

        if day_index == drift_day:
            drift_events.append({"date": date.strftime("%Y-%m-%d"), "day_index": day_index,
                                  "drift_type": drift_type})
            print(f"\n>>> DRIFT INJECTED on {date.strftime('%Y-%m-%d')} ({drift_type}) <<<\n")

        if day_errors:
            print(f"{date.strftime('%Y-%m-%d')}  mean_abs_error={np.mean(day_errors):.2f}  "
                  f"calls={len(day_errors)}")

    if drift_events:
        os.makedirs(os.path.dirname(DRIFT_LOG_PATH), exist_ok=True)
        pd.DataFrame(drift_events).to_csv(DRIFT_LOG_PATH, index=False)
        print(f"\nGround-truth drift log saved to {DRIFT_LOG_PATH}")

    print(f"\nDone. {total_calls} successful predictions, {total_errors} errors.")
    if total_errors > 0 and total_calls == 0:
        print("All requests failed - is the API running? (uvicorn serving.app:app --port 8000)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--n_items", type=int, default=50, help="how many known items to replay (max = model's full list)")
    p.add_argument("--n_stores", type=int, default=3)
    p.add_argument("--drift_day", type=int, default=15)
    p.add_argument("--drift_type", type=str, default="level_shift",
                    choices=["level_shift", "trend", "variance"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.days, args.n_items, args.n_stores, args.drift_day, args.drift_type)
