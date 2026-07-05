"""
Step 4 (the uniqueness layer): Simulate a "live" daily transaction stream that
picks up right where the real M5 history ends (2016-06-19) and continues
forward. This is what lets you demo a system that reacts to *new* data, and
later lets you demonstrate drift detection and auto-retraining honestly -
because we deliberately inject real drift events into the stream.

WHY THIS MATTERS FOR YOUR RESUME:
Static Kaggle notebooks stop at "trained a model, got RMSE X." This script
gives you an ongoing synthetic feed so you can honestly say "the system
ingests daily data, detects drift, and retrains" - because it actually does,
against data you control and understand.

HOW IT WORKS:
1. If real M5 sales data is present (data/raw/sales_train_validation.csv),
   it learns each item's baseline level + weekly seasonality from real
   history, so the simulated future looks statistically like the real past.
2. If M5 data isn't downloaded yet, it falls back to a reasonable synthetic
   baseline so you can still run and test the full pipeline today.
3. It generates ONE CSV per simulated day (like a daily batch land job would),
   written to data/streaming/YYYY-MM-DD.csv
4. It deliberately injects THREE kinds of drift at points you control:
   - level shift   (sudden demand jump/drop, e.g. a promo or supply shock)
   - trend drift    (gradual upward/downward creep)
   - variance drift (demand becomes noisier / less predictable)
   These are logged to data/streaming/_drift_log.csv so you can later verify
   your monitoring system actually caught them (ground truth to grade your
   own drift detector against - a very strong interview talking point).

RUN (defaults to simulating 60 days forward, 200 items):
    python src/simulate_stream.py --days 60 --n_items 200

RUN with drift injected at day 30 (level shift) and day 45 (variance drift):
    python src/simulate_stream.py --days 60 --n_items 200 \\
        --drift_day 30 --drift_type level_shift \\
        --drift_day2 45 --drift_type2 variance
"""

import os
import argparse
import numpy as np
import pandas as pd
from faker import Faker

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
STREAM_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "streaming")

SIM_START_DATE = "2016-06-20"  # day after real M5 data ends
STORES = ["CA_1", "CA_2", "TX_1", "TX_2", "WI_1"]

fake = Faker()
Faker.seed(42)
np.random.seed(42)


def load_real_baselines(n_items):
    """Learn per-item mean + weekday seasonality from real M5 data if present."""
    sales_path = os.path.join(RAW_DIR, "sales_train_validation.csv")
    if not os.path.exists(sales_path):
        print("NOTE: real M5 data not found -> using synthetic fallback baselines.")
        return None

    df = pd.read_csv(sales_path, nrows=n_items)
    day_cols = [c for c in df.columns if c.startswith("d_")]
    item_ids = df["item_id"] + "_" + df["store_id"]
    values = df[day_cols].values

    baselines = {}
    for i, item_id in enumerate(item_ids):
        series = values[i]
        mean_level = max(series[-90:].mean(), 0.5)  # last 90 real days
        weekday_effect = np.array([
            series[d::7][-13:].mean() / mean_level if mean_level > 0 else 1.0
            for d in range(7)
        ])
        baselines[item_id] = {
            "mean_level": mean_level,
            "weekday_effect": weekday_effect,
            "store_id": df.iloc[i]["store_id"],
            "item_id": df.iloc[i]["item_id"],
        }
    print(f"Learned baselines for {len(baselines)} real items from M5 history.")
    return baselines


def synthetic_fallback_baselines(n_items):
    baselines = {}
    for i in range(n_items):
        store = STORES[i % len(STORES)]
        item_id = f"SYN_ITEM_{i:04d}"
        key = f"{item_id}_{store}"
        baselines[key] = {
            "mean_level": np.random.uniform(1, 20),
            "weekday_effect": np.clip(np.random.normal(1.0, 0.15, 7), 0.5, 1.6),
            "store_id": store,
            "item_id": item_id,
        }
    return baselines


def apply_drift(base_level, day_index, drift_day, drift_type, magnitude=0.6):
    """Return (adjusted_level, extra_noise_std) given a drift event."""
    if drift_day is None or day_index < drift_day:
        return base_level, 0.0

    if drift_type == "level_shift":
        return base_level * (1 + magnitude), 0.0
    elif drift_type == "trend":
        days_since = day_index - drift_day
        return base_level * (1 + magnitude * (days_since / 30)), 0.0
    elif drift_type == "variance":
        return base_level, base_level * magnitude
    return base_level, 0.0


def simulate(days, n_items, drift_configs):
    os.makedirs(STREAM_DIR, exist_ok=True)

    baselines = load_real_baselines(n_items)
    if baselines is None:
        baselines = synthetic_fallback_baselines(n_items)

    dates = pd.date_range(SIM_START_DATE, periods=days, freq="D")
    drift_log_rows = []

    for day_index, date in enumerate(dates):
        rows = []
        weekday = date.weekday()

        for item_key, info in baselines.items():
            level = info["mean_level"] * info["weekday_effect"][weekday]

            extra_noise_std = 0.0
            for cfg in drift_configs:
                level, noise = apply_drift(
                    level, day_index, cfg["day"], cfg["type"], cfg["magnitude"]
                )
                extra_noise_std = max(extra_noise_std, noise)

            noise_std = max(level * 0.25, 0.3) + extra_noise_std
            qty = max(0, int(np.random.normal(level, noise_std)))

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "item_id": info["item_id"],
                "store_id": info["store_id"],
                "units_sold": qty,
                "transaction_id": fake.uuid4(),
            })

        day_df = pd.DataFrame(rows)
        out_path = os.path.join(STREAM_DIR, f"{date.strftime('%Y-%m-%d')}.csv")
        day_df.to_csv(out_path, index=False)

        for cfg in drift_configs:
            if day_index == cfg["day"]:
                drift_log_rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "day_index": day_index,
                    "drift_type": cfg["type"],
                    "magnitude": cfg["magnitude"],
                })

        if day_index % 10 == 0:
            print(f"  simulated {date.strftime('%Y-%m-%d')} ({len(rows)} rows)")

    if drift_log_rows:
        drift_log_path = os.path.join(STREAM_DIR, "_drift_log.csv")
        pd.DataFrame(drift_log_rows).to_csv(drift_log_path, index=False)
        print(f"Drift ground-truth log saved to {drift_log_path}")

    print(f"Done. {days} daily files written to {STREAM_DIR}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--n_items", type=int, default=200)
    p.add_argument("--drift_day", type=int, default=None, help="day index to inject 1st drift")
    p.add_argument("--drift_type", type=str, default="level_shift",
                    choices=["level_shift", "trend", "variance"])
    p.add_argument("--drift_day2", type=int, default=None, help="day index to inject 2nd drift")
    p.add_argument("--drift_type2", type=str, default="variance",
                    choices=["level_shift", "trend", "variance"])
    p.add_argument("--magnitude", type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()
    drift_configs = []
    if args.drift_day is not None:
        drift_configs.append({"day": args.drift_day, "type": args.drift_type, "magnitude": args.magnitude})
    if args.drift_day2 is not None:
        drift_configs.append({"day": args.drift_day2, "type": args.drift_type2, "magnitude": args.magnitude})

    simulate(args.days, args.n_items, drift_configs)


if __name__ == "__main__":
    main()
