"""
Phase 4, Step 1: Build the "reference" dataset for drift detection.

WHY: Drift detection always needs a baseline to compare against. The
reference is a sample of the feature distributions the model was actually
trained on. Everything arriving later (via /predict) gets compared against
this snapshot to answer: "does live traffic still look like what the model
learned from?"

We sample rather than use the full 5.6M-row feature table because:
  1. Evidently's statistical tests don't need millions of rows to be valid
  2. A stratified random sample keeps the comparison fast without losing
     the shape of the distribution

RUN:
    python monitoring/reference_data.py
"""

import os
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MONITORING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "monitoring")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features_sample.parquet")
OUT_PATH = os.path.join(MONITORING_DIR, "reference_data.parquet")

# same feature set the model was trained on (see feature_engineering.py / Colab Cell 3)
FEATURE_COLS = [
    "lag_1", "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28",
    "rolling_std_7", "sell_price", "price_change", "temp_max_c",
    "temp_min_c", "precipitation_mm", "is_holiday", "days_to_next_holiday",
    "weekday", "month", "year", "is_weekend", "item_id", "store_id", "state_id",
]

SAMPLE_SIZE = 20000


def main():
    os.makedirs(MONITORING_DIR, exist_ok=True)

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH}. Run src/feature_engineering.py first."
        )

    df = pd.read_parquet(FEATURES_PATH, columns=FEATURE_COLS + ["is_simulated"])
    real_df = df[df["is_simulated"] == 0].drop(columns=["is_simulated"])

    n = min(SAMPLE_SIZE, len(real_df))
    reference = real_df.sample(n=n, random_state=42)

    reference.to_parquet(OUT_PATH, index=False)
    print(f"Saved reference dataset: {len(reference):,} rows to {OUT_PATH}")
    print(reference.describe(include="all").T[["count", "mean", "std"]].head(10))


if __name__ == "__main__":
    main()
