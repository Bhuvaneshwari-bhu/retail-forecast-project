"""
Phase 5: Closed-loop retraining.

TRIGGERED BY: drift_check.py, when any_drift_detected == True (see the
--auto_retrain flag there).

WHAT IT DOES (and why each step exists):
1. Loads the CURRENT production model, so we have something to compare against
2. Builds a training set = original reference data + newly streamed labeled
   data (data/logs/streamed_training_data.jsonl) - the drifted period the
   old model struggled with
3. Trains a candidate model on this combined set
4. Evaluates BOTH old and new models on the SAME held-out slice of the new
   (drifted) data - this is the only fair comparison: does the candidate
   actually handle the new regime better?
5. Promotes the candidate ONLY if it's actually better. Never blindly swaps
   in a new model - a bad automatic retrain is worse than a stale model.
6. Logs every attempt (triggered/promoted/rejected) with before/after
   metrics to data/logs/retrain_events.jsonl - this log is your audit trail
7. If promoted, calls the API's /admin/reload endpoint for a zero-downtime
   model swap

RUN:
    python retraining/retrain.py --reason "drift detected in lag_1"
"""

import os
import json
import glob
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb
import requests

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VERSIONS_DIR = os.path.join(MODELS_DIR, "versions")
LOGS_DIR = os.path.join(BASE_DIR, "data", "logs")
CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")
CURRENT_MODEL_PATH = os.path.join(MODELS_DIR, "lightgbm_model.txt")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "monitoring", "reference_data.parquet")
STREAMED_TRAINING_LOG = os.path.join(LOGS_DIR, "streamed_training_data.jsonl")
RETRAIN_EVENTS_LOG = os.path.join(LOGS_DIR, "retrain_events.jsonl")

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}


def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1e-6)


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_streamed_data(config):
    if not os.path.exists(STREAMED_TRAINING_LOG):
        return pd.DataFrame()
    rows = []
    with open(STREAMED_TRAINING_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])

    # same fix as serving/app.py: None values (e.g. missing weather/holiday
    # fields during replay) make pandas infer "object" dtype instead of
    # float, which LightGBM rejects outright.
    numeric_cols = [c for c in config["feature_cols"] if c not in config["cat_features"]]
    numeric_cols += ["actual_units_sold"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in config["cat_features"]:
        df[col] = df[col].astype("category")
    return df


def load_reference_data(config, n_sample=20000):
    """Loads the small reference file also used by drift_check.py - it's
    already pre-filtered to real (non-simulated) rows, so no further
    filtering is needed here."""
    real_df = pd.read_parquet(FEATURES_PATH)
    n = min(n_sample, len(real_df))
    sample = real_df.sample(n=n, random_state=42)
    for col in config["cat_features"]:
        sample[col] = sample[col].astype("category")
    return sample


def apply_category_dtypes(df, config):
    for col in config["cat_features"]:
        categories = config["category_maps"][col]
        cat_dtype = pd.CategoricalDtype(categories=categories)
        df[col] = df[col].where(df[col].isin(categories), other=None).astype(cat_dtype)
    return df


def ensure_numeric_dtypes(df, config):
    """Defensive re-coercion after concatenating reference + streamed data -
    belt and suspenders against any dtype surprises from the merge."""
    numeric_cols = [c for c in config["feature_cols"] if c not in config["cat_features"]]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_next_version_number():
    existing = glob.glob(os.path.join(VERSIONS_DIR, "lightgbm_model_v*.txt"))
    if not existing:
        return 2  # v1 is the original Colab-trained model
    versions = [int(os.path.basename(p).split("_v")[1].split(".txt")[0]) for p in existing]
    return max(versions) + 1


def log_event(event: dict):
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(RETRAIN_EVENTS_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")


def main(reason: str, min_new_rows: int, reload_api: bool):
    os.makedirs(VERSIONS_DIR, exist_ok=True)
    config = load_config()
    feature_cols = config["feature_cols"]

    print(f"Retrain triggered. Reason: {reason}")

    streamed_df = load_streamed_data(config)
    if len(streamed_df) < min_new_rows:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "decision": "skipped",
            "why": f"only {len(streamed_df)} new labeled rows, need >= {min_new_rows}",
        }
        log_event(event)
        print(f"SKIPPED: not enough new data yet ({len(streamed_df)} rows).")
        return

    print(f"Loaded {len(streamed_df):,} newly streamed labeled rows")

    reference_df = load_reference_data(config)
    print(f"Loaded {len(reference_df):,} reference (original training) rows")

    # combined training set: old knowledge + new (drifted) reality
    streamed_df = apply_category_dtypes(streamed_df, config)
    reference_df = apply_category_dtypes(reference_df, config)

    combined_X = pd.concat([reference_df[feature_cols], streamed_df[feature_cols]], ignore_index=True)
    combined_y = pd.concat(
        [reference_df["units_sold"], streamed_df["actual_units_sold"]], ignore_index=True
    )

    # validation = a held-out slice of the NEW streamed data only. This is
    # the fair test: does the candidate handle the drifted regime better?
    val_frac = 0.3
    n_val = max(int(len(streamed_df) * val_frac), 1)
    streamed_shuffled = streamed_df.sample(frac=1.0, random_state=42)
    val_df = streamed_shuffled.iloc[:n_val]
    train_streamed_df = streamed_shuffled.iloc[n_val:]

    train_X = pd.concat([reference_df[feature_cols], train_streamed_df[feature_cols]], ignore_index=True)
    train_y = pd.concat(
        [reference_df["units_sold"], train_streamed_df["actual_units_sold"]], ignore_index=True
    )
    val_X = val_df[feature_cols]
    val_y = val_df["actual_units_sold"].values

    train_X = ensure_numeric_dtypes(train_X, config)
    val_X = ensure_numeric_dtypes(val_X, config)
    # re-apply category dtype after concat, since pd.concat can silently
    # upcast a category column back to object if categories differ slightly
    train_X = apply_category_dtypes(train_X, config)
    val_X = apply_category_dtypes(val_X, config)

    print(f"Training candidate model on {len(train_X):,} rows, "
          f"validating on {len(val_X):,} held-out drifted rows...")

    train_set = lgb.Dataset(train_X, label=train_y, categorical_feature=config["cat_features"])
    candidate = lgb.train(LGB_PARAMS, train_set, num_boost_round=300)

    candidate_preds = np.clip(candidate.predict(val_X), 0, None)
    candidate_wape = wape(val_y, candidate_preds)

    old_model = lgb.Booster(model_file=CURRENT_MODEL_PATH)
    old_preds = np.clip(old_model.predict(val_X), 0, None)
    old_wape = wape(val_y, old_preds)

    print(f"\nOld model WAPE on drifted validation data: {old_wape:.4f}")
    print(f"Candidate model WAPE on same data:          {candidate_wape:.4f}")

    promoted = candidate_wape < old_wape
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "n_new_rows": len(streamed_df),
        "n_val_rows": len(val_X),
        "old_model_wape": round(float(old_wape), 4),
        "candidate_wape": round(float(candidate_wape), 4),
        "decision": "promoted" if promoted else "rejected",
    }

    if promoted:
        version = get_next_version_number()
        versioned_path = os.path.join(VERSIONS_DIR, f"lightgbm_model_v{version}.txt")
        candidate.save_model(versioned_path)
        candidate.save_model(CURRENT_MODEL_PATH)  # promote to production
        event["promoted_version"] = version
        event["model_path"] = versioned_path
        print(f"\nPROMOTED: candidate is better ({candidate_wape:.4f} < {old_wape:.4f}). "
              f"Saved as v{version} and promoted to production.")

        if reload_api:
            try:
                resp = requests.post("http://localhost:8000/admin/reload", timeout=5)
                resp.raise_for_status()
                print("API reloaded the new model with zero downtime.")
                event["api_reloaded"] = True
            except requests.exceptions.RequestException as e:
                print(f"WARNING: model promoted but API reload failed: {e}")
                print("Restart the API manually to pick up the new model.")
                event["api_reloaded"] = False
    else:
        print(f"\nREJECTED: candidate is not better ({candidate_wape:.4f} >= {old_wape:.4f}). "
              "Keeping current production model.")

    log_event(event)
    print(f"\nEvent logged to {RETRAIN_EVENTS_LOG}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reason", type=str, default="manual trigger")
    p.add_argument("--min_new_rows", type=int, default=100,
                    help="minimum newly streamed labeled rows required before retraining")
    p.add_argument("--no_reload", action="store_true",
                    help="skip calling the API's /admin/reload after promotion")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.reason, args.min_new_rows, reload_api=not args.no_reload)
