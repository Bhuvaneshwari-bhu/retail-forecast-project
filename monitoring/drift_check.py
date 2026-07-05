"""
Phase 4, Step 2: Drift detection.

TWO LAYERS, ON PURPOSE:
  1. Evidently AI report (HTML) - the rich, human-readable diagnostic view.
     Great for exploring WHICH features drifted and HOW.
  2. A simple custom z-score check (JSON) - the machine-readable signal that
     your dashboard and later auto-retraining trigger actually act on.

Why not rely on Evidently's dict output alone for automation? Evidently's
internal JSON schema has changed across versions and can be brittle to parse
programmatically. Keeping a small, transparent, versioned check of our own
means the "should we retrain?" decision doesn't depend on a third-party
library's internal structure - a pattern real MLOps teams use often
(vendor tooling for visualization, in-house logic for automation).

RUN:
    python monitoring/drift_check.py
"""

import os
import json
import glob
from datetime import datetime, timezone

import pandas as pd
import numpy as np

MONITORING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "monitoring")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "logs")
REFERENCE_PATH = os.path.join(MONITORING_DIR, "reference_data.parquet")
PREDICTIONS_LOG = os.path.join(LOGS_DIR, "predictions.jsonl")
REPORTS_DIR = os.path.join(MONITORING_DIR, "reports")

NUMERIC_COLS_TO_CHECK = [
    "lag_1", "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28",
    "rolling_std_7", "sell_price", "temp_max_c", "temp_min_c",
    "precipitation_mm",
]
Z_SCORE_THRESHOLD = 2.0  # flag a column if |current_mean - ref_mean| > 2 * ref_std


def load_current_data():
    if not os.path.exists(PREDICTIONS_LOG):
        return pd.DataFrame()

    records = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row = dict(rec["input"])
            row["prediction"] = rec["prediction"]
            row["timestamp"] = rec["timestamp"]
            records.append(row)

    return pd.DataFrame(records)


def compute_zscore_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    flagged = {}
    for col in NUMERIC_COLS_TO_CHECK:
        if col not in current.columns or current[col].dropna().empty:
            continue
        ref_mean = reference[col].mean()
        ref_std = reference[col].std() or 1e-6  # avoid divide-by-zero
        cur_mean = current[col].mean()
        z = abs(cur_mean - ref_mean) / ref_std
        flagged[col] = {
            "reference_mean": round(float(ref_mean), 4),
            "current_mean": round(float(cur_mean), 4),
            "z_score": round(float(z), 4),
            "drift_flagged": bool(z > Z_SCORE_THRESHOLD),
        }
    return flagged


def run_evidently_report(reference: pd.DataFrame, current: pd.DataFrame) -> str | None:
    """Generates the human-readable HTML report. Returns the file path, or
    None if Evidently isn't installed / fails - the z-score check above
    still works independently either way."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        print("Evidently not installed - skipping HTML report (pip install evidently)")
        return None

    os.makedirs(REPORTS_DIR, exist_ok=True)
    shared_cols = [c for c in reference.columns if c in current.columns]

    try:
        report = Report([DataDriftPreset()])
        my_eval = report.run(current[shared_cols], reference[shared_cols])

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = os.path.join(REPORTS_DIR, f"drift_{timestamp}.html")
        my_eval.save_html(report_path)

        latest_path = os.path.join(MONITORING_DIR, "latest_drift_report.html")
        my_eval.save_html(latest_path)

        print(f"Evidently HTML report saved: {report_path}")
        return report_path
    except Exception as e:
        print(f"Evidently report generation failed (non-fatal): {e}")
        return None


def main():
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(
            f"Missing {REFERENCE_PATH}. Run monitoring/reference_data.py first."
        )

    reference = pd.read_parquet(REFERENCE_PATH)
    current = load_current_data()

    if current.empty:
        print(
            "No predictions logged yet in data/logs/predictions.jsonl. "
            "Call POST /predict a few times (or run monitoring/replay_stream.py) "
            "before checking drift."
        )
        return

    print(f"Reference rows: {len(reference):,} | Current rows: {len(current):,}")

    zscore_results = compute_zscore_drift(reference, current)
    any_drift = any(v["drift_flagged"] for v in zscore_results.values())

    html_report_path = run_evidently_report(reference, current)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_reference_rows": len(reference),
        "n_current_rows": len(current),
        "any_drift_detected": any_drift,
        "z_score_threshold": Z_SCORE_THRESHOLD,
        "columns": zscore_results,
        "html_report_path": html_report_path,
    }

    summary_path = os.path.join(MONITORING_DIR, "drift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDrift summary saved to {summary_path}")
    print(f"Overall drift detected: {any_drift}")
    for col, res in zscore_results.items():
        flag = "DRIFT" if res["drift_flagged"] else "ok"
        print(f"  [{flag:5s}] {col:20s} z={res['z_score']:.2f} "
              f"(ref={res['reference_mean']:.2f}, cur={res['current_mean']:.2f})")


if __name__ == "__main__":
    main()
