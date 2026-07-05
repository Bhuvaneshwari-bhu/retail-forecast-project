"""
Phase 4, Step 4: Live monitoring dashboard.

Shows:
  - Recent prediction volume + latency over time
  - A button to run drift_check.py on demand, with results displayed inline
  - The embedded Evidently HTML report (if generated)
  - A comparison against your ground-truth drift log, so you can visually
    confirm the detector caught what you actually injected

RUN:
    streamlit run monitoring/dashboard.py
"""

import os
import json
import subprocess
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PREDICTIONS_LOG = os.path.join(BASE_DIR, "data", "logs", "predictions.jsonl")
DRIFT_SUMMARY_PATH = os.path.join(BASE_DIR, "data", "monitoring", "drift_summary.json")
LATEST_HTML_REPORT = os.path.join(BASE_DIR, "data", "monitoring", "latest_drift_report.html")
GROUND_TRUTH_DRIFT_LOG = os.path.join(BASE_DIR, "data", "monitoring", "replay_drift_log.csv")

st.set_page_config(page_title="Retail Forecast Monitoring", layout="wide")
st.title("📈 Retail Demand Forecast — Live Dashboard")

# ============================================================
# FRONTEND: make a real prediction against the live API
# ============================================================
st.subheader("🔮 Make a Prediction")
st.caption(
    "This calls the actual deployed API in real time - point it at your "
    "local server (http://localhost:8000) or your live AWS Lambda URL."
)

api_url = st.text_input("API base URL", value="http://localhost:8000")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        item_id = st.text_input("Item ID", value="FOODS_1_004")
        store_id = st.text_input("Store ID", value="CA_1")
        state_id = st.text_input("State ID", value="CA")
        weekday = st.number_input("Weekday (0=Mon)", min_value=0, max_value=6, value=2)
    with col2:
        lag_1 = st.number_input("Units sold 1 day ago", value=3.0)
        lag_7 = st.number_input("Units sold 7 days ago", value=5.0)
        lag_28 = st.number_input("Units sold 28 days ago", value=2.0)
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
    with col3:
        rolling_mean_7 = st.number_input("7-day rolling mean", value=4.1)
        rolling_mean_28 = st.number_input("28-day rolling mean", value=3.8)
        sell_price = st.number_input("Sell price", value=2.5)
        year = st.number_input("Year", value=2016)

    submitted = st.form_submit_button("Get Prediction", type="primary")

if submitted:
    payload = {
        "item_id": item_id, "store_id": store_id, "state_id": state_id,
        "lag_1": lag_1, "lag_7": lag_7, "lag_28": lag_28,
        "rolling_mean_7": rolling_mean_7, "rolling_mean_28": rolling_mean_28,
        "rolling_std_7": 1.0, "sell_price": sell_price, "price_change": 0.0,
        "temp_max_c": None, "temp_min_c": None, "precipitation_mm": None,
        "is_holiday": 0, "days_to_next_holiday": None,
        "weekday": int(weekday), "month": int(month), "year": int(year),
        "is_weekend": int(weekday in (5, 6)),
    }
    try:
        import requests
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Predicted units", result["predicted_units"])
        pcol2.metric("Latency (ms)", result["latency_ms"])
        pcol3.metric("Model version", result["model_version"])
    except Exception as e:
        st.error(f"Prediction request failed: {e}")

st.divider()


def load_predictions():
    if not os.path.exists(PREDICTIONS_LOG):
        return pd.DataFrame()
    records = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


df = load_predictions()

if df.empty:
    st.warning(
        "No predictions logged yet. Call POST /predict a few times, or run "
        "`python monitoring/replay_stream.py` to generate live traffic."
    )
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total predictions logged", f"{len(df):,}")
    col2.metric("Avg latency (ms)", f"{df['latency_ms'].mean():.1f}")
    col3.metric("Unique items seen", df["item_id"].nunique())

    st.subheader("Predictions over time")
    chart_df = df.set_index("timestamp")[["prediction"]].rename(
        columns={"prediction": "predicted_units"}
    )
    st.line_chart(chart_df)

    st.subheader("Latency over time")
    st.line_chart(df.set_index("timestamp")[["latency_ms"]])

    st.subheader("Recent predictions")
    st.dataframe(
        df[["timestamp", "item_id", "store_id", "prediction", "latency_ms"]].tail(50),
        use_container_width=True,
    )

st.divider()
st.subheader("🔍 Drift Detection")

if st.button("Run drift check now"):
    with st.spinner("Comparing live traffic against training reference..."):
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "monitoring", "drift_check.py")],
            capture_output=True, text=True,
        )
        st.text(result.stdout)
        if result.returncode != 0:
            st.error(result.stderr)

if os.path.exists(DRIFT_SUMMARY_PATH):
    with open(DRIFT_SUMMARY_PATH) as f:
        summary = json.load(f)

    status_color = "🔴" if summary["any_drift_detected"] else "🟢"
    st.markdown(f"### {status_color} Drift status: "
                f"{'DRIFT DETECTED' if summary['any_drift_detected'] else 'No drift detected'}")
    st.caption(f"Last checked: {summary['timestamp']}  |  "
               f"Reference rows: {summary['n_reference_rows']:,}  |  "
               f"Current rows: {summary['n_current_rows']:,}")

    rows = []
    for col, res in summary["columns"].items():
        rows.append({
            "feature": col,
            "reference_mean": res["reference_mean"],
            "current_mean": res["current_mean"],
            "z_score": res["z_score"],
            "flagged": "🔴 yes" if res["drift_flagged"] else "ok",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No drift check has been run yet. Click the button above.")

if os.path.exists(LATEST_HTML_REPORT):
    st.subheader("Evidently AI detailed report")
    with open(LATEST_HTML_REPORT) as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=800, scrolling=True)

if os.path.exists(GROUND_TRUTH_DRIFT_LOG):
    st.divider()
    st.subheader("✅ Ground-truth drift events (from replay_stream.py)")
    st.caption(
        "These are the drift events you deliberately injected during replay. "
        "Compare their dates against when the drift check above first flagged "
        "an issue - that comparison is your proof the monitoring actually works."
    )
    st.dataframe(pd.read_csv(GROUND_TRUTH_DRIFT_LOG), use_container_width=True)
