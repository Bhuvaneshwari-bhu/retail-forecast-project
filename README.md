# Retail Demand Forecasting System with Live Drift Monitoring & Auto-Retraining

An end-to-end, production-style ML system that forecasts daily retail demand,
serves predictions through a REST API, and monitors itself for data drift —
built to run on modest hardware (a 4GB-available-RAM laptop) by design, not
by accident.

## Why this project is different

Most portfolio ML projects stop at "trained a model in a notebook, got a good
RMSE." This one goes further in three ways:

1. **Blended data sources, not a single static CSV.** Real historical sales
   (M5/Walmart, 5.4 years, 58M rows), real weather (Open-Meteo), real US
   holidays, and a *simulated live stream* engineered to mimic new data
   arriving daily — with deliberately injected drift events and a
   ground-truth log to grade the monitoring system against.
2. **A genuinely deployable service**, not just a `.pkl` file — a FastAPI
   app with input validation, correct categorical feature handling,
   Prometheus metrics, and structured prediction logging.
3. **Closed-loop monitoring**, not a static "it works" screenshot — a
   drift detector (Evidently AI + an independent z-score check) validated
   against known, self-injected drift events, feeding a live Streamlit
   dashboard.

## Architecture

```
                    ┌─────────────────────┐
                    │   Data Sources       │
                    │  M5 (real) + Weather │
                    │  + Holidays + Sim.   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  DuckDB ETL           │
                    │  (out-of-core,        │
                    │   58M rows -> 5.6M    │
                    │   feature rows)       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Google Colab         │
                    │  LightGBM + MLflow    │
                    │  (25.5% better than   │
                    │   naive baseline)     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  AWS Lambda + API GW  │
                    │  (Docker container)   │
                    │  Model loaded from S3 │
                    │  Predictions -> Dynamo│
                    │  Public HTTPS URL     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Monitoring           │
                    │  Evidently + z-score  │
                    │  drift detection      │
                    │  Streamlit dashboard  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  GitHub Actions       │
                    │  Scheduled drift      │
                    │  check + auto-retrain │
                    │  (free forever, no    │
                    │   AWS dependency)     │
                    └──────────────────────┘
```

## Live deployment

Deployed on AWS Lambda + API Gateway (serverless, container image, model loaded
from S3, predictions logged to DynamoDB):

```
GET  https://6c3hlz6vo9.execute-api.us-east-1.amazonaws.com/health
POST https://6c3hlz6vo9.execute-api.us-east-1.amazonaws.com/predict
```

> Note: this deployment is torn down after demo recording to stay within
> free-tier usage indefinitely (see "Honest limitations" below). The demo
> video and code are the permanent artifacts; the live URL may not always
> be active.

## Tech stack

| Layer | Tools |
|---|---|
| Data | M5 (Kaggle), Open-Meteo API, `holidays` package, custom drift simulator (Faker) |
| ETL / Feature engineering | DuckDB (out-of-core SQL, window functions) |
| Modeling | LightGBM, MLflow (experiment tracking), Google Colab (training compute) |
| Serving | FastAPI, Pydantic, Uvicorn |
| Monitoring | Evidently AI, custom z-score drift detection, Prometheus, Streamlit |
| Engineering constraints | Built and debugged entirely on a 4GB-available-RAM laptop — every pipeline stage is memory-safe by construction (chunked processing, out-of-core SQL, streaming parquet writes) |

## Results

- **Model**: LightGBM demand forecaster trained on 300 top-selling SKUs
  across 10 stores, 5+ years of history
- **Baseline**: naive seasonal (predict same weekday, last week) — WAPE 0.623
- **Model**: WAPE 0.464 — a **25.5% improvement** over baseline
- **Serving latency**: ~9-14ms per prediction (warm), both locally and on
  deployed Lambda
- **Monitoring validation**: drift detector correctly identified a
  deliberately injected level-shift event (z-score 6.1 vs. threshold 2.0)
  with zero false positives on undisturbed features
- **Deployment**: live on AWS Lambda + API Gateway, model served from S3,
  predictions logged to DynamoDB, confirmed working end-to-end via direct
  invoke and public HTTPS testing

## Project phases

1. **Data collection** — real M5 sales/prices/calendar, real weather
   (Open-Meteo), real US holidays, simulated live stream with injected drift
2. **Data engineering** — memory-safe merge of 58M rows via chunked,
   categorical-dtype, streaming-parquet-write pipeline
3. **Feature engineering** — DuckDB SQL window functions (lags, rolling
   stats, leakage-safe framing) on a representative 300-item sample
4. **Modeling** — LightGBM on Google Colab, tracked in MLflow, evaluated
   against a naive baseline
5. **Serving** — FastAPI with strict input validation, exact categorical
   code alignment between training and inference, Prometheus metrics
6. **Monitoring** — Evidently AI + custom z-score drift detection, a replay
   harness that validates the detector against self-injected ground truth,
   a live Streamlit dashboard
7. **Closed-loop retraining** — candidate model trained on drifted data,
   evaluated against the current production model on held-out drifted data,
   promoted only if genuinely better, zero-downtime swap via `/admin/reload`
8. **Cloud deployment** — same codebase (env-var driven) deployed as a
   Docker container image on AWS Lambda, served through API Gateway, model
   loaded from S3, predictions logged to DynamoDB
9. *(next)* GitHub Actions — scheduled drift check + auto-retrain, free
   forever, no dependency on AWS staying deployed

## Repo structure

```
retail-forecast-project/
├── src/                    # Phase 1-2: data collection, merging, feature engineering
│   ├── download_m5.py
│   ├── fetch_weather.py
│   ├── fetch_holidays.py
│   ├── simulate_stream.py
│   ├── merge_datasets.py
│   ├── verify_data.py
│   └── feature_engineering.py
├── serving/                 # Phase 5, 8: the deployable API (local + Lambda)
│   ├── app.py
│   ├── lambda_handler.py
│   ├── requirements-lambda.txt
│   └── sample_request.json
├── monitoring/               # Phase 6: drift detection + dashboard
│   ├── reference_data.py
│   ├── drift_check.py
│   ├── replay_stream.py
│   └── dashboard.py
├── retraining/               # Phase 7: closed-loop retraining
│   └── retrain.py
├── Dockerfile                # Phase 8: Lambda container image
├── models/                   # trained model + config (from Colab)
├── data/                      # raw, processed, streaming, logs, monitoring
└── requirements.txt
```

## Honest limitations / next steps

- Trained on the top 300 items by volume, not the full 3,049-item catalog —
  a deliberate tradeoff for laptop-friendly local development, with a clear
  scale-up path (the DuckDB feature pipeline already runs on the full
  dataset; only the Colab training sample was reduced)
- Weather/holiday features aren't available for the simulated future dates
  during replay (real forecasts wouldn't have this limitation with a live
  weather API key)
- The live AWS deployment is torn down after demo recording to stay
  genuinely free indefinitely — the code, architecture, and recorded demo
  are the permanent artifacts
- GitHub Actions auto-retraining (scheduled, no ongoing AWS dependency) is
  the final planned phase
