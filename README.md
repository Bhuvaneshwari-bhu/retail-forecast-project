# Retail Demand Forecasting System with Live Drift Monitoring & Closed-Loop Auto-Retraining

An end-to-end, production-style ML system that forecasts daily retail demand,
serves predictions through a deployed REST API, monitors itself for data
drift, and automatically retrains — with every step verified working, not
just claimed.

**Demo video:** [add your link here]
**Live code:** this repo
**Deployed API (may be torn down after demo recording — see note below):**
`https://6c3hlz6vo9.execute-api.us-east-1.amazonaws.com`

---

## Why this project is different

Most portfolio ML projects stop at "trained a model in a notebook, got a
decent RMSE." This one goes further in four ways:

1. **Blended data sources, not a single static CSV.** Real historical sales
   (M5/Walmart, 5.4 years, 58M rows), real weather (Open-Meteo), real US
   holidays, and a *simulated live stream* engineered to mimic new data
   arriving daily — with deliberately injected drift events and a
   ground-truth log to grade the monitoring system against.
2. **A genuinely deployable service, not a `.pkl` file.** FastAPI with
   input validation, exact categorical feature alignment between training
   and inference, Prometheus metrics, structured prediction logging —
   deployed for real on AWS Lambda + API Gateway, model served from S3,
   predictions logged to DynamoDB.
3. **Closed-loop monitoring, not a static screenshot.** A drift detector
   (Evidently AI + an independent z-score check) that was validated against
   self-injected, known drift events — and caught them.
4. **Closed-loop retraining that actually ran, in the cloud, unattended.**
   A GitHub Actions workflow detected real drift, trained a candidate
   model, compared it against production on held-out data, and committed
   the result back to the repo automatically — no manual trigger, no AWS
   dependency, free forever.

---

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
                    │  Model from S3        │
                    │  Predictions -> Dynamo│
                    │  Public HTTPS URL     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Streamlit Frontend   │
                    │  + Monitoring         │
                    │  Evidently + z-score  │
                    │  drift detection      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  GitHub Actions       │
                    │  Scheduled drift      │
                    │  check + auto-retrain │
                    │  VERIFIED WORKING:    │
                    │  drift caught, model  │
                    │  retrained, committed │
                    │  (free forever)       │
                    └──────────────────────┘
```

---

## Results — verified, not just claimed

| Metric | Result |
|---|---|
| Model | LightGBM, 300 top-selling SKUs, 10 stores, 5+ years history |
| Baseline (naive seasonal) | WAPE 0.623 |
| Trained model | WAPE 0.464 — **25.5% improvement** |
| Serving latency (warm) | ~9-14ms, both locally and on deployed Lambda |
| Drift detection | Correctly flagged an injected level-shift event with **zero false positives** on undisturbed features |
| Auto-retraining | GitHub Actions detected live drift, trained a candidate, compared it against production on held-out drifted data, and committed the outcome — **confirmed running successfully end-to-end** |
| Deployment | Live on AWS Lambda + API Gateway, confirmed via direct invoke, public HTTPS, and DynamoDB scan showing logged predictions |

---

## Tech stack

| Layer | Tools |
|---|---|
| Data | M5 (Kaggle), Open-Meteo API, `holidays` package, custom drift simulator (Faker) |
| ETL / Feature engineering | DuckDB (out-of-core SQL, window functions) |
| Modeling | LightGBM, MLflow (experiment tracking), Google Colab (training compute) |
| Serving | FastAPI, Pydantic, Uvicorn, Mangum (Lambda adapter) |
| Frontend | Streamlit (live prediction form + monitoring dashboard) |
| Monitoring | Evidently AI, custom z-score drift detection, Prometheus |
| Cloud | AWS Lambda, API Gateway, S3, DynamoDB, ECR, IAM (all within free tier) |
| CI/CD & automation | GitHub Actions (scheduled drift check + auto-retrain, free forever) |
| Engineering constraints | Built and debugged entirely on a 4GB-available-RAM laptop — every pipeline stage is memory-safe by construction (chunked processing, out-of-core SQL, streaming parquet writes) |

---

## Project phases (all complete)

1. **Data collection** — real M5 sales/prices/calendar, real weather, real
   US holidays, simulated live stream with injected drift
2. **Data engineering** — memory-safe merge of 58M rows (chunked,
   categorical-dtype, streaming-parquet-write pipeline; fixed multiple
   real OOM failures along the way)
3. **Feature engineering** — DuckDB SQL window functions (lags, rolling
   stats, leakage-safe framing) on a representative 300-item sample
4. **Modeling** — LightGBM on Google Colab, tracked in MLflow, evaluated
   against a naive baseline (25.5% improvement)
5. **Serving** — FastAPI with strict input validation, exact categorical
   code alignment, Prometheus metrics
6. **Monitoring** — Evidently AI + custom z-score drift detection, a replay
   harness validated against self-injected ground truth, a live Streamlit
   dashboard doubling as the prediction frontend
7. **Closed-loop retraining** — candidate model trained on drifted data,
   evaluated against production on held-out drifted data, promoted only if
   genuinely better, zero-downtime swap via `/admin/reload`
8. **Cloud deployment** — same codebase (env-var driven) deployed as a
   Docker container image on AWS Lambda, API Gateway, S3, DynamoDB —
   confirmed working end-to-end
9. **GitHub Actions automation** — scheduled + on-demand drift check and
   retraining, running entirely free, with no ongoing AWS dependency —
   confirmed catching real drift and completing a retrain cycle in
   production

---

## Repo structure

```
retail-forecast-project/
├── .github/workflows/retrain.yml   # Phase 9: scheduled auto-retrain
├── src/                            # Phase 1-3: data + feature engineering
│   ├── download_m5.py
│   ├── fetch_weather.py
│   ├── fetch_holidays.py
│   ├── simulate_stream.py
│   ├── merge_datasets.py
│   ├── verify_data.py
│   └── feature_engineering.py
├── serving/                        # Phase 5, 8: the deployable API
│   ├── app.py                      # runs local OR Lambda via env vars
│   ├── lambda_handler.py
│   ├── requirements-lambda.txt
│   └── sample_request.json
├── monitoring/                     # Phase 6: drift detection + frontend
│   ├── reference_data.py
│   ├── drift_check.py
│   ├── replay_stream.py
│   └── dashboard.py                # Streamlit: prediction form + monitoring
├── retraining/                     # Phase 7: closed-loop retraining
│   └── retrain.py
├── Dockerfile                      # Phase 8: Lambda container image
├── models/                         # trained model + config + version history
├── data/                           # logs, monitoring artifacts (large raw
│                                    # data gitignored, regenerate via src/)
└── requirements.txt
```

---

## Resume bullets

**Strong, general-purpose:**
- Built and deployed an end-to-end retail demand forecasting system —
  real historical sales data (58M rows), live weather/holiday APIs, and a
  custom drift-injection simulator — achieving a **25.5% WAPE improvement**
  over a seasonal-naive baseline with LightGBM.
- Deployed a serverless ML API on AWS (Lambda + API Gateway + S3 +
  DynamoDB), confirmed working end-to-end with ~9-14ms inference latency,
  entirely within AWS's free tier.
- Designed and validated a data drift monitoring system (Evidently AI +
  custom statistical checks) against self-injected ground-truth drift
  events, confirming correct detection with zero false positives.
- Built a fully automated, closed-loop retraining pipeline on GitHub
  Actions — scheduled drift checks trigger candidate model training, which
  is evaluated against production before being promoted, with the entire
  cycle confirmed running successfully unattended in production.

**If the job emphasizes MLOps / production ML:**
- Architected a full ML lifecycle (data -> features -> training -> serving
  -> monitoring -> retraining) with clear separation of concerns: DuckDB
  for out-of-core feature engineering, Colab for training compute, AWS
  Lambda for serverless serving, GitHub Actions for CI/CD-driven retraining.
- Implemented environment-parity deployment — the same FastAPI codebase
  runs locally (file-based logging) and on AWS Lambda (S3 model loading,
  DynamoDB logging) switched purely by environment variables.

**If the job emphasizes data engineering:**
- Debugged and resolved multiple out-of-memory failures in a pandas-based
  ETL pipeline by redesigning it around categorical dtypes, chunked
  per-store processing, and streaming Parquet writes — cutting peak memory
  usage roughly 10x, enabling a 58M-row pipeline to run on a 4GB-RAM laptop.

**One-line summary:**
**Retail Demand Forecasting System** — LightGBM + FastAPI + AWS Lambda +
Evidently AI + GitHub Actions | 25.5% improvement over baseline | Deployed,
drift-monitored, closed-loop auto-retraining confirmed working in
production | [github.com/Bhuvaneshwari-bhu/retail-forecast-project](https://github.com/Bhuvaneshwari-bhu/retail-forecast-project) | [Demo video]

---

## Honest limitations

- Trained on the top 300 items by volume, not the full 3,049-item catalog —
  a deliberate tradeoff for laptop-friendly local development; the DuckDB
  feature pipeline already runs on the full dataset, only the training
  sample was reduced
- The live AWS deployment is torn down after demo recording to stay
  genuinely free indefinitely — the code, GitHub Actions run history, and
  recorded demo are the permanent artifacts
- The z-score drift check uses a fixed recent-window heuristic (most
  recent 500 predictions); a production system would tune this window
  based on actual traffic volume and seasonality
