"""
Phase 3: FastAPI service that serves the LightGBM demand forecasting model.

Endpoints:
  GET  /health   -> liveness/readiness check
  POST /predict  -> single prediction
  GET  /metrics  -> Prometheus-format metrics (request count, latency) -
                     this is what your monitoring dashboard will scrape later

WHY THE CATEGORY HANDLING MATTERS:
LightGBM stores categorical features as integer codes internally, not the
original strings. If we don't reconstruct the EXACT SAME category-to-code
mapping used during training, the model will silently mispredict (it'll
still return a number, just a wrong one - the dangerous kind of bug).
That's why model_config.json carries "category_maps" and why every request
re-applies those exact categories before calling the model.

RUN LOCALLY:
    uvicorn serving.app:app --reload --port 8000

RUN ON LAMBDA:
    Same code, different environment variables:
      MODEL_SOURCE=s3
      LOG_BACKEND=dynamodb
      S3_BUCKET=<your-bucket>
      DYNAMODB_TABLE=<your-table>
    See serving/lambda_handler.py for the Lambda entry point.

TEST:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
"""

import os
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forecast-api")

# --- environment-driven config: same code runs locally (Phase 3/4) and on
# Lambda (Phase 6) - only these env vars change between the two. ---
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "local")   # "local" or "s3"
LOG_BACKEND = os.environ.get("LOG_BACKEND", "local")      # "local" or "dynamodb"
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_MODEL_KEY = os.environ.get("S3_MODEL_KEY", "models/lightgbm_model.txt")
S3_CONFIG_KEY = os.environ.get("S3_CONFIG_KEY", "models/model_config.json")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "retail-forecast-predictions")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
# Lambda's filesystem is read-only except /tmp - so on Lambda, the model
# gets downloaded from S3 into /tmp on cold start instead of read from disk.
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_model.txt")
LOCAL_CONFIG_PATH = os.path.join(BASE_DIR, "models", "model_config.json")
TMP_MODEL_PATH = "/tmp/lightgbm_model.txt"
TMP_CONFIG_PATH = "/tmp/model_config.json"
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
PREDICTIONS_LOG = os.path.join(LOG_DIR, "predictions.jsonl")

app = FastAPI(
    title="Retail Demand Forecast API",
    description="Serves next-day unit demand predictions per item/store",
    version="1.0.0",
)

# Prometheus metrics at GET /metrics - request count, latency histograms, etc.
Instrumentator().instrument(app).expose(app)

# --- globals populated at startup ---
_model: Optional[lgb.Booster] = None
_config: Optional[dict] = None


class PredictionRequest(BaseModel):
    item_id: str = Field(..., examples=["FOODS_1_004"])
    store_id: str = Field(..., examples=["CA_1"])
    state_id: str = Field(..., examples=["CA"])

    lag_1: float = Field(..., description="units sold 1 day ago")
    lag_7: float = Field(..., description="units sold 7 days ago")
    lag_28: float = Field(..., description="units sold 28 days ago")
    rolling_mean_7: float = Field(..., description="avg units sold, past 7 days (excl. today)")
    rolling_mean_28: float = Field(..., description="avg units sold, past 28 days (excl. today)")
    rolling_std_7: Optional[float] = None

    sell_price: Optional[float] = None
    price_change: Optional[float] = None

    temp_max_c: Optional[float] = None
    temp_min_c: Optional[float] = None
    precipitation_mm: Optional[float] = None

    is_holiday: int = 0
    days_to_next_holiday: Optional[float] = None

    weekday: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    year: int
    is_weekend: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    predicted_units: float
    item_id: str
    store_id: str
    model_version: str
    latency_ms: float
    timestamp: str


@app.on_event("startup")
def load_model():
    global _model, _config

    if MODEL_SOURCE == "s3":
        import boto3
        if not S3_BUCKET:
            raise RuntimeError("MODEL_SOURCE=s3 but S3_BUCKET env var is not set")

        logger.info(f"Downloading model from s3://{S3_BUCKET}/{S3_MODEL_KEY}")
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, TMP_MODEL_PATH)
        s3.download_file(S3_BUCKET, S3_CONFIG_KEY, TMP_CONFIG_PATH)
        model_path, config_path = TMP_MODEL_PATH, TMP_CONFIG_PATH
    else:
        model_path, config_path = LOCAL_MODEL_PATH, LOCAL_CONFIG_PATH
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model file not found at {model_path}. "
                "Download lightgbm_model.txt from Colab/Drive into the models/ folder."
            )
        if not os.path.exists(config_path):
            raise RuntimeError(
                f"Config file not found at {config_path}. "
                "Download model_config.json from Colab/Drive into the models/ folder."
            )

    _model = lgb.Booster(model_file=model_path)
    with open(config_path) as f:
        _config = json.load(f)

    if LOG_BACKEND == "local":
        os.makedirs(LOG_DIR, exist_ok=True)

    logger.info(f"Model loaded from {model_path} (source={MODEL_SOURCE})")
    logger.info(f"Feature columns: {_config['feature_cols']}")


@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "model_loaded": _model is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/admin/reload")
def reload_model():
    """Re-reads the model file from disk without restarting the process.
    Called by retraining/retrain.py after a new model is promoted, so a
    drift-triggered retrain can take effect with zero downtime."""
    try:
        load_model()
        return {
            "status": "reloaded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


def _build_feature_row(payload: PredictionRequest) -> pd.DataFrame:
    """Convert the request into a single-row DataFrame with the exact same
    column order and categorical encoding used during training."""
    row = payload.model_dump()
    df = pd.DataFrame([row])

    # numeric columns: force real float64 dtype so that None -> NaN (a type
    # LightGBM handles natively as "missing"), instead of staying as pandas
    # "object" dtype (which LightGBM rejects outright). This matters because
    # a lone None value in a column makes pandas infer "object" dtype rather
    # than "float64" by default.
    numeric_cols = [c for c in _config["feature_cols"] if c not in _config["cat_features"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # re-apply the EXACT category sets learned at training time, in the same
    # order, so LightGBM's internal codes match what it learned on.
    for col in _config["cat_features"]:
        categories = _config["category_maps"][col]
        cat_dtype = pd.CategoricalDtype(categories=categories)
        original_value = df[col].iloc[0]
        df[col] = df[col].where(df[col].isin(categories), other=None).astype(cat_dtype)
        if df[col].isna().any():
            logger.warning(
                f"Unseen category for '{col}': '{original_value}' not in "
                "training categories - LightGBM will treat it as missing."
            )

    df = df[_config["feature_cols"]]
    return df


def _log_prediction(payload: PredictionRequest, prediction: float, latency_ms: float):
    """Log every prediction for later drift monitoring. Two backends:
      - local: append to a JSONL file (Phase 3/4 development)
      - dynamodb: put_item into DynamoDB (Lambda deployment)

    Note on DynamoDB: boto3's resource API rejects native Python float
    (it requires Decimal). Rather than recursively convert every nested
    float in the input payload, we store the input as a JSON string - this
    sidesteps the float/Decimal issue entirely and is trivial to parse back
    in drift_check.py / retrain.py.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    if LOG_BACKEND == "dynamodb":
        try:
            import boto3
            table = boto3.resource("dynamodb").Table(DYNAMODB_TABLE)
            table.put_item(Item={
                "prediction_id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "item_id": payload.item_id,
                "store_id": payload.store_id,
                "prediction": str(round(prediction, 4)),
                "latency_ms": str(round(latency_ms, 2)),
                "input_json": json.dumps(payload.model_dump()),
            })
        except Exception as e:
            # never let logging failures break a prediction response
            logger.error(f"Failed to log prediction to DynamoDB: {e}")
    else:
        record = {
            "timestamp": timestamp,
            "item_id": payload.item_id,
            "store_id": payload.store_id,
            "prediction": prediction,
            "latency_ms": latency_ms,
            "input": payload.model_dump(),
        }
        try:
            with open(PREDICTIONS_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to log prediction locally: {e}")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    if _model is None or _config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        features_df = _build_feature_row(payload)
        raw_pred = _model.predict(features_df)[0]
        prediction = max(0.0, float(raw_pred))  # units sold can't be negative
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    latency_ms = (time.perf_counter() - start) * 1000
    _log_prediction(payload, prediction, latency_ms)

    return PredictionResponse(
        predicted_units=round(prediction, 2),
        item_id=payload.item_id,
        store_id=payload.store_id,
        model_version=os.environ.get("MODEL_VERSION", "v1"),
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
