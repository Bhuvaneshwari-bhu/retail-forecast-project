"""
Lambda entry point. AWS Lambda needs a `handler(event, context)` function -
Mangum is the adapter library that translates API Gateway's event format
into requests your existing FastAPI app already knows how to handle.

Nothing about app.py's actual logic changes for Lambda - only the
environment variables set in the Lambda configuration (MODEL_SOURCE=s3,
LOG_BACKEND=dynamodb, etc.) change its behavior. This is the same code
that runs locally with `uvicorn serving.app:app`.
"""

from mangum import Mangum
from serving.app import app

handler = Mangum(app)
