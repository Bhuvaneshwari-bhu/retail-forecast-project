# Lambda container image for the forecast API.
#
# WHY A CONTAINER IMAGE INSTEAD OF A PLAIN ZIP:
# Lambda's zip-based deployment caps out at 250MB unzipped. pandas + LightGBM
# + fastapi + their transitive dependencies exceed that comfortably. Container
# images support up to 10GB, so we use one here - a common real-world reason
# ML teams choose container-based Lambda deployment over zip.
#
# WHY THE MODEL ISN'T BAKED INTO THIS IMAGE:
# app.py downloads the model from S3 at cold start (MODEL_SOURCE=s3). This
# means updating the model (e.g. after a drift-triggered retrain) doesn't
# require rebuilding/redeploying this container - just upload the new file
# to S3. Decoupling "code" from "model artifact" is standard MLOps practice.

FROM public.ecr.aws/lambda/python:3.12

# gcc/gcc-c++ in case any dependency needs to compile from source on this
# platform - harmless if unused, saves a confusing build failure if needed.
RUN dnf install -y gcc gcc-c++ && dnf clean all

COPY serving/requirements-lambda.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY serving/__init__.py ${LAMBDA_TASK_ROOT}/serving/__init__.py
COPY serving/app.py ${LAMBDA_TASK_ROOT}/serving/app.py
COPY serving/lambda_handler.py ${LAMBDA_TASK_ROOT}/serving/lambda_handler.py

CMD ["serving.lambda_handler.handler"]
