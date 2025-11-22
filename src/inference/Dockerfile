# Simple Python Inference Service (No Kafka/Spark)
FROM python:3.10-slim

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    redis==4.5.4 \
    pandas==1.5.3 \
    numpy==1.24.3 \
    scikit-learn==1.2.2 \
    xgboost==1.7.5 \
    joblib==1.2.0 \
    pyyaml==6.0 \
    flask==2.3.0

# Create app directory
WORKDIR /app

# Copy inference code
COPY __init__.py /app/src/inference/
COPY config.py /app/src/inference/
COPY config.yaml /app/src/inference/
COPY schema.py /app/src/inference/
COPY logging_utils.py /app/src/inference/
COPY model_loader.py /app/src/inference/
COPY feature_pipeline_simple.py /app/src/inference/
COPY velocity_service.py /app/src/inference/
COPY ato_service.py /app/src/inference/
COPY decision_engine.py /app/src/inference/
COPY main_simple.py /app/src/inference/
COPY utils/ /app/src/inference/utils/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH
ENV LOG_LEVEL=INFO

# Default command - run Flask API
CMD ["python", "/app/src/inference/main_simple.py"]
