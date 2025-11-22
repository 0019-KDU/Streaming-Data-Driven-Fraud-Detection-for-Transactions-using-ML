# Clean Spark Inference Service Dockerfile
FROM bitnami/spark:3.3.2

USER root

# Install Python dependencies
RUN pip install --no-cache-dir \
    redis==4.5.4 \
    pandas==1.5.3 \
    numpy==1.24.3 \
    scikit-learn==1.2.2 \
    xgboost==1.7.5 \
    joblib==1.2.0 \
    pyyaml==6.0

# Create app directory
WORKDIR /app

# Copy inference code
COPY __init__.py /app/src/inference/
COPY config.py /app/src/inference/
COPY config.yaml /app/src/inference/
COPY schema.py /app/src/inference/
COPY logging_utils.py /app/src/inference/
COPY model_loader.py /app/src/inference/
COPY feature_pipeline_spark.py /app/src/inference/
COPY velocity_service.py /app/src/inference/
COPY ato_service.py /app/src/inference/
COPY decision_engine.py /app/src/inference/
COPY main_inference.py /app/src/inference/
COPY utils/ /app/src/inference/utils/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH
ENV LOG_LEVEL=INFO

# Default command
CMD ["spark-submit", \
     "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2", \
     "--conf", "spark.sql.shuffle.partitions=50", \
     "/app/src/inference/main_inference.py"]
