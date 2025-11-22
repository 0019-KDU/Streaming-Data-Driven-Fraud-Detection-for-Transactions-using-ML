# Stage 1: Builder stage
FROM python:3.10.16-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    curl \
    gcc \
    python3-dev \
    openjdk-17-jdk-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify network connectivity before downloading
RUN wget -q --spider http://google.com || (echo "Network issue: Unable to reach the internet" && exit 1)

# Define Spark version and URLs
ENV SPARK_VERSION=3.5.4
ENV HADOOP_VERSION=3
ENV SPARK_FILENAME="spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"
ENV SPARK_URL_PRIMARY="https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/${SPARK_FILENAME}"
ENV SPARK_URL_BACKUP="https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/${SPARK_FILENAME}"

# Download and install Spark with retry logic
RUN wget -q --retry-connrefused --waitretry=5 --tries=3 ${SPARK_URL_PRIMARY} -O ${SPARK_FILENAME} || \
    wget -q --retry-connrefused --waitretry=5 --tries=3 ${SPARK_URL_BACKUP} -O ${SPARK_FILENAME} || \
    (echo "Failed to download Spark from all sources!" && exit 1)

RUN tar xzf ${SPARK_FILENAME} -C /tmp && rm ${SPARK_FILENAME}
RUN mv /tmp/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /spark

# Add Kafka connector and client JARs
RUN wget -O /spark/jars/spark-sql-kafka-0-10_2.12-${SPARK_VERSION}.jar \
    https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/${SPARK_VERSION}/spark-sql-kafka-0-10_2.12-${SPARK_VERSION}.jar

RUN wget -O /spark/jars/kafka-clients-3.6.1.jar \
    https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.6.1/kafka-clients-3.6.1.jar

RUN wget -O /spark/jars/spark-token-provider-kafka-0-10_2.12-${SPARK_VERSION}.jar \
    https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/${SPARK_VERSION}/spark-token-provider-kafka-0-10_2.12-${SPARK_VERSION}.jar

RUN wget -O /spark/jars/commons-pool2-2.12.0.jar \
    https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.12.0/commons-pool2-2.12.0.jar

# Create virtual environment and working directory
RUN mkdir -p /app/src/inference
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir \
    redis==4.5.4 \
    pandas==1.5.3 \
    numpy==1.24.3 \
    scikit-learn==1.2.2 \
    xgboost==2.0.3 \
    joblib==1.2.0 \
    pyyaml==6.0 \
    confluent-kafka==2.3.0 \
    pyarrow==14.0.1

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

# Stage 2: Runtime stage
FROM python:3.10.16-slim-bookworm as runtime

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    procps \
    ca-certificates \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create application directory structure
RUN mkdir -p /app/src/inference
WORKDIR /app

# Copy artifacts from builder
COPY --from=builder /spark /spark
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src /app/src

# Environment configuration
ENV SPARK_HOME=/spark \
    PATH="/spark/bin:/opt/venv/bin:$PATH" \
    PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.9.7-src.zip:/app" \
    SPARK_DRIVER_MEMORY=2g \
    SPARK_EXECUTOR_MEMORY=2g \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

# Configure logging
COPY --from=builder /spark/conf/log4j2.properties.template /spark/conf/log4j2.properties
RUN sed -i 's/rootLogger.level = info/rootLogger.level = error/' /spark/conf/log4j2.properties

# Set permissions
RUN useradd -m sparkuser && \
    chown -R sparkuser:sparkuser /app /spark
USER sparkuser

# Run Spark Structured Streaming job
CMD ["spark-submit", \
     "--conf", "spark.driver.defaultJavaOptions=-Dlog4j2.configuration=file:$SPARK_HOME/conf/log4j2.properties", \
     "--conf", "spark.sql.shuffle.partitions=200", \
     "--master", "local[*]", \
     "--driver-memory", "2g", \
     "--executor-memory", "2g", \
     "/app/src/inference/main_inference.py"]