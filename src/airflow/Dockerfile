FROM apache/airflow:2.10.5

# Install system dependencies for ML libraries
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create models directory and set ownership
RUN mkdir -p /app/models && chown -R airflow:root /app

# Switch to airflow user
USER airflow

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt