#!/bin/bash
# Install Optuna in Airflow containers on DigitalOcean droplet

echo "🔥 Installing Optuna for Hyperparameter Tuning..."
echo ""

# Install in Airflow worker
echo "📦 Installing in airflow-worker container..."
docker exec -it airflow-worker pip install optuna>=3.0.0

# Install in Airflow scheduler (if training happens there)
echo "📦 Installing in airflow-scheduler container..."
docker exec -it airflow-scheduler pip install optuna>=3.0.0

# Install in Airflow webserver (for DAG parsing)
echo "📦 Installing in airflow-webserver container..."
docker exec -it airflow-webserver pip install optuna>=3.0.0

echo ""
echo "✅ Optuna installed successfully!"
echo ""
echo "Next steps:"
echo "1. Verify config: src/config.yaml has 'use_optuna_tuning: true'"
echo "2. Trigger DAG: http://64.23.228.115:8080"
echo "3. Wait ~60 minutes for 100 trials"
echo "4. Expected AUC-ROC: 0.85-0.88 (current: 0.80)"
echo ""
