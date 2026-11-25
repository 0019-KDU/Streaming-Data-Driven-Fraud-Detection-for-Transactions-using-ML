"""
Airflow DAG: train the decision-time model served by the live system.

simulate (first run only) -> replay features (same code as serving) ->
train / calibrate / cost thresholds -> out-of-time test -> save bundle + metrics.
Manual trigger:  airflow dags trigger decision_time_fraud_training
"""

import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, '/opt/airflow/dags')


def run_training(**context):
    from train_decision_time import train_decision_time_model

    metrics = train_decision_time_model(config_path="/app/config.yaml")
    summary = {"thresholds": metrics["thresholds"], "test": metrics["test"]["model_block"],
               "recall_by_scenario": metrics["recall_by_scenario"]}
    context['task_instance'].xcom_push(key='training_metrics', value=summary)
    return summary


with DAG(
    dag_id='decision_time_fraud_training',
    description='Train the decision-time fraud model (features computable at authorization time)',
    default_args={'owner': 'fraud-detection-team', 'retries': 0, 'execution_timeout': timedelta(hours=2)},
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['fraud-detection', 'ml-training', 'decision-time'],
) as dag:
    PythonOperator(task_id='train_decision_time_model', python_callable=run_training)
