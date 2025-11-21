"""
Airflow DAG for IEEE-CIS Fraud Detection Model Training

This DAG orchestrates the training pipeline with the following tasks:
1. Environment validation (config files, data availability)
2. Execute training pipeline
3. Cleanup temporary resources

Schedule: Daily at 3:00 AM (or on-demand)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Import training module from inference folder
import sys
sys.path.insert(0, '/opt/airflow/dags')
sys.path.insert(0, '/app/inference')  # Add inference folder to path

from ieee_cis_training import train_ieee_cis_model


# Default DAG arguments
default_args = {
    'owner': 'fraud-detection-team',
    'depends_on_past': False,
    'email': ['chirantharavishka@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 0,  # Disabled: No automatic retries on failure
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}


def validate_environment(**context):
    """
    Validate that all required files and configurations are present

    Checks:
    - config.yaml exists
    - IEEE-CIS data files exist
    - Model output directories are writable
    """
    import os
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Validating environment...")

    # Check config.yaml
    config_path = "/app/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    logger.info(f"✓ Config file found: {config_path}")

    # Load config to check data paths
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check IEEE-CIS data files
    trans_path = config["data"]["ieee_cis"]["train_transaction_path"]
    identity_path = config["data"]["ieee_cis"]["train_identity_path"]

    if not os.path.exists(trans_path):
        raise FileNotFoundError(f"Transaction data not found: {trans_path}")
    logger.info(f"✓ Transaction data found: {trans_path}")

    if not os.path.exists(identity_path):
        logger.warning(f"⚠ Identity data not found (optional): {identity_path}")
    else:
        logger.info(f"✓ Identity data found: {identity_path}")

    # Check model output directory is writable
    model_path = config["training"]["model_path"]
    model_dir = os.path.dirname(model_path)

    os.makedirs(model_dir, exist_ok=True)

    if not os.access(model_dir, os.W_OK):
        raise PermissionError(f"Model directory not writable: {model_dir}")
    logger.info(f"✓ Model directory writable: {model_dir}")

    logger.info("Environment validation complete ✓")


def execute_training(**context):
    """
    Execute the IEEE-CIS training pipeline

    Returns training metrics to XCom for downstream tasks
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Starting IEEE-CIS training pipeline...")

    try:
        metrics = train_ieee_cis_model(config_path="/app/config.yaml")

        logger.info(f"Training completed successfully!")
        logger.info(f"  Status: {metrics['status']}")

        if 'auc_pr' in metrics:
            logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")

        # Push metrics to XCom
        context['task_instance'].xcom_push(key='training_metrics', value=metrics)

        return metrics

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


# Define the DAG
with DAG(
    dag_id='ieee_cis_fraud_detection_training',
    default_args=default_args,
    description='Train fraud detection model on IEEE-CIS dataset',
    schedule_interval=None,  # Manual trigger only (disabled automatic schedule)
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,  # Prevent concurrent training runs
    tags=['fraud-detection', 'ml-training', 'ieee-cis'],
) as dag:

    # Task 1: Validate environment
    validate_env_task = PythonOperator(
        task_id='validate_environment',
        python_callable=validate_environment,
        provide_context=True,
    )

    # Task 2: Execute training
    train_model_task = PythonOperator(
        task_id='execute_training',
        python_callable=execute_training,
        provide_context=True,
    )

    # Task 3: Cleanup temporary files
    cleanup_task = BashOperator(
        task_id='cleanup_resources',
        bash_command="""
        echo "Cleaning up temporary files..."
        # Remove temporary pickle files if any
        find /app/models -name "*.tmp" -type f -delete 2>/dev/null || true
        # Clear Python cache
        find /opt/airflow/dags -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        echo "Cleanup complete"
        """,
        trigger_rule='all_done',  # Run even if training fails
    )

    # Define task dependencies
    validate_env_task >> train_model_task >> cleanup_task


# Add documentation
dag.doc_md = """
# IEEE-CIS Fraud Detection Training DAG

This DAG trains a fraud detection model on the IEEE-CIS dataset.

## Tasks

1. **validate_environment**: Checks that config files and data are available
2. **execute_training**: Runs the ENHANCED training pipeline with the following steps:
   - Load and merge transaction + identity data
   - Create user identifiers (UID)
   - Engineer 50+ advanced features:
     * Velocity features (1h, 6h, 24h, 7d time-windows)
     * Frequency encoding for categorical variables
     * Email risk indicators
     * Temporal patterns
   - Chronological train/validation split (80/20)
    - Train gradient boosting model (XGBoost/LightGBM/CatBoost) with GPU support
   - Calibrate probabilities
   - Initialize adaptive threshold system
   - Log to MLflow
   - Save enhanced artifacts to /app/models/
3. **cleanup_resources**: Clean up temporary files

## Configuration

Edit `/app/config.yaml` to configure:
- Data paths (`data.ieee_cis`)
- Model parameters (`model.params`)
- Training settings (`training.use_external_model`)

## Manual Trigger

To trigger manually via CLI:
```bash
airflow dags trigger ieee_cis_fraud_detection_training
```

## Outputs

- Model: `/app/models/fraud_detection_model.pkl`
- Feature Pipeline: `/app/models/feature_pipeline.pkl`
- MLflow Experiment: `ieee_cis_fraud_detection`
"""
