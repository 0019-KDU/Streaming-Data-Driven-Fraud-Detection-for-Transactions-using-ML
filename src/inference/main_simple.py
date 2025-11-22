"""
Simple REST API Inference Service for Fraud Detection (No Kafka/Spark).

Provides a Flask REST API endpoint for real-time fraud predictions.
"""

import json
from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd

from src.inference.config import Config
from src.inference.model_loader import ModelLoader
from src.inference.feature_pipeline_spark import create_feature_pipeline
from src.inference.velocity_service import VelocityService
from src.inference.ato_service import ATOService
from src.inference.decision_engine import DecisionEngine
from src.inference.logging_utils import setup_logger_from_config

# Initialize Flask app
app = Flask(__name__)

# Global services (initialized once)
config = None
model_loader = None
feature_pipeline = None
velocity_service = None
ato_service = None
decision_engine = None
logger = None


def initialize_services():
    """Initialize all services once at startup."""
    global config, model_loader, feature_pipeline, velocity_service, ato_service, decision_engine, logger

    # Load configuration
    config = Config.load()
    logger = setup_logger_from_config(__name__, config)

    logger.info("="*80)
    logger.info("Initializing Fraud Detection Inference Service")
    logger.info("="*80)

    # Load model
    logger.info("Loading ML model...")
    model_loader = ModelLoader(config)
    model_loader.load()
    logger.info("✓ Model loaded successfully")

    # Create feature pipeline
    logger.info("Creating feature pipeline...")
    feature_pipeline = create_feature_pipeline(config)
    logger.info("✓ Feature pipeline created")

    # Initialize velocity service
    logger.info("Initializing velocity service...")
    velocity_service = VelocityService(config)
    logger.info("✓ Velocity service initialized")

    # Initialize ATO service
    logger.info("Initializing ATO service...")
    ato_service = ATOService(config)
    logger.info("✓ ATO service initialized")

    # Initialize decision engine
    logger.info("Initializing decision engine...")
    decision_engine = DecisionEngine(config)
    logger.info("✓ Decision engine initialized")

    logger.info("="*80)
    logger.info("Fraud Detection Inference Service Ready!")
    logger.info("="*80)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'fraud-detection-inference',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fraud for a single transaction.

    Request body:
    {
        "TransactionID": "T123456",
        "TransactionAmt": 150.50,
        "card1": 12345,
        "addr1": 315,
        "P_emaildomain": "gmail.com",
        ...
    }

    Response:
    {
        "transaction_id": "T123456",
        "fraud_probability": 0.234,
        "decision": "APPROVE",
        "risk_level": "LOW",
        "risk_factors": ["..."],
        "ato_risk": 0.0,
        "velocity_risk": 0.1,
        "amount_risk": 0.05,
        "timestamp": "2025-11-22T12:34:56.789Z"
    }
    """
    try:
        # Parse input
        transaction = request.get_json()

        if not transaction:
            return jsonify({'error': 'No transaction data provided'}), 400

        transaction_id = transaction.get('TransactionID', f'unknown_{datetime.utcnow().timestamp()}')

        logger.info(f"Processing transaction: {transaction_id}")

        # 1. Apply feature engineering
        features_df = feature_pipeline.feature_pipeline.transform(
            pd.DataFrame([transaction])
        )

        # 2. Get ML model prediction
        fraud_prob = float(model_loader.predict(features_df)[0])

        # 3. Analyze velocity
        card1 = str(transaction.get('card1', 'unknown'))
        uid = f"{card1}_{transaction.get('addr1', 'na')}_{transaction.get('P_emaildomain', 'na')}"
        amount = float(transaction.get('TransactionAmt', 0.0))
        timestamp = float(transaction.get('TransactionDT', datetime.utcnow().timestamp()))

        velocity_result = velocity_service.analyze_velocity(
            card1, uid, amount, timestamp
        )

        # Record transaction for future velocity calculations
        velocity_service.record_transaction(card1, uid, amount, timestamp)

        # 4. Analyze ATO
        ato_result = ato_service.process_transaction(card1, transaction)

        # 5. Make final decision
        decision_result = decision_engine.make_decision(
            fraud_probability=fraud_prob,
            velocity_risk=velocity_result.velocity_risk,
            amount_risk=velocity_result.amount_risk,
            ato_risk=ato_result.ato_risk,
            ato_detected=ato_result.ato_detected,
            transaction=transaction,
            velocity_factors=velocity_result.factors,
            ato_factors=ato_result.factors
        )

        # 6. Build response
        response = {
            'transaction_id': transaction_id,
            'fraud_probability': fraud_prob,
            'decision': decision_result.decision.value,
            'risk_level': decision_result.risk_level.value,
            'risk_factors': decision_result.risk_factors,
            'ato_risk': ato_result.ato_risk,
            'velocity_risk': velocity_result.velocity_risk,
            'amount_risk': velocity_result.amount_risk,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        logger.info(f"Transaction {transaction_id}: {decision_result.decision.value} (prob={fraud_prob:.3f})")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing transaction: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict fraud for multiple transactions.

    Request body:
    {
        "transactions": [
            {"TransactionID": "T1", ...},
            {"TransactionID": "T2", ...}
        ]
    }

    Response:
    {
        "predictions": [
            {"transaction_id": "T1", "fraud_probability": 0.2, ...},
            {"transaction_id": "T2", "fraud_probability": 0.8, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        transactions = data.get('transactions', [])

        if not transactions:
            return jsonify({'error': 'No transactions provided'}), 400

        logger.info(f"Processing batch of {len(transactions)} transactions")

        results = []

        for transaction in transactions:
            try:
                transaction_id = transaction.get('TransactionID', f'unknown_{len(results)}')

                # Process each transaction
                features_df = feature_pipeline.feature_pipeline.transform(
                    pd.DataFrame([transaction])
                )

                fraud_prob = float(model_loader.predict(features_df)[0])

                card1 = str(transaction.get('card1', 'unknown'))
                uid = f"{card1}_{transaction.get('addr1', 'na')}_{transaction.get('P_emaildomain', 'na')}"
                amount = float(transaction.get('TransactionAmt', 0.0))
                timestamp = float(transaction.get('TransactionDT', datetime.utcnow().timestamp()))

                velocity_result = velocity_service.analyze_velocity(card1, uid, amount, timestamp)
                velocity_service.record_transaction(card1, uid, amount, timestamp)

                ato_result = ato_service.process_transaction(card1, transaction)

                decision_result = decision_engine.make_decision(
                    fraud_probability=fraud_prob,
                    velocity_risk=velocity_result.velocity_risk,
                    amount_risk=velocity_result.amount_risk,
                    ato_risk=ato_result.ato_risk,
                    ato_detected=ato_result.ato_detected,
                    transaction=transaction,
                    velocity_factors=velocity_result.factors,
                    ato_factors=ato_result.factors
                )

                results.append({
                    'transaction_id': transaction_id,
                    'fraud_probability': fraud_prob,
                    'decision': decision_result.decision.value,
                    'risk_level': decision_result.risk_level.value,
                    'risk_factors': decision_result.risk_factors,
                    'ato_risk': ato_result.ato_risk,
                    'velocity_risk': velocity_result.velocity_risk,
                    'amount_risk': velocity_result.amount_risk,
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                })

            except Exception as e:
                logger.error(f"Error processing transaction {transaction_id}: {e}")
                results.append({
                    'transaction_id': transaction_id,
                    'error': str(e)
                })

        return jsonify({'predictions': results}), 200

    except Exception as e:
        logger.error(f"Error processing batch: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    # Initialize services
    initialize_services()

    # Run Flask app
    port = 5000
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
