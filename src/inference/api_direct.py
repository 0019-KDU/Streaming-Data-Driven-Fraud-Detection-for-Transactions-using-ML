"""
Direct Fraud Detection API - For Testing and Demo

This provides a synchronous REST API that directly invokes the ML model
without going through Kafka/Spark. Useful for:
- Quick testing
- Demos
- Debugging model behavior

Endpoint: POST /predict
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add dags directory for ieee_cis_training module (needed to unpickle model)
dags_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dags')
sys.path.insert(0, dags_path)

from inference.config import Config
from inference.model_loader import ModelLoader
from inference.velocity_service import VelocityService
from inference.ato_service import ATOService
from inference.decision_engine import HybridDecisionEngine
from inference.utils.redis_client import get_redis_client
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Direct Fraud Detection API",
    description="Synchronous fraud detection API for testing and demos",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services (initialized once)
config = None
model_loader = None
feature_pipeline = None
velocity_service = None
ato_service = None
decision_engine = None


# Mock services for when Redis is unavailable
class MockVelocityResult:
    """Mock velocity result with zero risk"""
    def __init__(self):
        self.velocity_risk = 0.0
        self.amount_risk = 0.0
        self.factors = []


class MockATOResult:
    """Mock ATO result with zero risk"""
    def __init__(self):
        self.ato_risk = 0.0
        self.ato_detected = False
        self.factors = []


class MockVelocityService:
    """Mock velocity service when Redis unavailable"""
    def check_velocity(self, card1, uid, amount, timestamp):
        return MockVelocityResult()


class MockATOService:
    """Mock ATO service when Redis unavailable"""
    def check_ato(self, card1, transaction):
        return MockATOResult()


def initialize_services():
    """Initialize all services once at startup"""
    global config, model_loader, feature_pipeline, velocity_service, ato_service, decision_engine
    
    try:
        logger.info("Initializing fraud detection services...")
        
        # Load config
        config = Config.load()
        logger.info("✅ Config loaded")
        
        # Initialize Redis (optional - use mock if unavailable)
        try:
            redis_client = get_redis_client(config)
            logger.info("✅ Redis connected")
        except Exception as redis_error:
            logger.warning(f"⚠️  Redis unavailable: {redis_error}")
            logger.warning("⚠️  Using mock Redis client (velocity/ATO features disabled)")
            redis_client = None
        
        # Load model
        model_loader = ModelLoader(config)
        model_loader.load()
        model_metadata = model_loader.get_metadata()
        logger.info(f"✅ Model loaded: {model_metadata.get('threshold', 0.0):.4%} threshold")
        
        # Load feature pipeline directly from pickle file
        pipeline_path = config.model.feature_pipeline_path
        logger.info(f"Loading feature pipeline from: {pipeline_path}")
        feature_pipeline = joblib.load(pipeline_path)
        logger.info("✅ Feature pipeline loaded")
        
        # Initialize services (with mock redis if needed)
        if redis_client:
            velocity_service = VelocityService(redis_client, config)
            ato_service = ATOService(redis_client, config)
            logger.info("✅ Services initialized with Redis")
        else:
            # Create mock services that return zero risk
            velocity_service = MockVelocityService()
            ato_service = MockATOService()
            logger.info("✅ Services initialized with mock (Redis unavailable)")
        
        decision_engine = HybridDecisionEngine(config, model_metadata)
        logger.info("✅ Decision engine initialized")
        
        logger.info("="*80)
        logger.info("🚀 Direct Fraud Detection API Ready!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise


# Request/Response models
class PredictRequest(BaseModel):
    """Transaction prediction request"""
    TransactionID: Optional[str] = None
    TransactionDT: Optional[float] = None
    TransactionAmt: float = Field(..., gt=0)
    ProductCD: Optional[str] = "W"
    card1: Optional[int] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow all IEEE-CIS fields


class PredictResponse(BaseModel):
    """Prediction response"""
    transaction_id: str
    fraud_probability: float
    decision: str
    risk_level: str
    risk_factors: list
    velocity_risk: float
    amount_risk: float
    ato_risk: float
    base_threshold: float
    hybrid_threshold: float
    processing_time_ms: float
    timestamp: str
    # Original transaction fields
    TransactionAmt: float
    card1: str
    P_emaildomain: str
    ProductCD: str


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    initialize_services()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "direct-fraud-detection-api",
        "model_loaded": model_loader is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_fraud(request: PredictRequest):
    """
    Predict fraud probability for a transaction
    
    Args:
        request: Transaction data (IEEE-CIS format)
        
    Returns:
        PredictResponse with fraud probability and decision
    """
    import time
    start_time = time.time()
    
    try:
        # Convert request to dict
        transaction = request.dict()
        transaction_id = transaction.get('TransactionID') or f"TXN_{int(time.time())}"
        
        logger.info(f"Processing transaction: {transaction_id}")
        logger.info(f"  Amount: ${transaction.get('TransactionAmt', 0):.2f}")
        logger.info(f"  Card: {transaction.get('card1', 'N/A')}")
        logger.info(f"  Email: {transaction.get('P_emaildomain', 'N/A')}")
        
        # 1. Feature engineering
        logger.info("Step 1: Feature engineering...")
        try:
            # ✅ FIX #2: Ensure TransactionAmt is properly converted to float
            transaction['TransactionAmt'] = float(transaction.get('TransactionAmt', 0.0))
            
            # Transform using the loaded feature pipeline
            features_df = feature_pipeline.transform(
                pd.DataFrame([transaction])
            )
            logger.info(f"  Engineered {len(features_df.columns)} features")
        except Exception as e:
            logger.error(f"  ❌ Feature engineering failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Feature engineering error: {str(e)}"
            )
        
        # 2. ML model prediction
        logger.info("Step 2: ML prediction...")
        try:
            fraud_prob = float(model_loader.predict(features_df)[0])
            logger.info(f"  Fraud probability: {fraud_prob:.4%}")
            
            # ✅ FIX #3: Check if probability is suspiciously low
            if fraud_prob < 0.0001 and transaction.get('P_emaildomain', '').lower() in ['yopmail.com', 'tempmail.com', 'guerrillamail.com']:
                logger.warning("⚠️  Suspiciously low probability for disposable email!")
                logger.warning("  This suggests feature engineering may not be working correctly")
                
        except Exception as e:
            logger.error(f"  ❌ Model prediction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Model prediction error: {str(e)}"
            )
        
        # 3. Velocity analysis
        logger.info("Step 3: Velocity analysis...")
        card1 = str(transaction.get('card1', 'unknown'))
        uid = f"{card1}_{transaction.get('addr1', 'na')}_{transaction.get('P_emaildomain', 'na')}"
        amount = float(transaction.get('TransactionAmt', 0.0))
        timestamp = float(transaction.get('TransactionDT', time.time()))
        
        velocity_result = velocity_service.check_velocity(
            card1, uid, amount, timestamp
        )
        logger.info(f"  Velocity risk: {velocity_result.velocity_risk:.2f}")
        logger.info(f"  Amount risk: {velocity_result.amount_risk:.2f}")
        
        # 4. ATO analysis
        logger.info("Step 4: ATO analysis...")
        ato_result = ato_service.check_ato(card1, transaction)
        logger.info(f"  ATO risk: {ato_result.ato_risk:.2f}")
        
        # 5. Final decision
        logger.info("Step 5: Final decision...")
        decision_result = decision_engine.make_decision(
            fraud_probability=fraud_prob,
            velocity_risk=velocity_result.velocity_risk,
            amount_risk=velocity_result.amount_risk,
            ato_risk=ato_result.ato_risk,
            ato_detected=ato_result.ato_detected,
            transaction_data=transaction,
            velocity_factors=velocity_result.factors,
            ato_factors=ato_result.factors
        )
        
        logger.info(f"  Decision: {decision_result.decision.value}")
        logger.info(f"  Risk level: {decision_result.risk_level.value}")
        logger.info(f"  Hybrid threshold: {decision_result.hybrid_threshold:.4%}")
        
        processing_time_ms = (time.time() - start_time) * 1000
        logger.info(f"✅ Processing completed in {processing_time_ms:.2f}ms")
        
        # Build response
        return PredictResponse(
            transaction_id=transaction_id,
            fraud_probability=fraud_prob,
            decision=decision_result.decision.value,
            risk_level=decision_result.risk_level.value,
            risk_factors=decision_result.risk_factors,
            velocity_risk=velocity_result.velocity_risk,
            amount_risk=velocity_result.amount_risk,
            ato_risk=ato_result.ato_risk,
            base_threshold=decision_engine.base_threshold,
            hybrid_threshold=decision_result.hybrid_threshold,
            processing_time_ms=round(processing_time_ms, 2),
            timestamp=datetime.utcnow().isoformat() + 'Z',
            TransactionAmt=amount,
            card1=str(transaction.get('card1', 'N/A')),
            P_emaildomain=str(transaction.get('P_emaildomain', 'N/A')),
            ProductCD=str(transaction.get('ProductCD', 'W'))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
