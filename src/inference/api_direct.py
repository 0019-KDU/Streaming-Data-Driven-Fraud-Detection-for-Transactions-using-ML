"""
Synchronous fraud scoring API: the authorization path.

POST /predict   decision + reasons for one transaction (same FraudScorer as Spark)
POST /feedback  confirmed outcome (chargeback / analyst) for a past transaction
GET  /health    liveness, model status

Production behaviour:
- API-key auth (header X-API-Key, keys in SCORING_API_KEYS)
- latency budget (SCORING_TIMEOUT_MS): if the model misses it or fails, the API
  answers with a rules-only decision marked `degraded` instead of failing the
  payment (fallback policy FALLBACK_MODE = review | approve)
- idempotent: retrying the same TransactionID returns the original decision
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add dags directory for ieee_cis_training module (needed to unpickle model)
dags_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dags')
sys.path.insert(0, dags_path)

from inference.config import Config
from inference.decision_engine import HybridDecisionEngine
from inference.scoring import FraudScorer, connect_redis_or_none, rules_only_decision
from inference.security import is_authorized, load_keys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Direct Fraud Detection API",
    description="Synchronous fraud scoring (authorization path). Uses the same FraudScorer as the Spark job.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scorer: Optional[FraudScorer] = None
fallback_engine: Optional[HybridDecisionEngine] = None
API_KEYS = load_keys()
SCORING_TIMEOUT_S = float(os.getenv("SCORING_TIMEOUT_MS", "300")) / 1000.0
# Anything other than an explicit "approve" fails SAFE (review), never open
FALLBACK_MODE = "approve" if os.getenv("FALLBACK_MODE", "review").strip().lower() == "approve" else "review"
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Optional[str] = Security(_api_key_header)) -> None:
    if not is_authorized(key, API_KEYS):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid X-API-Key")


def initialize_services():
    """Load the model once at startup; if it cannot load, serve rules-only (degraded) decisions."""
    global scorer, fallback_engine
    config = Config.load()
    fallback_engine = HybridDecisionEngine(config, {})
    if not API_KEYS:
        logger.warning("SCORING_API_KEYS is empty: the API is OPEN (development only)")
    try:
        scorer = FraudScorer(config, connect_redis_or_none(config))
        logger.info(
            f"Scoring API ready (redis={'on' if scorer.redis_enabled else 'off'}, "
            f"model={scorer.metadata.get('model_version')}, budget={SCORING_TIMEOUT_S * 1000:.0f} ms, "
            f"fallback={FALLBACK_MODE})"
        )
    except Exception as exc:  # noqa: BLE001 - keep answering with rules only
        scorer = None
        logger.error(f"Model failed to load ({exc}); serving DEGRADED rules-only decisions")


# Request/Response models
class PredictRequest(BaseModel):
    """Transaction prediction request"""
    TransactionID: Optional[Any] = None
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
    transaction_id: Optional[Any] = None
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
    explanation: List[str] = []
    model_kind: str = ""
    model_version: str = ""
    rules_version: str = ""
    shadow_probability: float = -1.0
    degraded: bool = False   # True = model unavailable/too slow, rules-only decision
    replayed: bool = False   # True = TransactionID already scored, original decision returned
    # Original transaction fields
    TransactionAmt: float
    card1: str
    P_emaildomain: str
    ProductCD: str


class FeedbackRequest(BaseModel):
    """A confirmed outcome for a past transaction (chargeback or analyst decision)."""
    transaction_id: Optional[str] = None
    terminal_id: str
    tx_datetime: str = Field(..., description="ISO-8601 time of the ORIGINAL transaction")
    is_fraud: bool


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
        "model_loaded": scorer is not None,
        "mode": "normal" if scorer is not None else "degraded_rules_only",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
async def predict_fraud(request: PredictRequest):
    """Score one transaction synchronously and return the decision with reason codes."""
    start_time = time.time()
    payload = request.dict()

    result = None
    if scorer is not None:
        try:
            result = await asyncio.wait_for(run_in_threadpool(scorer.score_one, payload), SCORING_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.error(f"Scoring exceeded {SCORING_TIMEOUT_S * 1000:.0f} ms budget; rules-only fallback")
            result = rules_only_decision(fallback_engine, payload, FALLBACK_MODE, "timeout")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Scoring failed: {e}; rules-only fallback", exc_info=True)
            result = rules_only_decision(fallback_engine, payload, FALLBACK_MODE, "model_error")
    if result is None:
        result = rules_only_decision(fallback_engine, payload, FALLBACK_MODE, "model_unavailable")

    tid = request.TransactionID
    return PredictResponse(
        transaction_id=tid,
        fraud_probability=result['fraud_probability'],
        decision=result['decision'],
        risk_level=result['risk_level'],
        risk_factors=[f for f in result['risk_factors'].split(',') if f],
        velocity_risk=result['velocity_risk'],
        amount_risk=result['amount_risk'],
        ato_risk=result['ato_risk'],
        base_threshold=result['base_threshold'],
        hybrid_threshold=result['hybrid_threshold'],
        processing_time_ms=round((time.time() - start_time) * 1000, 2),
        timestamp=result['timestamp'],
        explanation=[e for e in result['explanation'].split(' | ') if e],
        model_kind=result['model_kind'],
        model_version=result['model_version'],
        rules_version=result['rules_version'],
        shadow_probability=result['shadow_probability'],
        degraded=bool(result.get('degraded', False)),
        replayed=bool(result.get('replayed', False)),
        TransactionAmt=result['TransactionAmt'],
        card1=result['card1'],
        P_emaildomain=result['P_emaildomain'],
        ProductCD=result['ProductCD'],
    )


@app.post("/feedback", dependencies=[Depends(require_api_key)])
async def record_feedback(feedback: FeedbackRequest):
    """
    Label feedback loop: confirmed frauds update the terminal-risk features used
    by later decisions (and become training labels at the next retrain).
    """
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    scorer.record_feedback(feedback.terminal_id, feedback.tx_datetime, feedback.is_fraud)
    return {"status": "recorded", "transaction_id": feedback.transaction_id}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
