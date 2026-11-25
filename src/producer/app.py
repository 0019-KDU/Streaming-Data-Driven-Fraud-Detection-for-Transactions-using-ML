"""
FastAPI REST Endpoint for Transaction Submission

Provides REST API for submitting transactions to Kafka for fraud detection.
Supports both Postman testing and programmatic integration.

Endpoints:
- POST /api/v1/transactions/submit - Submit transaction for async processing

Synchronous scoring is served by the inference API (src/inference/api_direct.py, POST /predict).
- GET /health - Health check endpoint
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union

import hmac

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from confluent_kafka import Producer
from dotenv import load_dotenv
import math

# Load environment variables
load_dotenv(dotenv_path="../.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection Transaction API",
    description="REST API for submitting transactions to real-time fraud detection pipeline",
    version="1.0.0"
)


# ===== Request/Response Models =====

class TransactionSubmitRequest(BaseModel):
    """
    Transaction submission payload (IEEE-CIS compatible)

    Supports IEEE-CIS fields and allows extra fields for flexibility
    """
    TransactionID: Optional[Union[str, int]] = Field(None, description="Unique transaction identifier (auto-generated if not provided)")
    TransactionDT: Optional[float] = Field(None, description="Seconds offset from reference time")
    TransactionAmt: float = Field(..., description="Transaction amount", gt=0)
    ProductCD: Optional[str] = Field("W", description="Product code (W, H, C, S, R)")
    card1: Optional[int] = Field(None, description="Card identifier 1")
    card2: Optional[int] = Field(None, description="Card identifier 2")
    addr1: Optional[int] = Field(None, description="Address identifier")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp")

    # Allow extra fields for forward compatibility
    class Config:
        extra = "allow"

    @validator('*', pre=True)
    def convert_nan_to_none(cls, v):
        """Convert NaN values to None for JSON serialization"""
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    @validator('TransactionID', always=True)
    def set_transaction_id(cls, v):
        """Auto-generate TransactionID if not provided"""
        return v or f"TXN_{uuid.uuid4().hex[:12].upper()}"

    @validator('timestamp', always=True)
    def set_timestamp(cls, v):
        """Auto-generate timestamp if not provided"""
        return v or datetime.now(timezone.utc).isoformat()


class TransactionSubmitResponse(BaseModel):
    """Response for async transaction submission"""
    status: str = Field(..., description="Status of submission (accepted, rejected)")
    transaction_id: str = Field(..., description="Assigned transaction ID")
    message: str = Field(..., description="Human-readable message")
    timestamp: str = Field(..., description="Server timestamp")


# ===== Kafka Producer Configuration =====

class KafkaProducerService:
    """Kafka producer service for publishing transactions"""

    def __init__(self):
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.kafka_username = os.getenv("KAFKA_USERNAME")
        self.kafka_password = os.getenv("KAFKA_PASSWORD")
        self.topic = os.getenv("KAFKA_TOPIC", "transactions")

        # Producer configuration
        self.producer_config = {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": "transaction-producer-api",
            "compression.type": "gzip",
            "linger.ms": 5,
            "batch.size": 16384,
        }

        if self.kafka_username and self.kafka_password:
            self.producer_config.update({
                "security.protocol": "SASL_SSL",
                "sasl.mechanism": "PLAIN",
                "sasl.username": self.kafka_username,
                "sasl.password": self.kafka_password,
            })
        else:
            self.producer_config["security.protocol"] = "PLAINTEXT"

        try:
            self.producer = Producer(self.producer_config)
            logger.info(f"Kafka Producer initialized for topic: {self.topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Producer: {str(e)}")
            raise

    def delivery_report(self, err, msg):
        """Delivery callback for Kafka messages"""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    def send_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Send transaction to Kafka topic

        Args:
            transaction: Transaction payload as dict

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.producer.produce(
                self.topic,
                key=str(transaction.get("TransactionID", transaction.get("transaction_id"))),
                value=json.dumps(transaction),
                callback=self.delivery_report
            )
            self.producer.poll(0)  # Trigger callbacks
            return True
        except Exception as e:
            logger.error(f"Error producing message: {str(e)}")
            return False

    def flush(self):
        """Flush pending messages"""
        self.producer.flush(timeout=10)


# Initialize Kafka producer
kafka_producer = KafkaProducerService()


# ===== Authentication =====
# Callers send X-API-Key; valid keys in PRODUCER_API_KEYS (comma-separated).
# Empty = open API, for local development only.
PRODUCER_API_KEYS = [k.strip() for k in os.getenv("PRODUCER_API_KEYS", "").split(",") if k.strip()]
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
if not PRODUCER_API_KEYS:
    logger.warning("PRODUCER_API_KEYS is empty: the producer API is OPEN (development only)")


def require_api_key(key: Optional[str] = Security(_api_key_header)) -> None:
    if not PRODUCER_API_KEYS:
        return
    if not key or not any(hmac.compare_digest(key.encode(), k.encode()) for k in PRODUCER_API_KEYS):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid X-API-Key")


# ===== API Endpoints =====

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint

    Returns:
        JSON with service status
    """
    return {
        "status": "healthy",
        "service": "fraud-detection-transaction-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kafka_topic": kafka_producer.topic
    }


@app.post(
    "/api/v1/transactions/submit",
    response_model=TransactionSubmitResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_api_key)],
)
async def submit_transaction(request: TransactionSubmitRequest):
    """
    Submit transaction for asynchronous fraud detection

    The transaction is published to Kafka and processed by the inference pipeline.
    Results are published to fraud_predictions or legit_predictions topics.

    Args:
        request: Transaction payload

    Returns:
        TransactionSubmitResponse with transaction_id and status
    """
    try:
        # Convert request to dict
        transaction = request.dict()
        
        # Clean NaN/Infinity values for JSON serialization
        transaction = {
            k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
            for k, v in transaction.items()
        }
        
        # Ensure TransactionID is a string (handles int input)
        if transaction.get("TransactionID") is not None:
            transaction["TransactionID"] = str(transaction["TransactionID"])

        # Log submission
        logger.info(f"Received transaction: {transaction['TransactionID']}")

        # Send to Kafka
        success = kafka_producer.send_transaction(transaction)

        if success:
            return TransactionSubmitResponse(
                status="accepted",
                transaction_id=transaction["TransactionID"],
                message="Transaction submitted successfully for fraud detection",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to publish transaction to Kafka"
            )

    except Exception as e:
        logger.error(f"Error submitting transaction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown - flush Kafka producer"""
    logger.info("Shutting down API...")
    kafka_producer.flush()
    logger.info("Kafka producer flushed")


# ===== Example Usage =====
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
