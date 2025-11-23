"""
FastAPI REST Endpoint for Transaction Submission

Provides REST API for submitting transactions to Kafka for fraud detection.
Supports both Postman testing and programmatic integration.

Endpoints:
- POST /api/v1/transactions/submit - Submit transaction for async processing
- POST /api/v1/transactions/score-sync - Submit transaction and get immediate fraud score (optional)
- GET /health - Health check endpoint
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from confluent_kafka import Producer, Consumer
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


class TransactionScoreSyncResponse(BaseModel):
    """Response for synchronous fraud scoring (optional feature)"""
    status: str = Field(..., description="Status (approved, blocked)")
    transaction_id: str = Field(..., description="Transaction ID")
    decision: str = Field(..., description="APPROVE or BLOCK")
    probability: float = Field(..., description="Fraud probability [0, 1]")
    risk_level: str = Field(..., description="HIGH, MEDIUM, or LOW")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


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
    status_code=status.HTTP_200_OK
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


@app.post(
    "/api/v1/transactions/score-sync",
    response_model=TransactionScoreSyncResponse,
    status_code=status.HTTP_200_OK
)
async def score_transaction_sync(request: TransactionSubmitRequest):
    """
    Submit transaction and wait for synchronous fraud score (OPTIONAL FEATURE)

    This endpoint publishes the transaction to Kafka and waits for a reply
    on the reply topic using a request-reply pattern.

    Timeout: 2 seconds (configurable)

    Args:
        request: Transaction payload

    Returns:
        TransactionScoreSyncResponse with decision and fraud probability

    Raises:
        HTTPException: If scoring times out or fails
    """
    import time

    try:
        start_time = time.time()

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
            
        transaction_id = transaction["TransactionID"]

        logger.info(f"Synchronous scoring request: {transaction_id}")

        # Send to Kafka
        success = kafka_producer.send_transaction(transaction)
        kafka_producer.flush()

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to publish transaction to Kafka"
            )

        # Set up consumer for reply topic (request-reply pattern)
        reply_topic = os.getenv("KAFKA_REPLY_TOPIC", "transaction_replies")
        timeout_ms = int(os.getenv("SYNC_SCORING_TIMEOUT_MS", "2000"))

        consumer_config = {
            "bootstrap.servers": kafka_producer.bootstrap_servers,
            "group.id": f"sync-scorer-{transaction_id}",
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        }

        if kafka_producer.kafka_username and kafka_producer.kafka_password:
            consumer_config.update({
                "security.protocol": "SASL_SSL",
                "sasl.mechanism": "PLAIN",
                "sasl.username": kafka_producer.kafka_username,
                "sasl.password": kafka_producer.kafka_password,
            })

        consumer = Consumer(consumer_config)
        consumer.subscribe([reply_topic])

        # Poll for response (with timeout)
        start = time.time()
        while (time.time() - start) * 1000 < timeout_ms:
            msg = consumer.poll(timeout=0.5)

            if msg is None:
                continue

            if msg.error():
                continue

            try:
                reply = json.loads(msg.value().decode("utf-8"))

                # Check if this reply matches our transaction_id
                if reply.get("transaction_id") == transaction_id:
                    consumer.close()

                    latency_ms = (time.time() - start_time) * 1000

                    # Determine HTTP status based on decision
                    http_status = status.HTTP_200_OK if reply.get("decision") == "APPROVE" else status.HTTP_403_FORBIDDEN

                    return JSONResponse(
                        status_code=http_status,
                        content={
                            "status": "approved" if reply.get("decision") == "APPROVE" else "blocked",
                            "transaction_id": transaction_id,
                            "decision": reply.get("decision", "APPROVE"),
                            "probability": reply.get("probability", 0.0),
                            "risk_level": reply.get("risk_level", "LOW"),
                            "latency_ms": round(latency_ms, 2)
                        }
                    )
            except Exception as e:
                logger.warning(f"Error parsing reply: {str(e)}")
                continue

        # Timeout - close consumer and return timeout error
        consumer.close()

        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Scoring request timed out after {timeout_ms}ms. Transaction is still being processed asynchronously."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in synchronous scoring: {str(e)}")
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
