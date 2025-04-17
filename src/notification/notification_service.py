import json
import logging
import os
import smtplib
import signal
from email.mime.text import MIMEText
from typing import Dict, Any, Optional

from confluent_kafka import Consumer, KafkaError
from dotenv import load_dotenv
from faker import Faker

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./notification_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Faker for generating random email addresses
fake = Faker()

class NotificationService:
    def __init__(self, config_path: str = "/app/config.yaml"):
        # Load environment variables
        load_dotenv(dotenv_path="/app/.env")

        # Kafka configuration
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.kafka_username = os.getenv("KAFKA_USERNAME")
        self.kafka_password = os.getenv("KAFKA_PASSWORD")
        self.topic = os.getenv("KAFKA_FRAUD_TOPIC", "fraud_predictions")
        self.group_id = "notification-service-group"

        # SMTP configuration for MailDev
        self.smtp_host = os.getenv("SMTP_HOST", "maildev")
        self.smtp_port = int(os.getenv("SMTP_PORT", 1025))
        self.smtp_from = os.getenv("SMTP_FROM", "fraud.alerts@bank.com")

        # Consumer configuration for Confluent Kafka
        self.consumer_config = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        }

        if self.kafka_username and self.kafka_password:
            self.consumer_config.update({
                "security.protocol": "SASL_SSL",
                "sasl.mechanism": "PLAIN",
                "sasl.username": self.kafka_username,
                "sasl.password": self.kafka_password,
            })
        else:
            self.consumer_config["security.protocol"] = "PLAINTEXT"

        try:
            self.consumer = Consumer(self.consumer_config)
            self.consumer.subscribe([self.topic])
            logger.info(f"Subscribed to Kafka topic: {self.topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Consumer: {str(e)}")
            raise e

        # Track running state for graceful shutdown
        self.running = False

        # Configure graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def generate_user_email(self, user_id: int) -> str:
        """Generate a random email address for a user based on user_id."""
        # Seed Faker with user_id for deterministic email generation
        Faker.seed(user_id)
        return fake.email()

    def send_email(self, to_email: str, transaction: Dict[str, Any]) -> bool:
        """Send email notification for a fraudulent transaction."""
        try:
            # Construct email content
            subject = f"Fraud Alert: Suspicious Transaction Detected (ID: {transaction['transaction_id']})"
            body = f"""
Dear Customer,

We have detected a potentially fraudulent transaction on your account. Please review the details below and contact our fraud prevention team if necessary.

Transaction Details:
- Transaction ID: {transaction['transaction_id']}
- User ID: {transaction['user_id']}
- Amount: {transaction['amount']} {transaction['currency']}
- Merchant: {transaction['merchant']}
- Timestamp: {transaction['timestamp']}
- Location: {transaction['location']}

If you did not authorize this transaction, please contact us immediately at support@bank.com.

Best regards,
Fraud Prevention Team
            """

            # Create MIMEText object for email
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.smtp_from
            msg["To"] = to_email

            # Connect to SMTP server (MailDev)
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.sendmail(self.smtp_from, to_email, msg.as_string())
                logger.info(f"Email sent to {to_email} for transaction {transaction['transaction_id']}")
                return True

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False

    def process_message(self, msg: Any) -> bool:
        """Process a single Kafka message and send notification."""
        try:
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug("Reached end of partition")
                    return False
                logger.error(f"Kafka error: {msg.error()}")
                return False

            # Parse transaction data
            transaction = json.loads(msg.value().decode("utf-8"))
            user_id = transaction.get("user_id")
            if not user_id:
                logger.warning("Missing user_id in transaction data")
                return False

            # Generate user email
            to_email = self.generate_user_email(user_id)
            logger.debug(f"Generated email for user {user_id}: {to_email}")

            # Send email notification
            return self.send_email(to_email, transaction)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return False

    def run(self):
        """Run the notification service to consume and process fraud predictions."""
        self.running = True
        logger.info("Starting notification service...")

        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                self.process_message(msg)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            self.shutdown()

    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown procedure."""
        if self.running:
            logger.info("Initiating shutdown...")
            self.running = False
            if self.consumer:
                self.consumer.close()
            logger.info("Notification service stopped")