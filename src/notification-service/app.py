import json
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Any
from dotenv import load_dotenv
import psycopg2
from confluent_kafka import Consumer, KafkaException

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path="../.env")

class NotificationService:
    def __init__(self):
        # Fixed Kafka config (removed self.topic assignment)
        self.kafka_config = {
            'bootstrap.servers': os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
            'group.id': 'notification-group',
            'security.protocol': 'SASL_SSL',
            'sasl.mechanism': 'PLAIN',
            'sasl.username': os.getenv("KAFKA_USERNAME"),
            'sasl.password': os.getenv("KAFKA_PASSWORD"),
            'auto.offset.reset': 'earliest'  # Fixed comma placement
        }
        self.consumer = Consumer(self.kafka_config)
        
        # Database connection with error handling
        try:
            self.db_conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST'),
                database=os.getenv('POSTGRES_DB'),
                user=os.getenv('POSTGRES_USER'),
                password=os.getenv('POSTGRES_PASSWORD'),
                connect_timeout=5
            )
            logger.info("Connected to PostgreSQL database")
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
        
        # SMTP configuration validation
        self.smtp_host = os.getenv('SMTP_HOST')
        self.smtp_port = int(os.getenv('SMTP_PORT', 1025))
        if not self.smtp_host:
            raise ValueError("SMTP_HOST environment variable not set")

    def get_user_email(self, user_id: int) -> str:
        """Get user email from PostgreSQL database with error handling"""
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("SELECT email FROM users WHERE user_id = %s", (user_id,))
                result = cur.fetchone()
                return result[0] if result else None
        except psycopg2.Error as e:
            logger.error(f"Database query failed: {str(e)}")
            self.db_conn.rollback()
            return None

    def send_notification_email(self, transaction: Dict[str, Any]):
        """Send email notification with improved error handling"""
        try:
            user_email = self.get_user_email(transaction['user_id'])
            if not user_email:
                logger.warning(f"No email found for user {transaction['user_id']}")
                return

            msg = MIMEMultipart('alternative')
            # Add email headers and content (same as before)
            
            # Test SMTP connection first
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()  # Add encryption if supported
                server.sendmail(msg['From'], user_email, msg.as_string())
            logger.info(f"Sent notification to {user_email}")
            
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            raise

    def consume_fraud_predictions(self):
        """Consume messages with proper error recovery"""
        try:
            self.consumer.subscribe(['fraud_predictions'])
            logger.info("Started consuming fraud predictions")
            
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaException._PARTITION_EOF:
                        continue
                    raise KafkaException(msg.error())
                
                try:
                    transaction = json.loads(msg.value().decode('utf-8'))
                    self.send_notification_email(transaction)
                    self.consumer.commit(msg)  # Manual commit after processing
                except Exception as e:
                    logger.error(f"Message processing failed: {str(e)}")

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully")
        finally:
            self.consumer.close()
            self.db_conn.close()
            logger.info("Resources cleaned up")

if __name__ == "__main__":
    service = NotificationService()
    service.consume_fraud_predictions()