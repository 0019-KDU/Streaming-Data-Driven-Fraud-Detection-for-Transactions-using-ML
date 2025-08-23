from confluent_kafka import Consumer, KafkaError
import json
import logging
from datetime import datetime
from typing import List
from app.schemas.transaction_schema import TransactionKafka, TransactionCreate
from app.services.transaction_service import TransactionService
from app.db.session import get_db
import os

logger = logging.getLogger(__name__)

class KafkaTransactionConsumer:
    def __init__(self):
        # Kafka configuration for Confluent Cloud (Fixed)
        self.config = {
            'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
            'security.protocol': 'SASL_SSL',
            'sasl.mechanism': 'PLAIN',
            'sasl.username': os.getenv('KAFKA_USERNAME'),
            'sasl.password': os.getenv('KAFKA_PASSWORD'),
            'group.id': 'bank-transaction-consumer',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
            'session.timeout.ms': 45000,
            'heartbeat.interval.ms': 3000,
            'max.poll.interval.ms': 300000,
            'fetch.min.bytes': 1,
            # Removed invalid property 'fetch.max.wait.ms'
        }
        
        self.consumer = Consumer(self.config)
        self.topic = os.getenv('KAFKA_TOPIC', 'transactions')
        
        logger.info(f"Kafka consumer initialized for topic: {self.topic}")
        logger.info(f"Bootstrap servers: {os.getenv('KAFKA_BOOTSTRAP_SERVERS')}")
        
    def parse_kafka_message(self, message_value: str) -> TransactionCreate:
        """Parse Kafka message to TransactionCreate object"""
        try:
            data = json.loads(message_value)
            kafka_transaction = TransactionKafka(**data)
            
            # Convert timestamp string to datetime
            timestamp_str = kafka_transaction.timestamp
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str.replace('Z', '+00:00')
            elif '+' not in timestamp_str and 'T' in timestamp_str:
                timestamp_str += '+00:00'
                
            timestamp = datetime.fromisoformat(timestamp_str)
            
            # Convert is_fraud int to boolean
            is_fraud = bool(kafka_transaction.is_fraud)
            
            return TransactionCreate(
                transaction_id=kafka_transaction.transaction_id,
                user_id=kafka_transaction.user_id,
                amount=kafka_transaction.amount,
                currency=kafka_transaction.currency,
                merchant=kafka_transaction.merchant,
                timestamp=timestamp,
                location=kafka_transaction.location,
                is_fraud=is_fraud
            )
            
        except Exception as e:
            logger.error(f"Error parsing Kafka message: {e}")
            logger.error(f"Message content: {message_value}")
            raise
    
    def consume_transactions(self, max_messages: int = 2000, timeout: int = 60):
        """Consume transactions from Kafka topic"""
        logger.info(f"Starting to consume up to {max_messages} transactions from topic '{self.topic}'")
        
        try:
            self.consumer.subscribe([self.topic])
            transactions = []
            message_count = 0
            consecutive_empty_polls = 0
            max_empty_polls = 30
            
            logger.info("Subscribed to topic, starting to poll messages...")
            
            while message_count < max_messages and consecutive_empty_polls < max_empty_polls:
                msg = self.consumer.poll(timeout=2.0)  # Poll every 2 seconds
                
                if msg is None:
                    consecutive_empty_polls += 1
                    logger.debug(f"No message received, empty polls: {consecutive_empty_polls}")
                    continue
                    
                consecutive_empty_polls = 0
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info("Reached end of partition")
                        continue  # Changed from break to continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        continue
                
                try:
                    message_value = msg.value().decode('utf-8')
                    logger.debug(f"Received message: {message_value[:100]}...")  # Log first 100 chars
                    
                    transaction = self.parse_kafka_message(message_value)
                    transactions.append(transaction)
                    message_count += 1
                    
                    if message_count % 50 == 0:
                        logger.info(f"Consumed {message_count} messages so far...")
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
            
            if consecutive_empty_polls >= max_empty_polls:
                logger.info(f"Stopped consuming after {max_empty_polls} consecutive empty polls")
            
            logger.info(f"Finished consuming. Total messages: {len(transactions)}")
            return transactions
            
        except Exception as e:
            logger.error(f"Error during consumption: {e}")
            raise
        finally:
            self.consumer.close()
    
    def save_transactions_to_db(self, transactions: List[TransactionCreate]) -> int:
        """Save transactions to database"""
        db = next(get_db())
        try:
            transaction_service = TransactionService(db)
            created_count = transaction_service.create_transactions_bulk(transactions)
            return created_count
        finally:
            db.close()
    
    def consume_and_save(self, max_messages: int = 2000, timeout: int = 60) -> dict:
        """Consume from Kafka and save to database"""
        logger.info("Starting Kafka consumption and database save process...")
        
        try:
            # Consume from Kafka
            transactions = self.consume_transactions(max_messages, timeout)
            
            if not transactions:
                return {
                    "status": "no_data",
                    "message": "No transactions found in Kafka topic",
                    "consumed": 0,
                    "saved": 0
                }
            
            # Save to database
            saved_count = self.save_transactions_to_db(transactions)
            
            result = {
                "status": "success",
                "consumed": len(transactions),
                "saved": saved_count,
                "duplicates": len(transactions) - saved_count,
                "message": f"Successfully processed {len(transactions)} messages, saved {saved_count} new transactions"
            }
            
            logger.info(result["message"])
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "message": f"Error during Kafka consumption: {str(e)}",
                "consumed": 0,
                "saved": 0
            }
            logger.error(error_result["message"])
            return error_result
