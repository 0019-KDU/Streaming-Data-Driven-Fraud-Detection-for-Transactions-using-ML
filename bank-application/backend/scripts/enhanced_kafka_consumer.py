#!/usr/bin/env python3
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime, date
from faker import Faker
import json
from confluent_kafka import Consumer, KafkaError

# Setup path and environment
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))
load_dotenv(backend_dir / '.env')

from app.services.customer_service import CustomerService
from app.schemas.customer_schema import CustomerCreate
from app.models.customer import Customer
from app.models.transaction import Transaction
from app.config.database import engine
from app.db.base import Base
from app.db.session import get_db
from sqlalchemy.exc import IntegrityError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_kafka_consumer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Faker for customer generation
Faker.seed(42)
fake = Faker()

class EnhancedKafkaConsumer:
    def __init__(self):
        # Kafka configuration for Confluent Cloud
        self.config = {
            'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
            'security.protocol': 'SASL_SSL',
            'sasl.mechanism': 'PLAIN',
            'sasl.username': os.getenv('KAFKA_USERNAME'),
            'sasl.password': os.getenv('KAFKA_PASSWORD'),
            'group.id': 'enhanced-bank-consumer',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
            'session.timeout.ms': 45000,
            'heartbeat.interval.ms': 3000,
            'max.poll.interval.ms': 300000,
        }
        
        self.consumer = Consumer(self.config)
        self.topic = os.getenv('KAFKA_TOPIC', 'transactions')
        self.db = next(get_db())
        
        logger.info(f"Enhanced Kafka consumer initialized for topic: {self.topic}")
        logger.info(f"Bootstrap servers: {os.getenv('KAFKA_BOOTSTRAP_SERVERS')}")
    
    def consume_kafka_messages(self, max_messages=2000, timeout=120):
        """Consume messages from Kafka and return raw data"""
        logger.info(f"Starting to consume up to {max_messages} transactions from topic '{self.topic}'")
        
        try:
            self.consumer.subscribe([self.topic])
            messages = []
            message_count = 0
            consecutive_empty_polls = 0
            max_empty_polls = 30
            
            logger.info("Subscribed to topic, starting to poll messages...")
            
            while message_count < max_messages and consecutive_empty_polls < max_empty_polls:
                msg = self.consumer.poll(timeout=2.0)
                
                if msg is None:
                    consecutive_empty_polls += 1
                    continue
                    
                consecutive_empty_polls = 0
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info("Reached end of partition")
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        continue
                
                try:
                    message_value = msg.value().decode('utf-8')
                    transaction_data = json.loads(message_value)
                    messages.append(transaction_data)
                    message_count += 1
                    
                    if message_count % 50 == 0:
                        logger.info(f"Consumed {message_count} messages so far...")
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
            
            logger.info(f"Finished consuming. Total messages: {len(messages)}")
            return messages
            
        except Exception as e:
            logger.error(f"Error during consumption: {e}")
            return []
        finally:
            self.consumer.close()
    
    def generate_customers_for_user_ids(self, user_ids):
        """Generate customers for the given user IDs"""
        logger.info(f"Generating customers for {len(user_ids)} unique user IDs...")
        
        # Check which customers already exist
        existing_customer_ids = set()
        if user_ids:
            existing_ids = self.db.query(Customer.customer_id).filter(
                Customer.customer_id.in_(user_ids)
            ).all()
            existing_customer_ids = set(id[0] for id in existing_ids)
        
        logger.info(f"Found {len(existing_customer_ids)} existing customers")
        
        # Generate customers for missing user_ids
        missing_customer_ids = user_ids - existing_customer_ids
        customers_created = 0
        
        if missing_customer_ids:
            logger.info(f"Creating {len(missing_customer_ids)} new customers...")
            new_customers = []
            
            for user_id in missing_customer_ids:
                customer = CustomerCreate(
                    customer_id=user_id,  # Match the user_id from Kafka
                    first_name=fake.first_name(),
                    last_name=fake.last_name(),
                    date_of_birth=fake.date_of_birth(minimum_age=18, maximum_age=80),
                    national_id=fake.ssn()[:20],
                    phone=fake.phone_number()[:15],
                    email=f"user{user_id}@{fake.domain_name()}",
                    address=fake.address().replace('\n', ', '),
                    country_code=fake.country_code()
                )
                new_customers.append(customer)
            
            # Save customers using customer service
            customer_service = CustomerService(self.db)
            customers_created = customer_service.create_customers_bulk(new_customers)
            logger.info(f"Successfully created {customers_created} customers")
        
        return customers_created
    
    def save_transactions(self, messages):
        """Save transaction messages to database"""
        logger.info(f"Saving {len(messages)} transactions to database...")
        
        transaction_objects = []
        
        for msg in messages:
            try:
                # Convert timestamp string to datetime if needed
                timestamp = msg.get('timestamp')
                if isinstance(timestamp, str):
                    # Handle different timestamp formats
                    if timestamp.endswith('Z'):
                        timestamp = timestamp.replace('Z', '+00:00')
                    elif '+' not in timestamp and 'T' in timestamp:
                        timestamp += '+00:00'
                    timestamp = datetime.fromisoformat(timestamp)
                
                transaction = Transaction(
                    transaction_id=msg.get('transaction_id'),
                    user_id=msg.get('user_id'),
                    amount=float(msg.get('amount', 0)),
                    currency=msg.get('currency', 'USD'),
                    merchant=msg.get('merchant', 'Unknown'),
                    timestamp=timestamp,
                    location=msg.get('location', 'US'),
                    is_fraud=bool(msg.get('is_fraud', 0))
                )
                transaction_objects.append(transaction)
                
            except Exception as e:
                logger.error(f"Error processing transaction message: {e}")
                continue
        
        # Bulk save transactions
        transactions_saved = 0
        if transaction_objects:
            try:
                self.db.add_all(transaction_objects)
                self.db.commit()
                transactions_saved = len(transaction_objects)
                logger.info(f"Successfully saved {transactions_saved} transactions")
                
            except IntegrityError as e:
                self.db.rollback()
                logger.error(f"Foreign key violation: {e}")
                return 0
        
        return transactions_saved
    
    def process_kafka_with_customers(self, max_messages=2000, timeout=120):
        """Main processing function - consume Kafka and generate customers"""
        try:
            # Step 1: Consume messages from Kafka
            messages = self.consume_kafka_messages(max_messages, timeout)
            
            if not messages:
                logger.warning("No messages received from Kafka")
                return {
                    "status": "no_messages", 
                    "customers_created": 0, 
                    "transactions_saved": 0,
                    "messages_processed": 0
                }
            
            # Step 2: Extract unique user_ids
            unique_user_ids = set(msg.get('user_id') for msg in messages if msg.get('user_id'))
            logger.info(f"Found {len(unique_user_ids)} unique user IDs")
            
            # Step 3: Generate customers for user_ids
            customers_created = self.generate_customers_for_user_ids(unique_user_ids)
            
            # Step 4: Save transactions
            transactions_saved = self.save_transactions(messages)
            
            return {
                "status": "success",
                "messages_processed": len(messages),
                "unique_user_ids": len(unique_user_ids),
                "customers_created": customers_created,
                "transactions_saved": transactions_saved
            }
            
        except Exception as e:
            logger.error(f"Error in processing: {e}")
            return {"status": "error", "error": str(e)}
        
        finally:
            self.db.close()

def validate_environment():
    """Validate required environment variables"""
    required_vars = [
        'KAFKA_BOOTSTRAP_SERVERS',
        'KAFKA_USERNAME', 
        'KAFKA_PASSWORD',
        'KAFKA_TOPIC',
        'DATABASE_URL'
    ]
    
    logger.info("Checking environment variables...")
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
            logger.error(f"❌ {var}: Not set")
        else:
            if 'PASSWORD' in var or 'SECRET' in var:
                logger.info(f"✅ {var}: ***")
            else:
                logger.info(f"✅ {var}: {value}")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("All required environment variables are set")
    return True

def main():
    """Main function"""
    try:
        logger.info("=== Enhanced Kafka Consumer with Customer Generation ===")
        
        # Validate environment
        if not validate_environment():
            return 1
        
        # Create database tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
        
        # Process Kafka messages with customer generation
        processor = EnhancedKafkaConsumer()
        result = processor.process_kafka_with_customers(max_messages=2000, timeout=120)
        
        # Display results
        print("\n" + "="*60)
        print("ENHANCED KAFKA PROCESSING RESULTS")
        print("="*60)
        print(f"Status: {result.get('status')}")
        print(f"Messages Processed: {result.get('messages_processed', 0)}")
        print(f"Unique User IDs: {result.get('unique_user_ids', 0)}")
        print(f"Customers Created: {result.get('customers_created', 0)}")
        print(f"Transactions Saved: {result.get('transactions_saved', 0)}")
        
        if result.get('status') == 'error':
            print(f"Error: {result.get('error')}")
        else:
            print("✅ Successfully processed Kafka data with customer generation!")
        print("="*60)
        
        return 0 if result.get('status') == 'success' else 1
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
