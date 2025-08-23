#!/usr/bin/env python3
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to Python path
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

# Load environment variables from backend/.env file
env_path = backend_dir / '.env'
load_dotenv(dotenv_path=env_path)

from app.services.kafka_consumer import KafkaTransactionConsumer
from app.config.database import engine
from app.db.base import Base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kafka_consumer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def validate_environment():
    """Validate required environment variables"""
    required_vars = [
        'KAFKA_BOOTSTRAP_SERVERS',
        'KAFKA_USERNAME', 
        'KAFKA_PASSWORD',
        'KAFKA_TOPIC',
        'DATABASE_URL'
    ]
    
    print("Checking environment variables:")
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
            print(f"❌ {var}: Not set")
        else:
            if 'PASSWORD' in var or 'SECRET' in var:
                print(f"✅ {var}: ***")
            else:
                print(f"✅ {var}: {value}")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("All required environment variables are set")
    return True

def main():
    """Main function to consume Kafka transactions"""
    try:
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script location: {Path(__file__).parent}")
        print(f"Backend directory: {backend_dir}")
        print(f"Looking for .env at: {env_path}")
        print(f".env file exists: {env_path.exists()}")
        
        # Validate environment
        if not validate_environment():
            return 1
            
        # Create database tables if they don't exist
        logger.info("Creating database tables...")
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            return 1
        
        # Initialize Kafka consumer
        logger.info("Initializing Kafka consumer...")
        try:
            consumer = KafkaTransactionConsumer()
            logger.info("Kafka consumer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Kafka consumer: {e}")
            return 1
        
        # Consume and save transactions
        logger.info("Starting Kafka consumer...")
        result = consumer.consume_and_save(max_messages=2000, timeout=120)
        
        # Print results
        print("\n" + "="*50)
        print("KAFKA CONSUMPTION RESULTS")
        print("="*50)
        print(f"Status: {result['status']}")
        print(f"Messages Consumed: {result['consumed']}")
        print(f"Transactions Saved: {result['saved']}")
        if 'duplicates' in result:
            print(f"Duplicates Skipped: {result['duplicates']}")
        print(f"Message: {result['message']}")
        print("="*50)
        
        return 0 if result['status'] in ['success', 'no_data'] else 1
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
