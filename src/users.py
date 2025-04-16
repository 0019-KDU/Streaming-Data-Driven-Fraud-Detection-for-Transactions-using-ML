import os
import logging
import psycopg2
from faker import Faker
from dotenv import load_dotenv
from contextlib import closing

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def generate_users(num_users: int, start_id: int = 1000):
    """Generate fake users and insert into PostgreSQL"""
    fake = Faker()
    
    # Get database configuration from environment variables
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'postgres'),
        'database': os.getenv('POSTGRES_DB', 'notify-data'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', '12345q')
    }

    try:
        with closing(psycopg2.connect(**db_config)) as conn:
            with conn.cursor() as cur:
                # Create table if not exists
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INT PRIMARY KEY,
                        email VARCHAR(255) NOT NULL UNIQUE
                    )
                """)
                
                # Generate user data
                users = [
                    (user_id, fake.unique.email())
                    for user_id in range(start_id, start_id + num_users)
                ]
                
                # Batch insert using executemany
                cur.executemany(
                    "INSERT INTO users (user_id, email) VALUES (%s, %s)",
                    users
                )
                conn.commit()
                
                logger.info(f"Successfully inserted {len(users)} users")

    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    generate_users(
        num_users=1000,
        start_id=1000
    )