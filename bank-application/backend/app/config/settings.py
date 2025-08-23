from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:12345q@localhost:5432/bank_management"
    SECRET_KEY: str = "gdhdgdhsgshshs"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
     # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: Optional[str] = None
    KAFKA_USERNAME: Optional[str] = None
    KAFKA_PASSWORD: Optional[str] = None
    KAFKA_TOPIC: Optional[str] = "transactions"
    
    class Config:
        env_file = ".env"

settings = Settings()
