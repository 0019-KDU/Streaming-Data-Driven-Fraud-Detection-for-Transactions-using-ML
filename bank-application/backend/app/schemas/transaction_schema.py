from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class TransactionBase(BaseModel):
    transaction_id: str
    user_id: int
    amount: float
    currency: str
    merchant: str
    timestamp: datetime
    location: str
    is_fraud: bool = False

class TransactionCreate(TransactionBase):
    pass

class TransactionResponse(TransactionBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class TransactionKafka(BaseModel):
    transaction_id: str
    user_id: int
    amount: float
    currency: str
    merchant: str
    timestamp: str  # Will be converted to datetime
    location: str
    is_fraud: int  # Will be converted to boolean
