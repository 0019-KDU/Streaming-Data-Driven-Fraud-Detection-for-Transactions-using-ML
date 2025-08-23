from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey('customers.customer_id', ondelete='CASCADE'), nullable=False, index=True)  # FOREIGN KEY
    amount = Column(Float, nullable=False)
    currency = Column(String(10), nullable=False, default="USD")
    merchant = Column(String(255), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    location = Column(String(10), nullable=False)  # This matches your Kafka data
    is_fraud = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to customer (many-to-one)
    customer = relationship("Customer", back_populates="transactions")
