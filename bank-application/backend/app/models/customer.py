from sqlalchemy import Column, Integer, String, Date, Text, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base

class Customer(Base):
    __tablename__ = "customers"
    
    customer_id = Column(Integer, primary_key=True, index=True)  # PRIMARY KEY
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    national_id = Column(String(20), unique=True, nullable=False)
    phone = Column(String(15), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    address = Column(Text, nullable=False)
    country_code = Column(String(3), nullable=False, server_default='US')  # NEW FIELD
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationship to transactions (one-to-many)
    transactions = relationship("Transaction", back_populates="customer", cascade="all, delete-orphan")
    accounts = relationship("Account", back_populates="customer")