from sqlalchemy import Column, Integer, String, DateTime, Boolean, DECIMAL, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base
import enum

class AccountType(enum.Enum):
    SAVINGS = "savings"
    CHECKING = "checking"
    BUSINESS = "business"
    JOINT = "joint"
    STUDENT = "student"
    SENIOR = "senior"

class AccountStatus(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    CLOSED = "closed"
    SUSPENDED = "suspended"

class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    account_number = Column(String(20), unique=True, index=True, nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), nullable=False)
    account_type = Column(Enum(AccountType), nullable=False)
    account_status = Column(Enum(AccountStatus), default=AccountStatus.ACTIVE)
    balance = Column(DECIMAL(15, 2), default=0.00)
    linked_services = Column(Text)  # JSON string of linked services
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_transaction_date = Column(DateTime(timezone=True))
    interest_rate = Column(DECIMAL(5, 4), default=0.0000)  # Annual interest rate
    minimum_balance = Column(DECIMAL(10, 2), default=0.00)
    overdraft_limit = Column(DECIMAL(10, 2), default=0.00)
    
    # Relationships
    customer = relationship("Customer", back_populates="accounts")
