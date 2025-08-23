from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from app.models.account import AccountType, AccountStatus

class AccountBase(BaseModel):
    account_type: AccountType
    account_status: AccountStatus = AccountStatus.ACTIVE
    balance: Decimal = Field(default=0.00, ge=0)
    linked_services: Optional[str] = None
    interest_rate: Decimal = Field(default=0.0000, ge=0)
    minimum_balance: Decimal = Field(default=0.00, ge=0)
    overdraft_limit: Decimal = Field(default=0.00, ge=0)

class AccountCreate(AccountBase):
    customer_id: int

class AccountResponse(AccountBase):
    id: int
    account_number: str
    customer_id: int
    created_at: datetime
    updated_at: Optional[datetime]
    last_transaction_date: Optional[datetime]
    
    class Config:
        from_attributes = True
        use_enum_values = True

class AccountWithCustomer(AccountResponse):
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None

class AccountStats(BaseModel):
    total_accounts: int
    total_balance: float
    active_accounts: int
    average_balance: float
    account_types: dict
