from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import datetime, date
from decimal import Decimal

class PaymentRequest(BaseModel):
    # Customer Information
    customer_id: int
    first_name: str
    last_name: str
    date_of_birth: date
    national_id: str
    phone: str
    email: EmailStr
    address: str
    country_code: str
    
    # Transaction Information
    amount: Decimal
    currency: str = "USD"
    merchant: str
    location: str
    
    # Account Information
    account_number: str
    account_type: str
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be greater than 0')
        return v
    
    @validator('currency')
    def validate_currency(cls, v):
        allowed_currencies = ['USD', 'EUR', 'GBP', 'CAD']
        if v not in allowed_currencies:
            raise ValueError(f'Currency must be one of: {allowed_currencies}')
        return v

class PaymentResponse(BaseModel):
    message: str
    transaction_id: str
    status: str = "success"
