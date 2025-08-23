from pydantic import BaseModel, EmailStr
from datetime import date, datetime
from typing import Optional

class CustomerBase(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: date
    national_id: str
    phone: str
    email: EmailStr
    address: str
    country_code: str  # NEW FIELD

class CustomerCreate(CustomerBase):
    customer_id: int

class CustomerResponse(CustomerBase):
    customer_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
