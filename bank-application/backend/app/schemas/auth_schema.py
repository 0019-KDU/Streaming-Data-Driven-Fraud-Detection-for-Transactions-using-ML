from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: 'UserResponse'

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    is_admin: bool
    is_active: bool
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class CreateAdminRequest(BaseModel):
    email: EmailStr
    full_name: str
    password: str

class TokenData(BaseModel):
    email: Optional[str] = None
