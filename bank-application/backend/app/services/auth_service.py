from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.auth_schema import CreateAdminRequest
from app.config.security import get_password_hash, verify_password
from datetime import datetime
from fastapi import HTTPException, status

class AuthService:
    def __init__(self, db: Session):
        self.db = db
    
    def create_admin_user(self, admin_data: CreateAdminRequest):
        # Check if admin already exists
        existing_admin = self.db.query(User).filter(User.email == admin_data.email).first()
        if existing_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Admin with this email already exists"
            )
        
        # Create new admin
        hashed_password = get_password_hash(admin_data.password)
        db_admin = User(
            email=admin_data.email,
            full_name=admin_data.full_name,
            hashed_password=hashed_password,
            is_admin=True,
            is_active=True
        )
        
        self.db.add(db_admin)
        self.db.commit()
        self.db.refresh(db_admin)
        return db_admin
    
    def authenticate_admin(self, email: str, password: str):
        admin = self.db.query(User).filter(
            User.email == email,
            User.is_admin == True,
            User.is_active == True
        ).first()
        
        if not admin or not verify_password(password, admin.hashed_password):
            return None
        
        # Update last login
        admin.last_login = datetime.utcnow()
        self.db.commit()
        
        return admin
    
    def get_admin_by_email(self, email: str):
        return self.db.query(User).filter(
            User.email == email,
            User.is_admin == True,
            User.is_active == True
        ).first()
