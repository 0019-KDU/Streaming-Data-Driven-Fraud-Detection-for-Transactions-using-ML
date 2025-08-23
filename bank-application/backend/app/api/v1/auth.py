from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.schemas.auth_schema import LoginRequest, LoginResponse, CreateAdminRequest, UserResponse
from app.services.auth_service import AuthService
from app.config.security import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from app.db.session import get_db
from app.api.deps import get_current_admin
from datetime import timedelta

router = APIRouter()

@router.post("/login", response_model=LoginResponse)
def login_admin(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    admin = auth_service.authenticate_admin(login_data.email, login_data.password)
    
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": admin.email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": admin
    }

@router.post("/create-admin", response_model=UserResponse)
def create_admin(
    admin_data: CreateAdminRequest,
    db: Session = Depends(get_db)
):
    auth_service = AuthService(db)
    admin = auth_service.create_admin_user(admin_data)
    return admin

@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_admin = Depends(get_current_admin)
):
    return current_admin

@router.post("/logout")
def logout_admin(
    current_admin = Depends(get_current_admin)
):
    return {"message": "Successfully logged out"}
