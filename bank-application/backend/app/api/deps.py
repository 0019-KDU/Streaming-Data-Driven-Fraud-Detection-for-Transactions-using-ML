from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.config.security import verify_token
from app.services.auth_service import AuthService
from app.db.session import get_db

security = HTTPBearer()

def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    token = credentials.credentials
    email = verify_token(token)
    
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    auth_service = AuthService(db)
    admin = auth_service.get_admin_by_email(email)
    
    if admin is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin not found or inactive"
        )
    
    return admin
