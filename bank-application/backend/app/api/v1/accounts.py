from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from app.services.account_service import AccountService
from app.schemas.account_schema import AccountResponse, AccountStats
from app.db.session import get_db
from app.api.deps import get_current_admin

router = APIRouter()

@router.get("/", response_model=List[AccountResponse])
def get_accounts(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_admin = Depends(get_current_admin)
):
    """Get all accounts with pagination"""
    account_service = AccountService(db)
    accounts = account_service.get_all_accounts(skip, limit)
    return accounts

@router.get("/customer/{customer_id}", response_model=List[AccountResponse])
def get_customer_accounts(
    customer_id: int,
    db: Session = Depends(get_db),
    current_admin = Depends(get_current_admin)
):
    """Get all accounts for a specific customer"""
    account_service = AccountService(db)
    accounts = account_service.get_accounts_by_customer(customer_id)
    return accounts

@router.get("/stats", response_model=AccountStats)
def get_account_stats(
    db: Session = Depends(get_db),
    current_admin = Depends(get_current_admin)
):
    """Get account statistics"""
    account_service = AccountService(db)
    stats = account_service.get_account_stats()
    return stats

@router.get("/{account_number}", response_model=AccountResponse)
def get_account(
    account_number: str,
    db: Session = Depends(get_db),
    current_admin = Depends(get_current_admin)
):
    """Get account by account number"""
    account_service = AccountService(db)
    account = account_service.get_account_by_number(account_number)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    return account
