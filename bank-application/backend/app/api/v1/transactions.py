from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from app.services.transaction_service import TransactionService
from app.schemas.transaction_schema import TransactionResponse
from app.db.session import get_db
from app.api.deps import get_current_admin

router = APIRouter()

# Fix: Handle both with and without trailing slash
@router.get("/", response_model=List[TransactionResponse])
@router.get("", response_model=List[TransactionResponse])  # Add this line
def get_transactions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_admin = Depends(get_current_admin)
):
    """Get all transactions with pagination"""
    transaction_service = TransactionService(db)
    transactions = transaction_service.get_transactions(skip, limit)
    return transactions

# ✅ ADD THIS NEW ENDPOINT - This was missing!
@router.get("/{transaction_id}", response_model=TransactionResponse)
def get_transaction(
    transaction_id: str,  # Use str since your transaction_id is a UUID string
    db: Session = Depends(get_db),
    current_admin = Depends(get_current_admin)
):
    """Get transaction by ID"""
    transaction_service = TransactionService(db)
    transaction = transaction_service.get_transaction_by_id(transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction