from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from app.services.customer_service import CustomerService
from app.services.transaction_service import TransactionService
from app.schemas.customer_schema import CustomerResponse
from app.schemas.transaction_schema import TransactionResponse
from app.db.session import get_db
from app.api.deps import get_current_admin

router = APIRouter()

@router.get("/", response_model=List[CustomerResponse])
def get_customers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_admin = Depends(get_current_admin)
):
    """Get all customers with pagination"""
    customer_service = CustomerService(db)
    customers = customer_service.get_customers(skip, limit)
    return customers

@router.get("/{customer_id}", response_model=CustomerResponse)
def get_customer(
    customer_id: int,
    db: Session = Depends(get_db),
    current_admin = Depends(get_current_admin)
):
    """Get customer by ID"""
    customer_service = CustomerService(db)
    customer = customer_service.get_customer_by_id(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer

# ✅ ADD THIS NEW ENDPOINT - This was missing!
@router.get("/{customer_id}/transactions", response_model=List[TransactionResponse])
def get_customer_transactions(
    customer_id: int,
    db: Session = Depends(get_db),
    current_admin = Depends(get_current_admin)
):
    """Get all transactions for a specific customer"""
    # First verify customer exists
    customer_service = CustomerService(db)
    customer = customer_service.get_customer_by_id(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # Get transactions for this customer
    transaction_service = TransactionService(db)
    transactions = transaction_service.get_transactions_by_customer(customer_id)
    
    return transactions
