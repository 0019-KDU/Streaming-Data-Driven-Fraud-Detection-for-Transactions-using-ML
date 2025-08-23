from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.services.customer_service import CustomerService
from app.services.transaction_service import TransactionService
from app.db.session import get_db
from app.api.deps import get_current_admin

router = APIRouter()

@router.get("/stats")
def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_admin = Depends(get_current_admin)
):
    """Get dashboard statistics"""
    customer_service = CustomerService(db)
    transaction_service = TransactionService(db)
    
    total_customers = customer_service.get_customer_count()
    total_transactions = transaction_service.get_transaction_count()
    fraud_count = transaction_service.get_fraud_count()
    total_amount = transaction_service.get_total_amount()
    
    return {
        "totalCustomers": total_customers,
        "totalTransactions": total_transactions,
        "fraudTransactions": fraud_count,
        "totalAmount": total_amount
    }
