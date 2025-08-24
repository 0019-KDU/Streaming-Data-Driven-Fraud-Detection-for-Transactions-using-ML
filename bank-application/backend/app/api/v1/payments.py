from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.services.payment_service import PaymentService, PaymentValidationError
from app.schemas.payment_schema import PaymentRequest, PaymentResponse
from app.db.session import get_db
import logging
import app.models.customer as customer

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=PaymentResponse)
def process_payment(
    payment_request: PaymentRequest,
    db: Session = Depends(get_db)
):
    """
    Process a payment request
    
    This endpoint validates customer and account details, then processes the payment
    if all validations pass. Only existing customers can make payments.
    """
    try:
        payment_service = PaymentService(db)
        result = payment_service.process_payment(payment_request)
        
        return PaymentResponse(
            message=result["message"],
            transaction_id=result["transaction_id"],
            status=result["status"]
        )
        
    except PaymentValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in payment processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during payment processing"
        )

@router.get("/validate-customer/{customer_id}")
def validate_customer_exists(
    customer_id: int,
    db: Session = Depends(get_db)
):
    """
    Check if a customer exists (for pre-validation)
    """
    try:
        payment_service = PaymentService(db)
        customer = db.query(customer.Customer).filter(customer.Customer.customer_id == customer_id).first()

        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer with ID {customer_id} not found"
            )
        
        return {
            "customer_id": customer.customer_id,
            "name": f"{customer.first_name} {customer.last_name}",
            "email": customer.email,
            "status": "exists"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating customer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error validating customer"
        )
