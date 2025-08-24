from sqlalchemy.orm import Session
from sqlalchemy import and_
from app.models.customer import Customer
from app.models.account import Account
from app.models.transaction import Transaction
from app.schemas.payment_schema import PaymentRequest
from app.schemas.transaction_schema import TransactionCreate
from datetime import datetime
from faker import Faker
import logging

logger = logging.getLogger(__name__)
fake = Faker()

class PaymentValidationError(Exception):
    pass

class PaymentService:
    def __init__(self, db: Session):
        self.db = db
    
    def validate_customer(self, payment_request: PaymentRequest) -> Customer:
        """Validate customer exists and details match"""
        customer = self.db.query(Customer).filter(
            Customer.customer_id == payment_request.customer_id
        ).first()
        
        if not customer:
            raise PaymentValidationError(f"Customer with ID {payment_request.customer_id} not found")
        
        # Validate customer details match
        validation_errors = []
        
        if customer.first_name.lower() != payment_request.first_name.lower():
            validation_errors.append("First name does not match")
        
        if customer.last_name.lower() != payment_request.last_name.lower():
            validation_errors.append("Last name does not match")
        
        if customer.date_of_birth != payment_request.date_of_birth:
            validation_errors.append("Date of birth does not match")
        
        if customer.national_id != payment_request.national_id:
            validation_errors.append("National ID does not match")
        
        if customer.phone != payment_request.phone:
            validation_errors.append("Phone number does not match")
        
        if customer.email.lower() != payment_request.email.lower():
            validation_errors.append("Email does not match")
        
        if customer.country_code != payment_request.country_code:
            validation_errors.append("Country code does not match")
        
        if validation_errors:
            raise PaymentValidationError(f"Customer validation failed: {', '.join(validation_errors)}")
        
        return customer
    
    def validate_account(self, payment_request: PaymentRequest, customer: Customer) -> Account:
        """Validate account exists and belongs to customer"""
        account = self.db.query(Account).filter(
            and_(
                Account.account_number == payment_request.account_number,
                Account.customer_id == customer.customer_id
            )
        ).first()
        
        if not account:
            raise PaymentValidationError(f"Account {payment_request.account_number} not found for customer {customer.customer_id}")
        
        # Check account status
        if account.account_status.value != 'active':
            raise PaymentValidationError(f"Account {payment_request.account_number} is not active")
        
        # Check account type matches
        if account.account_type.value != payment_request.account_type.lower():
            raise PaymentValidationError("Account type does not match")
        
        # Check sufficient balance (for debit transactions)
        if account.balance < payment_request.amount:
            raise PaymentValidationError("Insufficient account balance")
        
        return account
    
    def process_payment(self, payment_request: PaymentRequest) -> dict:
        """Process payment after validation"""
        try:
            # Step 1: Validate customer
            customer = self.validate_customer(payment_request)
            logger.info(f"Customer validation successful for ID: {customer.customer_id}")
            
            # Step 2: Validate account
            account = self.validate_account(payment_request, customer)
            logger.info(f"Account validation successful for: {account.account_number}")
            
            # Step 3: Generate transaction ID
            transaction_id = fake.uuid4()
            
            # Step 4: Create transaction record
            transaction_data = TransactionCreate(
                transaction_id=transaction_id,
                user_id=customer.customer_id,
                amount=float(payment_request.amount),
                currency=payment_request.currency,
                merchant=payment_request.merchant,
                timestamp=datetime.utcnow(),
                location=payment_request.location,
                is_fraud=False  # Real fraud detection would be implemented here
            )
            
            # Step 5: Save transaction to database
            db_transaction = Transaction(**transaction_data.dict())
            self.db.add(db_transaction)
            
            # Step 6: Update account balance (deduct payment amount)
            account.balance = float(account.balance) - float(payment_request.amount)
            account.last_transaction_date = datetime.utcnow()
            
            # Step 7: Commit all changes
            self.db.commit()
            self.db.refresh(db_transaction)
            
            logger.info(f"Payment processed successfully. Transaction ID: {transaction_id}")
            
            return {
                "message": "Payment successful",
                "transaction_id": transaction_id,
                "status": "success",
                "customer_id": customer.customer_id,
                "account_number": account.account_number,
                "amount": str(payment_request.amount),
                "currency": payment_request.currency
            }
            
        except PaymentValidationError as e:
            logger.error(f"Payment validation failed: {str(e)}")
            raise e
        except Exception as e:
            self.db.rollback()
            logger.error(f"Payment processing error: {str(e)}")
            raise Exception(f"Payment processing failed: {str(e)}")
