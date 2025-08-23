from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.transaction import Transaction
from app.schemas.transaction_schema import TransactionCreate
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class TransactionService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_transactions_by_customer(self, customer_id: int) -> List[Transaction]:
        """Get all transactions for a specific customer"""
        try:
            transactions = self.db.query(Transaction).filter(
                Transaction.user_id == customer_id
            ).order_by(Transaction.timestamp.desc()).all()
            
            logger.info(f"Found {len(transactions)} transactions for customer {customer_id}")
            return transactions
            
        except Exception as e:
            logger.error(f"Error fetching transactions for customer {customer_id}: {e}")
            return []
    
    def get_transaction_by_id(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction by transaction_id"""
        try:
            transaction = self.db.query(Transaction).filter(
                Transaction.transaction_id == transaction_id
            ).first()
            
            if transaction:
                logger.info(f"Found transaction {transaction_id}")
            else:
                logger.warning(f"Transaction {transaction_id} not found")
                
            return transaction
            
        except Exception as e:
            logger.error(f"Error fetching transaction {transaction_id}: {e}")
            return None
    
    def get_transactions(self, skip: int = 0, limit: int = 100) -> List[Transaction]:
        """Get paginated transactions"""
        try:
            transactions = self.db.query(Transaction).offset(skip).limit(limit).all()
            logger.info(f"Retrieved {len(transactions)} transactions (skip={skip}, limit={limit})")
            return transactions
            
        except Exception as e:
            logger.error(f"Error fetching transactions: {e}")
            return []
    
    def get_transaction_count(self) -> int:
        """Get total count of transactions"""
        try:
            count = self.db.query(Transaction).count()
            logger.info(f"Total transactions count: {count}")
            return count
            
        except Exception as e:
            logger.error(f"Error counting transactions: {e}")
            return 0
    
    def get_fraud_count(self) -> int:
        """Get count of fraudulent transactions"""
        try:
            count = self.db.query(Transaction).filter(Transaction.is_fraud == True).count()
            logger.info(f"Fraud transactions count: {count}")
            return count
            
        except Exception as e:
            logger.error(f"Error counting fraud transactions: {e}")
            return 0
    
    def get_total_amount(self) -> float:
        """Get total amount of all transactions"""
        try:
            result = self.db.query(func.sum(Transaction.amount)).scalar()
            total = float(result) if result else 0.0
            logger.info(f"Total transaction amount: {total}")
            return total
            
        except Exception as e:
            logger.error(f"Error calculating total amount: {e}")
            return 0.0

    # Add your existing methods here...
    def create_transactions_bulk(self, transactions_data: List[TransactionCreate]) -> int:
        """Create multiple transactions in bulk"""
        created_count = 0
        
        for transaction_data in transactions_data:
            try:
                existing = self.db.query(Transaction).filter(
                    Transaction.transaction_id == transaction_data.transaction_id
                ).first()
                
                if not existing:
                    db_transaction = Transaction(**transaction_data.dict())
                    self.db.add(db_transaction)
                    created_count += 1
                    
            except Exception as e:
                logger.error(f"Error creating transaction {transaction_data.transaction_id}: {e}")
                continue
        
        try:
            self.db.commit()
            logger.info(f"Successfully created {created_count} transactions")
            return created_count
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error committing transactions: {e}")
            return 0

def get_transactions_by_customer(self, customer_id: int) -> List[Transaction]:
    """Get all transactions for a specific customer"""
    try:
        transactions = self.db.query(Transaction).filter(
            Transaction.user_id == customer_id
        ).order_by(Transaction.timestamp.desc()).all()
        
        logger.info(f"Found {len(transactions)} transactions for customer {customer_id}")
        return transactions
        
    except Exception as e:
        logger.error(f"Error fetching transactions for customer {customer_id}: {e}")
        return []
