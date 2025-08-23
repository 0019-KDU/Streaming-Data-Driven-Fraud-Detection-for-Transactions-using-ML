from sqlalchemy.orm import Session
from app.models.customer import Customer
from app.schemas.customer_schema import CustomerCreate
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class CustomerService:
    def __init__(self, db: Session):
        self.db = db
    
    def create_customer(self, customer_data: CustomerCreate) -> Customer:
        """Create a new customer"""
        # Check if customer already exists
        existing = self.db.query(Customer).filter(
            Customer.customer_id == customer_data.customer_id
        ).first()
        
        if existing:
            logger.warning(f"Customer {customer_data.customer_id} already exists")
            return existing
        
        db_customer = Customer(**customer_data.dict())
        self.db.add(db_customer)
        self.db.commit()
        self.db.refresh(db_customer)
        
        logger.info(f"Created customer {db_customer.customer_id}")
        return db_customer
    
    def create_customers_bulk(self, customers_data: List[CustomerCreate]) -> int:
        """Create multiple customers in bulk"""
        created_count = 0
        
        for customer_data in customers_data:
            try:
                # Check if customer already exists
                existing = self.db.query(Customer).filter(
                    Customer.customer_id == customer_data.customer_id
                ).first()
                
                if not existing:
                    db_customer = Customer(**customer_data.dict())
                    self.db.add(db_customer)
                    created_count += 1
                    
            except Exception as e:
                logger.error(f"Error creating customer {customer_data.customer_id}: {e}")
                continue
        
        try:
            self.db.commit()
            logger.info(f"Successfully created {created_count} customers")
            return created_count
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error committing customers: {e}")
            return 0
    
    def get_customer_count(self) -> int:
        """Get total count of customers"""
        return self.db.query(Customer).count()
    
    def get_customers(self, skip: int = 0, limit: int = 100) -> List[Customer]:
        """Get paginated customers"""
        return self.db.query(Customer).offset(skip).limit(limit).all()
    
    def get_customer_by_id(self, customer_id: int) -> Optional[Customer]:
        """Get customer by ID"""
        return self.db.query(Customer).filter(Customer.customer_id == customer_id).first()
