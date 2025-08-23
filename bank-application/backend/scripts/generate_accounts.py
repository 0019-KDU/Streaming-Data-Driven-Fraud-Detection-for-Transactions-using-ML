#!/usr/bin/env python3
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Setup path and environment
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))
load_dotenv(backend_dir / '.env')

from app.services.account_service import AccountService
from app.models.customer import Customer
from app.models.account import Account  # <- ADD THIS IMPORT
from app.config.database import engine
from app.db.base import Base
from app.db.session import get_db

# Configure logging (removed emojis for Windows compatibility)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('account_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def generate_accounts_for_all_customers():
    """Generate accounts for all existing customers"""
    db = next(get_db())
    try:
        account_service = AccountService(db)
        
        # Get all customers without accounts
        customers_with_accounts = db.query(Customer.customer_id).join(
            Account, Customer.customer_id == Account.customer_id, isouter=True
        ).filter(Account.id.isnot(None)).distinct()
        
        customers_without_accounts = db.query(Customer).filter(
            ~Customer.customer_id.in_(customers_with_accounts)
        ).all()
        
        logger.info(f"Found {len(customers_without_accounts)} customers without accounts")
        
        total_accounts_created = 0
        
        for customer in customers_without_accounts:
            try:
                # Generate 1-3 accounts per customer
                accounts = account_service.create_accounts_for_customer(customer.customer_id)
                total_accounts_created += len(accounts)
                
                logger.info(f"Created {len(accounts)} accounts for customer {customer.customer_id} ({customer.first_name} {customer.last_name})")
                
                # Progress indicator
                if total_accounts_created % 25 == 0:
                    logger.info(f"Generated {total_accounts_created} accounts so far...")
                    
            except Exception as e:
                logger.error(f"Error generating accounts for customer {customer.customer_id}: {e}")
                continue
        
        # Show final statistics
        stats = account_service.get_account_stats()
        logger.info("=== FINAL STATISTICS ===")
        logger.info(f"Total Accounts: {stats['total_accounts']}")
        logger.info(f"Total Balance: ${stats['total_balance']:,.2f}")
        logger.info(f"Average Balance: ${stats['average_balance']:,.2f}")
        logger.info(f"Active Accounts: {stats['active_accounts']}")
        
        for account_type, count in stats['account_types'].items():
            logger.info(f"{account_type.title()} Accounts: {count}")
        
        return total_accounts_created
        
    except Exception as e:
        logger.error(f"Error in account generation: {e}")
        return 0
    finally:
        db.close()

def main():
    """Main function to generate accounts"""
    try:
        logger.info("=== BANK ACCOUNT GENERATION SYSTEM ===")
        logger.info("Starting account generation for all customers...")
        
        # Create database tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
        
        # Generate accounts
        total_created = generate_accounts_for_all_customers()
        
        print("\n" + "="*70)
        print("ACCOUNT GENERATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Total Accounts Created: {total_created}")
        print(f"Your banking system now has realistic account data!")
        print(f"Account types: Savings, Checking, Business, Student, Senior, Joint")
        print(f"Realistic balances and interest rates generated")
        print(f"Linked services automatically assigned")
        print(f"Account creation dates spanning last 5 years")
        print("="*70)
        print("Ready to view accounts in your frontend application!")
        print("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Critical error in main process: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
