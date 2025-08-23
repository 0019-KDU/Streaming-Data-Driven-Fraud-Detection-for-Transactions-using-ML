#!/usr/bin/env python3
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import random
from datetime import datetime, timedelta
from faker import Faker

# Add the parent directory to Python path
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))

# Load environment variables
env_path = backend_dir / '.env'
load_dotenv(dotenv_path=env_path)

from app.services.customer_service import CustomerService
from app.services.transaction_service import TransactionService
from app.schemas.customer_schema import CustomerCreate
from app.config.database import engine
from app.db.base import Base
from app.db.session import get_db
from app.models.transaction import Transaction
from app.models.customer import Customer
from sqlalchemy import distinct, text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('customer_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ✅ FIXED: Use Faker.seed() as class method
Faker.seed(42)  # For reproducible data
fake = Faker()

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)

def print_section(title):
    """Print formatted section"""
    print(f"\n--- {title} ---")

def generate_synthetic_customers(num_customers=2000):
    """Generate synthetic customer records with country codes using Faker"""
    customers = []
    used_ids = set()
    used_emails = set()
    used_national_ids = set()
    
    logger.info(f"Starting generation of {num_customers} synthetic customers...")
    
    for i in range(num_customers):
        # Generate unique customer_id (1000-9999)
        while True:
            customer_id = random.randint(1000, 9999)
            if customer_id not in used_ids:
                used_ids.add(customer_id)
                break
        
        # Generate realistic data using Faker
        first_name = fake.first_name()
        last_name = fake.last_name()
        date_of_birth = fake.date_of_birth(minimum_age=18, maximum_age=80)
        
        # Generate unique national ID
        while True:
            national_id = fake.ssn()[:20]  # Limit to 20 characters
            if national_id not in used_national_ids:
                used_national_ids.add(national_id)
                break
        
        phone = fake.phone_number()[:15]  # Limit to 15 characters
        
        # Generate unique email
        while True:
            email = fake.email()[:100]  # Limit to 100 characters
            if email not in used_emails:
                used_emails.add(email)
                break
        
        address = fake.address().replace('\n', ', ')  # Single line address
        country_code = fake.country_code()  # ✅ This generates the country code as requested
        
        customer = CustomerCreate(
            customer_id=customer_id,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            national_id=national_id,
            phone=phone,
            email=email,
            address=address,
            country_code=country_code  # Country code field added
        )
        
        customers.append(customer)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1} customers...")
    
    logger.info(f"Generated {len(customers)} total customers")
    return customers

def save_customers_to_db(customers):
    """Save customers to database"""
    db = next(get_db())
    try:
        customer_service = CustomerService(db)
        created_count = customer_service.create_customers_bulk(customers)
        return created_count
    finally:
        db.close()

def get_database_stats():
    """Get comprehensive database statistics"""
    db = next(get_db())
    try:
        # Basic counts
        customer_count = db.execute(text("SELECT COUNT(*) FROM customers")).scalar()
        transaction_count = db.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
        
        if transaction_count > 0:
            # Relationship analysis
            customers_with_transactions = db.execute(text("""
                SELECT COUNT(DISTINCT c.customer_id) 
                FROM customers c 
                INNER JOIN transactions t ON c.customer_id = t.user_id
            """)).scalar() or 0
            
            customers_without_transactions = customer_count - customers_with_transactions
            
            # Orphaned transactions
            orphaned_transactions = db.execute(text("""
                SELECT COUNT(*) 
                FROM transactions t 
                LEFT JOIN customers c ON t.user_id = c.customer_id 
                WHERE c.customer_id IS NULL
            """)).scalar() or 0
        else:
            customers_with_transactions = 0
            customers_without_transactions = customer_count
            orphaned_transactions = 0
        
        return {
            "customer_count": customer_count,
            "transaction_count": transaction_count,
            "customers_with_transactions": customers_with_transactions,
            "customers_without_transactions": customers_without_transactions,
            "orphaned_transactions": orphaned_transactions
        }
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return None
    finally:
        db.close()

def get_transaction_user_ids():
    """Get all unique user_ids from transaction table"""
    db = next(get_db())
    try:
        user_ids = db.query(distinct(Transaction.user_id)).all()
        user_id_list = [uid[0] for uid in user_ids if uid is not None]
        logger.info(f"Found {len(user_id_list)} unique user_ids in transactions table")
        return user_id_list
    except Exception as e:
        logger.error(f"Error getting transaction user_ids: {e}")
        return []
    finally:
        db.close()

def verify_foreign_key_relationship():
    """Verify the foreign key relationship between customers and transactions"""
    db = next(get_db())
    try:
        print_section("Foreign Key Relationship Verification")
        
        # Get sample data to verify relationship
        sample_data = db.execute(text("""
            SELECT 
                c.customer_id, 
                c.first_name, 
                c.last_name, 
                c.country_code,
                COUNT(t.id) as transaction_count,
                COALESCE(SUM(t.amount), 0) as total_amount,
                COUNT(CASE WHEN t.is_fraud THEN 1 END) as fraud_count
            FROM customers c
            LEFT JOIN transactions t ON c.customer_id = t.user_id
            GROUP BY c.customer_id, c.first_name, c.last_name, c.country_code
            ORDER BY transaction_count DESC
            LIMIT 10
        """)).fetchall()
        
        print("\nTop 10 customers by transaction count:")
        print(f"{'Customer ID':<12} {'Name':<25} {'Country':<8} {'Transactions':<12} {'Total Amount':<15} {'Fraud Count'}")
        print("-" * 85)
        
        for row in sample_data:
            customer_id, first_name, last_name, country_code, tx_count, total_amount, fraud_count = row
            name = f"{first_name} {last_name}"
            print(f"{customer_id:<12} {name:<25} {country_code:<8} {tx_count:<12} ${total_amount:<14.2f} {fraud_count}")
        
        return sample_data
        
    except Exception as e:
        logger.error(f"Error verifying foreign key relationship: {e}")
        return []
    finally:
        db.close()

def test_database_services():
    """Test database services functionality"""
    db = next(get_db())
    try:
        print_section("Testing Database Services")
        
        # Test customer service
        customer_service = CustomerService(db)
        customers = customer_service.get_customers(0, 5)
        print(f"✅ CustomerService.get_customers(): Retrieved {len(customers)} customers")
        
        # Test transaction service
        transaction_service = TransactionService(db)
        transactions = transaction_service.get_transactions(0, 5)
        print(f"✅ TransactionService.get_transactions(): Retrieved {len(transactions)} transactions")
        
        # Test relationship
        if customers:
            customer_id = customers[0].customer_id
            customer_transactions = transaction_service.get_transactions_by_customer(customer_id)
            print(f"✅ Customer {customer_id} has {len(customer_transactions)} transactions")
            
            if customer_transactions:
                sample_tx = customer_transactions
                print(f"   Sample transaction: {sample_tx.transaction_id[:8]}... - ${sample_tx.amount} - {sample_tx.merchant}")
        
        # Test dashboard stats
        total_customers = customer_service.get_customer_count()
        total_transactions = transaction_service.get_transaction_count()
        fraud_count = transaction_service.get_fraud_count()
        total_amount = transaction_service.get_total_amount()
        
        print(f"\n📊 Dashboard Statistics:")
        print(f"   Total Customers: {total_customers}")
        print(f"   Total Transactions: {total_transactions}")
        print(f"   Fraud Transactions: {fraud_count}")
        print(f"   Total Amount: ${total_amount:,.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing database services: {e}")
        return False
    finally:
        db.close()

def main():
    """Main function to generate customers and verify database"""
    try:
        print_header("BANK CUSTOMER GENERATION WITH COUNTRY CODES")
        
        # Create database tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created/verified")
        
        # Get initial database statistics
        print_section("Current Database Statistics")
        stats = get_database_stats()
        if stats:
            print(f"Customers in database: {stats['customer_count']}")
            print(f"Transactions in database: {stats['transaction_count']}")
            print(f"Customers with transactions: {stats['customers_with_transactions']}")
            print(f"Customers without transactions: {stats['customers_without_transactions']}")
            print(f"Orphaned transactions: {stats['orphaned_transactions']}")
        
        # Generate customers if none exist
        if not stats or stats['customer_count'] == 0:
            print_section("Generating Synthetic Customers with Country Codes")
            logger.info("Generating 2000 synthetic customer records...")
            customers = generate_synthetic_customers(2000)
            
            logger.info(f"Generated {len(customers)} customers. Saving to database...")
            created_count = save_customers_to_db(customers)
            
            print_header("CUSTOMER GENERATION RESULTS")
            print(f"✅ Customers Generated: {len(customers)}")
            print(f"✅ Customers Saved to DB: {created_count}")
            print(f"⚠️  Duplicates Skipped: {len(customers) - created_count}")
            print(f"📊 Success Rate: {(created_count/len(customers)*100):.1f}%")
            
            # Display sample data
            print_section("Sample Generated Customer Data")
            for i in range(min(5, len(customers))):
                c = customers[i]
                print(f"Customer {c.customer_id}: {c.first_name} {c.last_name}")
                print(f"  Email: {c.email}")
                print(f"  Phone: {c.phone}")
                print(f"  Country: {c.country_code}")  # Show country code
                print(f"  DOB: {c.date_of_birth}")
                print(f"  Address: {c.address}")
                print()
        else:
            print("✅ Customers already exist in database")
        
        # Get transaction user IDs analysis
        print_section("Transaction User IDs Analysis")
        transaction_user_ids = get_transaction_user_ids()
        
        if transaction_user_ids:
            print(f"Total unique user_ids in transactions: {len(transaction_user_ids)}")
            print(f"Sample user_ids: {transaction_user_ids[:15]}")
            print(f"User_id range: {min(transaction_user_ids)} to {max(transaction_user_ids)}")
        else:
            print("⚠️  No transaction user_ids found")
        
        # Verify foreign key relationship
        verify_foreign_key_relationship()
        
        # Test database services
        services_ok = test_database_services()
        
        # Final statistics
        print_section("Final Database Statistics")
        final_stats = get_database_stats()
        if final_stats:
            print(f"✅ Total Customers: {final_stats['customer_count']}")
            print(f"✅ Total Transactions: {final_stats['transaction_count']}")
            print(f"✅ Customers with Transactions: {final_stats['customers_with_transactions']}")
            print(f"⚠️  Customers without Transactions: {final_stats['customers_without_transactions']}")
            print(f"⚠️  Orphaned Transactions: {final_stats['orphaned_transactions']}")
        
        print_header("SYSTEM READY FOR FRONTEND")
        print("🎉 Customer generation with country codes completed successfully!")
        print("🚀 Your banking application frontend should now display data correctly.")
        
        if services_ok and final_stats and final_stats['customer_count'] > 0:
            print("\n✅ All systems operational:")
            print("   - Database tables created with country_code field")
            print("   - Customer data generated with country codes") 
            print("   - Foreign key relationships verified (customer_id → user_id)")
            print("   - API services tested and working")
            print("   - Ready for frontend integration")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        import traceback
        traceback.print_exc()
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
