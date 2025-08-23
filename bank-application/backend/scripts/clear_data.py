#!/usr/bin/env python3
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup path
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
sys.path.append(str(backend_dir))
load_dotenv(backend_dir / '.env')

from app.db.session import get_db
from app.models.customer import Customer
from app.models.transaction import Transaction

def clear_all_data():
    db = next(get_db())
    try:
        # Delete all transactions first (due to foreign key)
        deleted_transactions = db.query(Transaction).delete()
        
        # Delete all customers
        deleted_customers = db.query(Customer).delete()
        
        # Commit changes
        db.commit()
        
        print(f"✅ Deleted {deleted_transactions} transactions")
        print(f"✅ Deleted {deleted_customers} customers")
        print("✅ Database cleared successfully!")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error clearing data: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    clear_all_data()
