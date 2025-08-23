from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.models.account import Account, AccountType, AccountStatus
from app.models.customer import Customer
from app.schemas.account_schema import AccountCreate
from typing import List, Optional
import random
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AccountService:
    def __init__(self, db: Session):
        self.db = db
    
    def generate_account_number(self, account_type: AccountType) -> str:
        """Generate a unique account number based on account type"""
        type_prefix = {
            AccountType.SAVINGS: "SAV",
            AccountType.CHECKING: "CHK", 
            AccountType.BUSINESS: "BUS",
            AccountType.JOINT: "JNT",
            AccountType.STUDENT: "STU",
            AccountType.SENIOR: "SNR"
        }
        
        prefix = type_prefix.get(account_type, "ACC")
        
        # Generate unique 10-digit number
        while True:
            number = f"{prefix}{random.randint(1000000000, 9999999999)}"
            existing = self.db.query(Account).filter(Account.account_number == number).first()
            if not existing:
                return number
    
    def generate_linked_services(self, account_type: AccountType) -> List[str]:
        """Generate realistic linked services based on account type"""
        all_services = ["credit_card", "loan", "overdraft", "investment", "mortgage", "insurance"]
        
        if account_type == AccountType.CHECKING:
            base_services = ["credit_card", "overdraft"]
            if random.random() > 0.6:
                base_services.append("loan")
            if random.random() > 0.8:
                base_services.append("insurance")
                
        elif account_type == AccountType.SAVINGS:
            base_services = ["investment"]
            if random.random() > 0.7:
                base_services.append("loan")
            if random.random() > 0.9:
                base_services.append("insurance")
                
        elif account_type == AccountType.BUSINESS:
            base_services = ["credit_card", "loan", "overdraft"]
            if random.random() > 0.5:
                base_services.extend(["investment", "insurance"])
            if random.random() > 0.7:
                base_services.append("mortgage")
                
        elif account_type == AccountType.STUDENT:
            base_services = []
            if random.random() > 0.6:
                base_services.append("credit_card")
            if random.random() > 0.9:
                base_services.append("loan")
                
        else:  # Joint, Senior
            base_services = ["credit_card"]
            if random.random() > 0.5:
                base_services.append("investment")
            if random.random() > 0.7:
                base_services.extend(["loan", "insurance"])
        
        return list(set(base_services))  # Remove duplicates
    
    def generate_account_details(self, account_type: AccountType) -> dict:
        """Generate realistic account details based on type"""
        details = {}
        
        # Balance ranges based on account type
        if account_type == AccountType.SAVINGS:
            details['balance'] = round(random.uniform(500.00, 75000.00), 2)
            details['interest_rate'] = round(random.uniform(0.01, 0.045), 4)  # 1-4.5% APY
            details['minimum_balance'] = 100.00
            
        elif account_type == AccountType.CHECKING:
            details['balance'] = round(random.uniform(50.00, 25000.00), 2)
            details['interest_rate'] = round(random.uniform(0.001, 0.015), 4)  # 0.1-1.5% APY
            details['minimum_balance'] = 25.00
            details['overdraft_limit'] = round(random.uniform(100.00, 2000.00), 2)
            
        elif account_type == AccountType.BUSINESS:
            details['balance'] = round(random.uniform(1000.00, 500000.00), 2)
            details['interest_rate'] = round(random.uniform(0.005, 0.025), 4)  # 0.5-2.5% APY
            details['minimum_balance'] = 500.00
            details['overdraft_limit'] = round(random.uniform(5000.00, 50000.00), 2)
            
        elif account_type == AccountType.STUDENT:
            details['balance'] = round(random.uniform(10.00, 8000.00), 2)
            details['interest_rate'] = round(random.uniform(0.005, 0.02), 4)  # 0.5-2% APY
            details['minimum_balance'] = 0.00
            
        else:  # Joint, Senior
            details['balance'] = round(random.uniform(200.00, 50000.00), 2)
            details['interest_rate'] = round(random.uniform(0.01, 0.035), 4)  # 1-3.5% APY
            details['minimum_balance'] = 50.00
        
        return details
    
    def create_account(self, customer_id: int, account_type: AccountType = None) -> Account:
        """Create a new account for a customer"""
        try:
            # Verify customer exists
            customer = self.db.query(Customer).filter(Customer.customer_id == customer_id).first()
            if not customer:
                raise ValueError(f"Customer {customer_id} not found")
            
            # If no account type specified, choose randomly based on customer profile
            if not account_type:
                # Logic to determine account type based on customer age, etc.
                birth_date = customer.date_of_birth
                age = (datetime.now().date() - birth_date).days // 365
                
                if age < 25:
                    account_type = random.choice([AccountType.STUDENT, AccountType.CHECKING, AccountType.SAVINGS])
                elif age >= 65:
                    account_type = random.choice([AccountType.SENIOR, AccountType.SAVINGS, AccountType.CHECKING])
                else:
                    account_type = random.choice([AccountType.CHECKING, AccountType.SAVINGS, AccountType.BUSINESS])
            
            # Generate account details
            account_number = self.generate_account_number(account_type)
            linked_services = self.generate_linked_services(account_type)
            account_details = self.generate_account_details(account_type)
            
            # Create account creation date (within last 5 years)
            days_ago = random.randint(30, 1825)  # 30 days to 5 years
            creation_date = datetime.now() - timedelta(days=days_ago)
            
            account = Account(
                account_number=account_number,
                customer_id=customer_id,
                account_type=account_type,
                account_status=AccountStatus.ACTIVE,
                balance=account_details['balance'],
                linked_services=json.dumps(linked_services) if linked_services else None,
                interest_rate=account_details.get('interest_rate', 0.0),
                minimum_balance=account_details.get('minimum_balance', 0.0),
                overdraft_limit=account_details.get('overdraft_limit', 0.0),
                created_at=creation_date,
                last_transaction_date=creation_date + timedelta(days=random.randint(1, days_ago))
            )
            
            self.db.add(account)
            self.db.commit()
            self.db.refresh(account)
            
            logger.info(f"Created {account_type.value} account {account_number} for customer {customer_id}")
            return account
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating account for customer {customer_id}: {e}")
            raise
    
    def create_accounts_for_customer(self, customer_id: int, num_accounts: int = None) -> List[Account]:
        """Create multiple accounts for a customer with realistic distribution"""
        if num_accounts is None:
            # Realistic distribution: 50% have 1 account, 35% have 2, 15% have 3+
            rand = random.random()
            if rand < 0.5:
                num_accounts = 1
            elif rand < 0.85:
                num_accounts = 2
            else:
                num_accounts = random.randint(3, 5)
        
        accounts = []
        used_types = set()
        
        for i in range(num_accounts):
            # Avoid duplicate account types for the same customer
            available_types = [t for t in AccountType if t not in used_types]
            if not available_types:
                # If all types used, allow duplicates for checking/savings
                available_types = [AccountType.CHECKING, AccountType.SAVINGS]
            
            account_type = random.choice(available_types)
            account = self.create_account(customer_id, account_type)
            accounts.append(account)
            used_types.add(account_type)
        
        return accounts
    
    def get_accounts_by_customer(self, customer_id: int) -> List[Account]:
        """Get all accounts for a customer"""
        return (self.db.query(Account)
                .filter(Account.customer_id == customer_id)
                .order_by(desc(Account.created_at))
                .all())
    
    def get_account_by_number(self, account_number: str) -> Optional[Account]:
        """Get account by account number"""
        return self.db.query(Account).filter(Account.account_number == account_number).first()
    
    def get_all_accounts(self, skip: int = 0, limit: int = 100) -> List[Account]:
        """Get all accounts with pagination"""
        return (self.db.query(Account)
                .order_by(desc(Account.created_at))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_account_stats(self) -> dict:
        """Get account statistics"""
        total_accounts = self.db.query(func.count(Account.id)).scalar()
        total_balance = self.db.query(func.sum(Account.balance)).scalar() or 0
        active_accounts = self.db.query(func.count(Account.id)).filter(
            Account.account_status == AccountStatus.ACTIVE
        ).scalar()
        
        # Account type distribution
        type_stats = {}
        for account_type in AccountType:
            count = self.db.query(func.count(Account.id)).filter(
                Account.account_type == account_type
            ).scalar()
            type_stats[account_type.value] = count
        
        return {
            "total_accounts": total_accounts,
            "total_balance": float(total_balance),
            "active_accounts": active_accounts,
            "average_balance": float(total_balance / total_accounts) if total_accounts > 0 else 0,
            "account_types": type_stats
        }
