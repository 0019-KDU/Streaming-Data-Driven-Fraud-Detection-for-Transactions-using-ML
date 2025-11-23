"""
Send Real IEEE-CIS Test Transactions for Fraud Detection

This script:
1. Reads REAL transactions from ieee-fraud-detection/test_transaction.csv
2. Optionally merges with test_identity.csv for complete data
3. Sends transactions to the Producer API one-by-one or in batches
4. Shows real-time results from your trained XGBoost model

Usage:
    # Send 10 random test transactions
    python send_real_transactions.py --count 10
    
    # Send transactions with delay (simulate real-time)
    python send_real_transactions.py --count 20 --delay 2
    
    # Send specific transaction IDs
    python send_real_transactions.py --ids 3663549 3663550 3663551
    
    # Continuous mode (send indefinitely)
    python send_real_transactions.py --continuous --delay 5
"""

import pandas as pd
import requests
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import sys

# Configuration
PRODUCER_API_URL = "http://localhost:8000/api/v1/transactions/submit"
TRANSACTION_FILE = "../ieee-fraud-detection/test_transaction.csv"
IDENTITY_FILE = "../ieee-fraud-detection/test_identity.csv"


class RealTransactionSender:
    """Send real IEEE-CIS test transactions to fraud detection API"""
    
    def __init__(self, transaction_file=TRANSACTION_FILE, identity_file=IDENTITY_FILE):
        """Initialize with dataset paths"""
        self.transaction_file = Path(transaction_file)
        self.identity_file = Path(identity_file)
        
        print("📂 Loading real IEEE-CIS test dataset...")
        
        # Load transaction data
        if not self.transaction_file.exists():
            raise FileNotFoundError(f"Transaction file not found: {self.transaction_file}")
        
        self.transactions_df = pd.read_csv(self.transaction_file)
        print(f"   ✅ Loaded {len(self.transactions_df):,} test transactions")
        
        # Load identity data (optional - may not exist for all transactions)
        if self.identity_file.exists():
            self.identity_df = pd.read_csv(self.identity_file)
            print(f"   ✅ Loaded {len(self.identity_df):,} identity records")
            
            # Merge transaction with identity
            self.data = self.transactions_df.merge(
                self.identity_df, 
                on='TransactionID', 
                how='left'
            )
            print(f"   ✅ Merged dataset: {len(self.data):,} records")
        else:
            print(f"   ⚠️ Identity file not found (optional): {self.identity_file}")
            self.data = self.transactions_df.copy()
        
        print(f"   📊 Dataset shape: {self.data.shape}")
        print(f"   📋 Columns: {len(self.data.columns)} features")
        print(f"   💰 Amount range: ${self.data['TransactionAmt'].min():.2f} - ${self.data['TransactionAmt'].max():.2f}")
        print()
    
    def prepare_transaction(self, row):
        """
        Prepare transaction payload from DataFrame row
        
        Args:
            row: pandas Series with transaction data
            
        Returns:
            dict: API-compatible transaction payload
        """
        # Convert row to dict and handle NaN values
        transaction = row.to_dict()
        
        # Convert NaN to None (JSON compatible)
        for key, value in transaction.items():
            if pd.isna(value):
                transaction[key] = None
            elif isinstance(value, (float, int)):
                # Convert numpy types to Python native types
                transaction[key] = float(value) if isinstance(value, float) else int(value)
        
        return transaction
    
    def send_transaction(self, transaction, show_response=True):
        """
        Send transaction to API
        
        Args:
            transaction: dict with transaction data
            show_response: bool, whether to print response
            
        Returns:
            tuple: (success: bool, response_data: dict or None)
        """
        try:
            response = requests.post(
                PRODUCER_API_URL,
                json=transaction,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if show_response:
                    print(f"   ✅ Submitted: {data.get('transaction_id')}")
                    print(f"      Amount: ${transaction.get('TransactionAmt', 0):.2f}")
                    print(f"      Product: {transaction.get('ProductCD', 'N/A')}")
                    print(f"      Card: {transaction.get('card1', 'N/A')} ({transaction.get('card4', 'N/A')})")
                    print(f"      Status: {data.get('status')}")
                
                return True, data
            else:
                if show_response:
                    print(f"   ❌ Error: HTTP {response.status_code}")
                    print(f"      Response: {response.text[:200]}")
                return False, None
                
        except requests.exceptions.RequestException as e:
            if show_response:
                print(f"   ❌ Request failed: {e}")
            return False, None
    
    def send_random_transactions(self, count=10, delay=1.0, show_progress=True):
        """
        Send random sample of transactions
        
        Args:
            count: number of transactions to send
            delay: seconds to wait between transactions
            show_progress: whether to show progress
            
        Returns:
            list: submitted transaction IDs
        """
        sample = self.data.sample(n=min(count, len(self.data)))
        
        print(f"📤 Sending {len(sample)} random test transactions...")
        print(f"   Delay: {delay}s between transactions")
        print(f"   API: {PRODUCER_API_URL}")
        print()
        
        submitted_ids = []
        success_count = 0
        
        for idx, (_, row) in enumerate(sample.iterrows(), 1):
            if show_progress:
                print(f"[{idx}/{len(sample)}] Transaction ID: {row['TransactionID']}")
            
            transaction = self.prepare_transaction(row)
            success, response = self.send_transaction(transaction, show_response=show_progress)
            
            if success and response:
                submitted_ids.append(response.get('transaction_id'))
                success_count += 1
            
            if show_progress:
                print()
            
            # Delay between transactions (except for last one)
            if idx < len(sample):
                time.sleep(delay)
        
        print(f"✅ Submitted {success_count}/{len(sample)} transactions successfully")
        return submitted_ids
    
    def send_specific_transactions(self, transaction_ids, delay=1.0):
        """
        Send specific transactions by ID
        
        Args:
            transaction_ids: list of TransactionID values
            delay: seconds between transactions
            
        Returns:
            list: submitted transaction IDs
        """
        print(f"📤 Sending {len(transaction_ids)} specific transactions...")
        print()
        
        submitted_ids = []
        success_count = 0
        
        for idx, txn_id in enumerate(transaction_ids, 1):
            # Find transaction in dataset
            row = self.data[self.data['TransactionID'] == int(txn_id)]
            
            if row.empty:
                print(f"[{idx}/{len(transaction_ids)}] ⚠️ Transaction {txn_id} not found in dataset")
                continue
            
            print(f"[{idx}/{len(transaction_ids)}] Transaction ID: {txn_id}")
            
            transaction = self.prepare_transaction(row.iloc[0])
            success, response = self.send_transaction(transaction, show_response=True)
            
            if success and response:
                submitted_ids.append(response.get('transaction_id'))
                success_count += 1
            
            print()
            
            if idx < len(transaction_ids):
                time.sleep(delay)
        
        print(f"✅ Submitted {success_count}/{len(transaction_ids)} transactions successfully")
        return submitted_ids
    
    def send_continuous(self, delay=5.0, batch_size=5):
        """
        Continuously send transactions (press Ctrl+C to stop)
        
        Args:
            delay: seconds between batches
            batch_size: transactions per batch
        """
        print(f"🔄 CONTINUOUS MODE - Sending {batch_size} transactions every {delay}s")
        print(f"   Press Ctrl+C to stop")
        print()
        
        batch_num = 0
        total_sent = 0
        
        try:
            while True:
                batch_num += 1
                print(f"📦 Batch {batch_num} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Send batch
                sample = self.data.sample(n=batch_size)
                batch_success = 0
                
                for _, row in sample.iterrows():
                    transaction = self.prepare_transaction(row)
                    success, _ = self.send_transaction(transaction, show_response=False)
                    
                    if success:
                        batch_success += 1
                        total_sent += 1
                
                print(f"   ✅ Sent {batch_success}/{batch_size} transactions")
                print(f"   📊 Total sent: {total_sent}")
                print()
                
                time.sleep(delay)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Stopped by user")
            print(f"   Total transactions sent: {total_sent}")
    
    def show_dataset_stats(self):
        """Display dataset statistics"""
        print("\n" + "="*80)
        print("📊 DATASET STATISTICS")
        print("="*80)
        
        print(f"\n📋 General:")
        print(f"   Total transactions: {len(self.data):,}")
        print(f"   Features: {len(self.data.columns)}")
        
        print(f"\n💰 Transaction Amounts:")
        print(f"   Min: ${self.data['TransactionAmt'].min():.2f}")
        print(f"   Max: ${self.data['TransactionAmt'].max():.2f}")
        print(f"   Mean: ${self.data['TransactionAmt'].mean():.2f}")
        print(f"   Median: ${self.data['TransactionAmt'].median():.2f}")
        
        print(f"\n🏷️ Product Categories:")
        print(self.data['ProductCD'].value_counts())
        
        if 'card4' in self.data.columns:
            print(f"\n💳 Card Types:")
            print(self.data['card4'].value_counts().head(10))
        
        if 'P_emaildomain' in self.data.columns:
            print(f"\n📧 Top Email Domains:")
            print(self.data['P_emaildomain'].value_counts().head(10))
        
        print("\n" + "="*80 + "\n")


def main():
    """Main function with CLI arguments"""
    parser = argparse.ArgumentParser(
        description="Send real IEEE-CIS test transactions to fraud detection API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send 10 random transactions
  python send_real_transactions.py --count 10
  
  # Send 20 transactions with 2 second delay
  python send_real_transactions.py --count 20 --delay 2
  
  # Send specific transaction IDs
  python send_real_transactions.py --ids 3663549 3663550 3663551
  
  # Continuous mode (5 transactions every 10 seconds)
  python send_real_transactions.py --continuous --batch-size 5 --delay 10
  
  # Show dataset statistics only
  python send_real_transactions.py --stats-only
        """
    )
    
    parser.add_argument(
        '--count', 
        type=int, 
        default=10,
        help='Number of random transactions to send (default: 10)'
    )
    
    parser.add_argument(
        '--delay', 
        type=float, 
        default=1.0,
        help='Delay in seconds between transactions (default: 1.0)'
    )
    
    parser.add_argument(
        '--ids', 
        type=int, 
        nargs='+',
        help='Specific transaction IDs to send'
    )
    
    parser.add_argument(
        '--continuous', 
        action='store_true',
        help='Continuously send transactions (press Ctrl+C to stop)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=5,
        help='Batch size for continuous mode (default: 5)'
    )
    
    parser.add_argument(
        '--stats-only', 
        action='store_true',
        help='Show dataset statistics only (no transactions sent)'
    )
    
    args = parser.parse_args()
    
    # Initialize sender
    try:
        sender = RealTransactionSender()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\nMake sure the dataset files exist:")
        print(f"   {Path(TRANSACTION_FILE).absolute()}")
        print(f"   {Path(IDENTITY_FILE).absolute()}")
        sys.exit(1)
    
    # Show stats only
    if args.stats_only:
        sender.show_dataset_stats()
        return
    
    # Check API connectivity
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code != 200:
            print("⚠️ Warning: Producer API may not be healthy")
    except requests.exceptions.RequestException:
        print("❌ Error: Producer API is not reachable at http://localhost:8000")
        print("   Make sure the producer-api container is running:")
        print("   docker ps | grep producer-api")
        sys.exit(1)
    
    # Send transactions based on mode
    if args.continuous:
        sender.send_continuous(delay=args.delay, batch_size=args.batch_size)
    
    elif args.ids:
        submitted_ids = sender.send_specific_transactions(args.ids, delay=args.delay)
        
        print("\n" + "="*80)
        print(f"📝 Submitted Transaction IDs:")
        for txn_id in submitted_ids:
            print(f"   • {txn_id}")
        print("="*80)
    
    else:
        submitted_ids = sender.send_random_transactions(
            count=args.count, 
            delay=args.delay,
            show_progress=True
        )
        
        print("\n" + "="*80)
        print(f"📝 Submitted {len(submitted_ids)} Transaction IDs:")
        for txn_id in submitted_ids:
            print(f"   • {txn_id}")
        print("="*80)
    
    print("\n💡 Check results:")
    print("   1. Dashboard: http://localhost:8501")
    print("   2. Spark logs: docker logs fraud-inference-spark --tail 100")
    print("   3. Run test_e2e_workflow.py to consume Kafka output topics")
    print()


if __name__ == "__main__":
    main()
