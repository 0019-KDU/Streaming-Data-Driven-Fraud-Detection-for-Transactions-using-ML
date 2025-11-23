"""
Show Real IEEE-CIS Test Transaction Payload

This script displays one real transaction from test_transaction.csv
with all features formatted as JSON payload.
"""

import pandas as pd
import json
from pathlib import Path

# Load dataset
TRANSACTION_FILE = "../ieee-fraud-detection/test_transaction.csv"
IDENTITY_FILE = "../ieee-fraud-detection/test_identity.csv"

print("📂 Loading IEEE-CIS test dataset...\n")

# Load transaction data
transactions_df = pd.read_csv(TRANSACTION_FILE)
print(f"✅ Loaded {len(transactions_df):,} transactions")

# Load identity data (optional)
identity_df = pd.read_csv(IDENTITY_FILE)
print(f"✅ Loaded {len(identity_df):,} identity records")

# Merge
data = transactions_df.merge(identity_df, on='TransactionID', how='left')
print(f"✅ Merged: {len(data):,} records with {len(data.columns)} features\n")

# Get one random transaction
sample = data.sample(n=1).iloc[0]

print("="*80)
print(f"🎯 REAL TEST TRANSACTION (ID: {sample['TransactionID']})")
print("="*80)

# Convert to dict and handle NaN
transaction = sample.to_dict()

# Clean up payload
cleaned_transaction = {}
for key, value in transaction.items():
    if pd.isna(value):
        cleaned_transaction[key] = None
    elif isinstance(value, float):
        cleaned_transaction[key] = float(value)
    elif isinstance(value, int):
        cleaned_transaction[key] = int(value)
    else:
        cleaned_transaction[key] = str(value)

# Show key fields first
print(f"\n📋 KEY FIELDS:")
print(f"   TransactionID: {cleaned_transaction.get('TransactionID')}")
print(f"   TransactionAmt: ${cleaned_transaction.get('TransactionAmt', 0):.2f}")
print(f"   ProductCD: {cleaned_transaction.get('ProductCD')}")
print(f"   card1: {cleaned_transaction.get('card1')}")
print(f"   card4: {cleaned_transaction.get('card4')}")
print(f"   card6: {cleaned_transaction.get('card6')}")
print(f"   P_emaildomain: {cleaned_transaction.get('P_emaildomain')}")
print(f"   R_emaildomain: {cleaned_transaction.get('R_emaildomain')}")
print(f"   addr1: {cleaned_transaction.get('addr1')}")
print(f"   addr2: {cleaned_transaction.get('addr2')}")

# Show full JSON payload
print(f"\n📦 COMPLETE JSON PAYLOAD ({len(cleaned_transaction)} fields):")
print("="*80)
print(json.dumps(cleaned_transaction, indent=2))
print("="*80)

# Show summary
print(f"\n📊 SUMMARY:")
print(f"   Total features: {len(cleaned_transaction)}")
print(f"   Non-null features: {sum(1 for v in cleaned_transaction.values() if v is not None)}")
print(f"   Null features: {sum(1 for v in cleaned_transaction.values() if v is None)}")

print(f"\n💡 To test this transaction:")
print(f"   1. Start your services: docker-compose up -d")
print(f"   2. Run: python send_real_transactions.py --count 1")
print(f"   3. Check dashboard: http://localhost:8501")
