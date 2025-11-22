"""
Extract Real Fraud Examples from IEEE-CIS Dataset
==================================================
This script extracts real fraudulent and legitimate transactions
from the IEEE-CIS training dataset for testing the fraud detection system.

Author: Senior ML Engineering Consultant
"""
import pandas as pd
import json
import os

print("=" * 100)
print("EXTRACTING REAL FRAUD EXAMPLES FROM IEEE-CIS DATASET")
print("=" * 100)

# Paths
dataset_path = "D:/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/ieee-fraud-detection"
train_txn_path = os.path.join(dataset_path, "train_transaction.csv")
train_id_path = os.path.join(dataset_path, "train_identity.csv")

print(f"\n[STEP 1] Loading dataset...")
print(f"   Transaction file: {train_txn_path}")
print(f"   Identity file: {train_id_path}")

# Load data
txn_df = pd.read_csv(train_txn_path)
id_df = pd.read_csv(train_id_path)

# Merge
df = txn_df.merge(id_df, on='TransactionID', how='left')
print(f"   Loaded {len(df):,} transactions")
print(f"   Fraud rate: {df['isFraud'].mean()*100:.2f}%")

# Separate fraud and legitimate
fraud_df = df[df['isFraud'] == 1].copy()
legit_df = df[df['isFraud'] == 0].copy()

print(f"\n   Fraudulent: {len(fraud_df):,} transactions")
print(f"   Legitimate: {len(legit_df):,} transactions")

print(f"\n[STEP 2] Finding interesting fraud examples...")

# Find high-value fraud
high_value_fraud = fraud_df[fraud_df['TransactionAmt'] > 1000].copy()
print(f"   High-value fraud (>$1000): {len(high_value_fraud):,}")

# Find fraud with risky email domains
risky_domains = ['anonymous.com', 'mailinator.com', 'tempmail.com', 'yopmail.com']
risky_email_fraud = fraud_df[fraud_df['P_emaildomain'].isin(risky_domains)]
print(f"   Risky email fraud: {len(risky_email_fraud):,}")

# Find discover card fraud
discover_fraud = fraud_df[fraud_df['card4'] == 'discover']
print(f"   Discover card fraud: {len(discover_fraud):,}")

# Select diverse fraud examples
print(f"\n[STEP 3] Selecting diverse fraud examples...")

fraud_examples = []

# Example 1: High-value fraud
if len(high_value_fraud) > 0:
    sample = high_value_fraud.iloc[0]
    fraud_examples.append(("High-Value Fraud", sample))
    print(f"   [OK] High-value fraud: ${sample['TransactionAmt']:.2f}")

# Example 2: Discover card fraud
if len(discover_fraud) > 0:
    sample = discover_fraud.iloc[0]
    fraud_examples.append(("Discover Card Fraud", sample))
    print(f"   [OK] Discover card fraud: ${sample['TransactionAmt']:.2f}")

# Example 3: Geographic distance fraud (high dist1)
high_dist_fraud = fraud_df[fraud_df['dist1'] > 1000].dropna(subset=['dist1'])
if len(high_dist_fraud) > 0:
    sample = high_dist_fraud.iloc[0]
    fraud_examples.append(("Geographic Anomaly Fraud", sample))
    print(f"   [OK] Geographic fraud: dist1={sample['dist1']:.0f}km")

# Example 4: Cash withdrawal fraud
cash_fraud = fraud_df[fraud_df['ProductCD'] == 'C']
if len(cash_fraud) > 0:
    sample = cash_fraud.iloc[0]
    fraud_examples.append(("Cash Withdrawal Fraud", sample))
    print(f"   [OK] Cash fraud: ${sample['TransactionAmt']:.2f}")

# Example 5: Generic fraud (diverse characteristics)
generic_fraud = fraud_df[(fraud_df['TransactionAmt'] > 100) &
                         (fraud_df['TransactionAmt'] < 500)].sample(n=min(3, len(fraud_df)), random_state=42)

for i, (idx, sample) in enumerate(generic_fraud.iterrows(), 1):
    fraud_examples.append((f"Fraud Example {i}", sample))
    print(f"   [OK] Generic fraud {i}: ${sample['TransactionAmt']:.2f}, card4={sample.get('card4', 'N/A')}")

# Select legitimate examples
print(f"\n[STEP 4] Selecting legitimate examples...")

legit_examples = []

# Example 1: Low-value legitimate
low_value_legit = legit_df[(legit_df['TransactionAmt'] > 20) &
                            (legit_df['TransactionAmt'] < 100)].sample(n=1, random_state=42)
sample = low_value_legit.iloc[0]
legit_examples.append(("Low-Value Legitimate", sample))
print(f"   [OK] Low-value legit: ${sample['TransactionAmt']:.2f}")

# Example 2: Visa debit legitimate
visa_legit = legit_df[(legit_df['card4'] == 'visa') &
                      (legit_df['card6'] == 'debit')].sample(n=1, random_state=42)
sample = visa_legit.iloc[0]
legit_examples.append(("Visa Debit Legitimate", sample))
print(f"   [OK] Visa debit legit: ${sample['TransactionAmt']:.2f}")

# Example 3: Generic legitimate
generic_legit = legit_df.sample(n=2, random_state=42)
for i, (idx, sample) in enumerate(generic_legit.iterrows(), 1):
    legit_examples.append((f"Legitimate Example {i}", sample))
    print(f"   [OK] Generic legit {i}: ${sample['TransactionAmt']:.2f}")

# Convert to JSON format
print(f"\n[STEP 5] Converting to JSON format...")

def row_to_json(row):
    """Convert pandas row to JSON transaction format"""
    # Core fields
    txn = {
        "TransactionID": str(row.get('TransactionID', '')),
        "TransactionAmt": float(row.get('TransactionAmt', 0)),
        "TransactionDT": int(row.get('TransactionDT', 0)),
        "ProductCD": str(row.get('ProductCD', 'W')),

        # Card info
        "card1": int(row.get('card1', -1)) if pd.notna(row.get('card1')) else -1,
        "card2": int(row.get('card2', -1)) if pd.notna(row.get('card2')) else -1,
        "card3": int(row.get('card3', -1)) if pd.notna(row.get('card3')) else -1,
        "card4": str(row.get('card4', 'visa')) if pd.notna(row.get('card4')) else 'visa',
        "card5": int(row.get('card5', -1)) if pd.notna(row.get('card5')) else -1,
        "card6": str(row.get('card6', 'debit')) if pd.notna(row.get('card6')) else 'debit',

        # Address
        "addr1": int(row.get('addr1', -1)) if pd.notna(row.get('addr1')) else -1,
        "addr2": int(row.get('addr2', -1)) if pd.notna(row.get('addr2')) else -1,

        # Distance
        "dist1": float(row.get('dist1', -1)) if pd.notna(row.get('dist1')) else None,
        "dist2": float(row.get('dist2', -1)) if pd.notna(row.get('dist2')) else None,

        # Email
        "P_emaildomain": str(row.get('P_emaildomain', 'unknown')) if pd.notna(row.get('P_emaildomain')) else 'unknown',
        "R_emaildomain": str(row.get('R_emaildomain', 'unknown')) if pd.notna(row.get('R_emaildomain')) else 'unknown',

        # C columns
        "C1": float(row.get('C1', 0)) if pd.notna(row.get('C1')) else 0,
        "C2": float(row.get('C2', 0)) if pd.notna(row.get('C2')) else 0,
        "C3": float(row.get('C3', 0)) if pd.notna(row.get('C3')) else 0,
        "C4": float(row.get('C4', 0)) if pd.notna(row.get('C4')) else 0,
        "C5": float(row.get('C5', 0)) if pd.notna(row.get('C5')) else 0,
        "C6": float(row.get('C6', 0)) if pd.notna(row.get('C6')) else 0,
        "C7": float(row.get('C7', 0)) if pd.notna(row.get('C7')) else 0,
        "C8": float(row.get('C8', 0)) if pd.notna(row.get('C8')) else 0,
        "C9": float(row.get('C9', 0)) if pd.notna(row.get('C9')) else 0,
        "C10": float(row.get('C10', 0)) if pd.notna(row.get('C10')) else 0,
        "C11": float(row.get('C11', 0)) if pd.notna(row.get('C11')) else 0,
        "C12": float(row.get('C12', 0)) if pd.notna(row.get('C12')) else 0,
        "C13": float(row.get('C13', 0)) if pd.notna(row.get('C13')) else 0,
        "C14": float(row.get('C14', 0)) if pd.notna(row.get('C14')) else 0,

        # D columns
        "D1": float(row.get('D1', 0)) if pd.notna(row.get('D1')) else 0,
        "D2": float(row.get('D2', 0)) if pd.notna(row.get('D2')) else None,
        "D3": float(row.get('D3', 0)) if pd.notna(row.get('D3')) else None,
        "D4": float(row.get('D4', 0)) if pd.notna(row.get('D4')) else None,
        "D5": float(row.get('D5', 0)) if pd.notna(row.get('D5')) else None,
        "D10": float(row.get('D10', 0)) if pd.notna(row.get('D10')) else 0,
        "D11": float(row.get('D11', 0)) if pd.notna(row.get('D11')) else None,
        "D15": float(row.get('D15', 0)) if pd.notna(row.get('D15')) else 0,

        # Metadata
        "_is_fraud": int(row.get('isFraud', 0))
    }

    return txn

# Save fraud examples
print(f"\n[STEP 6] Saving examples to files...")

output_dir = "d:/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML"

# Save individual fraud examples
for i, (label, row) in enumerate(fraud_examples[:5], 1):  # Save top 5
    txn_json = row_to_json(row)
    filename = os.path.join(output_dir, f"fraud_example_{i}.json")

    with open(filename, 'w') as f:
        json.dump(txn_json, f, indent=2)

    print(f"   [OK] Saved: {filename}")
    print(f"      {label}: ${txn_json['TransactionAmt']:.2f}, "
          f"card4={txn_json['card4']}, email={txn_json['P_emaildomain']}")

# Save individual legitimate examples
for i, (label, row) in enumerate(legit_examples[:3], 1):  # Save top 3
    txn_json = row_to_json(row)
    filename = os.path.join(output_dir, f"legit_example_{i}.json")

    with open(filename, 'w') as f:
        json.dump(txn_json, f, indent=2)

    print(f"   [OK] Saved: {filename}")
    print(f"      {label}: ${txn_json['TransactionAmt']:.2f}, "
          f"card4={txn_json['card4']}, email={txn_json['P_emaildomain']}")

# Create a summary JSON with all examples
summary = {
    "fraud_examples": [
        {
            "label": label,
            "transaction": row_to_json(row)
        }
        for label, row in fraud_examples[:5]
    ],
    "legitimate_examples": [
        {
            "label": label,
            "transaction": row_to_json(row)
        }
        for label, row in legit_examples[:3]
    ]
}

summary_file = os.path.join(output_dir, "real_transaction_examples.json")
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n   [OK] Saved summary: {summary_file}")

# Create test script
print(f"\n[STEP 7] Creating test script...")

test_script = """#!/bin/bash
# Test script for real fraud examples

echo "Testing FRAUD examples..."
echo "========================="

for i in {1..5}; do
    if [ -f "fraud_example_$i.json" ]; then
        echo ""
        echo "Fraud Example $i:"
        curl -X POST http://localhost:8000/api/v1/transactions/submit \\
          -H "Content-Type: application/json" \\
          -d @fraud_example_$i.json
        echo ""
        sleep 2
    fi
done

echo ""
echo "Testing LEGITIMATE examples..."
echo "=============================="

for i in {1..3}; do
    if [ -f "legit_example_$i.json" ]; then
        echo ""
        echo "Legitimate Example $i:"
        curl -X POST http://localhost:8000/api/v1/transactions/submit \\
          -H "Content-Type: application/json" \\
          -d @legit_example_$i.json
        echo ""
        sleep 2
    fi
done
"""

test_script_file = os.path.join(output_dir, "test_real_examples.sh")
with open(test_script_file, 'w') as f:
    f.write(test_script)

print(f"   [OK] Saved: {test_script_file}")

print(f"\n" + "=" * 100)
print("EXTRACTION COMPLETE!")
print("=" * 100)

print(f"\nFiles created:")
print(f"   - fraud_example_1.json to fraud_example_5.json (5 fraud cases)")
print(f"   - legit_example_1.json to legit_example_3.json (3 legitimate cases)")
print(f"   - real_transaction_examples.json (summary)")
print(f"   - test_real_examples.sh (automated test script)")

print(f"\nTo test:")
print(f"   1. Single fraud example:")
print(f"      curl -X POST http://localhost:8000/api/v1/transactions/submit \\")
print(f"        -H 'Content-Type: application/json' -d @fraud_example_1.json")
print(f"\n   2. Run all examples:")
print(f"      bash test_real_examples.sh")

print(f"\nExpected results:")
print(f"   - Fraud examples: probability > 0.30, decision = REVIEW/BLOCK")
print(f"   - Legit examples: probability < 0.10, decision = APPROVE")
