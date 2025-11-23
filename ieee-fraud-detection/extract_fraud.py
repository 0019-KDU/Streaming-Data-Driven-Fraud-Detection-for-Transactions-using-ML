import pandas as pd
import json

# Load training data (first 50000 rows for speed)
print("Loading training data (first 50K rows)...")
train_txn = pd.read_csv('train_transaction.csv', nrows=50000)
train_id = pd.read_csv('train_identity.csv', nrows=50000)

# Merge
merged = train_txn.merge(train_id, on='TransactionID', how='left')

# Get fraud samples
fraud_samples = merged[merged['isFraud'] == 1].head(10)

print(f"\nFound {len(fraud_samples)} fraud samples")
print("\nFraud Transactions:")
print(fraud_samples[['TransactionID', 'TransactionAmt', 'ProductCD', 'card1', 'card4', 'card6', 'P_emaildomain']].to_string())

# Pick one high-value fraud transaction
fraud_txn = fraud_samples.iloc[0]

# Extract full transaction as JSON
output = {}
for col in fraud_txn.index:
    val = fraud_txn[col]
    if pd.isna(val):
        output[col] = None
    elif isinstance(val, (int, float)):
        output[col] = float(val) if pd.notna(val) else None
    else:
        output[col] = str(val)

# Remove isFraud label
output.pop('isFraud', None)

# Save to file
with open('../src/test_fraud_transaction.json', 'w') as f:
    json.dump(output, f, indent=4)

print(f"\n✅ Saved fraud transaction {output['TransactionID']} to test_fraud_transaction.json")
print(f"Amount: ${output['TransactionAmt']}, Product: {output.get('ProductCD', 'N/A')}")
