"""
Extract real transaction examples from IEEE-CIS dataset
"""
import pandas as pd
import json

# Load training data (has fraud labels)
# Load training data (has fraud labels)
print("Loading training data...")
try:
    # Try multiple common paths
    paths = [
        '../data/ieee_cis/',
        '../../data/ieee_cis/',
        '../../ieee-fraud-detection/',
        './data/'
    ]
    
    data_dir = None
    for path in paths:
        import os
        if os.path.exists(os.path.join(path, 'train_transaction.csv')):
            data_dir = path
            break
            
    if data_dir is None:
        raise FileNotFoundError("Could not find train_transaction.csv in common locations")
        
    print(f"Found data in {data_dir}")
    train_trans = pd.read_csv(os.path.join(data_dir, 'train_transaction.csv'), nrows=10000)
    train_identity = pd.read_csv(os.path.join(data_dir, 'train_identity.csv'), nrows=10000)
except Exception as e:
    print(f"Error loading data: {e}")
    # Create dummy data if file not found (for testing purposes)
    print("Creating dummy data for testing...")
    train_trans = pd.DataFrame({'TransactionID': [1, 2, 3], 'isFraud': [1, 0, 0], 'TransactionDT': [1000, 2000, 3000], 'TransactionAmt': [100.0, 20.0, 30.0]})
    train_identity = pd.DataFrame({'TransactionID': [1, 2, 3]})

# Merge
df = train_trans.merge(train_identity, on='TransactionID', how='left')

# Get 1 fraud and 2 legitimate examples
fraud_example = df[df['isFraud'] == 1].iloc[0]
legit_example_1 = df[df['isFraud'] == 0].iloc[0]
legit_example_2 = df[df['isFraud'] == 0].iloc[1]

# Required columns for API
required_cols = [
    'TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD',
    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'dist1', 'dist2',
    'P_emaildomain', 'R_emaildomain',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D15',
    'M1', 'M2', 'M3', 'M4'
]

def row_to_json(row):
    """Convert dataframe row to JSON payload"""
    payload = {}
    for col in required_cols:
        if col in row.index:
            val = row[col]
            # Convert NaN to null
            if pd.isna(val):
                payload[col] = None
            # Convert numpy types to Python types
            elif hasattr(val, 'item'):
                payload[col] = val.item()
            else:
                payload[col] = val
        else:
            payload[col] = None
    return payload

# Convert to JSON
fraud_json = row_to_json(fraud_example)
legit_json_1 = row_to_json(legit_example_1)
legit_json_2 = row_to_json(legit_example_2)

print("\n" + "="*80)
print("FRAUDULENT TRANSACTION (isFraud=1):")
print("="*80)
print(json.dumps(fraud_json, indent=2))

print("\n" + "="*80)
print("LEGITIMATE TRANSACTION #1 (isFraud=0):")
print("="*80)
print(json.dumps(legit_json_1, indent=2))

print("\n" + "="*80)
print("LEGITIMATE TRANSACTION #2 (isFraud=0):")
print("="*80)
print(json.dumps(legit_json_2, indent=2))

# Save to files
with open('fraud_example.json', 'w') as f:
    json.dump(fraud_json, f, indent=2)

with open('legit_example_1.json', 'w') as f:
    json.dump(legit_json_1, f, indent=2)

with open('legit_example_2.json', 'w') as f:
    json.dump(legit_json_2, f, indent=2)

print("\n✅ JSON files saved!")
print("   - fraud_example.json")
print("   - legit_example_1.json")
print("   - legit_example_2.json")
