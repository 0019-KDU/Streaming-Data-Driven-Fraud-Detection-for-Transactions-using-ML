import pandas as pd
import numpy as np
import joblib
import json
import sys
sys.path.append('../src/dags')

# Load the trained model
print("Loading trained model...")
model_path = '../src/models/fraud_detection_model.pkl'
model_bundle = joblib.load(model_path)
model = model_bundle['model']
feature_names = model_bundle['feature_names']

print(f"Model loaded with {len(feature_names)} features")

# Load feature pipeline
print("Loading feature pipeline...")
from feature_pipeline import IEEECISFeaturePipeline
pipeline = IEEECISFeaturePipeline()

# Load fraud transactions (first 100K for speed)
print("Loading fraud transactions...")
train_txn = pd.read_csv('train_transaction.csv', nrows=100000)
train_id = pd.read_csv('train_identity.csv', nrows=100000)

# Merge
merged = train_txn.merge(train_id, on='TransactionID', how='left')

# Get only fraud samples
fraud_samples = merged[merged['isFraud'] == 1].copy()
print(f"Found {len(fraud_samples)} fraud samples")

# Score a sample of them
test_sample = fraud_samples.head(50)

high_prob_txns = []

for idx, row in test_sample.iterrows():
    txn_dict = row.to_dict()
    
    try:
        # Apply feature engineering
        features_df = pipeline.transform(pd.DataFrame([txn_dict]))
        
        # Get prediction
        prob = float(model.predict_proba(features_df)[0][1])
        
        high_prob_txns.append({
            'TransactionID': txn_dict['TransactionID'],
            'probability': prob,
            'amount': txn_dict['TransactionAmt'],
            'product': txn_dict.get('ProductCD', 'N/A'),
            'card4': txn_dict.get('card4', 'N/A'),
            'email': txn_dict.get('P_emaildomain', 'N/A'),
            'data': txn_dict
        })
        
        print(f"Transaction {txn_dict['TransactionID']}: {prob:.3f} probability")
        
    except Exception as e:
        print(f"Error processing transaction {txn_dict.get('TransactionID', 'unknown')}: {e}")
        continue

# Sort by probability
high_prob_txns.sort(key=lambda x: x['probability'], reverse=True)

print("\n" + "="*80)
print("TOP 10 HIGH FRAUD PROBABILITY TRANSACTIONS:")
print("="*80)

for i, txn in enumerate(high_prob_txns[:10], 1):
    print(f"\n{i}. Transaction {txn['TransactionID']}")
    print(f"   Probability: {txn['probability']:.1%} ({txn['probability']:.4f})")
    print(f"   Amount: ${txn['amount']:.2f}")
    print(f"   Product: {txn['product']}, Card: {txn['card4']}, Email: {txn['email']}")

# Save the highest probability transaction
if high_prob_txns:
    best_txn = high_prob_txns[0]
    
    # Remove isFraud label
    output = best_txn['data'].copy()
    output.pop('isFraud', None)
    
    # Convert to JSON-serializable format
    final_output = {}
    for col, val in output.items():
        if pd.isna(val):
            final_output[col] = None
        elif isinstance(val, (np.integer, np.floating)):
            final_output[col] = float(val)
        else:
            final_output[col] = str(val) if val is not None else None
    
    # Save to file
    output_file = '../src/test_high_fraud_transaction.json'
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=4)
    
    print(f"\n✅ Saved HIGHEST fraud probability transaction to test_high_fraud_transaction.json")
    print(f"   Transaction ID: {best_txn['TransactionID']}")
    print(f"   Model Prediction: {best_txn['probability']:.1%} fraud probability")
    print(f"   Amount: ${best_txn['amount']:.2f}")
    print(f"   This should trigger BLOCK/HOLD decision!")
else:
    print("\n❌ No high probability transactions found")
