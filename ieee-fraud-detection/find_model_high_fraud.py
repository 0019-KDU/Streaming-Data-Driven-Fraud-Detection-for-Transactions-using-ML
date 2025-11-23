import pandas as pd
import numpy as np
import pickle
import sys
import json
import joblib

# Add paths for custom modules
sys.path.insert(0, '/home/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src/inference')
sys.path.insert(0, '/home/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src/dags')

# Import custom modules that the model was trained with
try:
    import ieee_cis_training
    print("✅ Imported ieee_cis_training module")
except ImportError as e:
    print(f"⚠️ Could not import ieee_cis_training: {e}")

print("Loading model and pipeline...")
# Load model - try different possible paths and names
import dill
import gzip

model_paths = [
    '/home/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src/models/fraud_detection_model.pkl',
    '/home/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src/models/xgboost_fraud_model.pkl',
    '../src/models/fraud_detection_model.pkl',
    '../src/models/xgboost_fraud_model.pkl'
]

model = None
model_bundle = None
threshold = 0.05639991909265518

for model_path in model_paths:
    if model is not None:
        break
    try:
        # Try joblib (recommended for sklearn/xgboost models)
        try:
            model_bundle = joblib.load(model_path)
            if isinstance(model_bundle, dict):
                model = model_bundle.get('model') or model_bundle.get('calibrated_model')
                feature_names = model_bundle.get('feature_names', [])
                threshold = model_bundle.get('threshold', 0.05639991909265518)
            else:
                model = model_bundle
                feature_names = []
                threshold = 0.05639991909265518
            print(f"✅ Model loaded with joblib from: {model_path}")
            break
        except (Exception,):
            pass
        
        # Try regular pickle
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                feature_names = model_data['feature_names']
                threshold = model_data.get('threshold', 0.05639991909265518)
                print(f"✅ Model loaded with pickle from: {model_path}")
                break
        except (pickle.UnpicklingError, KeyError):
            pass
        
        # Try dill
        try:
            with open(model_path, 'rb') as f:
                model_data = dill.load(f)
                model = model_data['model']
                feature_names = model_data['feature_names']
                threshold = model_data.get('threshold', 0.05639991909265518)
                print(f"✅ Model loaded with dill from: {model_path}")
                break
        except (Exception,):
            pass
            
    except FileNotFoundError:
        continue

if model is None:
    print("❌ Could not load model file")
    print("   Run: ls -la /home/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src/models/")
    print("   Check file format: file /home/.../src/models/fraud_detection_model.pkl")
    sys.exit(1)

print(f"Model loaded with {len(feature_names)} features")
print(f"Threshold: {threshold}")

# Load feature pipeline WITH aggregation maps
pipeline_path = '/home/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src/models/feature_pipeline.pkl'
print(f"Loading feature pipeline from: {pipeline_path}")
try:
    pipeline = joblib.load(pipeline_path)
    print("✅ Feature pipeline loaded with aggregation maps")
except Exception as e:
    print(f"❌ Failed to load feature pipeline: {e}")
    sys.exit(1)

print("\nLoading fraud transactions from training data...")
# Load only fraud transactions (isFraud=1)
train_trans = pd.read_csv('/home/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src/data/ieee_cis/train_transaction.csv', nrows=100000)
train_ident = pd.read_csv('/home/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src/data/ieee_cis/train_identity.csv')

# Filter for fraud only
fraud_df = train_trans[train_trans['isFraud'] == 1].copy()
print(f"Found {len(fraud_df)} fraud transactions in first 100K rows")

# Merge with identity
fraud_full = fraud_df.merge(train_ident, on='TransactionID', how='left')

print("\nScoring fraud transactions with model...")
high_fraud_results = []
errors = []

for idx, row in fraud_full.head(50).iterrows():
    try:
        # Convert to dict (drop isFraud)
        trans_dict = row.drop('isFraud').to_dict()
        
        # Apply feature pipeline
        features_df = pipeline.transform(pd.DataFrame([trans_dict]))
        
        # Predict
        if len(feature_names) > 0:
            fraud_prob = model.predict_proba(features_df[feature_names])[:, 1][0]
        else:
            fraud_prob = model.predict_proba(features_df)[:, 1][0]
        
        high_fraud_results.append({
            'TransactionID': int(row['TransactionID']),
            'fraud_probability': float(fraud_prob),
            'TransactionAmt': float(row['TransactionAmt']),
            'ProductCD': str(row['ProductCD']),
            'card1': int(row['card1']) if pd.notna(row['card1']) else None,
            'card4': str(row['card4']) if pd.notna(row['card4']) else None,
            'P_emaildomain': str(row['P_emaildomain']) if pd.notna(row['P_emaildomain']) else None
        })
        
        if fraud_prob > 0.03:  # Over 3% (above threshold)
            print(f"✅ Transaction {int(row['TransactionID'])}: {fraud_prob*100:.1f}% fraud | ${row['TransactionAmt']} | {row['card4']} {row['card1']} | {row['P_emaildomain']}")
    except Exception as e:
        errors.append(f"Transaction {row['TransactionID']}: {str(e)}")
        if len(errors) <= 3:  # Show first 3 errors
            print(f"❌ Error on transaction {row['TransactionID']}: {e}")
        continue

if errors:
    print(f"\n⚠️ Total errors: {len(errors)} out of 50 transactions")

# Sort by fraud probability
high_fraud_results.sort(key=lambda x: x['fraud_probability'], reverse=True)

print(f"\n{'='*80}")
print("TOP 20 HIGHEST FRAUD SCORES:")
print(f"{'='*80}")
for i, result in enumerate(high_fraud_results[:20], 1):
    print(f"{i}. Transaction {result['TransactionID']}: {result['fraud_probability']*100:.2f}% | "
          f"${result['TransactionAmt']:.0f} | {result['card4']} {result['card1']} | {result['P_emaildomain']}")

# Show statistics
if high_fraud_results:
    probs = [r['fraud_probability'] for r in high_fraud_results]
    print(f"\n{'='*80}")
    print("FRAUD PROBABILITY STATISTICS:")
    print(f"{'='*80}")
    print(f"Max:    {max(probs)*100:.2f}%")
    print(f"Mean:   {np.mean(probs)*100:.2f}%")
    print(f"Median: {np.median(probs)*100:.2f}%")
    print(f"Min:    {min(probs)*100:.2f}%")
    print(f"Above threshold (5.64%): {sum(1 for p in probs if p > 0.0564)} transactions")

# Save the highest scoring one
if high_fraud_results:
    best = high_fraud_results[0]
    best_trans_id = best['TransactionID']
    
    # Get full transaction data
    best_row = fraud_full[fraud_full['TransactionID'] == best_trans_id].iloc[0]
    best_dict = best_row.drop('isFraud').to_dict()
    
    # Convert numpy types to native Python
    for key, value in best_dict.items():
        if pd.isna(value):
            best_dict[key] = None
        elif isinstance(value, (np.integer, np.int64)):
            best_dict[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            best_dict[key] = float(value)
    
    output_file = '/home/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src/test_high_fraud_model.json'
    with open(output_file, 'w') as f:
        json.dump(best_dict, f, indent=4)
    
    print(f"\n✅ Saved highest fraud transaction to {output_file}")
    print(f"   Transaction {best_trans_id}: {best['fraud_probability']*100:.2f}% fraud probability")
    print(f"   Amount: ${best['TransactionAmt']:.0f}, Card: {best['card4']} {best['card1']}, Email: {best['P_emaildomain']}")
else:
    print("\n⚠️ No high-fraud transactions found in this sample")
