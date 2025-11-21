"""
Generate aggregation maps from IEEE-CIS training data to fix Train-Serve skew.
This allows us to calculate features like 'TransactionAmt_to_mean_card1' during inference
by using the historical averages from the training set.
"""
import pandas as pd
import numpy as np
import joblib
import gc

print("="*80)
print("GENERATING AGGREGATION MAPS FROM TRAINING DATA")
print("="*80)

# Load data
print("\n1. Loading training data...")
df_trans = pd.read_csv(r"D:\Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML\ieee-fraud-detection\train_transaction.csv")
df_id = pd.read_csv(r"D:\Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML\ieee-fraud-detection\train_identity.csv")
df = df_trans.merge(df_id, on='TransactionID', how='left')

print(f"   [OK] Loaded {len(df)} transactions")

# Define aggregation maps dictionary
agg_maps = {}

# Helper to calculate and store map
def create_map(df, group_col, target_col, agg_func):
    name = f"{target_col}_{agg_func}_{group_col}"
    print(f"   Generating {name}...")
    if agg_func == 'mean':
        mapping = df.groupby(group_col)[target_col].mean().to_dict()
    elif agg_func == 'std':
        mapping = df.groupby(group_col)[target_col].std().to_dict()
    elif agg_func == 'nunique':
        mapping = df.groupby(group_col)[target_col].nunique().to_dict()
    
    agg_maps[name] = mapping

print("\n2. Generating Aggregation Maps...")

# 1. Card1 Aggregates
create_map(df, 'card1', 'TransactionAmt', 'mean')
create_map(df, 'card1', 'TransactionAmt', 'std')
create_map(df, 'card1', 'D15', 'mean')
create_map(df, 'card1', 'D15', 'std')

# 2. Addr1 Aggregates
create_map(df, 'addr1', 'D15', 'mean')
create_map(df, 'addr1', 'D15', 'std')

# 3. Magic UID Aggregates
# Recreate magic_uid logic from training: card1_addr1 + (day - D1)
print("   Creating magic_uid (card1_addr1 + creation_date)...")
df['day'] = df['TransactionDT'] / (24 * 60 * 60)
d1_val = df['D1'].fillna(0)
df['card1_addr1'] = df['card1'].fillna(-999).astype(str) + '_' + df['addr1'].fillna(-999).astype(str)
df['uid'] = df['card1_addr1'] + '_' + np.floor(df['day'] - d1_val).fillna(-999).astype(str)

# Calculate aggregates for UID
# The model features are named 'magic_uid_X_mean'.
# We generate maps for 'uid' which corresponds to 'magic_uid' in the pipeline.
target_cols = ['C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
               'D4', 'D10', 'D15', 'M1', 'M2', 'M3', 'M4', 'M6', 'M7', 'M8', 'M9',
               'TransactionAmt']

for col in target_cols:
    if col in df.columns:
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            create_map(df, 'uid', col, 'mean')
            create_map(df, 'uid', col, 'std')
        else:
            print(f"   Skipping {col} for mean/std (non-numeric)")

# Also nunique features
create_map(df, 'uid', 'P_emaildomain', 'nunique')
create_map(df, 'uid', 'dist1', 'nunique')
create_map(df, 'uid', 'id_02', 'nunique')

# Fraud Rate Maps
print("\n   Generating Fraud Rate Maps...")
fraud_cols = ['DeviceInfo', 'DeviceType', 'id_12', 'id_15', 'id_16', 'id_28', 'id_29', 'id_31', 'id_35', 'id_36', 'id_37', 'id_38']
for col in fraud_cols:
    if col in df.columns:
        create_map(df, col, 'isFraud', 'mean') # Mean of isFraud is the fraud rate

print(f"\n3. Saving {len(agg_maps)} aggregation maps...")
joblib.dump(agg_maps, r"D:\Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML\src\inference\agg_maps.pkl")
print("   [OK] Saved to src/inference/agg_maps.pkl")

print("\n" + "="*80)
print("DONE")
print("="*80)
