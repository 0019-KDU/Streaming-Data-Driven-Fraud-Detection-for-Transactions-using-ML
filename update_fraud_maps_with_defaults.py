"""
UPDATE: Add Default Fraud Rates for Unseen Values
===================================================
This script adds default fraud rates for card numbers and addresses
not in our synthetic maps, calculated from the real IEEE-CIS dataset.
"""
import joblib
import pandas as pd
import shutil
from datetime import datetime

print("=" * 80)
print("UPDATING FRAUD RATE MAPS WITH DEFAULTS")
print("=" * 80)

# Load existing maps
agg_path = "src/inference/agg_maps.pkl"
backup_path = f"{agg_path}.backup_defaults_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"\n[1] Backing up...")
shutil.copy2(agg_path, backup_path)
print(f"   Backup: {backup_path}")

agg_maps = joblib.load(agg_path)
print(f"[2] Loaded {len(agg_maps)} maps")

# Load IEEE-CIS data to calculate real fraud rates
print(f"\n[3] Loading IEEE-CIS dataset...")
txn = pd.read_csv('ieee-fraud-detection/train_transaction.csv')
id_df = pd.read_csv('ieee-fraud-detection/train_identity.csv')
df = txn.merge(id_df, on='TransactionID', how='left')
print(f"   Loaded {len(df):,} transactions")

# Calculate fraud rates from REAL data
print(f"\n[4] Calculating fraud rates from real data...")

# Card1 fraud rates (top 100 cards by frequency)
card1_fraud = df.groupby('card1')['isFraud'].agg(['mean', 'count']).sort_values('count', ascending=False)
card1_fraud = card1_fraud[card1_fraud['count'] > 50]  # At least 50 transactions
top_card1_fraud = card1_fraud.head(100)['mean'].to_dict()

print(f"   card1: {len(top_card1_fraud)} fraud rates calculated")

# Card2 fraud rates
card2_fraud = df.groupby('card2')['isFraud'].agg(['mean', 'count']).sort_values('count', ascending=False)
card2_fraud = card2_fraud[card2_fraud['count'] > 50]
top_card2_fraud = card2_fraud.head(50)['mean'].to_dict()

print(f"   card2: {len(top_card2_fraud)} fraud rates calculated")

# Addr1 fraud rates
addr1_fraud = df.groupby('addr1')['isFraud'].agg(['mean', 'count']).sort_values('count', ascending=False)
addr1_fraud = addr1_fraud[addr1_fraud['count'] > 50]
top_addr1_fraud = addr1_fraud.head(100)['mean'].to_dict()

print(f"   addr1: {len(top_addr1_fraud)} fraud rates calculated")

# Addr2 fraud rates
addr2_fraud = df.groupby('addr2')['isFraud'].agg(['mean', 'count']).sort_values('count', ascending=False)
addr2_fraud = addr2_fraud[addr2_fraud['count'] > 50]
top_addr2_fraud = addr2_fraud.head(50)['mean'].to_dict()

print(f"   addr2: {len(top_addr2_fraud)} fraud rates calculated")

# Update maps with REAL fraud rates
print(f"\n[5] Updating fraud rate maps...")

updates = {
    'isFraud_mean_card1': top_card1_fraud,
    'isFraud_mean_card2': top_card2_fraud,
    'isFraud_mean_addr1': top_addr1_fraud,
    'isFraud_mean_addr2': top_addr2_fraud,
}

for map_name, new_rates in updates.items():
    if map_name in agg_maps:
        # Merge with existing (keep synthetic for unknown cards)
        existing = agg_maps[map_name]
        agg_maps[map_name] = {**new_rates, **existing}  # Existing takes precedence
        print(f"   Updated {map_name}: {len(agg_maps[map_name])} total entries")
    else:
        agg_maps[map_name] = new_rates
        print(f"   Created {map_name}: {len(new_rates)} entries")

# Save
print(f"\n[6] Saving...")
joblib.dump(agg_maps, agg_path)
print(f"   Saved to: {agg_path}")

# Verify key cards
print(f"\n[7] Verifying...")
test_cards = [2616, 8026, 4272, 9500]
for card in test_cards:
    if card in agg_maps['isFraud_mean_card1']:
        rate = agg_maps['isFraud_mean_card1'][card]
        print(f"   card1={card}: {rate:.3f} fraud rate")
    else:
        print(f"   card1={card}: NOT FOUND")

print(f"\n" + "=" * 80)
print("UPDATE COMPLETE!")
print(f"Total fraud rate maps: {len([k for k in agg_maps if 'isFraud_mean' in k])}")
print(f"Restart inference: docker restart fraud-inference")
print("=" * 80)
