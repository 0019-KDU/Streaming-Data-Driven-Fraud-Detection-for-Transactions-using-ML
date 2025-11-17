"""
Generate Real Test Payloads from IEEE-CIS Test Dataset

This script reads test_transaction.csv and creates JSON payloads for testing
the ML model with REAL data from the same distribution as training.
"""

import pandas as pd
import json
import os
import numpy as np

def generate_test_payloads():
    """Generate 10 diverse test payloads from IEEE-CIS test_transaction.csv"""
    
    # Check if test file exists
    test_file = 'ieee-fraud-detection/test_transaction.csv'
    if not os.path.exists(test_file):
        print(f"❌ ERROR: {test_file} not found!")
        print(f"📂 Current directory: {os.getcwd()}")
        print(f"\n💡 Please ensure test_transaction.csv is in ieee-fraud-detection/ folder")
        return
    
    print(f"✅ Found {test_file}")
    print("📖 Reading CSV file...")
    
    # Read test data
    df = pd.read_csv(test_file)
    print(f"✅ Loaded {len(df):,} transactions")
    print(f"📊 Columns: {list(df.columns)[:20]}...")  # Show first 20 columns
    
    # Required columns for our API
    required_cols = [
        'TransactionAmt', 'ProductCD', 
        'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'dist1', 'dist2',
        'P_emaildomain', 'R_emaildomain'
    ]
    
    # Check which columns exist
    available_cols = [col for col in required_cols if col in df.columns]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    print(f"\n✅ Available columns: {len(available_cols)}/{len(required_cols)}")
    if missing_cols:
        print(f"⚠️  Missing columns: {missing_cols}")
    
    # Strategy: Pick diverse transactions
    print("\n🎯 Selecting diverse test cases...")
    
    test_cases = []
    
    # 1. LOW AMOUNT transactions (likely legitimate)
    low_amt = df[df['TransactionAmt'] < 50].sample(n=2, random_state=42)
    test_cases.append(('LOW_AMOUNT_1', low_amt.iloc[0]))
    test_cases.append(('LOW_AMOUNT_2', low_amt.iloc[1]))
    
    # 2. MEDIUM AMOUNT transactions
    medium_amt = df[(df['TransactionAmt'] >= 50) & (df['TransactionAmt'] < 200)].sample(n=2, random_state=42)
    test_cases.append(('MEDIUM_AMOUNT_1', medium_amt.iloc[0]))
    test_cases.append(('MEDIUM_AMOUNT_2', medium_amt.iloc[1]))
    
    # 3. HIGH AMOUNT transactions (potential fraud)
    high_amt = df[df['TransactionAmt'] > 500].sample(n=2, random_state=42)
    test_cases.append(('HIGH_AMOUNT_1', high_amt.iloc[0]))
    test_cases.append(('HIGH_AMOUNT_2', high_amt.iloc[1]))
    
    # 4. DISCOVER CARD transactions (higher fraud rate)
    if 'card4' in df.columns:
        discover = df[df['card4'] == 'discover'].sample(n=2, random_state=42)
        test_cases.append(('DISCOVER_CARD_1', discover.iloc[0]))
        test_cases.append(('DISCOVER_CARD_2', discover.iloc[1]))
    
    # 5. PRODUCT C transactions (highest fraud rate)
    if 'ProductCD' in df.columns:
        product_c = df[df['ProductCD'] == 'C'].sample(n=2, random_state=42)
        test_cases.append(('PRODUCT_C_1', product_c.iloc[0]))
        test_cases.append(('PRODUCT_C_2', product_c.iloc[1]))
    
    print(f"✅ Selected {len(test_cases)} test cases")
    
    # Generate JSON files
    print("\n📝 Generating JSON payloads...\n")
    
    for idx, (name, row) in enumerate(test_cases, 1):
        payload = {}
        
        # Add available columns
        for col in available_cols:
            value = row[col]
            
            # Handle NaN values
            if pd.isna(value):
                if col in ['TransactionAmt', 'dist1', 'dist2']:
                    payload[col] = 0.0
                elif col in ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']:
                    payload[col] = -1
                elif col in ['card4', 'card6', 'ProductCD']:
                    payload[col] = "unknown"
                elif col in ['P_emaildomain', 'R_emaildomain']:
                    payload[col] = "unknown"
            else:
                # Convert numpy types to Python types for JSON serialization
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    payload[col] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    payload[col] = float(value)
                else:
                    payload[col] = value
        
        # Save to JSON file
        filename = f"TEST_{idx:02d}_{name}.json"
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=2)
        
        # Print summary
        print(f"✅ {filename}")
        print(f"   Amount: ${payload.get('TransactionAmt', 0):.2f}")
        print(f"   Product: {payload.get('ProductCD', 'N/A')}")
        print(f"   Card: {payload.get('card1', 'N/A')} ({payload.get('card4', 'N/A')})")
        print(f"   Email: {payload.get('P_emaildomain', 'N/A')}")
        print()
    
    print(f"✅ SUCCESS! Generated {len(test_cases)} JSON test files")
    print(f"\n📋 Test these in Postman:")
    print(f"   POST http://167.71.224.89:8000/transaction")
    print(f"   Body → raw → JSON → Paste file content")
    print(f"\n🎯 Expected: ML probabilities will vary (0.05, 0.30, 0.70, etc.)")
    print(f"   because these are REAL patterns from IEEE-CIS dataset!\n")
    
    # Create a summary CSV
    summary_data = []
    for name, row in test_cases:
        summary_data.append({
            'Test_Name': name,
            'TransactionAmt': row.get('TransactionAmt', 0),
            'ProductCD': row.get('ProductCD', 'N/A'),
            'card1': row.get('card1', 'N/A'),
            'card4': row.get('card4', 'N/A'),
            'P_emaildomain': row.get('P_emaildomain', 'N/A'),
            'dist1': row.get('dist1', 0),
            'dist2': row.get('dist2', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('TEST_PAYLOADS_SUMMARY.csv', index=False)
    print(f"📊 Summary saved to: TEST_PAYLOADS_SUMMARY.csv")
    
    # Show statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total transactions: {len(df):,}")
    print(f"   Amount range: ${df['TransactionAmt'].min():.2f} - ${df['TransactionAmt'].max():.2f}")
    print(f"   Avg amount: ${df['TransactionAmt'].mean():.2f}")
    
    if 'ProductCD' in df.columns:
        print(f"\n   Product distribution:")
        for prod, count in df['ProductCD'].value_counts().head(5).items():
            print(f"     {prod}: {count:,} ({count/len(df)*100:.1f}%)")
    
    if 'card4' in df.columns:
        print(f"\n   Card type distribution:")
        for card, count in df['card4'].value_counts().head(5).items():
            print(f"     {card}: {count:,} ({count/len(df)*100:.1f}%)")

if __name__ == '__main__':
    print("=" * 70)
    print("🎯 IEEE-CIS Test Payload Generator")
    print("=" * 70)
    print()
    
    generate_test_payloads()
    
    print("\n" + "=" * 70)
    print("✅ DONE! Test payloads ready for ML model validation")
    print("=" * 70)
