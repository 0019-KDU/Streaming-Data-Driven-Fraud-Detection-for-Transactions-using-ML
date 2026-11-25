#!/usr/bin/env python3
"""
Extract Transaction Data for Kafka Streaming
Purpose: Extract subset of transactions with merchant mapping for Kafka
Output: kafka_transactions.csv (ready for streaming)
"""

import pandas as pd
import sys
import os

# Configuration
INPUT_FILE = "ieee-fraud-detection/train_transaction.csv"
OUTPUT_FILE = "kafka_transactions.csv"
SAMPLE_SIZE = 10000  # Number of transactions to extract

# Merchant mapping based on ProductCD and card pattern
MERCHANT_MAP = {
    'W': ["Amazon", "eBay", "Walmart", "Target", "BestBuy", "NewEgg", "Etsy", "Shopify"],
    'H': ["BestBuy", "MicroCenter", "Newegg", "Fry_Electronics", "B&H_Photo"],
    'C': ["ATM_Chase", "ATM_BofA", "ATM_Wells_Fargo", "ATM_Citibank"],
    'S': ["Uber", "Lyft", "Netflix", "Spotify", "AmazonPrime", "Hulu"],
    'R': ["Airbnb", "Booking.com", "Hotels.com", "Expedia", "Enterprise_Rental"]
}

def get_merchant(product_cd, card1):
    """Map merchant based on ProductCD and card1 value"""
    if pd.isna(product_cd) or pd.isna(card1):
        return "Unknown"
    
    merchants = MERCHANT_MAP.get(product_cd, ["Unknown"])
    return merchants[int(card1) % len(merchants)]

def main():
    print("=" * 50)
    print("Kafka Data Extraction Script")
    print("=" * 50)
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Sample size: {SAMPLE_SIZE} transactions")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)
    
    print("Loading transaction data...")
    
    # Select columns to extract
    columns_to_extract = [
        'TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD',
        'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'P_emaildomain'
    ]
    
    # Read only needed columns and sample rows
    df = pd.read_csv(
        INPUT_FILE,
        usecols=columns_to_extract,
        nrows=SAMPLE_SIZE
    )
    
    print(f"✓ Loaded {len(df)} transactions")
    
    # Add merchant mapping
    print("Mapping merchants...")
    df['Merchant'] = df.apply(lambda row: get_merchant(row['ProductCD'], row['card1']), axis=1)
    
    # Save to output file
    print(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n✓ Extraction completed successfully!\n")
    
    # Display statistics
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total records: {len(df)}\n")
    
    print("Sample records:")
    print(df.head().to_string(index=False))
    print()
    
    print("Fraud distribution:")
    fraud_dist = df['isFraud'].value_counts().sort_index()
    for fraud_val, count in fraud_dist.items():
        print(f"  isFraud={fraud_val}: {count} transactions ({count/len(df)*100:.2f}%)")
    print()
    
    print("Product distribution:")
    product_dist = df['ProductCD'].value_counts().sort_index()
    for product, count in product_dist.items():
        print(f"  ProductCD={product}: {count} transactions")
    print()
    
    print("Top 5 merchants:")
    merchant_dist = df['Merchant'].value_counts().head(5)
    for merchant, count in merchant_dist.items():
        print(f"  {merchant}: {count} transactions")
    print()
    
    print("Transaction amount statistics:")
    print(f"  Mean: ${df['TransactionAmt'].mean():.2f}")
    print(f"  Median: ${df['TransactionAmt'].median():.2f}")
    print(f"  Min: ${df['TransactionAmt'].min():.2f}")
    print(f"  Max: ${df['TransactionAmt'].max():.2f}")
    print()
    
    print("=" * 50)
    print("Ready for Kafka streaming!")
    print("=" * 50)

if __name__ == "__main__":
    main()
