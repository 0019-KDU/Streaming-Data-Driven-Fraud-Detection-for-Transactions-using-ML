import pandas as pd
import os

def extract_examples():
    # Path to the dataset
    DATA_PATH = r"D:\Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML\ieee-fraud-detection\train_transaction.csv"
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    print("Reading dataset (this may take a moment)...")
    # Read only the first 10000 rows to be fast
    df = pd.read_csv(DATA_PATH, nrows=10000)
    
    # 1. Find a Real Fraud Example (isFraud=1)
    fraud_df = df[df['isFraud'] == 1]
    if not fraud_df.empty:
        fraud_example = fraud_df.iloc[0]
        print("\nREAL FRAUD EXAMPLE (from dataset):")
        print(fraud_example[['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'P_emaildomain', 'R_emaildomain']].to_json())
    else:
        print("\nNo fraud examples found in the first 10,000 rows.")

    # 2. Find a Real Legitimate Example (isFraud=0)
    legit_df = df[df['isFraud'] == 0]
    if not legit_df.empty:
        legit_example = legit_df.iloc[0]
        print("\nREAL LEGITIMATE EXAMPLE (from dataset):")
        print(legit_example[['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'P_emaildomain', 'R_emaildomain']].to_json())

if __name__ == "__main__":
    extract_examples()
