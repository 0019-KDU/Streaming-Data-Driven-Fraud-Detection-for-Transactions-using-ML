import requests
import json
import time

# Load the real transaction payloads
try:
    with open('fraud_example.json', 'r') as f:
        fraud_payload = json.load(f)
    with open('legit_example_1.json', 'r') as f:
        legit_payload = json.load(f)
    print("Loaded transaction payloads.")
except FileNotFoundError:
    print("Payload files not found. Please run extract_real_transactions.py first.")
    exit(1)

# API Endpoint
url = "http://localhost:8000/api/v1/transactions/submit"

def send_transaction(payload, label):
    print(f"\nSending {label} transaction (ID: {payload.get('TransactionID')})...")
    try:
        # Add timestamp if missing (needed for new fix)
        if 'TransactionDT' in payload:
            # Keep original DT for consistency
            pass
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            print(f"✅ Success! Status: {response.status_code}")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Failed. Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

# Send transactions
send_transaction(fraud_payload, "FRAUDULENT")
time.sleep(1)
send_transaction(legit_payload, "LEGITIMATE")
