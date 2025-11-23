"""
Test real IEEE-CIS transaction using Python requests
"""
import requests
import json
from pathlib import Path

# Load the payload
payload_file = Path(__file__).parent / "test_transaction_payload.json"
with open(payload_file, 'r') as f:
    transaction = json.load(f)

print("="*80)
print("Testing Real IEEE-CIS Transaction")
print("="*80)
print(f"\nTransaction Details:")
print(f"   ID: {transaction['TransactionID']}")
print(f"   Amount: ${transaction['TransactionAmt']:.2f}")
print(f"   Card: {transaction['card4']} {transaction['card6']}")
print(f"   Email: {transaction['P_emaildomain']}")
print(f"\nSending to: http://localhost:8000/api/v1/transactions/submit")

try:
    response = requests.post(
        "http://localhost:8000/api/v1/transactions/submit",
        json=transaction,
        timeout=10
    )
    
    print(f"\n✅ Response Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n📦 Response Data:")
        print(json.dumps(data, indent=2))
        print(f"\n✅ Transaction submitted successfully!")
        print(f"   Transaction ID: {data.get('transaction_id')}")
        print(f"   Status: {data.get('status')}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)

except requests.exceptions.ConnectionError:
    print(f"\n❌ Error: Cannot connect to API")
    print("   Make sure producer-api is running:")
    print("   docker-compose up -d producer-api")
    
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "="*80)
print("Check Results:")
print("   1. Dashboard: http://localhost:8501")
print("   2. Spark logs: docker logs fraud-inference-spark --tail 50")
print("="*80)
