import requests
import json
import time
import os
import sys

# Try to import Kafka consumer dependencies
try:
    from confluent_kafka import Consumer
    from dotenv import load_dotenv
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False
    print("⚠️  'confluent-kafka' or 'python-dotenv' not found. Cannot verify predictions automatically.")
    print("   Run: pip install confluent-kafka python-dotenv")

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

def get_kafka_consumer():
    if not HAS_KAFKA:
        return None
        
    # Load env vars from parent directory
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
    
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
    username = os.getenv('KAFKA_USERNAME')
    password = os.getenv('KAFKA_PASSWORD')
    
    if not bootstrap_servers:
        print("⚠️  KAFKA_BOOTSTRAP_SERVERS not found in .env")
        return None

    conf = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': f'verifier-script-{int(time.time())}',
        'auto.offset.reset': 'latest'
    }
    
    if username and password:
        conf.update({
            'security.protocol': 'SASL_SSL',
            'sasl.mechanism': 'PLAIN',
            'sasl.username': username,
            'sasl.password': password
        })
        
    try:
        consumer = Consumer(conf)
        consumer.subscribe(['fraud_predictions', 'legit_predictions'])
        return consumer
    except Exception as e:
        print(f"⚠️  Failed to create Kafka consumer: {e}")
        return None

def send_and_verify(payload, label):
    print(f"\n---------------------------------------------------")
    print(f"Sending {label} transaction (ID: {payload.get('TransactionID')})...")
    
    consumer = get_kafka_consumer()
    
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            print(f"✅ API Submission Success! Status: {response.status_code}")
            print(f"   Message: {response.json().get('message')}")
            
            if consumer:
                print("⏳ Waiting for prediction from Kafka (timeout 30s)...")
                start_time = time.time()
                found = False
                while (time.time() - start_time) < 30:
                    msg = consumer.poll(1.0)
                    if msg is None: continue
                    if msg.error():
                        print(f"   Kafka Error: {msg.error()}")
                        continue
                        
                    data = json.loads(msg.value().decode('utf-8'))
                    
                    # Check if this is our transaction
                    # The output format usually has 'transaction_id' or 'TransactionID'
                    tx_id = str(data.get('transaction_id') or data.get('TransactionID'))
                    target_id = str(payload.get('TransactionID'))
                    
                    if tx_id == target_id:
                        print(f"\n🎯 PREDICTION RECEIVED:")
                        print(f"   Transaction ID: {tx_id}")
                        print(f"   Probability:    {data.get('probability', 'N/A')}")
                        print(f"   Prediction:     {data.get('prediction', 'N/A')} (1=Fraud, 0=Legit)")
                        print(f"   Risk Level:     {data.get('risk_level', 'N/A')}")
                        found = True
                        break
                
                if not found:
                    print("❌ Timed out waiting for prediction.")
                
                consumer.close()
            else:
                print("\nTo verify manually:")
                print("1. Check logs: docker logs fraud-inference")
                
        else:
            print(f"❌ API Failed. Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

# Send transactions
send_and_verify(fraud_payload, "FRAUDULENT")
time.sleep(2)
send_and_verify(legit_payload, "LEGITIMATE")
