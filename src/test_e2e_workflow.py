"""
End-to-End Workflow Test Script

Tests the complete fraud detection pipeline:
1. Send transaction via Producer API (POST /api/v1/transactions/submit)
2. Transaction published to Kafka "transactions" topic
3. Spark Structured Streaming consumes and processes
4. XGBoost model scores transaction
5. Velocity + ATO services check Redis
6. Decision engine makes final decision
7. Result published to "fraud_predictions" or "legit_predictions" topics
8. Dashboard consumes and displays in real-time

Usage:
    python test_e2e_workflow.py
"""

import json
import time
import requests
from datetime import datetime
from confluent_kafka import Consumer
import os

# Configuration
PRODUCER_API_URL = "http://localhost:8000/api/v1/transactions/submit"
KAFKA_BOOTSTRAP_SERVERS = "pkc-921jm.us-east-2.aws.confluent.cloud:9092"
KAFKA_USERNAME = "TUISIFY5HCFLGXIH"
KAFKA_PASSWORD = "HIhrR1hP0Oj64llWYN8E4U3gnsJ83b64OGcrFDYvnkTppiMo1UkMwUUdfSFr6PLl"

# Test transaction payloads
TEST_TRANSACTIONS = [
    {
        "name": "Normal Transaction",
        "payload": {
            "TransactionAmt": 50.00,
            "ProductCD": "W",
            "card1": 12345,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com",
            "addr1": 100,
            "dist1": 50.0
        },
        "expected": "legit"
    },
    {
        "name": "High Amount Transaction",
        "payload": {
            "TransactionAmt": 5000.00,
            "ProductCD": "W",
            "card1": 54321,
            "card4": "mastercard",
            "card6": "credit",
            "P_emaildomain": "yahoo.com",
            "addr1": 200,
            "dist1": 100.0
        },
        "expected": "fraud (possible)"
    },
    {
        "name": "Suspicious Email Domain",
        "payload": {
            "TransactionAmt": 150.00,
            "ProductCD": "H",
            "card1": 99999,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "mailinator.com",  # Risky domain
            "addr1": 300,
            "dist1": 200.0
        },
        "expected": "fraud (possible)"
    }
]


def test_producer_api_health():
    """Test 1: Check if Producer API is healthy"""
    print("\n" + "="*80)
    print("TEST 1: Producer API Health Check")
    print("="*80)
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Producer API is HEALTHY")
            print(f"   Service: {data.get('service')}")
            print(f"   Status: {data.get('status')}")
            print(f"   Kafka Topic: {data.get('kafka_topic')}")
            return True
        else:
            print(f"❌ Producer API returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Producer API is NOT reachable: {e}")
        print("   Make sure the producer-api container is running:")
        print("   docker ps | grep producer-api")
        return False


def test_submit_transactions():
    """Test 2: Submit test transactions via API"""
    print("\n" + "="*80)
    print("TEST 2: Submit Test Transactions")
    print("="*80)
    
    transaction_ids = []
    
    for i, test_case in enumerate(TEST_TRANSACTIONS, 1):
        print(f"\n📤 Test Case {i}: {test_case['name']}")
        print(f"   Expected: {test_case['expected']}")
        
        try:
            response = requests.post(
                PRODUCER_API_URL,
                json=test_case['payload'],
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                txn_id = data.get('transaction_id')
                transaction_ids.append(txn_id)
                
                print(f"   ✅ Transaction submitted successfully")
                print(f"   Transaction ID: {txn_id}")
                print(f"   Status: {data.get('status')}")
                print(f"   Message: {data.get('message')}")
            else:
                print(f"   ❌ Failed with status code: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error submitting transaction: {e}")
    
    return transaction_ids


def test_kafka_output_consumer(transaction_ids, timeout=30):
    """Test 3: Consume results from Kafka output topics"""
    print("\n" + "="*80)
    print("TEST 3: Consume Results from Kafka Output Topics")
    print("="*80)
    print(f"⏳ Waiting up to {timeout} seconds for results...")
    print(f"   Looking for transaction IDs: {transaction_ids}")
    
    consumer_config = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "e2e-test-consumer",
        "auto.offset.reset": "earliest",
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "PLAIN",
        "sasl.username": KAFKA_USERNAME,
        "sasl.password": KAFKA_PASSWORD,
    }
    
    consumer = Consumer(consumer_config)
    consumer.subscribe(["fraud_predictions", "legit_predictions"])
    
    found_transactions = {}
    start_time = time.time()
    
    print("\n📥 Consuming messages...")
    
    while len(found_transactions) < len(transaction_ids) and (time.time() - start_time) < timeout:
        msg = consumer.poll(timeout=1.0)
        
        if msg is None:
            continue
            
        if msg.error():
            print(f"   ⚠️ Consumer error: {msg.error()}")
            continue
        
        try:
            data = json.loads(msg.value().decode('utf-8'))
            txn_id = data.get('transaction_id')
            
            if txn_id in transaction_ids and txn_id not in found_transactions:
                found_transactions[txn_id] = {
                    'topic': msg.topic(),
                    'data': data
                }
                
                print(f"\n   ✅ Found result for transaction: {txn_id}")
                print(f"      Topic: {msg.topic()}")
                print(f"      Decision: {data.get('decision')}")
                print(f"      Probability: {data.get('fraud_probability', data.get('probability', 'N/A'))}")
                print(f"      Risk Level: {data.get('risk_level')}")
                print(f"      Risk Factors: {data.get('risk_factors', 'none')[:100]}")
                
        except Exception as e:
            print(f"   ⚠️ Error parsing message: {e}")
    
    consumer.close()
    
    # Summary
    print("\n" + "-"*80)
    print(f"📊 RESULTS SUMMARY:")
    print(f"   Submitted: {len(transaction_ids)} transactions")
    print(f"   Received: {len(found_transactions)} results")
    
    if len(found_transactions) == len(transaction_ids):
        print("   ✅ ALL TRANSACTIONS PROCESSED SUCCESSFULLY!")
        
        # Breakdown by topic
        fraud_count = sum(1 for t in found_transactions.values() if t['topic'] == 'fraud_predictions')
        legit_count = sum(1 for t in found_transactions.values() if t['topic'] == 'legit_predictions')
        
        print(f"\n   Fraud Predictions: {fraud_count}")
        print(f"   Legit Predictions: {legit_count}")
        
        return True, found_transactions
    else:
        print(f"   ⚠️ MISSING {len(transaction_ids) - len(found_transactions)} RESULTS!")
        print(f"   Possible issues:")
        print(f"      - Spark inference service not running")
        print(f"      - Model/feature pipeline files missing")
        print(f"      - Redis connection issues")
        print(f"      - Kafka connectivity problems")
        
        return False, found_transactions


def test_spark_logs():
    """Test 4: Check Spark Structured Streaming logs"""
    print("\n" + "="*80)
    print("TEST 4: Check Spark Inference Service Logs")
    print("="*80)
    
    print("Run the following command to check Spark logs:")
    print("   docker logs fraud-inference-spark --tail 50")
    print("\nLook for:")
    print("   ✅ 'Streaming queries started successfully'")
    print("   ✅ 'numInputRows' > 0 (batch processing)")
    print("   ✅ 'MicroBatchExecution: Streaming query made progress'")
    print("   ❌ Any Python errors or exceptions")


def test_dashboard():
    """Test 5: Check Dashboard connectivity"""
    print("\n" + "="*80)
    print("TEST 5: Dashboard Accessibility")
    print("="*80)
    
    try:
        response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
        
        if response.status_code == 200:
            print("✅ Dashboard is ACCESSIBLE")
            print("   URL: http://localhost:8501")
            print("   Open in browser to see real-time results!")
            return True
        else:
            print(f"❌ Dashboard returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Dashboard is NOT reachable: {e}")
        print("   Make sure the dashboard container is running:")
        print("   docker ps | grep dashboard")
        return False


def main():
    """Run complete end-to-end workflow test"""
    print("\n" + "="*80)
    print("🚀 FRAUD DETECTION E2E WORKFLOW TEST")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'producer_health': False,
        'transactions_submitted': False,
        'results_received': False,
        'dashboard_accessible': False
    }
    
    # Test 1: Producer API health
    results['producer_health'] = test_producer_api_health()
    
    if not results['producer_health']:
        print("\n❌ Cannot proceed - Producer API is not healthy")
        return
    
    # Test 2: Submit transactions
    transaction_ids = test_submit_transactions()
    results['transactions_submitted'] = len(transaction_ids) > 0
    
    if not results['transactions_submitted']:
        print("\n❌ Cannot proceed - Failed to submit transactions")
        return
    
    # Wait for Spark to process
    print("\n⏳ Waiting 5 seconds for Spark to process transactions...")
    time.sleep(5)
    
    # Test 3: Check Kafka output
    success, found_txns = test_kafka_output_consumer(transaction_ids, timeout=30)
    results['results_received'] = success
    
    # Test 4: Spark logs info
    test_spark_logs()
    
    # Test 5: Dashboard
    results['dashboard_accessible'] = test_dashboard()
    
    # Final summary
    print("\n" + "="*80)
    print("📋 FINAL TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("   Your end-to-end fraud detection pipeline is working correctly!")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("   Check the logs above for troubleshooting steps")
    
    print("\n" + "="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
