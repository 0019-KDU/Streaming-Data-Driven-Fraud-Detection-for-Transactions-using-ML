#!/bin/bash
# Complete Test Suite for Fraud Detection System
# Uses the FULL IEEE-CIS transaction format

API_URL="http://localhost:8000/api/v1/transactions/submit"

echo "================================="
echo "Fraud Detection Test Suite"
echo "================================="
echo ""

# Test 1: Very Low Amount + Risky Email (SHOULD BLOCK)
echo "Test 1: Low amount ($0.50) + risky email domains"
echo "Expected: BLOCK/HOLD with HIGH risk"
echo "---"
curl -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "TEST_001",
    "TransactionDT": null,
    "TransactionAmt": 0.5,
    "ProductCD": "C",
    "card1": 1,
    "card2": 1,
    "card3": 1,
    "card4": "visa",
    "card5": 1,
    "card6": "debit",
    "addr1": 1,
    "addr2": 1,
    "P_emaildomain": "anonymous.com",
    "R_emaildomain": "mailinator.com"
  }'
echo -e "\n\n"
sleep 2

# Test 2: Very High Amount + Risky Email (SHOULD BLOCK)
echo "Test 2: High amount ($25,000) + risky email domains"
echo "Expected: BLOCK with HIGH risk"
echo "---"
curl -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "TEST_002",
    "TransactionDT": null,
    "TransactionAmt": 25000.00,
    "ProductCD": "W",
    "card1": 2,
    "card2": 2,
    "card3": 2,
    "card4": "visa",
    "card5": 2,
    "card6": "credit",
    "addr1": 2,
    "addr2": 2,
    "P_emaildomain": "10minutemail.com",
    "R_emaildomain": "guerrillamail.com"
  }'
echo -e "\n\n"
sleep 2

# Test 3: Normal Transaction (SHOULD APPROVE)
echo "Test 3: Normal amount ($45.99) + legitimate email"
echo "Expected: APPROVE with LOW risk"
echo "---"
curl -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "TEST_003",
    "TransactionDT": null,
    "TransactionAmt": 45.99,
    "ProductCD": "W",
    "card1": 12345,
    "card2": 100,
    "card3": 150,
    "card4": "mastercard",
    "card5": 200,
    "card6": "credit",
    "addr1": 100,
    "addr2": 50,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }'
echo -e "\n\n"
sleep 2

# Test 4: Velocity Test - Rapid Transactions (User ID: card1=99999)
echo "Test 4: Velocity Detection - Rapid transactions from same user"
echo "Sending 5 transactions in quick succession..."
echo "---"

echo "Transaction 4a: $100"
curl -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "TEST_004A",
    "TransactionDT": null,
    "TransactionAmt": 100.00,
    "ProductCD": "W",
    "card1": 99999,
    "card2": 888,
    "card3": 777,
    "card4": "visa",
    "card5": 666,
    "card6": "debit",
    "addr1": 999,
    "addr2": 111,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }'
echo -e "\n"
sleep 5

echo "Transaction 4b: $150 (5 seconds later)"
curl -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "TEST_004B",
    "TransactionDT": null,
    "TransactionAmt": 150.00,
    "ProductCD": "W",
    "card1": 99999,
    "card2": 888,
    "card3": 777,
    "card4": "visa",
    "card5": 666,
    "card6": "debit",
    "addr1": 999,
    "addr2": 111,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }'
echo -e "\n"
sleep 5

echo "Transaction 4c: $200 (10 seconds later)"
curl -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "TEST_004C",
    "TransactionDT": null,
    "TransactionAmt": 200.00,
    "ProductCD": "W",
    "card1": 99999,
    "card2": 888,
    "card3": 777,
    "card4": "visa",
    "card5": 666,
    "card6": "debit",
    "addr1": 999,
    "addr2": 111,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }'
echo -e "\n"
sleep 5

echo "Transaction 4d: $250 (15 seconds later)"
curl -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "TEST_004D",
    "TransactionDT": null,
    "TransactionAmt": 250.00,
    "ProductCD": "W",
    "card1": 99999,
    "card2": 888,
    "card3": 777,
    "card4": "visa",
    "card5": 666,
    "card6": "debit",
    "addr1": 999,
    "addr2": 111,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }'
echo -e "\n"
sleep 5

echo "Transaction 4e: $5,000 (20 seconds later) - HUGE SPIKE!"
echo "Expected: BLOCK/HOLD with HIGH risk (velocity + amount spike)"
curl -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "TEST_004E",
    "TransactionDT": null,
    "TransactionAmt": 5000.00,
    "ProductCD": "W",
    "card1": 99999,
    "card2": 888,
    "card3": 777,
    "card4": "visa",
    "card5": 666,
    "card6": "debit",
    "addr1": 999,
    "addr2": 111,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }'
echo -e "\n\n"
sleep 2

# Test 5: Night Transaction + Risky Email (SHOULD BLOCK/HOLD)
echo "Test 5: Night transaction (if current time is night) + risky email"
echo "Expected: BLOCK/HOLD with MEDIUM/HIGH risk"
echo "---"
curl -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "TEST_005",
    "TransactionDT": null,
    "TransactionAmt": 500.00,
    "ProductCD": "C",
    "card1": 5555,
    "card2": 5555,
    "card3": 5555,
    "card4": "discover",
    "card5": 5555,
    "card6": "credit",
    "addr1": 555,
    "addr2": 555,
    "P_emaildomain": "tempmail.com",
    "R_emaildomain": "yopmail.com"
  }'
echo -e "\n\n"

echo "================================="
echo "Test Suite Complete!"
echo "================================="
echo ""
echo "Expected Results Summary:"
echo "  Test 1: BLOCK/HOLD (low amount + risky email)"
echo "  Test 2: BLOCK (high amount + risky email)"
echo "  Test 3: APPROVE (normal transaction)"
echo "  Test 4a-d: APPROVE (normal pattern)"
echo "  Test 4e: BLOCK/HOLD (velocity spike)"
echo "  Test 5: BLOCK/HOLD (night + risky email)"
echo ""
echo "Check your dashboard or Kafka topics for results!"
