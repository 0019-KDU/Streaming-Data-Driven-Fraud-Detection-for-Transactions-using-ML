#!/bin/bash
#
# Email Notification Service - Quick Test Script
# Tests the complete email notification flow
#

echo "======================================================================"
echo "Email Notification Service - Testing"
echo "======================================================================"
echo ""

cd /root/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src

# Step 1: Check if services are running
echo "Step 1: Checking if notification and maildev services are running..."
echo ""
docker compose ps | grep -E "(notification|maildev)"
echo ""

# Step 2: Check MailDev accessibility
echo "Step 2: Checking MailDev SMTP port..."
docker compose exec notification nc -zv maildev 1025 2>&1 | grep -q "open" && echo "✅ MailDev SMTP is accessible" || echo "❌ MailDev SMTP not accessible"
echo ""

# Step 3: Check notification service Kafka subscription
echo "Step 3: Checking notification service Kafka subscription..."
docker compose logs notification | grep -q "Subscribed to Kafka topic" && echo "✅ Notification service subscribed to Kafka" || echo "❌ Not subscribed to Kafka"
echo ""

# Step 4: Submit a test fraud transaction
echo "Step 4: Submitting test fraud transaction (will trigger email)..."
echo ""

curl -X POST http://localhost:8000/submit_transaction \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "EMAIL_TEST_'$(date +%s)'",
    "TransactionDT": 86400,
    "TransactionAmt": 8888.88,
    "ProductCD": "W",
    "card1": 77777,
    "card2": 777,
    "card3": 777,
    "card4": "visa",
    "card5": 777,
    "card6": "credit",
    "addr1": 777,
    "addr2": 777,
    "P_emaildomain": "fraud-test.xyz",
    "R_emaildomain": "temporary.com",
    "user_id": 55555,
    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
  }'

echo ""
echo ""
echo "Waiting 5 seconds for processing..."
sleep 5

# Step 5: Check notification service logs
echo ""
echo "Step 5: Checking notification service logs (last 10 lines)..."
echo "================================================================"
docker compose logs --tail=10 notification
echo "================================================================"
echo ""

# Step 6: Check if email was sent
echo "Step 6: Checking if email was sent..."
docker compose logs --tail=20 notification | grep -q "Email sent" && echo "✅ Email sent successfully!" || echo "❌ No email sent (check logs above)"
echo ""

# Step 7: Display MailDev URL
echo "======================================================================"
echo "✅ Test Complete!"
echo "======================================================================"
echo ""
echo "To view the email, open MailDev UI in your browser:"
echo ""
echo "  🌐 http://64.23.228.115:1080"
echo ""
echo "You should see an email with subject:"
echo "  🚨 Fraud Alert: [RISK_LEVEL] Risk Transaction Detected"
echo ""
echo "======================================================================"
