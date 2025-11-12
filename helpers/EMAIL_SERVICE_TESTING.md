# Email Notification Service - Testing Guide

## 📧 Overview

Your fraud detection system includes:
1. **Notification Service** - Consumes fraud alerts from Kafka and sends emails
2. **MailDev** - Email testing server (SMTP + Web UI to view emails)

---

## 🔍 Step 1: Check if Services are Running

### On Your VM (64.23.228.115)

```bash
cd /root/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src

# Check if both services are running
docker compose ps | grep -E "(notification|maildev)"
```

**Expected Output**:
```
notification-xxx        running
maildev                 running   0.0.0.0:1025->1025/tcp, 0.0.0.0:1080->1080/tcp
```

---

## 🌐 Step 2: Access MailDev Web UI

### Open in Your Browser:
```
http://64.23.228.115:1080
```

This is the **email inbox** where all fraud alert emails will appear.

**What You'll See**:
- Web interface showing all emails sent by the notification service
- Emails are NOT actually sent to real addresses
- Perfect for testing without spamming real inboxes

---

## 🚀 Step 3: Test Email Notification

### Method 1: Submit a Fraud Transaction (Triggers Email)

```bash
# Submit a high-risk transaction that will trigger fraud alert
curl -X POST http://64.23.228.115:8000/submit_transaction \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "EMAIL_TEST_001",
    "TransactionDT": 86400,
    "TransactionAmt": 5000.00,
    "ProductCD": "W",
    "card1": 99999,
    "card2": 999,
    "card3": 999,
    "card4": "visa",
    "card5": 999,
    "card6": "credit",
    "addr1": 999,
    "addr2": 999,
    "P_emaildomain": "suspicious.xyz",
    "R_emaildomain": "temporary.tk",
    "user_id": 12345,
    "timestamp": "2025-10-21T18:00:00Z"
  }'
```

**Note**: Include `"user_id": 12345` so the notification service can generate an email address.

### Method 2: Check Notification Service Logs

```bash
# Real-time logs
docker compose logs -f notification

# Recent logs
docker compose logs --tail=50 notification
```

**Expected Log Output** (when fraud detected):
```
INFO - Subscribed to Kafka topic: fraud_predictions
INFO - Generated email for user 12345: john.doe@example.com
INFO - Email sent to john.doe@example.com for transaction EMAIL_TEST_001
```

---

## 📊 Step 4: View Emails in MailDev

1. **Open**: `http://64.23.228.115:1080`
2. **Look for**: Email with subject `🚨 Fraud Alert: HIGH Risk Transaction Detected`
3. **Email should contain**:
   - Transaction ID
   - User ID
   - Amount
   - Risk Level
   - Fraud Probability
   - Decision (REVIEW/DECLINE)

### Example Email Content:
```
Subject: 🚨 Fraud Alert: HIGH Risk Transaction Detected (ID: EMAIL_TEST_001)

Dear Customer,

We have detected a potentially fraudulent transaction on your account.

🔴 FRAUD DETECTION ALERT

Transaction Details:
- Transaction ID: EMAIL_TEST_001
- User ID: 12345
- Amount: $5000.00 USD
- Timestamp: 2025-10-21T18:00:00Z

Fraud Analysis:
- Risk Level: HIGH
- Fraud Probability: 85.5%
- Decision: DECLINE
- Action Taken: Transaction has been BLOCKED for your protection

If you did not authorize this transaction, please contact us immediately.

Best regards,
Fraud Prevention Team
```

---

## 🐛 Troubleshooting

### Issue 1: No Emails Appearing in MailDev

**Check if notification service is running:**
```bash
docker compose ps notification
```

**If not running, start it:**
```bash
docker compose up -d notification
```

**Check logs for errors:**
```bash
docker compose logs notification | grep -i error
```

---

### Issue 2: Notification Service Not Consuming from Kafka

**Check Kafka topic has messages:**
```bash
# Check if fraud_predictions topic exists and has messages
docker compose logs inference | grep "fraud_predictions"
```

**Restart notification service:**
```bash
docker compose restart notification
docker compose logs -f notification
```

---

### Issue 3: SMTP Connection Errors

**Check MailDev is accessible:**
```bash
docker compose exec notification nc -zv maildev 1025
```

**Expected Output:**
```
maildev (172.18.0.x:1025) open
```

**If failed, restart MailDev:**
```bash
docker compose restart maildev
sleep 5
docker compose restart notification
```

---

## 🔧 Advanced: Manual Email Test

### Send Test Email Directly (bypasses Kafka)

```bash
# Connect to notification container
docker compose exec notification python3 << 'EOF'
import smtplib
from email.mime.text import MIMEText

msg = MIMEText("This is a test email from the notification service.")
msg["Subject"] = "Test Email - Fraud Detection System"
msg["From"] = "fraud.alerts@bank.com"
msg["To"] = "test@example.com"

with smtplib.SMTP("maildev", 1025) as server:
    server.sendmail("fraud.alerts@bank.com", "test@example.com", msg.as_string())
    print("✅ Test email sent successfully!")
EOF
```

Then check MailDev UI: `http://64.23.228.115:1080`

---

## 📈 Verify End-to-End Flow

### Complete Test Sequence:

1. **Submit fraud transaction** → API (port 8000)
2. **Producer sends to Kafka** → `transactions` topic
3. **Inference service processes** → Spark Streaming
4. **Fraud detected** → Sends to `fraud_predictions` topic
5. **Notification service consumes** → Kafka consumer
6. **Email sent** → MailDev SMTP (port 1025)
7. **View email** → MailDev UI (port 1080)

### Monitor All Services:
```bash
# Terminal 1: Watch inference logs
docker compose logs -f inference

# Terminal 2: Watch notification logs
docker compose logs -f notification

# Terminal 3: Submit test transaction
curl -X POST http://localhost:8000/submit_transaction -H "Content-Type: application/json" -d '{...}'

# Browser: Open MailDev
http://64.23.228.115:1080
```

---

## 📋 Service Status Checklist

Run these commands on your VM to verify everything:

```bash
cd /root/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src

echo "=== Service Status ==="
docker compose ps | grep -E "(notification|maildev)"

echo ""
echo "=== MailDev Ports ==="
netstat -tlnp | grep -E "(1080|1025)"

echo ""
echo "=== Notification Service Logs (Last 10 lines) ==="
docker compose logs --tail=10 notification

echo ""
echo "=== Check Kafka Consumer Group ==="
docker compose logs notification | grep -i "Subscribed to Kafka topic"

echo ""
echo "=== MailDev Web UI ==="
echo "Open in browser: http://64.23.228.115:1080"
```

---

## 🎯 Expected Behavior

### When Fraud is Detected:

1. **Inference Service Log**:
   ```
   INFO: Prediction successful - Fraud Probability: 0.855
   INFO: Writing to topic: fraud_predictions
   ```

2. **Notification Service Log**:
   ```
   INFO: Subscribed to Kafka topic: fraud_predictions
   INFO: Generated email for user 12345: alice.smith@example.com
   INFO: Email sent to alice.smith@example.com for transaction EMAIL_TEST_001
   ```

3. **MailDev UI**:
   - Shows new email in inbox
   - Subject: 🚨 Fraud Alert: HIGH Risk...
   - Contains transaction details

4. **Dashboard**:
   - Transaction appears in "Recent Transactions" table
   - Type: FRAUD (red row)
   - Decision: REVIEW or DECLINE

---

## 🔐 Configuration Details

### Current Setup (from docker-compose.yml):

**Notification Service**:
- Kafka Topic: `fraud_predictions`
- Consumer Group: `notification-service-group`
- SMTP Host: `maildev`
- SMTP Port: `1025`
- From Email: `fraud.alerts@bank.com`

**MailDev**:
- Web UI Port: `1080` (external: `http://64.23.228.115:1080`)
- SMTP Port: `1025` (internal only)
- Storage: In-memory (emails cleared on restart)

---

## 💡 Tips

1. **Email addresses are auto-generated** based on `user_id` using Faker library
   - Same `user_id` = Same email (deterministic)
   - user_id: 12345 → always generates same email

2. **MailDev is for testing only**
   - Emails are NOT sent to real addresses
   - Perfect for development/testing
   - To use real SMTP (Gmail, SendGrid), update `SMTP_HOST` in docker-compose.yml

3. **To test with different users**:
   - Change `user_id` in your test transactions
   - Each user_id will get a different generated email address

4. **Check email generation**:
   ```bash
   docker compose exec notification python3 -c "
   from faker import Faker
   Faker.seed(12345)
   print(Faker().email())
   "
   ```

---

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| No emails appearing | Check notification service is running: `docker compose ps notification` |
| SMTP connection refused | Restart MailDev: `docker compose restart maildev` |
| Consumer not receiving messages | Check inference is writing to `fraud_predictions` topic |
| MailDev UI not accessible | Check port 1080 is not blocked by firewall |
| Missing user_id in transaction | Add `"user_id": 12345` to your JSON payload |

---

## 📞 Quick Test Command

Run this complete test on your VM:

```bash
# Submit fraud transaction with user_id
curl -X POST http://localhost:8000/submit_transaction \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "QUICK_EMAIL_TEST",
    "TransactionDT": 86400,
    "TransactionAmt": 9999.99,
    "ProductCD": "W",
    "card1": 88888,
    "card4": "visa",
    "card6": "credit",
    "user_id": 99999,
    "timestamp": "2025-10-21T20:00:00Z"
  }' && echo ""

# Wait 3 seconds
sleep 3

# Check notification logs
echo "=== Checking notification service logs ==="
docker compose logs --tail=5 notification

echo ""
echo "=== Open MailDev UI to see email ==="
echo "http://64.23.228.115:1080"
```

---

**Last Updated**: October 21, 2025  
**System**: Fraud Detection Platform - Email Notification Service
