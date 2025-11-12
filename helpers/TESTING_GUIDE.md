# Testing Guide - Why All Fields Are Required

## ❓ Your Question: "Why do tests have so many fields?"

Great question! The simplified tests I showed earlier (with only 4 fields) were **wrong**. Here's why all fields matter:

---

## 🎯 Why All Fields Are Required

Your **actual transaction format** from the REST API has **14 fields**:

```json
{
  "TransactionID": "TXN_1794AFEC6213",      // ✅ Required: Unique identifier
  "TransactionDT": null,                     // ⚠️  Optional: Time offset
  "TransactionAmt": 0.5,                     // ✅ CRITICAL: Transaction amount
  "ProductCD": "C",                          // ✅ Required: Product category
  "card1": 1,                                // ✅ CRITICAL: Card identifier (for velocity)
  "card2": 1,                                // ✅ Required: Card detail
  "card3": 1,                                // ✅ Required: Card detail
  "card4": "visa",                           // ✅ Required: Card type
  "card5": 1,                                // ✅ Required: Card detail
  "card6": "debit",                          // ✅ Required: Card category
  "addr1": 1,                                // ✅ CRITICAL: Address (for velocity)
  "addr2": 1,                                // ✅ Required: Address detail
  "P_emaildomain": "anonymous.com",          // ✅ CRITICAL: Purchaser email (fraud signal!)
  "R_emaildomain": "mailinator.com"          // ✅ CRITICAL: Recipient email (fraud signal!)
}
```

### Here's What Each Field Does

| Field | Purpose | Why It Matters |
|-------|---------|----------------|
| **TransactionAmt** | Amount spent | Detects high/low amounts, amount spikes |
| **card1** | Primary card ID | **Creates user identity for velocity tracking** |
| **card2-6** | Card details | Helps identify unique cards, frequency encoding |
| **addr1** | Primary address | **Part of user identity for velocity tracking** |
| **addr2** | Secondary address | Additional identity signal |
| **P_emaildomain** | Purchaser email | **CRITICAL**: Detects risky domains (anonymous.com, mailinator.com) |
| **R_emaildomain** | Recipient email | **CRITICAL**: Checks if emails match, detects risky domains |
| **ProductCD** | Product category | Different fraud patterns per product type |

---

## 🔍 What Happens If You Only Send 4 Fields?

**Simplified Test (WRONG)**:
```json
{
  "TransactionAmt": 0.5,
  "card1": 1,
  "P_emaildomain": "anonymous.com",
  "R_emaildomain": "mailinator.com"
}
```

**Problems**:
1. ❌ **Missing card2-6** → Frequency encoding fails (features = 0)
2. ❌ **Missing addr1** → User identity incomplete → velocity features wrong
3. ❌ **Missing addr2** → Another missing feature
4. ❌ **Missing ProductCD** → Model gets wrong product category

**Result**: Model makes predictions with **20+ missing features** → unreliable results

---

## ✅ Correct Test Format

**Use this format** (matches your actual API):

```json
{
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
}
```

**Now the model gets**:
- ✅ All card features for frequency encoding
- ✅ Complete user identity (card1 + addr1 + P_emaildomain) for velocity
- ✅ Product category for context
- ✅ Email risk signals

---

## 🧪 How to Run Tests

### Option 1: Use the Shell Script (All Fields Included)

```bash
chmod +x test_transactions.sh
./test_transactions.sh
```

This runs all 6 test cases with **complete field sets**.

### Option 2: Manual cURL (Copy from test_cases.json)

```bash
# Test 1: Low Amount + Risky Email
curl -X POST http://localhost:8000/api/v1/transactions/submit \
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
```

### Option 3: Using Postman

1. Import `test_cases.json`
2. Create POST request to `http://localhost:8000/api/v1/transactions/submit`
3. Set `Content-Type: application/json`
4. Copy entire transaction object from test_cases.json

---

## 📊 What Each Test Case Verifies

| Test | Amount | Email | Expected Result | What It Tests |
|------|--------|-------|-----------------|---------------|
| **Test 1** | $0.50 | Risky | BLOCK/HOLD | Very low amount + risky email |
| **Test 2** | $25,000 | Risky | BLOCK | Very high amount + risky email |
| **Test 3** | $45.99 | Gmail | APPROVE | Normal transaction baseline |
| **Test 4a-d** | $100-$250 | Gmail | APPROVE | Building normal velocity pattern |
| **Test 4e** | $5,000 | Gmail | BLOCK/HOLD | **Velocity spike detection** |
| **Test 5** | $500 | Risky | BLOCK/HOLD | Night transaction + risky email |

---

## 🎯 Understanding Velocity Tests (Test 4)

**Why Test 4 has 5 transactions:**

The velocity feature service tracks **transaction history per user**. A user is identified by:
```
uid = card1 + card2 + addr1 + P_emaildomain
```

**Test 4 Scenario** (all have `card1=99999, addr1=999`):

```
10:00 AM → $100   ✅ APPROVE (first transaction, no history)
10:05 AM → $150   ✅ APPROVE (txn_count_1h=1, normal)
10:10 AM → $200   ✅ APPROVE (txn_count_1h=2, normal growth)
10:15 AM → $250   ✅ APPROVE (txn_count_1h=3, still normal)
10:20 AM → $5000  ❌ BLOCK! (txn_count_1h=4, amt_spike_1h=28x!)
```

**What the 5th transaction sees**:
- `txn_count_1h = 4` (rapid transactions)
- `amt_mean_1h = 175` (average of $100, $150, $200, $250)
- `amt_spike_1h = 5000/175 = 28.5x` 🚨 **HUGE SPIKE!**
- `velocity_risk_score = 0.85` (very high)

**Result**: `decision = "BLOCK"`, `risk_factors = "high_amount,rapid_transactions,amount_spike,high_velocity_risk"`

---

## 🔧 Troubleshooting

### "My tests are still approving everything"

**Check**:
1. ✅ Did you re-train the model? (`python src/dags/ieee_cis_training.py`)
2. ✅ Did you restart inference? (`docker-compose restart inference`)
3. ✅ Are you using the **full field format** from test_transactions.sh?
4. ✅ Check logs: `docker logs inference-service | grep "Pipeline has transform"`

### "Test 4e (velocity) didn't block"

**Check**:
1. ✅ All 5 transactions have **same card1, addr1, P_emaildomain**
2. ✅ Transactions sent **within 1 hour** (preferably < 1 minute apart)
3. ✅ Velocity service initialized: `docker logs inference-service | grep "Velocity service"`

---

## 📋 Quick Test Checklist

Before testing:
- [ ] Model re-trained with fixed pipeline
- [ ] Inference service restarted
- [ ] Using **full field format** (14 fields, not 4)
- [ ] Test 1 (low amount) → expects BLOCK
- [ ] Test 2 (high amount) → expects BLOCK
- [ ] Test 3 (normal) → expects APPROVE
- [ ] Test 4e (velocity spike) → expects BLOCK
- [ ] All tests include card1, card2-6, addr1-2, email domains

---

## 🎉 Expected Results

After running `test_transactions.sh`, you should see:

```bash
Test 1: {"decision": "BLOCK", "risk_level": "HIGH", "risk_factors": "risky_email_domain,very_low_amount"}
Test 2: {"decision": "BLOCK", "risk_level": "HIGH", "risk_factors": "risky_email_domain,high_amount"}
Test 3: {"decision": "APPROVE", "risk_level": "LOW", "risk_factors": "none"}
Test 4a-d: {"decision": "APPROVE", ...}
Test 4e: {"decision": "BLOCK", "risk_level": "HIGH", "risk_factors": "high_amount,rapid_transactions,amount_spike,high_velocity_risk"}
Test 5: {"decision": "BLOCK", "risk_level": "MEDIUM/HIGH", "risk_factors": "risky_email_domain"}
```

---

## 📝 Summary

**Why all fields?**
- ✅ Model was trained on **60+ features** derived from these 14 fields
- ✅ Missing fields → missing features → bad predictions
- ✅ Velocity tracking needs **complete user identity** (card1 + addr1 + email)
- ✅ Email domains are **critical fraud signals**
- ✅ Card details enable frequency encoding

**Always use the full format** from `test_transactions.sh` or `test_cases.json`! 🎯
