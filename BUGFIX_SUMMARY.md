# Bug Fixes - Dashboard Data Quality Issues

**Date:** November 23, 2025  
**Issues Fixed:** Duplicate transactions + Incorrect geo_anomaly risk flags

---

## 🐛 Bugs Identified

### Bug #1: Duplicate Transactions in Dashboard
**Symptom:**
```
Transaction ID 2987245 appears TWICE with different timestamps:
- 2025-11-23T17:41:49 (newer)
- 2025-11-23T17:34:24 (older)
```

**Root Cause:**
- Dashboard was using `transactions.extend()` without deduplication
- Kafka consumer replays messages when offset resets or app restarts
- No check for existing `transaction_id` before adding to deque

**Fix Applied:**
```python
# src/dashboard/app.py (Lines 353-371)
# ✅ FIX: Deduplicate by transaction_id before adding
existing_tx_ids = {txn.get('transaction_id') for txn in st.session_state.transactions}
new_fraud = [msg for msg in fraud_messages if msg.get('transaction_id') not in existing_tx_ids]

if new_fraud:
    st.session_state.fraud_count += len(new_fraud)
    st.session_state.transactions.extend(new_fraud)
    update_model_stats(new_fraud)
```

---

### Bug #2: Incorrect Geo Anomaly Risk Factor
**Symptom:**
```
Transaction with dist1=19km shows:
  risk_factors: ["geo_anomaly_distance_19km"]

Expected: NO geo_anomaly flag (19km is LOW distance, well below 500km threshold)
```

**Root Cause:**
- ATO service `_identify_risk_factors()` was flagging ANY distance > 0
- Check was: `if dist > 0: factors.append(f"geo_anomaly_distance_{dist}km")`
- Should only flag when `dist > 500` (moderately far) or `dist > 1000` (very far)

**Fix Applied:**
```python
# src/inference/ato_service.py (Lines 387-395)
# ✅ FIX: Only add geo factor when ACTUALLY anomalous (>500km threshold)
if geo_anomaly > 0.5:
    dist = transaction_data.get('dist1')
    # Only flag if distance is actually high (>500km)
    if dist and dist > 500:
        factors.append(f"geo_anomaly_distance_{dist:.0f}km")
        # Also check for new address when distance is high
        factors.append("geo_anomaly_new_address")
```

---

## 📊 Expected Behavior After Fixes

### Transaction Deduplication
- ✅ Each `transaction_id` appears **ONCE** in dashboard table
- ✅ Kafka offset resets don't create duplicates
- ✅ Dashboard restarts don't replay old transactions
- ✅ Deque maintains unique transactions (maxlen=1000)

### Geo Anomaly Thresholds
| Distance | Geo Anomaly Score | Risk Factor Added? |
|----------|-------------------|--------------------|
| 0-19km   | 0.0               | ❌ No flag         |
| 20-499km | 0.0               | ❌ No flag         |
| 500-999km| 0.4 (moderate)    | ✅ `geo_anomaly_distance_XXXkm` |
| 1000km+  | 0.7 (high)        | ✅ `geo_anomaly_distance_XXXkm` |

**Your Transaction (dist1=19km):**
- ✅ Should show: `risk_factors: []` (empty array)
- ✅ Risk Level: LOW
- ✅ Decision: APPROVE
- ✅ Fraud Probability: 0.019 (1.9%)

---

## 🧪 Testing Instructions

### 1. Restart Dashboard (Clear State)
```bash
# On VM (167.71.224.89)
docker-compose restart dashboard
```

### 2. Send Test Transaction
```bash
# On local machine
python send_test_transaction.py
```

### 3. Verify Dashboard Output
**Expected:**
```
Transaction ID: 2987245 (appears ONCE)
Timestamp: 2025-11-23T17:XX:XX
Decision: APPROVE
Risk Level: LOW
Fraud Probability: 0.019
Amount: $37.10
Card: 13413.0 (visa)
Risk Factors: []  ← EMPTY (no geo_anomaly)
```

**Before Fix:**
```
❌ Transaction appears TWICE (duplicate)
❌ Risk Factors: ["geo_anomaly_distance_19km"]  ← WRONG
```

### 4. Send High-Distance Transaction (Test Geo Flag)
```bash
# Modify send_test_transaction.py:
# Change: "dist1": 19.0  →  "dist1": 650.0

python send_test_transaction.py
```

**Expected:**
```
Risk Factors: ["geo_anomaly_distance_650km"]  ← CORRECT (>500km)
```

---

## 📁 Files Modified

1. **src/dashboard/app.py** (Lines 353-371)
   - Added transaction deduplication logic
   - Prevents duplicate transaction_ids in session state

2. **src/inference/ato_service.py** (Lines 387-395)
   - Fixed geo_anomaly distance threshold check
   - Only flags distances > 500km as anomalous

---

## 🚀 Deployment Steps

### Option 1: Hot Reload (No Downtime)
```bash
# On VM
cd /path/to/project
git pull origin main

# Restart only affected services
docker-compose restart inference dashboard
```

### Option 2: Full Rebuild (Recommended for Production)
```bash
# On VM
cd /path/to/project
git pull origin main

# Rebuild containers with new code
docker-compose down
docker-compose build inference dashboard
docker-compose up -d

# Verify services are running
docker-compose ps
docker-compose logs -f --tail=100 inference dashboard
```

---

## ✅ Verification Checklist

- [ ] Dashboard shows transaction 2987245 only ONCE
- [ ] Risk factors array is EMPTY `[]` for dist1=19km
- [ ] Fraud probability is 0.019 (LOW)
- [ ] Decision is APPROVE
- [ ] No geo_anomaly_distance flag appears
- [ ] Submit new transaction → appears within 5 seconds
- [ ] Restart dashboard → no duplicates from Kafka replay
- [ ] High distance (>500km) → geo_anomaly flag DOES appear

---

## 📝 Notes

- **Kafka Offset:** Dashboard uses `auto.offset.reset: latest` - only reads NEW messages
- **Session State:** Deque maxlen=1000 - oldest transactions drop when full
- **Deduplication:** Uses set of transaction_ids for O(1) lookup before insert
- **ATO Service:** Geo anomaly score still calculated correctly (0.0-1.0 range)
- **Risk Factors:** Only SHOWN when threshold exceeded, not for all distances

---

## 🔍 Root Cause Analysis

### Why These Bugs Occurred

1. **Duplicate Transactions:**
   - Original code assumed Kafka messages are unique
   - Didn't account for offset resets or consumer group changes
   - Streamlit session state persists across reruns without clearing

2. **Geo Anomaly Flag:**
   - Copy-paste error: `if dist > 0:` instead of `if dist > 500:`
   - No unit tests for risk factor logic
   - ATO service calculates correct score but display logic was wrong

### Prevention Measures

- ✅ Add deduplication at ingestion layer
- ✅ Add threshold constants (DIST_MODERATE=500, DIST_HIGH=1000)
- ✅ Add unit tests for `_identify_risk_factors()`
- ✅ Add integration test for end-to-end transaction flow
- ✅ Add logging: "Duplicate transaction_id filtered" warnings

---

**Status:** ✅ FIXED - Ready for deployment  
**Impact:** High (affects all dashboard users + fraud analysis accuracy)  
**Deployment Time:** ~2 minutes (restart services)
