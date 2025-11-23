#!/bin/bash
# Debug Inference Service Output
# Run this on the VM to see what's actually being predicted

echo "=========================================="
echo "CHECKING INFERENCE SERVICE LOGS"
echo "=========================================="

# Get the last 200 lines with actual predictions (not just Spark progress)
echo ""
echo "1. Searching for XGBoost predictions..."
docker logs fraud-inference-spark --tail 500 2>&1 | grep -i "xgboost\|prediction\|fraud_prob\|decision\|APPROVE\|BLOCK\|error" | tail -50

echo ""
echo "=========================================="
echo "2. Checking Kafka output topics..."
echo "=========================================="

# Check fraud_predictions topic
echo ""
echo "Fraud predictions topic (last 5 messages):"
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic fraud_predictions \
  --from-beginning \
  --max-messages 5 2>/dev/null | tail -10

echo ""
echo "Legit predictions topic (last 5 messages):"
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic legit_predictions \
  --from-beginning \
  --max-messages 5 2>/dev/null | tail -10

echo ""
echo "=========================================="
echo "3. Checking transaction input (last message):"
echo "=========================================="

docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic transactions \
  --from-beginning \
  --max-messages 1 2>/dev/null | tail -5

echo ""
echo "=========================================="
echo "4. Inference service configuration:"
echo "=========================================="

docker exec fraud-inference-spark env | grep -E "KAFKA|MODEL|THRESHOLD"

echo ""
echo "=========================================="
echo "✅ DEBUG COMPLETE"
echo "=========================================="
