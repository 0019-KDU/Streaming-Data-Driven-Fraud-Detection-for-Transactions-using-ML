#!/bin/bash
# Quick diagnostic script to check what's actually in the inference logs

echo "=========================================="
echo "CHECKING INFERENCE LOGS (NO GREP FILTER)"
echo "=========================================="

echo ""
echo "Last 50 lines of inference logs (unfiltered):"
docker logs fraud-inference-spark --tail 50

echo ""
echo "=========================================="
echo "CHECKING FOR PYTHON ERRORS"
echo "=========================================="

docker logs fraud-inference-spark 2>&1 | grep -i "error\|exception\|traceback" | tail -20

echo ""
echo "=========================================="
echo "CHECKING KAFKA OUTPUT"
echo "=========================================="

echo "Last message in legit_predictions topic:"
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic legit_predictions \
  --max-messages 1 \
  --from-beginning 2>/dev/null | tail -1 | jq .

echo ""
echo "Last message in fraud_predictions topic:"
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic fraud_predictions \
  --max-messages 1 \
  --from-beginning 2>/dev/null | tail -1 | jq .
