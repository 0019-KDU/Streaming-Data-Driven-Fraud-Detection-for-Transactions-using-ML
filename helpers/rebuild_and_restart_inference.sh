#!/bin/bash
#
# Rebuild and restart inference service
# Fixes: AdaptiveThresholdSystem.get_hybrid_threshold() missing method error
#

cd /root/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src

echo "======================================================================"
echo "Rebuilding Inference Container (with updated ieee_cis_training.py)"
echo "======================================================================"

# Stop current inference
docker compose stop inference

# Rebuild inference (no cache to ensure fresh build)
docker compose build --no-cache inference

# Start inference
docker compose up -d inference

echo ""
echo "======================================================================"
echo "Inference Container Rebuilt and Restarted"
echo "======================================================================"
echo ""
echo "Monitor logs with:"
echo "  docker compose logs -f inference"
echo ""
echo "Check status:"
echo "  docker compose ps inference"
echo ""
