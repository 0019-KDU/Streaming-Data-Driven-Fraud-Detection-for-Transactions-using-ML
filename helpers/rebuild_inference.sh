#!/bin/bash
# Script to rebuild and restart the inference service with new fixes

echo "========================================"
echo "Rebuilding Inference Service"
echo "========================================"
echo ""

cd ~/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src

# Stop current inference service
echo "1. Stopping current inference service..."
docker compose stop inference
docker compose rm -f inference

# Rebuild inference image
echo ""
echo "2. Rebuilding inference Docker image..."
docker compose build inference

# Start inference service
echo ""
echo "3. Starting inference service..."
docker compose up -d inference

# Wait for service to start
echo ""
echo "4. Waiting for service to start (10 seconds)..."
sleep 10

# Check logs
echo ""
echo "5. Checking inference logs..."
docker compose logs inference --tail=30

echo ""
echo "========================================"
echo "Done! Check logs above for any errors."
echo "========================================"
echo ""
echo "To watch live logs:"
echo "  docker compose logs -f inference"
echo ""
echo "To check if velocity service loaded:"
echo "  docker compose logs inference | grep 'Velocity service initialized'"
