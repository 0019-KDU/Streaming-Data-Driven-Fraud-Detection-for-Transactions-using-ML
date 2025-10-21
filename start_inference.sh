#!/bin/bash
# Script to start the inference service

echo "🔧 Starting Fraud Detection Inference Service..."
echo ""

cd src

# Stop and remove existing inference container if it exists
echo "📦 Removing old inference container (if exists)..."
docker compose rm -f inference

# Rebuild inference image with latest code
echo "🔨 Building inference image..."
docker compose build inference

# Start inference service
echo "🚀 Starting inference service..."
docker compose up -d inference

echo ""
echo "✅ Inference service started!"
echo ""
echo "📊 Check status:"
echo "   docker compose ps | grep inference"
echo ""
echo "📋 View logs:"
echo "   docker compose logs -f inference"
echo ""
echo "🔍 Monitor processing:"
echo "   docker compose logs inference | tail -50"
