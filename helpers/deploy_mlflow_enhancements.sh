#!/bin/bash

# MLflow Visualization Enhancement Deployment Script
# This script deploys the enhanced MLflow tracking with visualizations

set -e  # Exit on error

echo "=========================================="
echo "MLflow Visualization Enhancement Deployment"
echo "=========================================="
echo ""

# Step 1: Pull latest code
echo "Step 1: Pulling latest code from repository..."
git pull origin main
echo "✓ Code updated"
echo ""

# Step 2: Check if containers are running
echo "Step 2: Checking Docker containers..."
cd src
if ! docker compose ps | grep -q "Up"; then
    echo "⚠ Containers not running. Starting..."
    docker compose up -d
    echo "✓ Containers started"
else
    echo "✓ Containers already running"
fi
echo ""

# Step 3: Rebuild Airflow containers (to install matplotlib & seaborn)
echo "Step 3: Rebuilding Airflow containers with new dependencies..."
echo "This will install matplotlib and seaborn..."
docker compose build airflow-scheduler airflow-worker
echo "✓ Containers rebuilt"
echo ""

# Step 4: Restart Airflow services
echo "Step 4: Restarting Airflow services..."
docker compose restart airflow-scheduler airflow-worker
echo "✓ Services restarted"
echo ""

# Step 5: Wait for services to be ready
echo "Step 5: Waiting for services to initialize (30 seconds)..."
sleep 30
echo "✓ Services should be ready"
echo ""

# Step 6: Verify dependencies
echo "Step 6: Verifying visualization dependencies..."
echo "Checking matplotlib..."
if docker compose exec -T airflow-worker pip list | grep -q matplotlib; then
    echo "✓ matplotlib installed"
else
    echo "⚠ matplotlib NOT found - may need manual installation"
fi

echo "Checking seaborn..."
if docker compose exec -T airflow-worker pip list | grep -q seaborn; then
    echo "✓ seaborn installed"
else
    echo "⚠ seaborn NOT found - may need manual installation"
fi
echo ""

# Step 7: Check MLflow
echo "Step 7: Checking MLflow service..."
MLFLOW_PORT=$(docker compose port mlflow 5000 2>/dev/null | cut -d: -f2 || echo "5000")
if [ -n "$MLFLOW_PORT" ]; then
    echo "✓ MLflow running on port $MLFLOW_PORT"
    echo "  Access at: http://localhost:$MLFLOW_PORT"
else
    echo "⚠ MLflow port not detected"
fi
echo ""

# Step 8: Show DAG status
echo "Step 8: Checking Airflow DAG status..."
docker compose exec -T airflow-scheduler airflow dags list 2>/dev/null | grep ieee_cis || echo "DAG list not available"
echo ""

echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "1. Access Airflow UI: http://localhost:8080"
echo "   Username: airflow"
echo "   Password: airflow"
echo ""
echo "2. Enable and trigger DAG: ieee_cis_training_dag"
echo ""
echo "3. Monitor training logs:"
echo "   docker compose logs -f airflow-worker"
echo ""
echo "4. Once complete, view MLflow UI: http://localhost:$MLFLOW_PORT"
echo "   Check for:"
echo "   - 22+ metrics"
echo "   - 6 visualization plots"
echo "   - Feature importance CSV"
echo ""
echo "5. Download plots for thesis from MLflow UI Artifacts tab"
echo ""
echo "=========================================="
echo ""

# Optional: Trigger training automatically
read -p "Do you want to trigger training now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Triggering ieee_cis_training_dag..."
    docker compose exec -T airflow-scheduler airflow dags trigger ieee_cis_training_dag
    echo "✓ Training triggered!"
    echo ""
    echo "Monitor progress:"
    echo "  docker compose logs -f airflow-worker"
fi

echo ""
echo "Deployment script finished!"
