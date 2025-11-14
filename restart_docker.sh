#!/bin/bash
# Safe Docker Restart Script for Digital Ocean
# This script restarts Docker containers WITHOUT losing your code changes

set -e  # Exit on error

echo "=========================================="
echo "🔄 Safe Docker Restart Script"
echo "=========================================="
echo ""

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker-compose not found"
    echo "Please install docker-compose first"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found"
    echo "Please run this script from the /src directory"
    echo "Example: cd /path/to/project/src && ./restart_docker.sh"
    exit 1
fi

echo "✅ Found docker-compose.yml"
echo ""

# Show current container status
echo "📊 Current container status:"
docker-compose ps
echo ""

# Ask user what to do
echo "Choose restart option:"
echo "1) Quick restart (30 seconds) - Restarts only Airflow services"
echo "2) Full restart (2-3 minutes) - Stops and starts all services"
echo "3) Cancel"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "🔄 Restarting Airflow services..."
        echo "   - airflow-scheduler"
        echo "   - airflow-worker"
        echo ""

        docker-compose restart airflow-scheduler airflow-worker

        echo ""
        echo "⏳ Waiting for services to start (10 seconds)..."
        sleep 10

        echo ""
        echo "✅ Quick restart complete!"
        echo ""
        echo "📊 New container status:"
        docker-compose ps | grep -E "NAME|airflow-scheduler|airflow-worker"
        ;;

    2)
        echo ""
        echo "🛑 Stopping all containers..."
        docker-compose down

        echo ""
        echo "🚀 Starting all containers..."
        docker-compose up -d

        echo ""
        echo "⏳ Waiting for services to initialize (30 seconds)..."
        sleep 30

        echo ""
        echo "✅ Full restart complete!"
        echo ""
        echo "📊 New container status:"
        docker-compose ps
        ;;

    3)
        echo ""
        echo "❌ Restart cancelled"
        exit 0
        ;;

    *)
        echo ""
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Docker Restart Complete!"
echo "=========================================="
echo ""
echo "📝 Next steps:"
echo "1. Check Airflow UI: http://YOUR_IP:8080"
echo "2. Check logs: docker-compose logs -f airflow-scheduler"
echo "3. Trigger training DAG"
echo ""
echo "🔍 Verify changes applied:"
echo "docker-compose logs airflow-scheduler | grep 'Forward Feature Selection'"
echo ""
