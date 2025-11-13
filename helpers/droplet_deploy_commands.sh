#!/bin/bash
# Deploy to DigitalOcean Droplet - Run these commands in order

# 1. Update system and install Docker
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# 2. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 3. Clone repo
cd ~
git clone https://github.com/0019-KDU/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML.git
cd Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML

# 4. Create .env file (edit with your credentials)
nano src/.env

# 5. Build and start all services
cd src
sudo docker-compose build
sudo docker-compose up -d

# 6. Check service status
sudo docker-compose ps

# 7. View logs
sudo docker-compose logs -f inference
sudo docker-compose logs -f mlflow-server

# 8. Create MinIO bucket
sudo docker-compose exec mc mc alias set minio http://minio:9000 minio minio123
sudo docker-compose exec mc mc mb minio/mlflow

# 9. Initialize Airflow
sudo docker-compose exec airflow-webserver airflow db init
sudo docker-compose exec airflow-webserver airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com

# 10. Restart services
sudo docker-compose restart

# 11. Trigger training DAG
sudo docker-compose exec airflow-webserver airflow dags unpause ieee_cis_training_dag
sudo docker-compose exec airflow-webserver airflow dags trigger ieee_cis_training_dag

# Service URLs (replace <DROPLET_IP> with your droplet IP)
# MLflow UI: http://<DROPLET_IP>:5500
# Airflow UI: http://<DROPLET_IP>:8080
# MinIO UI: http://<DROPLET_IP>:9001
# Dashboard: http://<DROPLET_IP>:8501
# MailDev: http://<DROPLET_IP>:1080
