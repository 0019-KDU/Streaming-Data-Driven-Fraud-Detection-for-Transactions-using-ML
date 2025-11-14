# DigitalOcean Droplet Deployment Guide

## 🎯 Recommended Droplet Specs

### **Minimum Requirements:**
- **Droplet Type**: CPU-Optimized
- **Plan**: 8 GB RAM / 4 vCPUs / 25 GB SSD ($48/month)
- **OS**: Ubuntu 22.04 LTS
- **Region**: Closest to your users

### **Recommended (for FFS + Production):**
- **Droplet Type**: CPU-Optimized
- **Plan**: 16 GB RAM / 8 vCPUs / 50 GB SSD ($96/month)
- **OS**: Ubuntu 22.04 LTS
- **Reason**: Forward Feature Selection + RandomizedSearchCV is CPU-intensive

### **Production (Heavy Load):**
- **Droplet Type**: CPU-Optimized
- **Plan**: 32 GB RAM / 16 vCPUs / 100 GB SSD ($192/month)
- **OS**: Ubuntu 22.04 LTS
- **Reason**: Multiple workers, high throughput, concurrent training

---

## 📦 Initial Droplet Setup

### 1. Create Droplet
```bash
# SSH into your droplet
ssh root@YOUR_DROPLET_IP
```

### 2. Update System
```bash
apt update && apt upgrade -y
apt install -y git curl wget htop
```

### 3. Install Docker
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Start Docker
systemctl start docker
systemctl enable docker

# Verify
docker --version
```

### 4. Install Docker Compose
```bash
# Install Docker Compose v2
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Verify
docker compose version
```

### 5. Configure Firewall
```bash
# Allow SSH, HTTP, HTTPS
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 8080/tcp  # Airflow
ufw allow 8501/tcp  # Dashboard
ufw allow 8000/tcp  # Producer API
ufw allow 5500/tcp  # MLflow
ufw enable
```

---

## 🚀 Deploy Application

### 1. Clone Repository
```bash
cd /opt
git clone https://github.com/0019-KDU/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML.git
cd Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src
```

### 2. Configure Environment
```bash
# Create .env file
cat > .env << 'EOF'
# Kafka Configuration (Confluent Cloud)
KAFKA_BOOTSTRAP_SERVERS=pkc-921jm.us-east-2.aws.confluent.cloud:9092
KAFKA_USERNAME=TUISIFY5HCFLGXIH
KAFKA_PASSWORD=HIhrR1hP0Oj64llWYN8E4U3gnsJ83b64OGcrFDYvnkTppiMo1UkMwUUdfSFr6PLl
KAFKA_TOPIC=transactions
KAFKA_OUTPUT_TOPIC=fraud_predictions
KAFKA_LEGIT_TOPIC=legit_predictions
KAFKA_REPLY_TOPIC=transaction_replies
KAFKA_SECURITY_PROTOCOL=SASL_SSL

# MinIO Configuration
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MINIO_USERNAME=minioadmin
MINIO_PASSWORD=minioadmin

# PostgreSQL Configuration
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Airflow Configuration
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin123

# Email Configuration (MailDev for testing)
SMTP_HOST=maildev
SMTP_PORT=1025
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM=fraud.alerts@bank.com
EOF

chmod 600 .env
```

### 3. Update MLflow Host in config.yaml
```bash
# Get your droplet IP
DROPLET_IP=$(curl -s ifconfig.me)

# Update MLflow allowed hosts
sed -i "s/64.23.228.115/${DROPLET_IP}/g" docker-compose.yml
```

### 4. Build and Start Services
```bash
# Build all images (takes 10-15 minutes)
docker compose build

# Start all services
docker compose up -d

# Check status
docker compose ps
```

### 5. Verify Services
```bash
# Check logs
docker compose logs -f --tail=100

# Individual service logs
docker logs airflow-scheduler -f
docker logs fraud-inference -f
docker logs producer-api -f
```

---

## 🔍 Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **Airflow** | http://YOUR_IP:8080 | admin / admin123 |
| **MLflow** | http://YOUR_IP:5500 | - |
| **Dashboard** | http://YOUR_IP:8501 | - |
| **Producer API** | http://YOUR_IP:8000/docs | - |
| **MailDev** | http://YOUR_IP:1080 | - |

---

## 📊 Upload IEEE-CIS Dataset

### Option 1: SCP from Local Machine
```bash
# From your local machine
scp -r /path/to/ieee_cis root@YOUR_DROPLET_IP:/opt/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src/data/
```

### Option 2: Download Directly on Droplet
```bash
# SSH into droplet
cd /opt/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src/data
mkdir -p ieee_cis

# Download from Kaggle (you need Kaggle API token)
pip install kaggle
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
kaggle competitions download -c ieee-fraud-detection -p ieee_cis/
unzip ieee_cis/ieee-fraud-detection.zip -d ieee_cis/
```

### Option 3: Upload via Docker Volume
```bash
# Copy to running Airflow container
docker cp /local/path/to/ieee_cis airflow-scheduler:/app/data/
```

---

## 🏋️ Train Model

### 1. Fix MLflow Bucket (IMPORTANT!)
```bash
docker exec minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker exec minio mc mb local/mlflow --ignore-existing
```

### 2. Trigger Training
```bash
# Via Airflow CLI
docker exec airflow-scheduler airflow dags trigger ieee_cis_training_dag

# Via Airflow UI
# Go to http://YOUR_IP:8080
# Click on "ieee_cis_training_dag"
# Click "Trigger DAG"
```

### 3. Monitor Training
```bash
# Watch logs
docker logs -f airflow-scheduler

# Check MLflow
# http://YOUR_IP:5500
```

### Expected Timeline:
- **Phase 1 (Basic)**: ~30 min, AUC: 0.8956 (89.56%)
- **Phase 1 + FFS**: ~60 min, AUC: 0.9556 (95.56%)
- **Phase 1 + FFS + RandomSearch**: ~120 min, AUC: 0.9656+ (96.56%+)

---

## 🧪 Test Fraud Detection

### 1. Submit Test Transaction
```bash
curl -X POST http://YOUR_DROPLET_IP:8000/submit \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "test001",
    "TransactionDT": 12345678,
    "TransactionAmt": 1500.50,
    "ProductCD": "W",
    "card1": 13926,
    "card2": 111,
    "card3": 150,
    "card4": "discover",
    "card5": 226,
    "card6": "debit",
    "addr1": 315.0,
    "addr2": 87.0,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }'
```

### 2. View Dashboard
```bash
# Open in browser
http://YOUR_DROPLET_IP:8501
```

---

## 🔧 Maintenance Commands

### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker logs -f fraud-inference
docker logs -f airflow-scheduler
docker logs -f producer-api
```

### Restart Services
```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart fraud-inference
docker compose restart airflow-scheduler
```

### Update Code
```bash
cd /opt/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML
git pull origin main
cd src
docker compose build
docker compose up -d
```

### Backup Models
```bash
# Backup trained models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
scp models_backup_*.tar.gz user@backup-server:/backups/
```

### Check Resource Usage
```bash
# CPU, Memory, Disk
htop

# Docker stats
docker stats

# Disk usage
df -h
docker system df
```

### Clean Up
```bash
# Remove unused images
docker image prune -a

# Remove old logs
docker compose logs --no-color > logs_backup.txt
docker compose down
docker compose up -d

# Clean Airflow logs
docker exec airflow-scheduler rm -rf /opt/airflow/logs/*
```

---

## 🚨 Troubleshooting

### Out of Memory
```bash
# Check memory
free -h

# Increase swap
fallocate -l 8G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

### Disk Full
```bash
# Check disk usage
df -h

# Clean Docker
docker system prune -a --volumes

# Clean logs
rm -rf src/logs/*
```

### Container Won't Start
```bash
# Check logs
docker compose logs <service_name>

# Rebuild
docker compose down
docker compose build --no-cache <service_name>
docker compose up -d
```

### Training Fails
```bash
# Check Airflow logs
docker logs airflow-scheduler | grep ERROR

# Check MLflow bucket
docker exec minio mc ls local/

# Restart training
docker exec airflow-scheduler airflow dags trigger ieee_cis_training_dag
```

---

## 📈 Performance Optimization

### For 8 GB Droplet (Basic)
```yaml
# config.yaml
training:
  use_forward_feature_selection: false  # Disable FFS (too slow)
  use_smote: true
  smote_sampling_strategy: 0.5          # Reduce SMOTE ratio
```

### For 16 GB Droplet (Recommended)
```yaml
# config.yaml
training:
  use_forward_feature_selection: true   # Enable FFS
  ffs_max_features: 50
  use_randomized_search: false          # Still too slow
```

### For 32 GB Droplet (Production)
```yaml
# config.yaml
training:
  use_forward_feature_selection: true   # Enable FFS
  ffs_max_features: 50
  use_randomized_search: true           # Enable full tuning
  random_search_iterations: 100
```

---

## 🔒 Security Hardening

### 1. Change Default Passwords
```bash
# Update .env
nano .env
# Change: _AIRFLOW_WWW_USER_PASSWORD, POSTGRES_PASSWORD, MINIO_PASSWORD
```

### 2. Enable HTTPS (Let's Encrypt)
```bash
# Install Nginx
apt install -y nginx certbot python3-certbot-nginx

# Get certificate
certbot --nginx -d your-domain.com

# Update Nginx config for services
nano /etc/nginx/sites-available/default
```

### 3. Restrict Access
```bash
# Only allow your IP
ufw delete allow 8080/tcp
ufw allow from YOUR_IP to any port 8080 proto tcp
```

---

## 📊 Monitoring

### Setup Monitoring
```bash
# Install Docker stats exporter
docker run -d \
  --name cadvisor \
  --volume=/:/rootfs:ro \
  --volume=/var/run:/var/run:ro \
  --volume=/sys:/sys:ro \
  --volume=/var/lib/docker/:/var/lib/docker:ro \
  --publish=8088:8080 \
  google/cadvisor:latest
```

### Check Metrics
- CPU/Memory: http://YOUR_IP:8088
- Airflow: http://YOUR_IP:8080
- MLflow: http://YOUR_IP:5500

---

## 🎯 Next Steps

1. ✅ Deploy to DigitalOcean Droplet
2. ✅ Upload IEEE-CIS dataset
3. ✅ Train model (60-120 min)
4. ✅ Test fraud detection
5. ✅ Monitor dashboard
6. ✅ Enable HTTPS
7. ✅ Setup backups
8. ✅ Production deployment

**Your fraud detection system is now running in the cloud!** 🚀
