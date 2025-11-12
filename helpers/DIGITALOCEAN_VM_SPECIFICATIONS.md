# DigitalOcean VM Resource Specifications

## System Requirements for Production Deployment

### 📊 Recommended VM Configuration

Based on your full ML training system with VAE ensemble, velocity features, and real-time inference:

---

## 🎯 RECOMMENDED: Production Droplet

### **CPU Optimized Droplet - 8 vCPUs**

**Specifications:**
```
Droplet Type: CPU-Optimized
vCPUs: 8 cores
RAM: 16 GB
Storage: 100 GB SSD
Bandwidth: 6 TB transfer
Price: ~$96/month
```

**Why This Configuration:**
- ✅ **8 vCPUs**: Handles parallel processing for:
  - Airflow workers (2 replicas)
  - Spark streaming inference
  - VAE ensemble training (3 models)
  - XGBoost/LightGBM/CatBoost training

- ✅ **16 GB RAM**: Required for:
  - IEEE-CIS dataset (~590k rows) in memory
  - Spark executor memory
  - Multiple Docker containers (12 services)
  - Gradient boosting model training
  - VAE training with TensorFlow

- ✅ **100 GB SSD**: Sufficient for:
  - Docker images (~20 GB)
  - IEEE-CIS dataset (~1-2 GB)
  - Model artifacts (~500 MB)
  - Logs and checkpoints (~10 GB)
  - MLflow artifacts (~5 GB)
  - Operating system (~10 GB)
  - Buffer (~50 GB)

---

## 🔧 Alternative Configurations

### Option 1: MINIMUM (Development/Testing)
```
Droplet Type: General Purpose
vCPUs: 4 cores
RAM: 8 GB
Storage: 50 GB SSD
Price: ~$48/month

⚠️ Limitations:
- Slower training (45-60 minutes)
- May need to reduce dataset size
- Fewer Airflow workers (1 replica)
- No GPU acceleration
```

### Option 2: OPTIMAL (Production with GPU - if available)
```
Droplet Type: GPU Droplet
GPU: 1x NVIDIA GPU
vCPUs: 8 cores
RAM: 30 GB
Storage: 100 GB SSD
Price: ~$300-500/month

✅ Benefits:
- 5-10x faster VAE training
- GPU-accelerated XGBoost/LightGBM/CatBoost
- Training time: 5-10 minutes (vs 20-45 minutes)
```

### Option 3: HIGH AVAILABILITY (Enterprise)
```
Droplet Type: CPU-Optimized
vCPUs: 16 cores
RAM: 32 GB
Storage: 200 GB SSD
Price: ~$192/month

✅ Benefits:
- Handle larger datasets
- More Airflow workers (4 replicas)
- Better Spark performance
- Production-grade reliability
```

---

## 💾 Storage Breakdown

### Disk Space Requirements (100 GB SSD):

```
Operating System (Ubuntu 22.04)      10 GB
Docker Images                        20 GB
├─ airflow-training                   5 GB
├─ inference                          4 GB
├─ mlflow-server                      2 GB
├─ postgres:13                        1 GB
├─ minio                              1 GB
├─ dashboard                          2 GB
├─ producer-api                       1 GB
├─ notification                       1 GB
└─ other services                     3 GB

IEEE-CIS Dataset                      2 GB
├─ train_transaction.csv            1.5 GB
└─ train_identity.csv               0.5 GB

Model Artifacts                      0.5 GB
├─ fraud_detection_model.pkl        0.3 GB
│  └─ (includes VAE models)
├─ feature_pipeline.pkl             0.1 GB
└─ MLflow artifacts                 0.1 GB

Logs & Checkpoints                   10 GB
├─ Airflow logs                      5 GB
├─ Spark checkpoints                 3 GB
└─ Docker logs                       2 GB

MLflow Experiments                    5 GB
Working Space / Buffer               50 GB
──────────────────────────────────────────
TOTAL                               ~100 GB
```

---

## 🧠 Memory (RAM) Breakdown

### RAM Requirements (16 GB):

```
Docker Services (Base)               6 GB
├─ airflow-webserver                1.0 GB
├─ airflow-scheduler                0.8 GB
├─ airflow-worker (2x)              2.0 GB
├─ postgres                         0.5 GB
├─ redis                            0.2 GB
├─ mlflow-server                    0.5 GB
├─ minio                            0.5 GB
├─ producer-api                     0.3 GB
├─ dashboard                        0.4 GB
└─ notification                     0.3 GB

Training Process (Peak)              8 GB
├─ IEEE-CIS data loading            2.0 GB
├─ Feature engineering              1.5 GB
├─ VAE training (3 models)          2.0 GB
├─ Gradient boosting                2.0 GB
└─ Model calibration                0.5 GB

Inference Process                    1.5 GB
├─ Spark executor                   1.0 GB
└─ Model in memory                  0.5 GB

Operating System                     0.5 GB
──────────────────────────────────────────
TOTAL PEAK                          ~16 GB
```

---

## ⚙️ CPU Breakdown

### vCPU Allocation (8 cores):

```
Service                    Cores  Usage Pattern
─────────────────────────────────────────────────
airflow-scheduler            1    Continuous (DAG scheduling)
airflow-worker (2x)          4    Peak during training
inference (Spark)            2    Continuous (streaming)
producer-api                 0.5  Low (event-driven)
dashboard                    0.5  Low (UI refresh)
notification                 0.2  Low (email sending)
postgres/redis/minio         0.3  Low (I/O bound)
mlflow-server               0.5  Low (API calls)
─────────────────────────────────────────────────
TOTAL                        8    (with buffer)

Peak Training Load:
- VAE training: 3-4 cores (parallel ensemble)
- XGBoost: 4-6 cores (multi-threaded)
- Velocity features: 2-3 cores (groupby operations)
```

---

## 🌐 Network Requirements

### Bandwidth & Transfer:

```
Inbound Traffic:
├─ IEEE-CIS dataset download              2 GB (one-time)
├─ Docker images pull                    20 GB (one-time)
├─ Transaction ingestion via API          1 GB/day
└─ Confluent Cloud Kafka consume          2 GB/day

Outbound Traffic:
├─ Kafka produce (predictions)            2 GB/day
├─ Email notifications                  0.1 GB/day
├─ MLflow artifact sync                 0.5 GB/day
└─ Dashboard access                     0.5 GB/day

Monthly Transfer: ~150 GB/month
DigitalOcean Allowance: 6 TB/month ✅ MORE THAN ENOUGH
```

### Ports to Open (Firewall):

```
INBOUND:
22    - SSH (restrict to your IP)
80    - HTTP (optional - redirect to HTTPS)
443   - HTTPS (for secure access)
8080  - Airflow UI (restrict to your IP or use SSH tunnel)
5500  - MLflow UI (restrict to your IP or use SSH tunnel)
8501  - Dashboard (public if needed, or restrict)
8000  - Producer API (public or restrict to app servers)
1080  - MailDev UI (restrict to your IP)

OUTBOUND:
9092  - Confluent Cloud Kafka (pkc-921jm.us-east-2.aws.confluent.cloud)
443   - HTTPS for package downloads
```

---

## 🐳 Docker Resource Limits

### Recommended docker-compose Resource Constraints:

```yaml
# Add to each service in docker-compose.yml

airflow-worker:
  deploy:
    resources:
      limits:
        cpus: '2.0'
        memory: 4G
      reservations:
        cpus: '1.0'
        memory: 2G

inference:
  deploy:
    resources:
      limits:
        cpus: '2.0'
        memory: 4G
      reservations:
        cpus: '1.0'
        memory: 2G

postgres:
  deploy:
    resources:
      limits:
        cpus: '1.0'
        memory: 1G
      reservations:
        cpus: '0.5'
        memory: 512M
```

---

## 📦 DigitalOcean Droplet Selection Guide

### Step-by-Step Provisioning:

```bash
# 1. Choose Region (closest to Confluent Cloud)
Region: New York 3 (nyc3)
   └─ Closest to Confluent Cloud us-east-2

# 2. Choose Image
OS: Ubuntu 22.04 LTS x64

# 3. Choose Droplet Type
Type: CPU-Optimized
Size: c-8 (8 vCPUs, 16 GB RAM, 100 GB SSD)

# 4. Add SSH Key
✅ Upload your SSH public key

# 5. Additional Options
☑ Monitoring (free)
☑ IPv6
☐ Backups (+$19.20/month) - Optional but recommended
☐ Droplet Agent (for metrics)

# 6. Finalize
Hostname: fraud-detection-ml
Tags: production, ml, fraud-detection
```

---

## 💰 Cost Breakdown (Monthly)

### Recommended Configuration:

```
Droplet (c-8)                        $96.00/month
Backups (optional)                   $19.20/month
──────────────────────────────────────────────────
Subtotal                            $115.20/month

External Services:
├─ Confluent Cloud Kafka             $0-10/month (basic tier)
└─ Domain name (optional)             $12/year

──────────────────────────────────────────────────
TOTAL ESTIMATED                     ~$125/month
```

### Cost Optimization Tips:

1. **Use Snapshots Instead of Backups:**
   - Take manual snapshots before major changes
   - Cost: $0.06/GB/month (~$6/month for 100 GB snapshot)
   - Savings: $13/month

2. **Start with Smaller Droplet for Testing:**
   - Use 4 vCPU / 8 GB RAM initially ($48/month)
   - Resize to 8 vCPU / 16 GB RAM when going to production
   - DigitalOcean allows easy resizing (requires reboot)

3. **Use Reserved Instances (if available):**
   - Commit to 1 year: 10% discount
   - Commit to 2 years: 15% discount

---

## 🚀 Initial Setup Script for DigitalOcean

### After creating the Droplet, run these commands:

```bash
#!/bin/bash
# fraud-detection-setup.sh

# 1. Update system
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 3. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 4. Install additional tools
sudo apt-get install -y git curl htop ncdu

# 5. Create project directory
mkdir -p ~/fraud-detection
cd ~/fraud-detection

# 6. Clone your repository
git clone https://github.com/YOUR_USERNAME/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML.git
cd Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML

# 7. Create data directory
mkdir -p src/data/ieee_cis

# 8. Upload IEEE-CIS dataset (use scp or wget)
# scp -r local_data/* root@YOUR_DROPLET_IP:~/fraud-detection/.../src/data/ieee_cis/

# 9. Configure environment
cp .env.example .env
nano .env  # Edit with your actual credentials

# 10. Start services
cd src
docker-compose up -d

# 11. Check status
docker-compose ps

# 12. View logs
docker-compose logs -f
```

---

## 📊 Performance Benchmarks (Expected)

### On Recommended Droplet (8 vCPU, 16 GB RAM):

```
Training Pipeline (IEEE-CIS full dataset):
├─ Data loading & merge              :  30 sec
├─ UID creation                       :  10 sec
├─ Basic feature engineering          :  20 sec
├─ Velocity features (optimized)      :  8 min
├─ Frequency encoding                 :  30 sec
├─ VAE ensemble (3 models, 50 epochs) :  6 min
├─ XGBoost training (2000 trees)      : 12 min
├─ Calibration                        :  30 sec
└─ MLflow logging & save              :  30 sec
──────────────────────────────────────────────
TOTAL                                 : ~28 min

Inference (Real-time):
├─ Latency per transaction            : < 100 ms
├─ Throughput                         : 200-500 TPS
└─ Spark micro-batch interval         : 5 seconds

Dashboard:
├─ Refresh rate                       : 2 seconds
└─ UI response time                   : < 500 ms
```

---

## 🔒 Security Recommendations

### 1. **SSH Security:**
```bash
# Disable password authentication
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no

# Use SSH keys only
# Restrict root login: PermitRootLogin no
```

### 2. **Firewall (ufw):**
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 8080/tcp    # Airflow (restrict with ufw allow from YOUR_IP)
sudo ufw allow 8501/tcp    # Dashboard
sudo ufw enable
```

### 3. **SSL/TLS:**
```bash
# Use Nginx reverse proxy with Let's Encrypt
sudo apt-get install nginx certbot python3-certbot-nginx
sudo certbot --nginx -d fraud-detection.yourdomain.com
```

### 4. **Environment Variables:**
```bash
# Never commit .env file
echo ".env" >> .gitignore

# Use strong passwords for:
- Airflow UI
- MLflow (if authentication enabled)
- Kafka credentials
```

---

## 📈 Monitoring & Maintenance

### DigitalOcean Monitoring (Free):

```
Metrics to Watch:
├─ CPU Usage (keep < 80% average)
├─ Memory Usage (keep < 90%)
├─ Disk Usage (alert at 80%)
├─ Bandwidth (track against 6 TB limit)
└─ Disk I/O (identify bottlenecks)

Alerts to Set:
☑ CPU > 90% for 5 minutes
☑ Memory > 95% for 5 minutes
☑ Disk > 85% full
☑ Droplet down/unreachable
```

### Log Rotation:

```bash
# Configure Docker log rotation
sudo nano /etc/docker/daemon.json

{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}

sudo systemctl restart docker
```

---

## 🔄 Upgrade Path

### When to Upgrade:

```
Upgrade to 16 vCPU / 32 GB RAM when:
├─ Training takes > 60 minutes
├─ CPU consistently > 85%
├─ Out of Memory (OOM) errors
├─ Handling > 1000 TPS in inference
└─ Running multiple experiments simultaneously

Upgrade to GPU Droplet when:
├─ Need to reduce training time by 5-10x
├─ Training VAE on larger datasets
├─ Running hyperparameter tuning
└─ Budget allows ($300-500/month)
```

---

## ✅ Pre-Deployment Checklist

Before deploying to DigitalOcean:

- [ ] IEEE-CIS dataset prepared and accessible
- [ ] `.env` file configured with all credentials
- [ ] Kafka credentials from Confluent Cloud
- [ ] SSH key pair generated
- [ ] Domain name purchased (optional)
- [ ] Email SMTP configured (or using MailDev)
- [ ] Backups strategy planned
- [ ] Monitoring alerts configured
- [ ] Cost budget approved (~$125/month)
- [ ] Security hardening plan reviewed

---

## 🎯 FINAL RECOMMENDATION

**For Your Use Case (Full ML Training with VAE + Velocity Features):**

```
✅ RECOMMENDED DROPLET:
   Type: CPU-Optimized
   Size: c-8 (8 vCPUs, 16 GB RAM, 100 GB SSD)
   Price: $96/month
   Region: New York 3 (closest to your Kafka cluster)

🎯 This configuration provides:
   ✅ Sufficient CPU for parallel training
   ✅ Enough RAM for large dataset + multiple models
   ✅ Fast SSD for I/O operations
   ✅ Room for growth
   ✅ Cost-effective for production

🚀 Start with this, monitor performance, and scale if needed!
```

---

**Questions to Consider:**

1. **Dataset Size:** Will you use the full IEEE-CIS dataset (590k rows) or a subset?
   - Full dataset: 8 vCPU / 16 GB RAM ✅
   - Subset (< 100k): 4 vCPU / 8 GB RAM might work

2. **Training Frequency:** How often will you retrain?
   - Daily: Need full resources
   - Weekly: Could start smaller and resize when training

3. **Inference Load:** Expected transactions per second?
   - < 100 TPS: Current config is fine
   - > 500 TPS: Consider 16 vCPU

4. **Budget:** What's your monthly budget?
   - $50-100: Start with 4-8 vCPU
   - $100-200: Go with 8-16 vCPU
   - $300+: Add GPU for faster training

---

**Need Help?**
- DigitalOcean Docs: https://docs.digitalocean.com/
- Community Forum: https://www.digitalocean.com/community
- Support: Available with Professional plans ($100+/month)

Good luck with your deployment! 🚀
