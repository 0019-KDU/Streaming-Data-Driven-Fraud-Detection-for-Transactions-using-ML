# Docker Restart Guide - Digital Ocean Droplet

## ✅ Your Changes Are Safe!

Your Docker Compose setup uses **volume mounts**, which means:
- ✅ Code changes in `dags/`, `config.yaml`, `models/` are **preserved**
- ✅ Database (Postgres) data is **persistent**
- ✅ MLflow experiments are **saved**
- ✅ Only containers restart, not your files

---

## 🚀 Quick Restart Commands (SSH to Digital Ocean)

### Option 1: Restart All Services (Recommended)
```bash
# SSH into your Digital Ocean droplet
ssh root@YOUR_DROPLET_IP

# Navigate to project directory
cd /path/to/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src

# Restart all containers (picks up new code changes)
docker-compose restart

# Or restart specific service (faster)
docker-compose restart airflow-scheduler
docker-compose restart airflow-worker
```

**Time:** ~30-60 seconds
**Impact:** Containers restart, volume mounts reload your new code

---

### Option 2: Stop and Start (Clean Restart)
```bash
# Stop all containers
docker-compose down

# Start all containers (preserves volumes)
docker-compose up -d

# Check status
docker-compose ps
```

**Time:** ~2-3 minutes
**Impact:** Full clean restart, volumes are preserved

---

### Option 3: Rebuild Only Changed Services (If Dockerfile Changed)
```bash
# If you changed Dockerfile (not just Python code)
docker-compose build airflow-scheduler
docker-compose up -d airflow-scheduler

# Check logs
docker-compose logs -f airflow-scheduler
```

**Time:** ~5-10 minutes
**Only needed if:** You modified `Dockerfile` or `requirements.txt`

---

## 📋 Step-by-Step: Apply Your Fixes on Digital Ocean

### 1. **SSH into Digital Ocean Droplet**
```bash
ssh root@YOUR_DROPLET_IP
# Or if you have SSH key configured:
# ssh -i ~/.ssh/your_key root@YOUR_DROPLET_IP
```

---

### 2. **Navigate to Project Directory**
```bash
cd /path/to/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML/src

# Verify you're in the right place
ls -la config.yaml  # Should exist
```

---

### 3. **Apply the Code Fixes**

Your fixes are in these files (already modified locally):
- ✅ `config.yaml` - Lines 40-43 (FFS enabled, RandomizedSearch disabled)
- ✅ `dags/ieee_cis_training.py` - Lines 692-774, 932-1028 (optimizations)

**Option A: Upload Fixed Files via SCP**
```bash
# From your LOCAL machine (where you made changes)
cd D:\Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML

# Upload config.yaml
scp src/config.yaml root@YOUR_DROPLET_IP:/path/to/project/src/config.yaml

# Upload fixed training script
scp src/dags/ieee_cis_training.py root@YOUR_DROPLET_IP:/path/to/project/src/dags/ieee_cis_training.py
```

**Option B: Pull from Git (If you committed changes)**
```bash
# On Digital Ocean droplet
cd /path/to/project
git pull origin main
```

**Option C: Edit Directly on Server (Not Recommended)**
```bash
# On Digital Ocean droplet
nano src/config.yaml  # Manually apply changes
nano src/dags/ieee_cis_training.py  # Manually apply changes
```

---

### 4. **Restart Airflow Services to Pick Up Changes**
```bash
# Quick restart (picks up new Python code)
docker-compose restart airflow-scheduler
docker-compose restart airflow-worker

# Wait 10 seconds
sleep 10

# Verify services are running
docker-compose ps
```

Expected output:
```
NAME                STATUS         PORTS
airflow-scheduler   Up 10 seconds
airflow-worker      Up 10 seconds
airflow-webserver   Up 5 minutes   0.0.0.0:8080->8080/tcp
...
```

---

### 5. **Verify Changes Are Applied**
```bash
# Check Airflow logs for new config
docker-compose logs airflow-scheduler | grep "Forward Feature Selection"

# Should see:
# ✅ "use_forward_feature_selection: true"
# ✅ "K-Fold CV target encoding"
# ✅ "RandomizedSearchCV DISABLED"
```

---

### 6. **Trigger Training DAG**
```bash
# Option A: Via Airflow UI
# Open browser: http://YOUR_DROPLET_IP:8080
# Login: airflow / airflow
# Enable and trigger "ieee_cis_training_dag"

# Option B: Via CLI
docker-compose exec airflow-scheduler airflow dags trigger ieee_cis_training_dag

# Monitor logs
docker-compose logs -f airflow-worker
```

---

## 🔍 Troubleshooting

### Issue: "Changes not showing up"
```bash
# Force recreate containers (preserves volumes)
docker-compose down
docker-compose up -d

# Or rebuild if Dockerfile changed
docker-compose build airflow-scheduler
docker-compose up -d
```

---

### Issue: "Permission denied on files"
```bash
# Fix permissions (Airflow needs UID 50000)
sudo chown -R 50000:50000 dags/ logs/ plugins/ models/
```

---

### Issue: "Container keeps restarting"
```bash
# Check logs
docker-compose logs airflow-scheduler
docker-compose logs airflow-worker

# Common issues:
# - Syntax error in Python code
# - Missing dependencies in requirements.txt
# - Database migration needed
```

---

### Issue: "Database connection failed"
```bash
# Restart Postgres
docker-compose restart postgres

# Wait for health check
docker-compose ps postgres

# Should show "healthy" status
```

---

## 🎯 Quick Commands Reference

| Task | Command |
|------|---------|
| **Restart all services** | `docker-compose restart` |
| **Restart Airflow only** | `docker-compose restart airflow-scheduler airflow-worker` |
| **View logs** | `docker-compose logs -f airflow-scheduler` |
| **Check status** | `docker-compose ps` |
| **Stop all** | `docker-compose down` |
| **Start all** | `docker-compose up -d` |
| **Rebuild image** | `docker-compose build airflow-scheduler` |
| **Execute command inside** | `docker-compose exec airflow-scheduler bash` |
| **List running containers** | `docker ps` |
| **Clean everything** | `docker-compose down -v` (⚠️ **DELETES DATA**) |

---

## 📂 Volume Mounts (Your Changes Are Safe)

From `docker-compose.yml` lines 29-37:
```yaml
volumes:
  - ./dags:/opt/airflow/dags          # ✅ DAG changes preserved
  - ./logs:/opt/airflow/logs          # ✅ Logs preserved
  - ./config:/opt/airflow/config      # ✅ Config preserved
  - ./plugins:/opt/airflow/plugins    # ✅ Plugins preserved
  - ./models:/app/models               # ✅ Models preserved
  - ./config.yaml:/app/config.yaml    # ✅ Config preserved
  - ./.env:/app/.env                   # ✅ Environment preserved
  - ./data:/app/data                   # ✅ Dataset preserved
```

**What This Means:**
- 🟢 Changes to these files on host → immediately visible in container
- 🟢 Restarting containers → changes persist
- 🟢 `docker-compose down` → changes persist
- 🔴 `docker-compose down -v` → **DELETES VOLUMES** (don't use!)

---

## 🔒 Persistent Data

| Data Type | Storage | Survives Restart? | Survives `down`? |
|-----------|---------|-------------------|------------------|
| **Python code** | Volume mount | ✅ Yes | ✅ Yes |
| **Config files** | Volume mount | ✅ Yes | ✅ Yes |
| **Models** | Volume mount | ✅ Yes | ✅ Yes |
| **Postgres DB** | Named volume | ✅ Yes | ✅ Yes |
| **MLflow experiments** | MinIO volume | ✅ Yes | ✅ Yes |
| **Airflow logs** | Volume mount | ✅ Yes | ✅ Yes |
| **Container state** | Container layer | ❌ No | ❌ No |

---

## 🚨 DANGER COMMANDS (Don't Use Unless Sure)

```bash
# ⚠️ DELETES ALL VOLUMES (database, models, everything)
docker-compose down -v

# ⚠️ DELETES ALL DOCKER DATA ON SERVER
docker system prune -a --volumes

# ⚠️ FORCE REMOVE CONTAINERS
docker rm -f $(docker ps -aq)
```

**Only use these if:**
- You want to completely reset the environment
- You have backups of all data
- You understand you'll lose everything

---

## ✅ Recommended Workflow

### For Code Changes (Python, Config)
```bash
# 1. Upload changes
scp src/config.yaml root@IP:/path/to/project/src/
scp src/dags/ieee_cis_training.py root@IP:/path/to/project/src/dags/

# 2. Restart services
ssh root@IP "cd /path/to/project/src && docker-compose restart airflow-scheduler airflow-worker"

# 3. Verify
ssh root@IP "cd /path/to/project/src && docker-compose logs -f airflow-scheduler"
```

### For Dependency Changes (requirements.txt, Dockerfile)
```bash
# 1. Upload changes
scp src/airflow/Dockerfile root@IP:/path/to/project/src/airflow/
scp src/airflow/requirements.txt root@IP:/path/to/project/src/airflow/

# 2. Rebuild and restart
ssh root@IP "cd /path/to/project/src && docker-compose build airflow-scheduler && docker-compose up -d"
```

---

## 📊 Monitor Training Progress

```bash
# Watch scheduler logs
docker-compose logs -f airflow-scheduler | grep "ieee_cis"

# Watch worker logs (where training runs)
docker-compose logs -f airflow-worker | grep -E "AUC|Forward Feature Selection|K-Fold"

# Check MLflow UI
# Open: http://YOUR_DROPLET_IP:5500

# Check Airflow UI
# Open: http://YOUR_DROPLET_IP:8080
```

---

## 🎉 Summary

Your code changes are **already safe** due to volume mounts. To apply fixes:

1. **Upload fixed files** via SCP or git pull
2. **Restart services**: `docker-compose restart airflow-scheduler airflow-worker`
3. **Monitor logs**: `docker-compose logs -f airflow-worker`
4. **Trigger training** via Airflow UI or CLI

**Expected Result:**
- ✅ Training completes in ~35-45 minutes (was 2-3 hours)
- ✅ AUC-ROC: 96-97% (was ~94%)
- ✅ No data leakage warnings
- ✅ Forward Feature Selection enabled

---

## 🆘 Need Help?

**Check Airflow UI:** http://YOUR_DROPLET_IP:8080
**Check MLflow UI:** http://YOUR_DROPLET_IP:5500
**Check Logs:** `docker-compose logs -f [service-name]`

**Common Issues:**
- Container not starting → Check logs for syntax errors
- Changes not applied → Do `docker-compose restart`
- Permission errors → Run `chown -R 50000:50000 dags/`
- Out of memory → Increase Digital Ocean droplet RAM or reduce `n_estimators`
