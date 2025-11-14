# 🔒 Security Configuration

## ⚠️ CRITICAL: Never Commit Credentials!

This project uses `.env` and `config.yaml` files to store sensitive credentials. These files are **gitignored** and should **NEVER** be committed to GitHub.

---

## 📋 Setup Instructions

### 1. Create Your `.env` File

```bash
cd src
cp .env.example .env
```

### 2. Update with Your Credentials

Edit `src/.env` and replace placeholders:

```bash
# Replace these with your actual credentials:
KAFKA_BOOTSTRAP_SERVERS=your-kafka-cluster.aws.confluent.cloud:9092
KAFKA_USERNAME=YOUR_KAFKA_API_KEY
KAFKA_PASSWORD=YOUR_KAFKA_API_SECRET
```

### 3. Verify `.env` is Ignored

```bash
git status
# Should NOT show .env or config.yaml
```

---

## 🚨 What's Protected

### Files that are `.gitignore`d:
- ✅ `src/.env` - All environment variables and secrets
- ✅ `src/config.yaml` - Kafka credentials
- ✅ `.env.*` - Any .env variants
- ✅ `*.env` - All .env files

### Files that ARE committed:
- ✅ `src/.env.example` - Template with placeholders
- ✅ `src/config.yaml.example` - Template (if created)

---

## 🔍 If You Accidentally Committed Credentials

### 1. Remove from Git History (URGENT!)

```bash
# Remove file from Git tracking
git rm --cached src/.env
git rm --cached src/config.yaml

# Commit the removal
git add .gitignore
git commit -m "chore: remove sensitive files from tracking"
git push origin main
```

### 2. Rotate ALL Credentials Immediately

If credentials were pushed to GitHub, assume they're compromised:

1. **Kafka (Confluent Cloud)**:
   - Go to Confluent Cloud Console
   - Delete old API key
   - Create new API key
   - Update `src/.env`

2. **MinIO**:
   - Change `MINIO_PASSWORD` in `src/.env`

3. **Airflow**:
   - Change `_AIRFLOW_WWW_USER_PASSWORD` in `src/.env`

4. **PostgreSQL**:
   - Change `POSTGRES_PASSWORD` in `src/.env`

### 3. Verify No Secrets in GitHub

```bash
# Search your repo on GitHub for keywords:
# - KAFKA_PASSWORD
# - KAFKA_USERNAME
# - HIhrR1hP0Oj (your current password prefix)
```

---

## ✅ Best Practices

### For Development (Local)
```bash
# Use .env file (gitignored)
cp src/.env.example src/.env
# Edit src/.env with real credentials
```

### For Production (DigitalOcean)
```bash
# Option 1: Copy .env securely
scp src/.env root@YOUR_DROPLET_IP:/opt/.../src/

# Option 2: Use secrets manager (recommended)
# - AWS Secrets Manager
# - HashiCorp Vault
# - Docker Secrets
```

### For CI/CD (GitHub Actions)
```yaml
# Use GitHub Secrets
env:
  KAFKA_USERNAME: ${{ secrets.KAFKA_USERNAME }}
  KAFKA_PASSWORD: ${{ secrets.KAFKA_PASSWORD }}
```

---

## 📝 Checklist

Before pushing to GitHub:

- [ ] `src/.env` exists and has your credentials
- [ ] `src/.env` is in `.gitignore`
- [ ] Run `git status` - should NOT show `.env`
- [ ] Run `git check-ignore src/.env` - should return a match
- [ ] `src/.env.example` exists with placeholders
- [ ] No credentials in `config.yaml` (use env vars instead)

---

## 🆘 Need Help?

If you committed credentials:
1. Follow removal steps above
2. Rotate ALL credentials immediately
3. Consider using `git-secrets` or `pre-commit` hooks to prevent future leaks

---

**Remember: Once credentials are pushed to GitHub, consider them compromised. Always rotate immediately!**
