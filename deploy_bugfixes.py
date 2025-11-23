"""
Deploy Bug Fixes to Production VM

This script helps deploy the dashboard and ATO service fixes to your production VM.
Run this AFTER pushing code changes to git.
"""

import subprocess
import sys
from pathlib import Path

print("=" * 80)
print("DEPLOYING BUG FIXES TO PRODUCTION")
print("=" * 80)

# Check if we're in the right directory
current_dir = Path.cwd()
if not (current_dir / "src" / "dashboard" / "app.py").exists():
    print("\n❌ ERROR: Must run from project root directory")
    print(f"Current directory: {current_dir}")
    sys.exit(1)

print("\n📋 FILES MODIFIED:")
print("  1. src/dashboard/app.py (deduplication fix)")
print("  2. src/inference/ato_service.py (geo_anomaly threshold fix)")

print("\n" + "=" * 80)
print("STEP 1: COMMIT AND PUSH CHANGES TO GIT")
print("=" * 80)

# Check git status
result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
if result.stdout.strip():
    print("\n📝 Modified files:")
    print(result.stdout)
    
    response = input("\n❓ Commit and push these changes? (y/n): ").lower()
    if response == 'y':
        # Add files
        subprocess.run(["git", "add", "src/dashboard/app.py"])
        subprocess.run(["git", "add", "src/inference/ato_service.py"])
        subprocess.run(["git", "add", "BUGFIX_SUMMARY.md"])
        subprocess.run(["git", "add", "send_test_transaction.py"])
        
        # Commit
        commit_msg = "Fix: Dashboard deduplication + geo_anomaly threshold"
        subprocess.run(["git", "commit", "-m", commit_msg])
        
        # Push
        print("\n📤 Pushing to git...")
        result = subprocess.run(["git", "push"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Changes pushed successfully!")
        else:
            print(f"❌ Push failed: {result.stderr}")
            sys.exit(1)
    else:
        print("\n⚠️ Skipping git commit. You'll need to push manually.")
else:
    print("\n✅ No uncommitted changes (already pushed?)")

print("\n" + "=" * 80)
print("STEP 2: DEPLOY TO VM (167.71.224.89)")
print("=" * 80)

print("\n📝 SSH Commands to run on VM:\n")

commands = """
# 1. Pull latest code
cd /path/to/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML
git pull origin main

# 2. Restart affected services
docker-compose restart inference dashboard

# 3. Verify services are running
docker-compose ps

# 4. Check logs for errors
docker-compose logs -f --tail=50 inference dashboard
"""

print(commands)

print("\n" + "=" * 80)
print("ALTERNATIVE: Full Rebuild (Recommended for Production)")
print("=" * 80)

rebuild_commands = """
# 1. Pull latest code
cd /path/to/Streaming-Data-Driven-Fraud-Detection-for-Transactions-using-ML
git pull origin main

# 2. Stop and remove containers
docker-compose down

# 3. Rebuild with new code
docker-compose build inference dashboard

# 4. Start all services
docker-compose up -d

# 5. Verify all services healthy
docker-compose ps
docker-compose logs -f --tail=100 inference dashboard
"""

print(rebuild_commands)

print("\n" + "=" * 80)
print("STEP 3: TEST THE FIXES")
print("=" * 80)

print("\n1. Send test transaction:")
print("   python send_test_transaction.py")

print("\n2. Check dashboard at: http://167.71.224.89:8501")

print("\n3. Verify:")
print("   ✅ Transaction 2987245 appears ONCE (no duplicates)")
print("   ✅ Risk Factors: [] (empty - no geo_anomaly_distance_19km)")
print("   ✅ Decision: APPROVE")
print("   ✅ Risk Level: LOW")
print("   ✅ Fraud Probability: 0.019")

print("\n" + "=" * 80)
print("✅ DEPLOYMENT INSTRUCTIONS READY")
print("=" * 80)

print("\n📌 Next Steps:")
print("1. SSH to VM: ssh user@167.71.224.89")
print("2. Run the deployment commands above")
print("3. Test with: python send_test_transaction.py")
print("4. Verify dashboard shows correct data")

print("\n" + "=" * 80)
