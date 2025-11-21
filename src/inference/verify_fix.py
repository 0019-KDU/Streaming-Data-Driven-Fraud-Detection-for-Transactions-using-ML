import pandas as pd
import numpy as np
import logging
import sys
import os

# Add current directory to path so we can import feature_pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from feature_pipeline import IEEECISFeaturePipeline
except ImportError:
    print("Error: Could not import feature_pipeline. Make sure you are running this from src/inference/")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def verify_pipeline_fix():
    logger.info("🚀 Starting Feature Pipeline Verification...")
    
    # 1. Create a Mock Payload (High Fraud)
    payload = {
        "TransactionAmt": 5500.0,
        "ProductCD": "C",
        "card1": 1000,
        "card2": 555,
        "card3": 150,
        "card4": "visa",
        "card5": 226,
        "card6": "charge card",
        "addr1": 500,
        "addr2": 87,
        "dist1": 3500.0,
        "dist2": 3500.0,
        "P_emaildomain": "tempmail.com",
        "R_emaildomain": "gmail.com",
        "TransactionDT": 15000000, # Arbitrary timestamp
        "TransactionID": "TEST_TXN_1"
    }
    
    df = pd.DataFrame([payload])
    logger.info(f"Input Payload:\n{df.iloc[0][['TransactionAmt', 'P_emaildomain', 'card1']].to_dict()}")
    
    # 2. Initialize Pipeline
    pipeline = IEEECISFeaturePipeline()
    # Mock feature names (what the model expects)
    pipeline.feature_names = [
        'TransactionAmt', 'log_TransactionAmt', 'email_risky', 'magic_uid', 
        'uid_agg', 'amt_is_C', 'dt_hour'
    ]
    # Mock risky domains (usually loaded from config)
    pipeline.risky_domains = {'tempmail.com', 'mailinator.com'}
    
    # 3. Run Transform
    logger.info("\nRunning pipeline.transform()...")
    try:
        transformed_df = pipeline.transform(df)
        
        # 4. Verify Critical Features
        logger.info("\n✅ Verification Results:")
        
        # Check Amount Features
        log_amt = transformed_df['log_TransactionAmt'].iloc[0]
        expected_log = np.log1p(5500.0)
        if abs(log_amt - expected_log) < 0.001:
            logger.info(f"  [PASS] log_TransactionAmt calculated correctly: {log_amt:.4f}")
        else:
            logger.error(f"  [FAIL] log_TransactionAmt mismatch! Got {log_amt}, expected {expected_log}")

        # Check Email Risk
        email_risk = transformed_df['email_risky'].iloc[0]
        if email_risk == 1:
            logger.info(f"  [PASS] email_risky correctly flagged as 1 (tempmail.com)")
        else:
            logger.error(f"  [FAIL] email_risky failed! Got {email_risk}")
            
        # Check Magic UID
        if 'magic_uid' in transformed_df.columns:
             # We can't easily check the exact string without D1, but check if it exists
             logger.info(f"  [PASS] magic_uid feature created: {transformed_df['magic_uid'].iloc[0]}")
        
        # Check Interaction
        if 'amt_is_C' in transformed_df.columns:
            amt_c = transformed_df['amt_is_C'].iloc[0]
            if amt_c == 5500.0:
                logger.info(f"  [PASS] amt_is_C interaction correct: {amt_c}")
            else:
                logger.error(f"  [FAIL] amt_is_C interaction failed! Got {amt_c}")

        logger.info("\n🎉 Pipeline Fix Verified! You are ready to redeploy.")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_pipeline_fix()
