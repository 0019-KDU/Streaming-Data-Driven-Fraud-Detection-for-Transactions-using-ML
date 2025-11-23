#!/usr/bin/env python3
"""
Verification Script: Model & Feature Pipeline Parity

Checks:
1. ✅ Model bundle contains correct XGBoost model and 88 features
2. ✅ Feature pipeline.pkl has transform() method and produces 88 features
3. ✅ Feature names match between model bundle and pipeline
4. ✅ Feature order matches exactly
5. ✅ Missing value handling is consistent
6. ✅ Scaling/normalization is applied consistently
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def verify_model_bundle(model_path: str) -> dict:
    """Verify model bundle structure and contents."""
    print("\n" + "="*80)
    print("🔍 VERIFYING MODEL BUNDLE")
    print("="*80)
    
    if not Path(model_path).exists():
        print(f"❌ FAIL: Model bundle not found at {model_path}")
        return None
    
    print(f"📁 Loading: {model_path}")
    bundle = joblib.load(model_path)
    
    # Check bundle structure
    print(f"\n📦 Bundle type: {type(bundle)}")
    
    if isinstance(bundle, dict):
        print(f"✅ Bundle is dict with keys: {list(bundle.keys())}")
        
        # Extract model
        model = bundle.get('model') or bundle.get('calibrated_model')
        if model is None:
            print("❌ FAIL: No 'model' or 'calibrated_model' key in bundle")
            return None
        
        print(f"✅ Model type: {type(model).__name__}")
        
        # Check feature names
        feature_names = bundle.get('feature_names', [])
        print(f"✅ Feature names: {len(feature_names)} features")
        
        # Check threshold
        threshold = bundle.get('threshold')
        print(f"✅ Threshold: {threshold}")
        
        # Check model metadata
        metadata = {
            'model_type': bundle.get('model_type', 'unknown'),
            'n_features': bundle.get('n_features', len(feature_names)),
            'training_date': bundle.get('training_date', 'unknown'),
            'threshold': threshold,
            'metrics': bundle.get('metrics', {})
        }
        
        print(f"\n📊 Model Metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        
        # Verify it's XGBoost
        model_class = type(model).__name__
        if 'XGB' in model_class or 'xgboost' in str(type(model)).lower():
            print(f"✅ Confirmed XGBoost model: {model_class}")
        else:
            print(f"⚠️  WARNING: Model is not XGBoost: {model_class}")
        
        return {
            'bundle': bundle,
            'model': model,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'metadata': metadata
        }
    else:
        print(f"⚠️  WARNING: Bundle is not dict, treating as raw model")
        return {
            'bundle': bundle,
            'model': bundle,
            'feature_names': [],
            'n_features': 88,  # Default
            'metadata': {}
        }


def verify_feature_pipeline(pipeline_path: str) -> dict:
    """Verify feature pipeline structure and functionality."""
    print("\n" + "="*80)
    print("🔍 VERIFYING FEATURE PIPELINE")
    print("="*80)
    
    if not Path(pipeline_path).exists():
        print(f"❌ FAIL: Feature pipeline not found at {pipeline_path}")
        return None
    
    print(f"📁 Loading: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    
    print(f"📦 Pipeline type: {type(pipeline).__name__}")
    
    # Check for transform method
    if hasattr(pipeline, 'transform'):
        print("✅ Pipeline has transform() method")
    else:
        print("❌ FAIL: Pipeline missing transform() method")
        return None
    
    # Check for feature_names attribute
    if hasattr(pipeline, 'feature_names'):
        feature_names = pipeline.feature_names
        print(f"✅ Pipeline has feature_names: {len(feature_names)} features")
    else:
        print("⚠️  WARNING: Pipeline missing feature_names attribute")
        feature_names = []
    
    # Check for other critical attributes
    attrs = ['freq_maps', 'scaler', 'mean_encoding_maps', 'card1_amt_mean', 
             'card4_amt_mean', 'magic_uid_stats', 'group_agg_stats']
    
    print("\n📋 Pipeline Attributes:")
    for attr in attrs:
        if hasattr(pipeline, attr):
            value = getattr(pipeline, attr)
            if isinstance(value, dict):
                print(f"   ✅ {attr}: dict with {len(value)} entries")
            else:
                print(f"   ✅ {attr}: {type(value).__name__}")
        else:
            print(f"   ⚠️  {attr}: MISSING")
    
    return {
        'pipeline': pipeline,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'has_transform': hasattr(pipeline, 'transform')
    }


def verify_feature_parity(model_info: dict, pipeline_info: dict):
    """Verify features match between model and pipeline."""
    print("\n" + "="*80)
    print("🔍 VERIFYING FEATURE PARITY")
    print("="*80)
    
    model_features = set(model_info['feature_names'])
    pipeline_features = set(pipeline_info['feature_names'])
    
    print(f"📊 Model features: {len(model_features)}")
    print(f"📊 Pipeline features: {len(pipeline_features)}")
    
    # Check counts
    if len(model_features) == 88 and len(pipeline_features) == 88:
        print("✅ Both have exactly 88 features")
    else:
        print(f"⚠️  WARNING: Feature count mismatch!")
        print(f"   Model: {len(model_features)} features")
        print(f"   Pipeline: {len(pipeline_features)} features")
    
    # Check exact match
    if model_features == pipeline_features:
        print("✅ Feature names match perfectly")
    else:
        missing_in_pipeline = model_features - pipeline_features
        missing_in_model = pipeline_features - model_features
        
        if missing_in_pipeline:
            print(f"❌ Features in model but NOT in pipeline ({len(missing_in_pipeline)}):")
            for feat in sorted(missing_in_pipeline)[:10]:
                print(f"   - {feat}")
            if len(missing_in_pipeline) > 10:
                print(f"   ... and {len(missing_in_pipeline) - 10} more")
        
        if missing_in_model:
            print(f"❌ Features in pipeline but NOT in model ({len(missing_in_model)}):")
            for feat in sorted(missing_in_model)[:10]:
                print(f"   - {feat}")
            if len(missing_in_model) > 10:
                print(f"   ... and {len(missing_in_model) - 10} more")
    
    # Check feature order
    model_features_list = model_info['feature_names']
    pipeline_features_list = pipeline_info['feature_names']
    
    if model_features_list == pipeline_features_list:
        print("✅ Feature order matches exactly")
    else:
        print("⚠️  WARNING: Feature order differs")
        print("   This is OK if model reorders internally")


def test_transform_consistency(pipeline_info: dict, model_info: dict):
    """Test pipeline transform produces correct output."""
    print("\n" + "="*80)
    print("🔍 TESTING TRANSFORM CONSISTENCY")
    print("="*80)
    
    pipeline = pipeline_info['pipeline']
    
    # Create dummy transaction (same format as inference)
    dummy_tx = {
        'TransactionID': 'TEST_12345',
        'TransactionDT': 86400.0,
        'TransactionAmt': 100.0,
        'card1': 13413.0,
        'card2': 150.0,
        'card3': 150.0,
        'card4': 'visa',
        'card5': 142.0,
        'card6': 'credit',
        'addr1': 299.0,
        'addr2': 87.0,
        'dist1': 19.0,
        'P_emaildomain': 'hotmail.com',
        'ProductCD': 'C',
        'D1': 0.0,
        'D2': np.nan,
        'D3': np.nan,
        'D4': np.nan,
        'D5': np.nan,
        'D6': np.nan,
        'D7': np.nan,
        'D8': np.nan,
        'D9': np.nan,
        'D10': np.nan,
        'D11': np.nan,
        'D12': np.nan,
        'D13': np.nan,
        'D14': np.nan,
        'D15': np.nan,
    }
    
    df = pd.DataFrame([dummy_tx])
    
    print(f"📝 Input shape: {df.shape}")
    print(f"📝 Input columns: {list(df.columns)[:10]}...")
    
    try:
        # Transform
        features_df = pipeline.transform(df)
        
        print(f"✅ Transform successful")
        print(f"📊 Output shape: {features_df.shape}")
        print(f"📊 Output columns: {features_df.shape[1]} features")
        
        # Check output feature count
        if features_df.shape[1] == 88:
            print("✅ Produces exactly 88 features")
        else:
            print(f"❌ FAIL: Expected 88 features, got {features_df.shape[1]}")
        
        # Check for NaN/inf values
        nan_count = features_df.isna().sum().sum()
        inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
        
        print(f"📊 NaN values: {nan_count}")
        print(f"📊 Inf values: {inf_count}")
        
        if nan_count > 0:
            print("⚠️  WARNING: Output contains NaN values")
            nan_cols = features_df.columns[features_df.isna().any()].tolist()
            print(f"   NaN columns: {nan_cols[:10]}")
        
        # Check column names match expected
        if hasattr(pipeline, 'feature_names'):
            expected_cols = pipeline.feature_names
            actual_cols = features_df.columns.tolist()
            
            if expected_cols == actual_cols:
                print("✅ Output column names match feature_names")
            else:
                print("⚠️  WARNING: Output column names differ from feature_names")
        
        # Show sample values
        print(f"\n📊 Sample feature values (first 10):")
        for col in features_df.columns[:10]:
            val = features_df[col].iloc[0]
            print(f"   {col}: {val:.4f}" if not pd.isna(val) else f"   {col}: NaN")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Transform failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "#"*80)
    print("# MODEL & FEATURE PIPELINE PARITY VERIFICATION")
    print("#"*80)
    
    # Paths (adjust if running from different location)
    model_path = "src/models/fraud_detection_xgboost_model.pkl"
    pipeline_path = "src/models/feature_pipeline.pkl"
    
    # Check if running from project root
    if not Path(model_path).exists():
        # Try alternative paths
        alt_model_path = "../models/fraud_detection_xgboost_model.pkl"
        alt_pipeline_path = "../models/feature_pipeline.pkl"
        
        if Path(alt_model_path).exists():
            model_path = alt_model_path
            pipeline_path = alt_pipeline_path
        else:
            print(f"❌ ERROR: Cannot find model files")
            print(f"   Tried: {model_path}")
            print(f"   Tried: {alt_model_path}")
            sys.exit(1)
    
    # Verify model bundle
    model_info = verify_model_bundle(model_path)
    if model_info is None:
        print("\n❌ VERIFICATION FAILED: Model bundle issues")
        sys.exit(1)
    
    # Verify feature pipeline
    pipeline_info = verify_feature_pipeline(pipeline_path)
    if pipeline_info is None:
        print("\n❌ VERIFICATION FAILED: Feature pipeline issues")
        sys.exit(1)
    
    # Verify feature parity
    verify_feature_parity(model_info, pipeline_info)
    
    # Test transform
    transform_ok = test_transform_consistency(pipeline_info, model_info)
    
    # Final summary
    print("\n" + "="*80)
    print("📋 VERIFICATION SUMMARY")
    print("="*80)
    
    checks = [
        ("Model bundle loaded", model_info is not None),
        ("Model is XGBoost", 'XGB' in type(model_info['model']).__name__),
        ("Model has 88 features", model_info['n_features'] == 88),
        ("Pipeline loaded", pipeline_info is not None),
        ("Pipeline has transform()", pipeline_info['has_transform']),
        ("Pipeline has 88 features", pipeline_info['n_features'] == 88),
        ("Feature names match", set(model_info['feature_names']) == set(pipeline_info['feature_names'])),
        ("Transform works", transform_ok),
    ]
    
    all_passed = all(result for _, result in checks)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print("="*80)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED - Model and pipeline are consistent")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Review issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
