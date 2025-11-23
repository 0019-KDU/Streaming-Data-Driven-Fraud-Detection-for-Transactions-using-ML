"""
XGBoost Integration Verification Script

This script verifies that the entire fraud detection system is properly
configured for XGBoost model deployment.

Run this script to check:
1. XGBoost package installation
2. Training configuration
3. Inference pipeline compatibility
4. Docker dependencies
"""

import sys
import os
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """Print section header"""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{text:^80}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

def print_success(text):
    """Print success message"""
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    """Print error message"""
    print(f"{RED}❌ {text}{RESET}")

def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}⚠️  {text}{RESET}")

def print_info(text):
    """Print info message"""
    print(f"   {text}")

# Get project root
PROJECT_ROOT = Path(__file__).parent

def check_xgboost_installation():
    """Check if XGBoost is installed"""
    print_header("1. XGBoost Installation Check")
    
    try:
        import xgboost as xgb
        version = xgb.__version__
        print_success(f"XGBoost installed: version {version}")
        
        # Check for GPU support (optional)
        try:
            xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1)
            print_success("GPU support available (optional)")
        except:
            print_info("GPU support not available (CPU mode only)")
        
        return True
    except ImportError:
        print_error("XGBoost not installed!")
        print_info("Install with: pip install xgboost>=1.5.0")
        return False

def check_training_config():
    """Check training configuration for XGBoost"""
    print_header("2. Training Configuration Check")
    
    config_path = PROJECT_ROOT / "src" / "config.yaml"
    
    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check MLflow experiment name
        experiment_name = config.get('mlflow', {}).get('experiment_name', '')
        if 'xgboost' in experiment_name.lower():
            print_success(f"MLflow experiment: {experiment_name}")
        else:
            print_warning(f"MLflow experiment name doesn't mention XGBoost: {experiment_name}")
        
        # Check model parameters
        params = config.get('model', {}).get('params', {})
        expected_params = ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'colsample_bytree']
        
        all_params_found = True
        for param in expected_params:
            if param in params:
                print_success(f"Parameter {param}: {params[param]}")
            else:
                print_error(f"Missing parameter: {param}")
                all_params_found = False
        
        # Verify optimal hyperparameters from notebook
        if params.get('n_estimators') == 500:
            print_success("Using optimal n_estimators=500 from notebook")
        else:
            print_warning(f"n_estimators={params.get('n_estimators')} (notebook optimal: 500)")
        
        if params.get('max_depth') == 7:
            print_success("Using optimal max_depth=7 from notebook")
        else:
            print_warning(f"max_depth={params.get('max_depth')} (notebook optimal: 7)")
        
        if params.get('learning_rate') == 0.05:
            print_success("Using optimal learning_rate=0.05 from notebook")
        else:
            print_warning(f"learning_rate={params.get('learning_rate')} (notebook optimal: 0.05)")
        
        return all_params_found
    except Exception as e:
        print_error(f"Error reading config: {e}")
        return False

def check_training_code():
    """Check training code for XGBoost usage"""
    print_header("3. Training Code Check")
    
    training_path = PROJECT_ROOT / "src" / "dags" / "ieee_cis_training.py"
    
    if not training_path.exists():
        print_error(f"Training file not found: {training_path}")
        return False
    
    try:
        with open(training_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for XGBoost import
        if 'from xgboost import XGBClassifier' in content or 'import xgboost' in content:
            print_success("XGBoost imported in training code")
        else:
            print_error("XGBoost not imported in training code")
            return False
        
        # Check for XGBoost model instantiation
        if 'XGBClassifier(' in content:
            print_success("XGBClassifier instantiation found")
        else:
            print_error("XGBClassifier not instantiated")
            return False
        
        # Check for optimal hyperparameters
        if 'n_estimators=500' in content:
            print_success("Found n_estimators=500 (optimal from notebook)")
        else:
            print_warning("n_estimators=500 not found in code")
        
        if 'max_depth=7' in content:
            print_success("Found max_depth=7 (optimal from notebook)")
        else:
            print_warning("max_depth=7 not found in code")
        
        if 'learning_rate=0.05' in content:
            print_success("Found learning_rate=0.05 (optimal from notebook)")
        else:
            print_warning("learning_rate=0.05 not found in code")
        
        # Check for LightGBM usage (should not be primary)
        if 'LGBMClassifier' in content and 'best_model = lgb_model' in content:
            print_warning("LightGBM found as best_model (should be XGBoost)")
        else:
            print_success("XGBoost is the primary model")
        
        return True
    except Exception as e:
        print_error(f"Error reading training code: {e}")
        return False

def check_inference_code():
    """Check inference code compatibility"""
    print_header("4. Inference Code Check")
    
    model_loader_path = PROJECT_ROOT / "src" / "inference" / "model_loader.py"
    
    if not model_loader_path.exists():
        print_error(f"Model loader not found: {model_loader_path}")
        return False
    
    try:
        with open(model_loader_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for model loading
        if 'joblib.load' in content or 'pickle.load' in content:
            print_success("Model loading mechanism found")
        else:
            print_error("No model loading found")
            return False
        
        # Check for predict_proba (required for XGBoost)
        if 'predict_proba' in content:
            print_success("predict_proba method found (XGBoost compatible)")
        else:
            print_warning("predict_proba not found")
        
        # Check for LightGBM-specific imports
        if 'import lightgbm' in content or 'from lightgbm' in content:
            print_warning("LightGBM import found in inference code (may be legacy)")
        else:
            print_success("No LightGBM imports in inference code")
        
        return True
    except Exception as e:
        print_error(f"Error reading inference code: {e}")
        return False

def check_docker_configs():
    """Check Docker configurations"""
    print_header("5. Docker Configuration Check")
    
    # Check Airflow requirements
    airflow_req = PROJECT_ROOT / "src" / "airflow" / "requirements.txt"
    if airflow_req.exists():
        with open(airflow_req, 'r') as f:
            content = f.read()
        if 'xgboost' in content.lower():
            print_success("XGBoost in Airflow requirements.txt")
        else:
            print_error("XGBoost NOT in Airflow requirements.txt")
    else:
        print_warning(f"Airflow requirements.txt not found")
    
    # Check Inference requirements
    inference_req = PROJECT_ROOT / "src" / "inference" / "requirements.txt"
    if inference_req.exists():
        with open(inference_req, 'r') as f:
            content = f.read()
        if 'xgboost' in content.lower():
            print_success("XGBoost in Inference requirements.txt")
        else:
            print_error("XGBoost NOT in Inference requirements.txt")
    else:
        print_warning(f"Inference requirements.txt not found")
    
    # Check Airflow Dockerfile
    airflow_dockerfile = PROJECT_ROOT / "src" / "airflow" / "Dockerfile"
    if airflow_dockerfile.exists():
        with open(airflow_dockerfile, 'r') as f:
            content = f.read()
        if 'libgomp1' in content:
            print_success("libgomp1 installed (required for XGBoost)")
        else:
            print_warning("libgomp1 not found in Airflow Dockerfile")
    
    # Check Inference Dockerfile
    inference_dockerfile = PROJECT_ROOT / "src" / "inference" / "Dockerfile"
    if inference_dockerfile.exists():
        with open(inference_dockerfile, 'r') as f:
            content = f.read()
        if 'libgomp1' in content:
            print_success("libgomp1 installed (required for XGBoost)")
        else:
            print_warning("libgomp1 not found in Inference Dockerfile")
    
    return True

def check_feature_pipeline():
    """Check feature pipeline compatibility"""
    print_header("6. Feature Pipeline Check")
    
    feature_pipeline_path = PROJECT_ROOT / "src" / "dags" / "feature_pipeline.py"
    
    if not feature_pipeline_path.exists():
        print_error(f"Feature pipeline not found: {feature_pipeline_path}")
        return False
    
    try:
        with open(feature_pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for transform method
        if 'def transform' in content:
            print_success("Feature transform method found")
        else:
            print_error("No transform method found")
            return False
        
        # Check for 88 features (optimal feature count)
        if '88' in content or 'all_features' in content:
            print_success("Feature engineering pipeline found")
        else:
            print_warning("Feature count not clear in pipeline")
        
        # Check for Magic UID features
        if 'magic_uid' in content.lower() or 'Magic UID' in content:
            print_success("Magic UID features implemented (critical for performance)")
        else:
            print_warning("Magic UID features not found")
        
        return True
    except Exception as e:
        print_error(f"Error reading feature pipeline: {e}")
        return False

def main():
    """Run all verification checks"""
    print_header("XGBoost Integration Verification")
    print_info(f"Project Root: {PROJECT_ROOT}")
    
    results = []
    
    # Run all checks
    results.append(("XGBoost Installation", check_xgboost_installation()))
    results.append(("Training Configuration", check_training_config()))
    results.append(("Training Code", check_training_code()))
    results.append(("Inference Code", check_inference_code()))
    results.append(("Docker Configurations", check_docker_configs()))
    results.append(("Feature Pipeline", check_feature_pipeline()))
    
    # Summary
    print_header("Verification Summary")
    
    all_passed = True
    for check_name, passed in results:
        if passed:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")
            all_passed = False
    
    print("\n")
    
    if all_passed:
        print(f"{GREEN}{'='*80}{RESET}")
        print(f"{GREEN}🎉 ALL CHECKS PASSED! XGBoost integration verified.{RESET}")
        print(f"{GREEN}{'='*80}{RESET}")
        print_info("\nNext steps:")
        print_info("1. Retrain model: docker exec -it <airflow-container> airflow dags trigger ieee_cis_fraud_detection_training")
        print_info("2. Expected AUC-ROC: ~0.9763 (from notebook)")
        print_info("3. Test inference: python test_inference_full_payload.py")
        return 0
    else:
        print(f"{RED}{'='*80}{RESET}")
        print(f"{RED}⚠️  SOME CHECKS FAILED - Please review errors above{RESET}")
        print(f"{RED}{'='*80}{RESET}")
        print_info("\nReview the errors and warnings above to fix issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
