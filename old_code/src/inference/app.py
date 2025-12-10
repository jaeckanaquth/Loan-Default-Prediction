from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Any, Optional
import mlflow.pyfunc
import os
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import time
from functools import wraps

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available, metrics endpoint will be disabled")

app = FastAPI(title="Loan Default Prediction API", version="1.0.0")
MODEL_PATH = os.getenv('MODEL_PATH', 'models/model')
_model_cache = None
_model_features = None  # Cache expected feature names

# API Key authentication
security = HTTPBearer(auto_error=False)
API_KEY = os.getenv('API_KEY', '')  # Set via environment variable
API_KEY_ENABLED = os.getenv('API_KEY_ENABLED', 'false').lower() == 'true'

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    prediction_counter = Counter(
        'loan_prediction_total',
        'Total number of predictions',
        ['status']  # 'success' or 'error'
    )
    prediction_latency = Histogram(
        'loan_prediction_duration_seconds',
        'Prediction latency in seconds',
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    )
    prediction_value = Gauge(
        'loan_prediction_value',
        'Prediction value (0=no default, 1=default)'
    )
    active_requests = Gauge(
        'loan_api_active_requests',
        'Number of active requests'
    )
else:
    # Dummy metrics if prometheus_client not available
    prediction_counter = None
    prediction_latency = None
    prediction_value = None
    active_requests = None

class Payload(BaseModel):
    """Strict Pydantic schema for prediction input validation."""
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names to numeric values",
        min_length=1
    )
    
    @field_validator('features')
    @classmethod
    def validate_features_numeric(cls, v: Dict[str, Any]) -> Dict[str, float]:
        """Ensure all feature values are numeric."""
        validated = {}
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError(f"Feature name must be a string, got {type(key)}")
            try:
                # Convert to float, rejecting NaN/Inf
                float_val = float(value)
                if not np.isfinite(float_val):
                    raise ValueError(f"Feature '{key}' has non-finite value: {value}")
                validated[key] = float_val
            except (ValueError, TypeError) as e:
                raise ValueError(f"Feature '{key}' must be a numeric value, got {type(value)}: {value}")
        return validated
    
    @field_validator('features')
    @classmethod
    def validate_feature_ranges(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate feature values are within reasonable ranges for loan data."""
        # Common loan feature ranges (can be extended)
        range_checks = {
            # Amounts (typically positive, but allow some negative for differences)
            'loan_amnt': (0, 1e7),
            'funded_amnt': (0, 1e7),
            'total_pymnt': (0, 1e7),
            'installment': (0, 1e5),
            # Percentages/ratios
            'int_rate': (0, 100),
            'revol_util': (0, 100),
            'dti': (0, 100),
            # Counts
            'open_acc': (0, 100),
            'total_acc': (0, 200),
            'pub_rec': (0, 50),
            'delinq_2yrs': (0, 50),
            'inq_last_6mths': (0, 50),
            # Months/years
            'term': (0, 600),
            'emp_length': (0, 50),
            # Credit scores
            'fico_range_low': (300, 850),
            'fico_range_high': (300, 850),
        }
        
        for key, value in v.items():
            key_lower = key.lower()
            # Check if any range pattern matches
            for pattern, (min_val, max_val) in range_checks.items():
                if pattern in key_lower:
                    if not (min_val <= value <= max_val):
                        raise ValueError(
                            f"Feature '{key}' value {value} is outside expected range "
                            f"[{min_val}, {max_val}]"
                        )
                    break
            # General sanity check: reject extremely large values
            if abs(value) > 1e10:
                raise ValueError(f"Feature '{key}' has suspiciously large value: {value}")
        
        return v
    
    @model_validator(mode='after')
    def validate_required_features(self):
        """Validate that all required features are present (checked at runtime against model)."""
        # This will be checked in the predict endpoint after model is loaded
        return self

def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify API key if authentication is enabled."""
    if not API_KEY_ENABLED:
        return True
    
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")
    
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return True

@app.get('/health')
def health():
    """Health check endpoint."""
    return {
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': _model_cache is not None
    }

@app.get('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return Response(
            content="Prometheus client not available",
            status_code=503,
            media_type="text/plain"
        )
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

def get_model_features(model):
    """Extract expected feature names from model or pipeline."""
    global _model_features
    if _model_features is not None:
        return _model_features
    
    # Try to get features from sklearn pipeline
    if hasattr(model, 'feature_names_in_'):
        _model_features = list(model.feature_names_in_)
        return _model_features
    
    # Try to get from pipeline steps
    if hasattr(model, 'steps') or hasattr(model, 'named_steps'):
        # It's a pipeline, try to get feature names from transformers
        pipeline = model
        if hasattr(pipeline, 'named_steps'):
            # Check for ColumnSelector or similar
            for step_name, step in pipeline.named_steps.items():
                if hasattr(step, 'columns'):
                    _model_features = list(step.columns)
                    return _model_features
                if hasattr(step, 'feature_names_in_'):
                    _model_features = list(step.feature_names_in_)
                    return _model_features
    
    # Fallback: try to load from metadata file
    meta_path = Path('data/processed/loan.meta.json')
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                if 'selected_features' in meta:
                    _model_features = meta['selected_features']
                    return _model_features
        except Exception as e:
            logger.warning(f"Could not load features from metadata: {e}")
    
    # Last resort: return None and validate at prediction time
    return None

def load_model():
    """Load model, supporting both MLflow and joblib formats"""
    global _model_cache, _model_features
    if _model_cache is not None:
        return _model_cache
    
    model_path = Path(MODEL_PATH)
    
    # Try MLflow model first
    try:
        _model_cache = mlflow.pyfunc.load_model(MODEL_PATH)
        _model_features = get_model_features(_model_cache)
        return _model_cache
    except Exception as e1:
        # Try joblib file if MLflow fails
        if model_path.suffix == '.pkl' or not model_path.exists():
            # Try default joblib path
            joblib_path = Path('src/models/model_rf.pkl')
            if joblib_path.exists():
                try:
                    _model_cache = joblib.load(joblib_path)
                    _model_features = get_model_features(_model_cache)
                    return _model_cache
                except Exception as e2:
                    raise Exception(f'MLflow load failed: {e1}, Joblib load failed: {e2}')
            elif model_path.exists() and model_path.suffix == '.pkl':
                try:
                    _model_cache = joblib.load(model_path)
                    _model_features = get_model_features(_model_cache)
                    return _model_cache
                except Exception as e2:
                    raise Exception(f'MLflow load failed: {e1}, Joblib load failed: {e2}')
        raise e1

def log_prediction(features: Dict[str, float], prediction: Any, log_to_file: bool = True, log_to_mlflow: bool = True):
    """Log prediction to file and optionally MLflow."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'features': features,
        'prediction': prediction
    }
    
    # Log to local file
    if log_to_file:
        log_file = Path('predictions.log')
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write to prediction log file: {e}")
    
    # Log to MLflow if available
    if log_to_mlflow:
        try:
            import mlflow
            mlflow.log_dict(log_entry, f"predictions/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        except ImportError:
            logger.debug("MLflow not available for logging")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

@app.post('/predict')
def predict(
    payload: Payload,
    _: bool = Depends(verify_api_key)
):
    """Make prediction with strict input validation and logging."""
    start_time = time.time()
    
    # Track active requests
    if active_requests:
        active_requests.inc()
    
    try:
        # Load model on demand
        try:
            model = load_model()
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            if prediction_counter:
                prediction_counter.labels(status='error').inc()
            raise HTTPException(status_code=500, detail=f'Cannot load model: {e}')
        
        # Validate features match model expectations
        expected_features = get_model_features(model)
        if expected_features is not None:
            provided_features = set(payload.features.keys())
            expected_set = set(expected_features)
            
            missing = expected_set - provided_features
            extra = provided_features - expected_set
            
            if missing:
                if prediction_counter:
                    prediction_counter.labels(status='error').inc()
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required features: {sorted(missing)}"
                )
            if extra:
                logger.warning(f"Extra features provided (will be ignored): {sorted(extra)}")
        
        # Create DataFrame with correct feature order
        try:
            if expected_features:
                # Ensure correct order and only include expected features
                feature_dict = {f: payload.features.get(f, 0.0) for f in expected_features}
                df = pd.DataFrame([feature_dict])
            else:
                df = pd.DataFrame([payload.features])
            
            # Make prediction
            preds = model.predict(df)
            pred_proba = None
            if hasattr(model, 'predict_proba'):
                pred_proba = model.predict_proba(df)
            
            result = {
                'predictions': preds.tolist(),
                'probabilities': pred_proba.tolist() if pred_proba is not None else None
            }
            
            # Update metrics
            if prediction_counter:
                prediction_counter.labels(status='success').inc()
            if prediction_latency:
                prediction_latency.observe(time.time() - start_time)
            if prediction_value is not None and len(preds) > 0:
                prediction_value.set(float(preds[0]))
            
            # Log prediction
            log_prediction(payload.features, result)
            
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            if prediction_counter:
                prediction_counter.labels(status='error').inc()
            raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')
    finally:
        # Decrement active requests
        if active_requests:
            active_requests.dec()
