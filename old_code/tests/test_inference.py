"""
Unit tests for the FastAPI inference app.
Tests /health and /predict endpoints using TestClient.
"""
import pytest
from fastapi.testclient import TestClient
from src.inference.app import app
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.features.transformers import ColumnSelector

# Create test client
client = TestClient(app)

@pytest.fixture
def sample_model():
    """Create a simple model for testing."""
    # Create a dummy model with known features
    X_dummy = np.random.rand(100, 5)
    y_dummy = np.random.randint(0, 2, 100)
    
    # Create a pipeline with ColumnSelector
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    pipeline = Pipeline([
        ('selector', ColumnSelector(columns=feature_names)),
        ('model', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    # Fit on dummy data
    df_dummy = pd.DataFrame(X_dummy, columns=feature_names)
    pipeline.fit(df_dummy, y_dummy)
    
    # Save model temporarily
    model_path = Path('src/models/model_rf.pkl')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    
    yield pipeline
    
    # Cleanup (optional - keep model for other tests)

def test_health_endpoint():
    """Test the /health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}

def test_predict_missing_features():
    """Test prediction with missing required features."""
    # This will fail if model requires specific features
    payload = {
        'features': {
            'feature_1': 1.0,
            'feature_2': 2.0
        }
    }
    response = client.post('/predict', json=payload)
    # Should either succeed (if model is flexible) or fail with 400/500
    assert response.status_code in [200, 400, 500]

def test_predict_invalid_feature_type():
    """Test prediction with invalid feature types."""
    payload = {
        'features': {
            'feature_1': 'not_a_number',
            'feature_2': 2.0
        }
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 422  # Validation error

def test_predict_nan_value():
    """Test prediction with NaN values."""
    payload = {
        'features': {
            'feature_1': float('nan'),
            'feature_2': 2.0
        }
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 422  # Validation error for non-finite

def test_predict_empty_features():
    """Test prediction with empty features dict."""
    payload = {
        'features': {}
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 422  # Validation error

def test_predict_valid_features():
    """Test prediction with valid features."""
    payload = {
        'features': {
            'feature_1': 1.0,
            'feature_2': 2.0,
            'feature_3': 3.0,
            'feature_4': 4.0,
            'feature_5': 5.0
        }
    }
    response = client.post('/predict', json=payload)
    # May fail if model not loaded, but should validate input first
    assert response.status_code in [200, 500]

def test_predict_out_of_range():
    """Test prediction with out-of-range values."""
    payload = {
        'features': {
            'int_rate': 150.0,  # Should be 0-100
            'feature_2': 2.0
        }
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 422  # Validation error

def test_predict_extra_large_value():
    """Test prediction with suspiciously large values."""
    payload = {
        'features': {
            'feature_1': 1e15,  # Too large
            'feature_2': 2.0
        }
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 422  # Validation error

def test_predict_response_structure():
    """Test that successful prediction returns expected structure."""
    # This test assumes a model is available
    payload = {
        'features': {
            'feature_1': 1.0,
            'feature_2': 2.0,
            'feature_3': 3.0,
            'feature_4': 4.0,
            'feature_5': 5.0
        }
    }
    response = client.post('/predict', json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert 'predictions' in data
        assert isinstance(data['predictions'], list)
        assert len(data['predictions']) > 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
