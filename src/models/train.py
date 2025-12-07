import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import joblib
from pathlib import Path
from src.features.transformers import ColumnSelector, MedianImputer

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_URI)

def train():
    """Train model with preprocessing pipeline."""
    df_path = 'data/processed/loan.csv'
    if not os.path.exists(df_path):
        print('Processed data not found at', df_path)
        return
    
    df = pd.read_csv(df_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Get feature column names
    feature_cols = X.columns.tolist()
    print(f'Training with {len(feature_cols)} features: {feature_cols}')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline with preprocessing + model
    # Use RandomForest as default (can be changed)
    model_type = os.getenv('MODEL_TYPE', 'random_forest')
    
    if model_type.lower() == 'logistic':
        estimator = LogisticRegression(max_iter=200, random_state=42)
    else:
        estimator = RandomForestClassifier(n_estimators=200, random_state=42)
    
    # Build pipeline: ColumnSelector -> MedianImputer -> Model
    # Note: ColumnSelector ensures correct feature order
    pipeline = Pipeline([
        ('selector', ColumnSelector(columns=feature_cols)),
        ('imputer', MedianImputer()),
        ('model', estimator)
    ])
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    acc = pipeline.score(X_test, y_test)
    print(f'Training done. Accuracy: {acc:.4f}')
    
    # Save pipeline to joblib
    model_dir = Path('src/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'model_rf.pkl'
    joblib.dump(pipeline, model_path)
    print(f'Saved pipeline to {model_path.resolve()}')
    
    # Log to MLflow
    try:
        with mlflow.start_run():
            mlflow.log_metric('accuracy', acc)
            mlflow.log_param('n_features', len(feature_cols))
            mlflow.log_param('model_type', model_type)
            mlflow.sklearn.log_model(pipeline, 'model')
            print('Logged model to MLflow')
    except Exception as e:
        print(f'Warning: MLflow logging failed: {e}')
        print('Model saved locally but not logged to MLflow')

if __name__ == '__main__':
    train()
