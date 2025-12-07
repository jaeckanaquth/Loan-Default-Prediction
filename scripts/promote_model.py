"""
Script to promote models between stages (development -> staging -> production).

Usage:
    python scripts/promote_model.py staging
    python scripts/promote_model.py production
"""
import sys
import os
import mlflow
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_URI)

MODEL_NAME = "loan_default_prediction"
EXPERIMENT_NAME = "loan_default_prediction"


def get_latest_model_run():
    """Get the latest model run from MLflow."""
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            logger.error(f"Experiment '{EXPERIMENT_NAME}' not found")
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            logger.error("No runs found in experiment")
            return None
        
        return runs.iloc[0]
    except Exception as e:
        logger.error(f"Failed to get latest run: {e}")
        return None


def promote_model(stage: str):
    """
    Promote model to specified stage.
    
    Args:
        stage: Target stage ('staging' or 'production')
    """
    if stage not in ['staging', 'production']:
        logger.error(f"Invalid stage: {stage}. Must be 'staging' or 'production'")
        return False
    
    logger.info(f"Promoting model to {stage}...")
    
    # Get latest run
    run = get_latest_model_run()
    if run is None:
        return False
    
    run_id = run['run_id']
    logger.info(f"Using run ID: {run_id}")
    
    try:
        # Get model URI
        model_uri = f"runs:/{run_id}/model"
        
        # Register model if not already registered
        try:
            mv = mlflow.register_model(model_uri, MODEL_NAME)
            model_version = mv.version
            logger.info(f"Registered new model version: {model_version}")
        except Exception as e:
            # Model might already be registered, get latest version
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if model_versions:
                model_version = model_versions[0].version
                logger.info(f"Using existing model version: {model_version}")
            else:
                raise e
        
        # Transition model to stage
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Transition to staging
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version,
            stage=stage,
            archive_existing_versions=True  # Archive previous versions in this stage
        )
        
        logger.info(f"✅ Model version {model_version} promoted to {stage}")
        
        # Log promotion details
        model_version_details = client.get_model_version(MODEL_NAME, model_version)
        logger.info(f"Model URI: {model_version_details.source}")
        logger.info(f"Current stage: {model_version_details.current_stage}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python promote_model.py <staging|production>")
        sys.exit(1)
    
    stage = sys.argv[1].lower()
    success = promote_model(stage)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
