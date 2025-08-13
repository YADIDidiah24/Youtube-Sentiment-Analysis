import json
import mlflow
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Configuration
MLFLOW_TRACKING_URI = "http://ec2-44-201-207-151.compute-1.amazonaws.com:5000/"
MODEL_NAME = "yt_chrome_plugin_model"
DEFAULT_STAGE = "Staging"

# Logging configuration
def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('model_registration')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler('model_registration.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def load_experiment_info(file_path: str = 'experiment_info.json') -> Dict:
    """Load experiment information from JSON file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Experiment info file not found: {file_path}")
        
        with open(file_path, 'r') as file:
            experiment_info = json.load(file)
        
        # Validate required fields
        required_fields = ['run_id', 'model_uri', 's3_model_path']
        missing_fields = [field for field in required_fields if field not in experiment_info]
        
        if missing_fields:
            raise ValueError(f"Missing required fields in experiment info: {missing_fields}")
        
        logger.info(f"Experiment info loaded from {file_path}")
        logger.info(f"Run ID: {experiment_info['run_id']}")
        logger.info(f"Model URI: {experiment_info['model_uri']}")
        
        return experiment_info
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading experiment info: {e}")
        raise

def validate_model_exists(model_uri: str) -> bool:
    """Validate that the model exists in MLflow."""
    try:
        # Try to load model metadata to verify it exists
        mlflow.artifacts.download_artifacts(model_uri)
        logger.info(f"Model validated successfully: {model_uri}")
        return True
    except Exception as e:
        logger.warning(f"Could not validate model existence: {e}")
        return False

def register_model_version(model_name: str, experiment_info: Dict, 
                         stage: str = DEFAULT_STAGE, 
                         description: Optional[str] = None) -> object:
    """Register a new model version to MLflow Model Registry."""
    try:
        model_uri = experiment_info['model_uri']
        
        # Create description if not provided
        if description is None:
            timestamp = experiment_info.get('timestamp', datetime.now().isoformat())
            model_type = experiment_info.get('model_type', 'Unknown')
            description = f"Model registered on {timestamp} - Type: {model_type}"
        
        # Register the model
        logger.info(f"Registering model: {model_name}")
        logger.info(f"Model URI: {model_uri}")
        
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name)
        
        logger.info(f"Model {model_name} version {model_version.version} registered successfully")
        
        # Transition to specified stage
        if stage:
            transition_model_stage(model_name, model_version.version, stage)
        
        return model_version
        
    except Exception as e:
        logger.error(f"Error during model registration: {e}")
        raise

def transition_model_stage(model_name: str, version: str, stage: str) -> None:
    """Transition model version to specified stage."""
    try:
        client = mlflow.tracking.MlflowClient()
        
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=False  # Keep existing versions in stage
        )
        
        logger.info(f"Model {model_name} version {version} transitioned to '{stage}' stage")
        
    except Exception as e:
        logger.error(f"Error transitioning model to {stage}: {e}")
        raise

def add_model_tags(model_name: str, version: str, experiment_info: Dict) -> None:
    """Add relevant tags to the registered model version."""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Define tags from experiment info
        tags = {
            "experiment_id": experiment_info.get('experiment_id', ''),
            "s3_bucket": experiment_info.get('s3_bucket', ''),
            "model_type": experiment_info.get('model_type', ''),
            "registration_date": datetime.now().strftime('%Y-%m-%d'),
            "source": "dvc_pipeline"
        }
        
        # Add tags to model version
        for key, value in tags.items():
            if value:  # Only add non-empty tags
                client.set_model_version_tag(model_name, version, key, str(value))
        
        logger.info(f"Tags added to model {model_name} version {version}")
        
    except Exception as e:
        logger.warning(f"Could not add tags to model: {e}")

def update_experiment_info_with_registration(experiment_info: Dict, model_version: object, 
                                           file_path: str = 'experiment_info.json') -> None:
    """Update experiment info file with registration details."""
    try:
        # Add registration info
        experiment_info.update({
            'registered_model_name': model_version.name,
            'registered_model_version': model_version.version,
            'registration_timestamp': datetime.now().isoformat(),
            'model_stage': DEFAULT_STAGE,
            'model_registry_uri': f"models:/{model_version.name}/{model_version.version}"
        })
        
        # Save updated info
        with open(file_path, 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        logger.info(f"Experiment info updated with registration details")
        
    except Exception as e:
        logger.warning(f"Could not update experiment info file: {e}")

def main():
    """Main function to register model."""
    global logger
    logger = setup_logging()
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
        
        # Load experiment information
        experiment_info = load_experiment_info()
        
        # Validate model exists (optional check)
        if not validate_model_exists(experiment_info['model_uri']):
            logger.warning("Model validation failed, but proceeding with registration...")
        
        # Register model
        model_version = register_model_version(
            model_name=MODEL_NAME,
            experiment_info=experiment_info,
            stage=DEFAULT_STAGE
        )
        
        # Add tags to model version
        add_model_tags(MODEL_NAME, model_version.version, experiment_info)
        
        # Update experiment info with registration details
        update_experiment_info_with_registration(experiment_info, model_version)
        
        # Success message
        print(f"✅ Model registration completed successfully!")
        print(f"📝 Model Name: {MODEL_NAME}")
        print(f"🔢 Version: {model_version.version}")
        print(f"🏷️  Stage: {DEFAULT_STAGE}")
        print(f"🔗 MLflow UI: {MLFLOW_TRACKING_URI}/#/models/{MODEL_NAME}")
        print(f"☁️  S3 Path: {experiment_info.get('s3_model_path', 'N/A')}")
        
        logger.info("Model registration process completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to complete model registration: {e}")
        print(f"❌ Registration failed: {e}")
        raise

if __name__ == '__main__':
    main()