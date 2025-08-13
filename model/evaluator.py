import numpy as np
import pandas as pd
import json
import yaml
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from mlflow.models import infer_signature
import os

# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load and clean test data."""
    try:
        df = pd.read_csv(file_path)
        
        # Basic validation and cleaning
        required_columns = ['clean_comment', 'category']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        
        df = df.dropna(subset=required_columns)
        df.fillna('', inplace=True)
        logger.info(f"Test data loaded: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        vectorizer = joblib.load(vectorizer_path)
        logger.info(f"Vectorizer loaded from {vectorizer_path}")
        return vectorizer
    except Exception as e:
        logger.error(f"Error loading vectorizer from {vectorizer_path}: {e}")
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters from {params_path}: {e}")
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and return metrics."""
    try:
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        return accuracy, report, cm
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def log_confusion_matrix(cm, dataset_name="Test Data"):
    """Create and log confusion matrix visualization."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            square=True
        )
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save and log to MLflow
        cm_file_path = f'confusion_matrix_{dataset_name.replace(" ", "_").lower()}.png'
        plt.savefig(cm_file_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(cm_file_path)
        plt.close()
        
        logger.info(f"Confusion matrix saved and logged: {cm_file_path}")
    except Exception as e:
        logger.error(f"Error creating confusion matrix: {e}")
        raise


def save_experiment_info(run_id: str, experiment_name: str, mlflow_tracking_uri: str, file_path: str = 'experiment_info.json') -> None:
    """Save experiment information to JSON file (required by DVC)."""
    try:
        # Get MLflow experiment ID for S3 path construction
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id if experiment else "unknown"
        
        # Construct S3 paths based on mlflow-yadi bucket structure
        model_uri = f"runs:/{run_id}/lgbm_model"
        s3_model_path = f"s3://mlflow-yadi/{experiment_id}/{run_id}/artifacts/lgbm_model"
        s3_artifacts_path = f"s3://mlflow-yadi/{experiment_id}/{run_id}/artifacts"
        
        experiment_info = {
            'run_id': run_id,
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'mlflow_tracking_uri': mlflow_tracking_uri,
            'experiment_name': experiment_name,
            'model_type': 'LightGBM',
            'model_uri': model_uri,
            's3_model_path': s3_model_path,
            's3_artifacts_path': s3_artifacts_path,
            's3_bucket': 'mlflow-yadi',
            'model_version': f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        with open(file_path, 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        logger.info(f"Experiment info saved to {file_path}")
        logger.info(f"Model stored at S3: {s3_model_path}")
        logger.info(f"Experiment folder: {experiment_id}")
    except Exception as e:
        logger.error(f"Error saving experiment info: {e}")
        raise


def get_root_directory() -> str:
    """Get the root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))


def main():
    """Main function to run model evaluation."""
    # MLflow configuration
    mlflow_tracking_uri = "http://ec2-44-201-207-151.compute-1.amazonaws.com:5000/"
    experiment_name = "dvc-pipeline-runs"
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        try:
            logger.info("Starting evaluation pipeline")
            
            # Get root directory
            root_dir = get_root_directory()
            
            # Load parameters
            params_path = os.path.join(root_dir, 'params.yaml')
            params = load_params(params_path)
            
            # Log basic parameters
            mlflow.log_param("model_type", "LightGBM")
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            # Load model artifacts
            models_dir = os.path.join(root_dir, 'models')
            model_path = os.path.join(models_dir, 'lgbm_model.joblib')
            vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
            
            model = load_model(model_path)
            vectorizer = load_vectorizer(vectorizer_path)
            
            # Load and prepare test data
            test_data_path = os.path.join(root_dir, 'data', 'interim', 'test_processed.csv')
            test_data = load_data(test_data_path)
            
            X_test = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values
            
            # Create input example for model signature
            input_example = pd.DataFrame(
                X_test.toarray()[:5], 
                columns=vectorizer.get_feature_names_out()
            )
            signature = infer_signature(input_example, model.predict(X_test[:5]))
            
            # Log model with signature
            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,
                input_example=input_example
            )
            
            # Log vectorizer as artifact
            mlflow.log_artifact(vectorizer_path)
            
            # Evaluate model
            accuracy, report, cm = evaluate_model(model, X_test, y_test)
            
            # Log key metrics only
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("macro_avg_f1", report['macro avg']['f1-score'])
            mlflow.log_metric("weighted_avg_f1", report['weighted avg']['f1-score'])
            
            # Log classification report metrics for each class
            for label, metrics in report.items():
                if isinstance(metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })
            
            # Create and log confusion matrix
            log_confusion_matrix(cm, "Test Data")
            
            # Set tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")
            mlflow.set_tag("evaluation_stage", "test")
            
            # Save experiment info (required by DVC)
            save_experiment_info(run.info.run_id, experiment_name, mlflow_tracking_uri)
            
            logger.info("Evaluation completed successfully")
            print(f"✅ Evaluation completed!")
            print(f"📊 MLflow Run ID: {run.info.run_id}")
            print(f"🔗 MLflow UI: {mlflow_tracking_uri}")
            
        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"❌ Error: {e}")
            raise


if __name__ == '__main__':
    main()