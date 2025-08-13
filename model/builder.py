import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  # More efficient than pickle for sklearn objects
import sys
print("=== SCRIPT STARTING ===")

@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    max_features: int
    ngram_range: Tuple[int, int]
    learning_rate: float
    max_depth: int
    n_estimators: int
    test_size: float = 0.2
    random_state: int = 42


class ModelBuilder:
    """Class to handle model building pipeline."""
    
    def __init__(self, config: ModelConfig, root_dir: Path):
        self.config = config
        self.root_dir = Path(root_dir)
        self.logger = self._setup_logger()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[lgb.LGBMClassifier] = None
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('model_building')
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Less verbose for console
        
        # File handler
        log_dir = self.root_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'model_building.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def load_params(params_path: Path) -> Dict[str, Any]:
        """Load parameters from a YAML file."""
        try:
            with open(params_path, 'r', encoding='utf-8') as file:
                params = yaml.safe_load(file)
            return params
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Parameters file not found: {params_path}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {params_path}: {e}") from e
    
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load and validate data from a CSV file."""
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['clean_comment', 'category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Handle missing values
            initial_shape = df.shape
            df = df.dropna(subset=['clean_comment', 'category'])
            df['clean_comment'] = df['clean_comment'].fillna('')
            
            self.logger.info(f"Data loaded from {file_path}")
            self.logger.info(f"Shape: {df.shape} (dropped {initial_shape[0] - df.shape[0]} rows with missing values)")
            
            return df
            
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV file {file_path}: {e}") from e
    
    def create_tfidf_features(self, train_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create TF-IDF features from text data."""
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                stop_words='english',  # Remove common English stop words
                lowercase=True,
                strip_accents='unicode',
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens with 2+ chars
            )
            
            X_train = train_data['clean_comment'].values
            y_train = train_data['category'].values
            
            # Fit and transform
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            
            self.logger.info(f"TF-IDF transformation complete. Shape: {X_train_tfidf.shape}")
            self.logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            
            # Save vectorizer
            vectorizer_path = self.root_dir / 'models' / 'tfidf_vectorizer.joblib'
            vectorizer_path.parent.mkdir(exist_ok=True)
            joblib.dump(self.vectorizer, vectorizer_path)
            self.logger.info(f"Vectorizer saved to {vectorizer_path}")
            
            return X_train_tfidf, y_train
            
        except Exception as e:
            self.logger.error(f"Error during TF-IDF transformation: {e}")
            raise
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMClassifier:
        """Train LightGBM model with validation."""
        try:
            # Split data for validation
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state,
                stratify=y_train
            )
            
            # Get number of unique classes
            n_classes = len(np.unique(y_train))
            self.logger.info(f"Training model with {n_classes} classes")
            
            self.model = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=n_classes,
                metric='multi_logloss',
                is_unbalance=True,
                class_weight='balanced',
                reg_alpha=0.1,
                reg_lambda=0.1,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                verbose=-1  # Suppress training output
            )
            
            # Train with early stopping
            self.model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Validate model
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            self.logger.info(f"Model training completed with validation accuracy: {accuracy:.4f}")
            self.logger.debug(f"Classification report:\n{classification_report(y_val, y_pred)}")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise
    
    def save_model(self, model_name: str = 'lgbm_model.joblib') -> Path:
        """Save the trained model."""
        try:
            if self.model is None:
                raise ValueError("No model to save. Train a model first.")
            
            models_dir = self.root_dir / 'models'
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / model_name
            
            joblib.dump(self.model, model_path)
            self.logger.info(f"Model saved to {model_path}")
            
            return model_path
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def run_pipeline(self) -> None:
        """Run the complete model building pipeline."""
        try:
            self.logger.info("Starting model building pipeline")
            
            # Load training data
            train_data_path = self.root_dir / 'data' / 'interim' / 'train_processed.csv'
            train_data = self.load_data(train_data_path)
            
            # Create features
            X_train_tfidf, y_train = self.create_tfidf_features(train_data)
            
            # Train model
            self.train_model(X_train_tfidf, y_train)
            
            # Save model
            self.save_model()
            
            self.logger.info("Model building pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


def get_root_directory() -> Path:
    """Get the root directory (two levels up from this script's location)."""
    return Path(__file__).resolve().parents[1] 



def create_config_from_params(params: Dict[str, Any]) -> ModelConfig:
    """Create ModelConfig from loaded parameters."""
    model_params = params.get('model_building', {})
    
    return ModelConfig(
        max_features=model_params.get('max_features', 10000),
        ngram_range=tuple(model_params.get('ngram_range', [1, 3])),
        learning_rate=model_params.get('learning_rate', 0.1),
        max_depth=model_params.get('max_depth', 6),
        n_estimators=model_params.get('n_estimators', 100),
        test_size=model_params.get('test_size', 0.2),
        random_state=model_params.get('random_state', 42)
    )

print("=== IMPORTS COMPLETED ===")
print("=== ABOUT TO CHECK IF __name__ == '__main__' ===")

def main():
    """Main function to run the model building process."""
    print("=== CALLING MAIN FUNCTION ===")

    try:
        # Get root directory
        root_dir = get_root_directory()
        print(f"[DEBUG] Resolved root directory: {root_dir}")
        sys.stdout.flush() 
        params_path = os.path.join(root_dir, "params.yaml")
        print(f"[DEBUG] Full params path: {params_path}")
        print(f"[DEBUG] Exists? {os.path.exists(params_path)}")

        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Parameters file not found: {params_path}")
        
        # Load parameters from YAML file
        import yaml
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            
        print(f"[DEBUG] Loaded params: {params}")  # Add this debug line
            
        # Create configuration
        config = create_config_from_params(params)
        print(f"[DEBUG] Created config: {config}")  # Add this debug line
        
        # Initialize and run model builder
        print("[DEBUG] Starting model builder...")  # Add this debug line
        model_builder = ModelBuilder(config, root_dir)
        model_builder.run_pipeline()
        print("[DEBUG] Model builder completed")  # Add this debug line
        
        # Check if output files exist
        model_path = os.path.join(root_dir, "models", "lgbm_model.joblib")
        vectorizer_path = os.path.join(root_dir, "models", "tfidf_vectorizer.joblib")
        print(f"[DEBUG] Model file exists: {os.path.exists(model_path)}")
        print(f"[DEBUG] Vectorizer file exists: {os.path.exists(vectorizer_path)}")
        
        print("✅ Model building completed successfully!")
        sys.stdout.flush() 
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.stdout.flush() 
        import traceback
        traceback.print_exc()  # Add full traceback for debugging
        raise
if __name__ == "__main__":
    print("=== SCRIPT EXECUTED AS MAIN ===")
    main()  