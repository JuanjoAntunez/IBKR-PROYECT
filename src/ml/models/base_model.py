from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Tuple
import joblib
import os
from loguru import logger

class BaseModel(ABC):
    """
    Abstract Base Class for all ML Trading Models.
    """
    
    def __init__(self, config: Dict = None, model_name: str = "base_model"):
        self.config = config or {}
        self.model_name = model_name
        self.model = None
        self.features_used = []
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray], **kwargs) -> Dict:
        """
        Train the model.
        Returns a dictionary of training metrics.
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Union[np.ndarray, pd.Series]:
        """
        Make predictions.
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make probability predictions (for classification).
        """
        pass
    
    def save(self, path: str):
        """
        Save the model and metadata.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'model': self.model,
            'config': self.config,
            'features_used': self.features_used,
            'model_name': self.model_name
        }
        joblib.dump(data, path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path: str):
        """
        Load the model and metadata.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        data = joblib.load(path)
        self.model = data['model']
        self.config = data['config']
        self.features_used = data['features_used']
        self.model_name = data.get('model_name', self.model_name)
        logger.info(f"Model loaded from {path}")

    def evaluate(self, X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Evaluate model performance.
        """
        pass
