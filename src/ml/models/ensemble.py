from .base_model import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, List, Union
from loguru import logger

class EnsembleModel(BaseModel):
    """
    Combines multiple models.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config, model_name="ensemble")
        self.models: List[BaseModel] = []
        self.weights = []
        
    def add_model(self, model: BaseModel, weight: float = 1.0):
        self.models.append(model)
        self.weights.append(weight)
        
    def train(self, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray], **kwargs) -> Dict:
        # Ensemble usually trains sub-models or just aggregates.
        # Here we assume sub-models are already trained or we train them here.
        metrics = {}
        for i, model in enumerate(self.models):
            logger.info(f"Training sub-model {i}: {model.model_name}")
            m_metrics = model.train(X_train, y_train, **kwargs)
            for k, v in m_metrics.items():
                metrics[f"model_{i}_{k}"] = v
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("No models in ensemble")
            
        preds = []
        for model in self.models:
            preds.append(model.predict(X))
            
        # Weighted Average for regression, Voting for classification
        # We need to know mode. Assume regression for direct predict average
        # Or if classification, use predict_proba average then threshold
        
        # Simple majority vote for classification if output is class
        preds = np.array(preds) # shape: (n_models, n_samples)
        
        # Weighted average?
        # For simplicity: mode (voting)
        from scipy import stats
        final_pred, count = stats.mode(preds, axis=0) # This works for classes
        return final_pred[0]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("No models in ensemble")
            
        probs = []
        total_weight = sum(self.weights)
        
        for i, model in enumerate(self.models):
            p = model.predict_proba(X)
            probs.append(p * self.weights[i])
            
        avg_prob = np.sum(probs, axis=0) / total_weight
        return avg_prob

    def evaluate(self, X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray]) -> Dict:
        # Similar to base evaluation
        # Simplified for now
        return {}
