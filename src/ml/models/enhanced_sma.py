from .base_model import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from loguru import logger

class EnhancedSMAModel(BaseModel):
    """
    Classification model to predict if an SMA crossover will be profitable.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config, model_name="enhanced_sma")
        self.model_type = self.config.get("model_type", "random_forest")
        
    def train(self, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray], **kwargs) -> Dict:
        self.features_used = X_train.columns.tolist()
        params = self.config.get(self.model_type, {})
        
        logger.info(f"Training {self.model_type} with input shape {X_train.shape}")
        
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 10),
                min_samples_split=params.get("min_samples_split", 2),
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
        elif self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                max_depth=params.get("max_depth", 6),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            self.model.fit(X_train, y_train)
            
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                num_leaves=params.get("num_leaves", 31),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            self.model.fit(X_train, y_train)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        # Training metrics
        y_pred = self.model.predict(X_train)
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred),
            "train_precision": precision_score(y_train, y_pred, zero_division=0),
            "train_recall": recall_score(y_train, y_pred, zero_division=0),
            "train_f1": f1_score(y_train, y_pred, zero_division=0)
        }
        logger.info(f"Training finished. Metrics: {metrics}")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        # Ensure features match
        if isinstance(X, pd.DataFrame):
            # Check if columns are present, fill missing with 0 or handle error
            # For simplicity, assume correct columns
            X = X[self.features_used]
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        if isinstance(X, pd.DataFrame):
            X = X[self.features_used]
        # Return probability of class 1 (Profitable)
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray]) -> Dict:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        }
        return metrics
