from .base_model import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger

class IntradayPredictor(BaseModel):
    """
    Intraday price/direction predictor using XGBoost.
    (LSTM removed to avoid tensorflow/pytorch dependency issues in this env for now, 
    XGBoost is state of the art for tabular time series anyway).
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config, model_name="intraday_predictor")
        self.mode = self.config.get("prediction_mode", "regression") # or classification
        
    def train(self, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray], **kwargs) -> Dict:
        self.features_used = X_train.columns.tolist()
        params = self.config.get("xgboost", {})
        
        logger.info(f"Training Intraday XGBoost ({self.mode}) with input shape {X_train.shape}")
        
        if self.mode == "classification":
            self.model = xgb.XGBClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.05),
                max_depth=params.get("max_depth", 6),
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.05),
                max_depth=params.get("max_depth", 6),
                random_state=42,
                n_jobs=-1
            )
            
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_train)
        
        if self.mode == "classification":
            from sklearn.metrics import accuracy_score
            metrics = {"train_accuracy": accuracy_score(y_train, y_pred)}
        else:
            metrics = {
                "train_mse": mean_squared_error(y_train, y_pred),
                "train_mae": mean_absolute_error(y_train, y_pred),
                "train_r2": r2_score(y_train, y_pred)
            }
            
        logger.info(f"Training finished. Metrics: {metrics}")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
             raise ValueError("Model not trained")
        if isinstance(X, pd.DataFrame):
            X = X[self.features_used]
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.mode != "classification":
            raise ValueError("predict_proba only available for classification mode")
        if self.model is None:
             raise ValueError("Model not trained")
        if isinstance(X, pd.DataFrame):
             X = X[self.features_used]
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray]) -> Dict:
        y_pred = self.predict(X_test)
        
        if self.mode == "classification":
            from sklearn.metrics import accuracy_score, precision_score
            return {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0)
            }
        else:
            return {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred)
            }
