import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, Any, Optional
from loguru import logger
from .features import FeatureEngineer
from .models.base_model import BaseModel

class MLEngine:
    """
    Serves ML predictions to trading strategies.
    Handles model loading, feature generation inference.
    """
    
    def __init__(self, config_path: str = "config/ml_config.yaml"):
        import yaml
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
            
        self.models_path = self.config.get("models_storage_path", "data/models")
        self.models: Dict[str, Any] = {} # Map "strategy_symbol" -> model
        self.feature_engineer = FeatureEngineer(self.config.get("features", {}))
        
        self.reload_models()
        
    def reload_models(self):
        """Load all available models from disk."""
        logger.info(f"Loading models from {self.models_path}")
        if not os.path.exists(self.models_path):
            logger.warning("Models directory does not exist.")
            return

        for filename in os.listdir(self.models_path):
            if filename.endswith(".joblib"):
                try:
                    path = os.path.join(self.models_path, filename)
                    data = joblib.load(path)
                    # We store the dictionary with 'model', 'config' etc.
                    # Or we can wrap it back into BaseModel, but for inference we just need the model object usually.
                    # But we implemented a custom save/load in BaseModel.
                    # Let's instantiate the generic wrapper or just store the raw model if sufficient.
                    
                    # Better: Instantiate the correct class based on metadata or name
                    model_name = data.get('model_name', 'unknown')
                    # This logic allows using the specific predict methods
                    
                    # For now, store the raw dict or object
                    # Key: filename minus extension? Or parsed?
                    # Format: enhanced_sma_SPY.joblib
                    key = filename.replace(".joblib", "")
                    self.models[key] = data
                    logger.info(f"Loaded model: {key} ({model_name})")
                    
                except Exception as e:
                    logger.error(f"Failed to load model {filename}: {e}")

    def get_prediction(self, model_key: str, data: pd.DataFrame) -> Dict:
        """
        Generate prediction for live data.
        
        Args:
            model_key: e.g. "enhanced_sma_SPY"
            data: Recent OHLCV data (DataFrame). Must be enough to generate features.
            
        Returns:
            Dict with 'prediction', 'probability', 'confidence'
        """
        if model_key not in self.models:
            return {"error": f"Model {model_key} not found"}
            
        model_data = self.models[model_key]
        raw_model = model_data['model']
        required_features = model_data.get('features_used', [])
        
        # 1. Generate Features
        # We process the whole dataframe, then take the last row
        try:
            df_features = self.feature_engineer.generate_features(data)
        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            return {"error": "Feature generation failed"}
            
        if df_features.empty:
             return {"error": "Not enough data for features"}
             
        # Take last row (current moment)
        current_features = df_features.iloc[[-1]] 
        
        # Ensure columns match
        # Fill missing with 0? Or error?
        # Ideally, feature engineer output matches training.
        # Filter to required
        try:
            input_vector = current_features[required_features]
        except KeyError as e:
            logger.error(f"Missing features: {e}")
            return {"error": f"Missing features: {e}"}
            
        # 2. Predict
        try:
            # check if classifier or regressor
            # loosely based on model type or existence of predict_proba
            is_classifier = hasattr(raw_model, "predict_proba")
            
            pred = raw_model.predict(input_vector)[0]
            result = {"prediction": pred}
            
            if is_classifier:
                probs = raw_model.predict_proba(input_vector)
                # Binary: prob of class 1
                prob_1 = probs[0][1] if len(probs[0]) > 1 else probs[0][0]
                result["probability"] = float(prob_1)
                result["confidence"] = float(abs(prob_1 - 0.5) * 2) # 0.5->0, 1.0->1, 0.0->1
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": f"Prediction failed: {e}"}
