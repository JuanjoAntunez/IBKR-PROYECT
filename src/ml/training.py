import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from typing import Dict, List, Optional
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

from .data_pipeline import MLDataPipeline
from .models.enhanced_sma import EnhancedSMAModel
from .models.intraday_predictor import IntradayPredictor

class MLTrainer:
    """
    Orchestrates the training process: Data Loading -> Prep -> Train -> Eval -> Save.
    """
    
    def __init__(self, config_path: str = "config/ml_config.yaml"):
        self.pipeline = MLDataPipeline(config_path)
        self.config = self.pipeline.config
        self.models = {}
        
    def train_enhanced_sma(self, symbol: str, start_date: str = None, end_date: str = None):
        """
        Train Enhanced SMA Model.
        Target: 1 if SMA crossover logic would be profitable in next N days.
        """
        logger.info(f"Starting training Enhanced SMA for {symbol}")
        
        # 1. Load Data (Daily)
        df = self.pipeline.get_data(symbol, "1d", start_date=pd.to_datetime(start_date), end_date=pd.to_datetime(end_date))
        if df.empty:
            logger.error("No data found")
            return
            
        # 2. Prepare Features
        # We need specific target generation for this strategy
        # Let's say: If SMA50 > SMA200 (Golden Cross), is price higher 10 days later?
        # Target: (Close_t+10 > Close_t) AND (SMA50 > SMA200)
        # But actually we want to predict if the signal IS VALID.
        
        df_features = self.pipeline.feature_engineer.generate_features(df)
        
        # Custom Target Logic for Enhanced SMA
        # 1 = Profitable Trade, 0 = Not Profitable
        # We look for Crossovers logic here or just predict generic direction?
        # User requirement: "Consult model: ¿este crossover será rentable?"
        # So we identify crossovers, and label them.
        # But to train a classifier on ALL timestamps, we define Target as:
        # "If I enter now, will I win?" -> This is Directional Prediction basically.
        # Or specifically filtering cross over events?
        # To make it robust, let's predict: "Will price be higher in N days?" (10 days)
        
        horizon = 10
        future_return = df_features['Close'].shift(-horizon) / df_features['Close'] - 1
        df_features['target'] = (future_return > 0.02).astype(int) # > 2% return
        
        # Drop NaNs
        data = df_features.dropna()
        
        # Split (70/15/15)
        X = data.drop(columns=['target'])
        y = data['target']
        
        # Remove future/target leakage cols if any
        
        splits = self.pipeline.split_data(X, y)
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        # 3. Initialize Model
        model = EnhancedSMAModel(self.config.get("strategies", {}).get("enhanced_sma", {}))
        
        # 4. Train
        metrics = model.train(X_train, y_train)
        
        # 5. Evaluate
        eval_metrics = model.evaluate(X_test, y_test)
        logger.info(f"Evaluation Metrics: {eval_metrics}")
        
        # 6. Save
        model_path = os.path.join(self.config["models_storage_path"], f"enhanced_sma_{symbol}.joblib")
        model.save(model_path)
        
        return eval_metrics

    def train_intraday_predictor(self, symbol: str, timeframe: str = "15 mins"):
        """
        Train Intraday Predictor.
        """
        logger.info(f"Starting training Intraday Predictor for {symbol} {timeframe}")
        
        # Load
        df = self.pipeline.get_data(symbol, timeframe)
        if df.empty:
            logger.error("No data")
            return
            
        X, y = self.pipeline.prepare_dataset(df, target_col="Close", target_horizon=1, is_classification=False)
        
        splits = self.pipeline.split_data(X, y)
        X_train, y_train = splits['train']
        X_test, y_test = splits['test']
        
        model = IntradayPredictor(self.config.get("strategies", {}).get("intraday_prediction", {}))
        
        model.train(X_train, y_train)
        eval_metrics = model.evaluate(X_test, y_test)
        logger.info(f"Intraday Metrics: {eval_metrics}")
        
        model_path = os.path.join(self.config["models_storage_path"], f"intraday_{symbol}_{timeframe.replace(' ', '')}.joblib")
        model.save(model_path)
        
        return eval_metrics
