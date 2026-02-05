import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict, Union, Callable
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import yaml
import joblib

from .features import FeatureEngineer

class MLDataPipeline:
    """
    Manages data loading, caching, cleaning, feature engineering, and splitting.
    """
    
    def __init__(self, config_path: str = "config/ml_config.yaml", feature_engineer: FeatureEngineer = None):
        """
        Initialize the data pipeline.
        
        Args:
            config_path: Path to ML config file.
            feature_engineer: Instance of FeatureEngineer. If None, creates a new one.
        """
        self.config = self._load_config(config_path)
        self.feature_engineer = feature_engineer or FeatureEngineer(self.config.get("features", {}))
        
        self.data_path = self.config.get("data_storage_path", "data/history")
        os.makedirs(self.data_path, exist_ok=True)
        
        self.scaler = None
        
    def _load_config(self, path: str) -> Dict:
        """Load configuration from YAML."""
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)
        return {}
        
    def get_data(self, 
                 symbol: str, 
                 timeframe: str, 
                 start_date: Optional[datetime] = None, 
                 end_date: Optional[datetime] = None,
                 fetch_func: Optional[Callable] = None,
                 force_refresh: bool = False) -> pd.DataFrame:
        """
        Get data for a symbol/timeframe. Checks cache first.
        
        Args:
            symbol: Ticker symbol.
            timeframe: Bar size (e.g. '1 day', '5 mins').
            start_date: Start date filter.
            end_date: End date filter.
            fetch_func: Callback/Function to fetch data from IBKR if not in cache.
                        Should accept (symbol, timeframe, duration) and return DataFrame.
            force_refresh: Ignore cache and re-fetch.
            
        Returns:
            DataFrame with OHLCV data.
        """
        file_path = self._get_file_path(symbol, timeframe)
        df = pd.DataFrame()
        
        # 1. Try Load from Disk
        if os.path.exists(file_path) and not force_refresh:
            try:
                df = pd.read_parquet(file_path)
                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"])
                        df.set_index("Date", inplace=True)
                
                logger.info(f"Loaded {len(df)} rows from cache for {symbol} {timeframe}")
            except Exception as e:
                logger.error(f"Error reading cache {file_path}: {e}")
                df = pd.DataFrame() # Reset on error
        
        # 2. Check if we need to fetch new data
        # Logic: If df is empty OR recent data missing (based on end_date or now)
        # For simplicity, if force_refresh or empty, we fetch.
        # In a real scenario, we might append new data.
        
        need_fetch = df.empty or force_refresh
        # Or check if "fresh enough" (e.g. has yesterday's data)
        if not df.empty and not force_refresh:
            last_date = df.index[-1]
            if end_date:
                # If we asked for data up to end_date, and we have less than that
                if last_date < end_date and (end_date - last_date).days > 1:
                    need_fetch = True
            else:
                # If no end_date specified, assume we want up to now
                now = datetime.now()
                # If usage is live trading, we might not use this pipeline for real-time bars anyway
                # But for training, if data is > 1 week old, refresh?
                if (now - last_date).days > 7:
                     need_fetch = True

        if need_fetch and fetch_func:
            logger.info(f"Fetching fresh data for {symbol} {timeframe}...")
            # Calculate duration string for IBKR
            # Simple heuristic: 2 years for Daily, 1 month for Intraday if not specified
            duration = "2 Y" if "d" in timeframe.lower() else "1 M"
            # If start_date provided, calculate duration more precisely if possible
            
            new_data = fetch_func(symbol, timeframe, duration)
            
            if new_data is not None and not new_data.empty:
                # Merge with existing or replace
                if not df.empty and not force_refresh:
                    # Combine and drop duplicates
                    df = pd.concat([df, new_data])
                    df = df[~df.index.duplicated(keep='last')]
                    df.sort_index(inplace=True)
                else:
                    df = new_data
                
                # Save to cache
                self.save_data(df, symbol, timeframe)
        
        if df.empty:
            logger.warning(f"No data available for {symbol} {timeframe}")
            return df
            
        # Filter by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        return df

    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save DataFrame to parquet."""
        file_path = self._get_file_path(symbol, timeframe)
        df.to_parquet(file_path)
        logger.info(f"Saved cache to {file_path}")

    def _get_file_path(self, symbol: str, timeframe: str) -> str:
        """Generate file path for storage."""
        safe_tf = timeframe.replace(" ", "_")
        return os.path.join(self.data_path, f"{symbol}_{safe_tf}.parquet")

    def prepare_dataset(self, 
                       df: pd.DataFrame, 
                       target_col: str = None, 
                       target_horizon: int = 1,
                       is_classification: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Full preparation: Feature Engineering -> Clean -> Target Gen.
        
        Returns:
            X (features), y (target)
        """
        if df.empty:
            return pd.DataFrame(), pd.Series()
            
        # 1. Feature Engineering
        df_features = self.feature_engineer.generate_features(df)
        
        # 2. Target Generation
        if target_col:
            # If we are predicting price direction/value based on a future column
            # For now, let's assume we predict 'Close'
            target_src = df_features['Close']
            
            if is_classification:
                # Example: 1 if Returns > 0, else 0
                future_returns = target_src.shift(-target_horizon) / target_src - 1
                y = (future_returns > 0).astype(int)
            else:
                # Regression: Future Price
                y = target_src.shift(-target_horizon)
        else:
            # Default target handling or provided internally
             y = pd.Series(index=df_features.index, data=0) # Placeholder

        # Remove NaNs created by lagging/shifting
        combined = pd.concat([df_features, y.rename('target')], axis=1).dropna()
        
        X = combined.drop(columns=['target'])
        y = combined['target']
        
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Dict:
        """
        Temporal split of data.
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }

    def scale_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, method: str = 'standard'):
        """
        Fit scaler on Train, transform Val/Test.
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            return X_train, X_val, X_test
            
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(self.scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        
        return X_train_scaled, X_val_scaled, X_test_scaled

    def save_scaler(self, path: str):
        if self.scaler:
            joblib.dump(self.scaler, path)
            
    def load_scaler(self, path: str):
        if os.path.exists(path):
            self.scaler = joblib.load(path)
