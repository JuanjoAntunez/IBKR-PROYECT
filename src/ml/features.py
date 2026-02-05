import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from loguru import logger

from src.utils.market_data import normalize_ohlcv

# Import specific indicators from 'ta' library
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

class FeatureEngineer:
    """
    Generates technical and derived features from OHLCV data using 'ta' library.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config: Configuration dictionary for features. If None, uses defaults.
        """
        self.config = config or {}
        self.feature_names = []
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all configured features.
        
        Args:
            df: DataFrame with at least 'Date', 'Open', 'High', 'Low', 'Close', 'Volume' columns.
            
        Returns:
            DataFrame with added features.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to generate_features")
            return df
            
        # Normalize OHLCV columns and ensure datetime index
        df = normalize_ohlcv(df, schema="upper", set_index=True)
                
        # Basic validation
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"DataFrame missing required columns. Has: {df.columns.tolist()}")
            raise ValueError(f"DataFrame must contain {required_cols}")

        initial_cols = df.columns.tolist()
        
        # 1. Technical Indicators
        df = self._add_technical_indicators(df)
        
        # 2. Volume Features
        df = self._add_volume_features(df)
        
        # 3. Time-based Features
        df = self._add_time_features(df)
        
        # 4. Lag Features (if configured)
        df = self._add_lag_features(df)
        
        # Update feature names list
        self.feature_names = [c for c in df.columns if c not in initial_cols]
        
        # Drop NaNs created by indicators (e.g. SMA200 needs 200 data points)
        if self.config.get('dropna', True):
            df.dropna(inplace=True)
            
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standard technical indicators using ta library."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Trend
        # SMA
        for length in [10, 20, 50, 200]:
            indicator = SMAIndicator(close=close, window=length)
            df[f'SMA_{length}'] = indicator.sma_indicator()
            
        # EMA
        for length in [12, 26]:
            indicator = EMAIndicator(close=close, window=length)
            df[f'EMA_{length}'] = indicator.ema_indicator()
            
        # ADX
        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        df['ADX_14'] = adx.adx()
        
        # Momentum
        # RSI
        rsi = RSIIndicator(close=close, window=14)
        df['RSI_14'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df['MACD_12_26_9'] = macd.macd()
        df['MACD_SIGNAL_12_26_9'] = macd.macd_signal()
        df['MACD_HIST_12_26_9'] = macd.macd_diff()
        
        # Stochastic
        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        df['STOCH_k_14_3_3'] = stoch.stoch()
        df['STOCH_d_14_3_3'] = stoch.stoch_signal()
        
        # ROC (Rate of Change)
        roc = ROCIndicator(close=close, window=10)
        df['ROC_10'] = roc.roc()
        
        # Volatility
        # BBands
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df['BBL_20_2.0'] = bb.bollinger_lband()
        df['BBM_20_2.0'] = bb.bollinger_mavg()
        df['BBU_20_2.0'] = bb.bollinger_hband()
        
        # ATR
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        df['ATR_14'] = atr.average_true_range()
        
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # OBV
        obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
        df['OBV'] = obv.on_balance_volume()
        
        # Volume SMA
        vol_sma = SMAIndicator(close=df['Volume'], window=20)
        df['VOL_SMA_20'] = vol_sma.sma_indicator()
        
        # Volume Ratio
        if 'VOL_SMA_20' in df.columns:
            # Avoid division by zero
            df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA_20'].replace(0, np.nan)
            
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar/time features."""
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['dayofmonth'] = df.index.day
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged values of price and returns."""
        lags = self.config.get('lags', [1, 2, 3, 5])
        
        if not lags:
            return df
            
        # Log Returns
        # Use numpy for safe log
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        for lag in lags:
            df[f'log_ret_lag_{lag}'] = df['log_ret'].shift(lag)
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            
        return df

    def validate_features(self, df: pd.DataFrame) -> bool:
        """Check for infinite values or remaining NaNs."""
        if df.isnull().values.any():
            # It's common to have NaNs after indicator calculation if not dropped
            # But generate_features drops them by default.
            logger.warning(f"Features contain NaNs: {df.isnull().sum()[df.isnull().sum() > 0]}")
            return False
            
        if np.isinf(df.select_dtypes(include=np.number)).values.any():
            logger.warning("Features contain Infinite values")
            return False
            
        return True
