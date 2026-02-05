# Machine Learning Trading System
===============================

This module implements a comprehensive ML trading system integrated with the Interactive Brokers (IBKR) project.

## Components

### 1. Data Pipeline (`src/ml/data_pipeline.py`)
- Handles data loading (historical Parquet + IBKR fetching).
- Manages feature engineering integration.
- Splits data (Train/Val/Test).
- Scales data (StandardScaler).

### 2. Feature Engineering (`src/ml/features.py`)
- Generates Technical Indicators (RSI, MACD, Bollinger Bands, ATR, etc.) using `ta` library.
- Adds Lag features and Time-based features.

### 3. Models (`src/ml/models/`)
- `BaseModel`: Abstract base class.
- `EnhancedSMAModel`: Classifier (RandomForest/XGBoost/LightGBM) to filter SMA crossovers.
- `IntradayPredictor`: Regressor/Classifier (XGBoost) for short-term price prediction.
- `EnsembleModel`: Combines multiple models.

### 4. Training (`src/ml/training.py`)
- `MLTrainer`: Orchestrates the training workflow (Load -> Prepare -> Train -> Evaluate -> Save).

### 5. Inference Engine (`src/ml/ml_engine.py`)
- `MLEngine`: Loads trained models and serves predictions for live data.

### 6. Strategies (`src/strategies/ml/`)
- `EnhancedSMAStrategy`: Inherits `BaseStrategy`. Uses ML model to validate "Golden Cross" signals.
- `IntradayMLStrategy`: Inherits `BaseStrategy`. Uses ML prediction to trade intraday price movements.

### 7. Monitoring (`src/ml/monitoring.py`)
- `MLMonitor`: Basics for drift detection and health checks.

### 8. Backtesting (`src/ml/backtesting.py`)
- `MLBacktester`: Event-driven backtester for ML strategies.

## Usage

### Training
```python
from src.ml.training import MLTrainer
trainer = MLTrainer()
metrics = trainer.train_enhanced_sma("SPY", start_date="2020-01-01", end_date="2023-01-01")
print(metrics)
```

### Backtesting
```python
from src.ml.backtesting import MLBacktester
from src.ml.data_pipeline import MLDataPipeline
from src.strategies.ml.enhanced_sma_strategy import EnhancedSMAStrategy
# ... setup config ...
backtester = MLBacktester(MLDataPipeline())
results = backtester.run(EnhancedSMAStrategy, config, "SPY", ...)
```

### Dashboard
Run the dashboard:
```bash
streamlit run dashboard/app.py
```
Navigate to "ML Trading" tab to view live predictions.

## Dependencies
- plotting: matplotlib, seaborn, plotly
- ml: scikit-learn, xgboost, lightgbm
- data: pandas, numpy, ta
