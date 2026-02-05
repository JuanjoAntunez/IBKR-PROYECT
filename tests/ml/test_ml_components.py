import pytest
import pandas as pd
import numpy as np
import os
from src.ml.features import FeatureEngineer
from src.ml.data_pipeline import MLDataPipeline
from src.ml.models.enhanced_sma import EnhancedSMAModel
from src.ml.models.intraday_predictor import IntradayPredictor

@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
    df = pd.DataFrame({
        "Open": np.random.rand(300) * 100,
        "High": np.random.rand(300) * 105,
        "Low": np.random.rand(300) * 95,
        "Close": np.random.rand(300) * 100,
        "Volume": np.random.randint(1000, 10000, 300)
    }, index=dates)
    return df

def test_feature_engineering_structure(sample_data):
    fe = FeatureEngineer()
    df_features = fe.generate_features(sample_data)
    assert not df_features.empty
    assert "SMA_10" in df_features.columns
    assert "RSI_14" in df_features.columns
    # SMA 200 will cause drops, so if len is 100, we might strictly have 0 rows if dropna=True
    # Sample has 100 rows. SMA 200 needs 200.
    # So DataFrame will be empty if we requested SMA 200 and dropna=True exists.
    # FeatureEngineer config default generates SMA200.
    # Let's verify behavior.
    
def test_data_pipeline_split(sample_data):
    pipeline = MLDataPipeline()
    # Dummy save
    pipeline.save_data(sample_data, "TEST_SYM", "1d")
    
    # Reload
    df = pipeline.get_data("TEST_SYM", "1d")
    assert len(df) == 300
    
    X, y = pipeline.prepare_dataset(df, target_col="Close")
    splits = pipeline.split_data(X, y)
    
    assert len(splits['train'][0]) > 0
    assert len(splits['val'][0]) > 0
    assert len(splits['test'][0]) > 0

def test_enhanced_sma_model_flow():
    # Create valid training data
    X_train = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
    y_train = np.random.randint(0, 2, 50)
    
    model = EnhancedSMAModel({"model_type": "random_forest"})
    metrics = model.train(X_train, y_train)
    
    assert "train_accuracy" in metrics
    
    # Predict
    preds = model.predict(X_train)
    assert len(preds) == 50
    
def test_intraday_model_flow():
    X_train = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
    y_train = np.random.rand(50) # Regression
    
    model = IntradayPredictor({"prediction_mode": "regression"})
    metrics = model.train(X_train, y_train)
    
    assert "train_mse" in metrics
    
    y_pred = model.predict(X_train)
    assert len(y_pred) == 50
