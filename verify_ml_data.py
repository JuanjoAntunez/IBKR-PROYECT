
import pandas as pd
import numpy as np
from src.ml.features import FeatureEngineer
from src.ml.data_pipeline import MLDataPipeline

# Create dummy data
dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
df = pd.DataFrame({
    "Open": np.random.rand(300) * 100,
    "High": np.random.rand(300) * 105,
    "Low": np.random.rand(300) * 95,
    "Close": np.random.rand(300) * 100,
    "Volume": np.random.randint(1000, 10000, 300)
}, index=dates)

print("Original Data Schema:")
print(df.head())

# Test Feature Engineering
fe = FeatureEngineer()
df_features = fe.generate_features(df)
print("\nFeature Engineering Result:")
print(f"Shape: {df_features.shape}")
print(f"Columns: {df_features.columns.tolist()}")

# Test Data Pipeline
pipeline = MLDataPipeline()
# Save dummy data
pipeline.save_data(df, "TEST", "1d")

# Load data
df_loaded = pipeline.get_data("TEST", "1d")
print(f"\nLoaded Data Shape: {df_loaded.shape}")

# Prepare
X, y = pipeline.prepare_dataset(df_loaded, target_col="Close", target_horizon=1)
print(f"\nPrepared X Shape: {X.shape}, y Shape: {y.shape}")

# Split
splits = pipeline.split_data(X, y)
print(f"\nTrain sizes - X: {splits['train'][0].shape}, y: {splits['train'][1].shape}")

print("\nVerification Passed!")
