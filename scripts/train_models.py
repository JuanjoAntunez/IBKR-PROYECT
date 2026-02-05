import yfinance as yf
from src.ml.training import MLTrainer
from src.ml.data_pipeline import MLDataPipeline
import pandas as pd
import os
from loguru import logger

def fetch_yfinance_data(symbol, period="2y"):
    """
    Fetch data from Yahoo Finance and format it for our pipeline.
    """
    logger.info(f"Downloading {symbol} data from Yahoo Finance...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    if df.empty:
        logger.error("No data downloaded")
        return None
        
    # Simplify columns and timezone
    df = df.reset_index()
    # Ensure Date is timezone naive or consistent
    df['Date'] = df['Date'].dt.tz_localize(None)
    
    # Rename columns to match what MLDataPipeline expects (Open, High, Low, Close, Volume)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df

def main():
    symbol = "AAPL"
    logger.info(f"Starting auto-training for {symbol}...")
    
    # 1. Fetch Data
    df = fetch_yfinance_data(symbol)
    if df is None:
        return
        
    # 2. Save directly to cache manualy so pipeline picks it up
    cache_dir = "data/history"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}_1d.parquet")
    df.to_parquet(cache_path)
    logger.info(f"Data saved to {cache_path}")
    
    # 3. Initialize Trainer
    trainer = MLTrainer()
    
    # 4. Train Enhanced SMA
    logger.info("Training Enhanced SMA Model...")
    metrics_sma = trainer.train_enhanced_sma(symbol)
    logger.info(f"Enhanced SMA Metrics: {metrics_sma}")
    
    # 5. Train Intraday
    # Note: Yahoo daily data might not work great for '15 mins' intraday model unless we fetch intraday
    # Yahoo supports max 60 days of intraday data.
    logger.info("Downloading Intraday data (15m)...")
    df_intra = yf.Ticker(symbol).history(period="60d", interval="15m")
    if not df_intra.empty:
        df_intra = df_intra.reset_index()
        # Rename Datetime to Date if present
        if 'Datetime' in df_intra.columns:
            df_intra.rename(columns={'Datetime': 'Date'}, inplace=True)
            
        df_intra['Date'] = df_intra['Date'].dt.tz_localize(None) # Removing timezone
        df_intra = df_intra[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Save cache
        # Note: Pipeline expects "15 mins" or "15min"?
        # Trainer calls pipeline.get_data(symbol, timeframe)
        # Dashboard requests "15min".
        # Let's support "15min".
        
        cache_path_intra = os.path.join(cache_dir, f"{symbol}_15min.parquet")
        df_intra.to_parquet(cache_path_intra)
        
        logger.info("Training Intraday Predictor...")
        metrics_intra = trainer.train_intraday_predictor(symbol, timeframe="15min")
        logger.info(f"Intraday Metrics: {metrics_intra}")
    else:
        logger.warning("Could not fetch intraday data")

if __name__ == "__main__":
    main()
