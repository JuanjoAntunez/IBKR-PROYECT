import pandas as pd
from typing import Dict, Type, Optional, Callable
from .data_pipeline import MLDataPipeline
from src.strategies.base import BaseStrategy, StrategyConfig
from loguru import logger

class MLBacktester:
    """
    Backtester specific for ML strategies.
    Simulates the strategy loop over historical data.
    """
    
    def __init__(self, pipeline: MLDataPipeline):
        self.pipeline = pipeline
        
    def run(
        self,
        strategy_class: Type[BaseStrategy],
        config: StrategyConfig,
        symbol: str,
        start_date: str,
        end_date: str,
        ml_engine,
        fetch_func: Optional[Callable] = None,
        force_refresh: bool = False,
    ):
        
        logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        
        # 1. Load Data
        df = self.pipeline.get_data(
            symbol,
            "1d",
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            fetch_func=fetch_func,
            force_refresh=force_refresh,
        )
        
        # 2. Initialize Strategy
        # Strategy expects MLEngine
        strategy = strategy_class(config, ml_engine)
        strategy.start()
        
        # 3. Loop
        equity = 100000
        position = 0
        trades = []
        
        # We simulate bar by bar
        # Note: Optimization would use vectorization, but strategy logic often requires looping
        
        for i in range(50, len(df)):
            # Feed data up to i
            # Simulate "on_bar"
            # We construct a bar dict
            row = df.iloc[i]
            bar = {
                "symbol": symbol,
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"],
                "volume": row["Volume"],
                "date": row.name
            }
            
            # This calls logic which calls ML Engine
            signal = strategy.on_bar(symbol, bar)
            
            if signal:
                # Execute Logic (Simulation)
                if signal.signal_type.value == "BUY" and position == 0:
                     entry_price = row["Close"]
                     position = 100 # fixed
                     trades.append({"type": "BUY", "price": entry_price, "date": row.name})
                elif signal.signal_type.value == "CLOSE_LONG" and position > 0:
                     exit_price = row["Close"]
                     pnl = (exit_price - trades[-1]["price"]) * position
                     equity += pnl
                     position = 0
                     trades.append({"type": "SELL", "price": exit_price, "date": row.name, "pnl": pnl})
                     
        logger.info(f"Backtest finished. Final Equity: {equity}")
        return {"equity": equity, "trades": trades}
