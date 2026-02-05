from src.strategies.base import (
    BaseStrategy,
    StrategyConfig,
    Signal,
    SignalType,
    PositionSide
)
from src.ml.ml_engine import MLEngine
from datetime import datetime
from typing import List, Optional, Dict, Any
from loguru import logger
import pandas as pd

class IntradayMLStrategy(BaseStrategy):
    """
    Intraday trading using ML prediction.
    """
    
    def __init__(self, config: StrategyConfig, ml_engine: MLEngine, timeframe: str = "15 mins"):
        super().__init__(config)
        self.ml_engine = ml_engine
        self.timeframe = timeframe
        self.symbol = self.symbols[0]
        self.model_key = f"intraday_{self.symbol}_{self.timeframe.replace(' ', '')}"
        
    def calculate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals: List[Signal] = []
        if data.empty:
            return signals
            
        # Query ML
        pred = self.ml_engine.get_prediction(self.model_key, data)
        if "error" in pred:
            return signals
            
        pred_price = pred.get("prediction")
        current_price = data['close'].iloc[-1]
        
        pct_change = (pred_price - current_price) / current_price
        
        buy_thresh = 0.005
        sell_thresh = -0.005
        
        if pct_change > buy_thresh:
            if not self.has_position(self.symbol):
                 signal = Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    timestamp=datetime.now(),
                    price=current_price,
                    quantity=100, # Sizing logic
                    confidence=abs(pct_change)*100, # scaled
                    metadata={"pred_price": pred_price}
                )
                 signals.append(signal)
        
        elif pct_change < sell_thresh:
             if self.has_position(self.symbol):
                 # Close long or Open Short
                 signal = Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.CLOSE_LONG, # or SHORT
                    timestamp=datetime.now(),
                    price=current_price,
                    metadata={"pred_price": pred_price}
                )
                 signals.append(signal)

        return signals

    def on_bar(self, symbol: str, bar: Dict[str, Any]) -> Optional[Signal]:
        if not self.is_running:
            return None
            
        df = self.get_data(symbol)
        new_row = pd.DataFrame([bar])
        if df is None:
            df = new_row
        else:
            df = pd.concat([df, new_row], ignore_index=True)
            
        # Limit history
        if len(df) > 100: # We need enough for features
            df = df.tail(100)
            
        self.update_data(symbol, df)
        
        signals = self.calculate_signals(df)
        for s in signals:
            self.process_signal(s)
            
        return signals[0] if signals else None
