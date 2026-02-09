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

class EnhancedSMAStrategy(BaseStrategy):
    """
    SMA Crossover strategy filtered by ML model.
    """
    
    def __init__(self, config: StrategyConfig, ml_engine: MLEngine, fast_period: int = 50, slow_period: int = 200, threshold: float = 0.65):
        super().__init__(config)
        self.ml_engine = ml_engine
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.threshold = threshold

    def _model_key(self, symbol: str) -> str:
        return f"enhanced_sma_{symbol}"
        
    def calculate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals: List[Signal] = []
        if data.empty:
            return signals
            
        symbol = data["symbol"].iloc[0] if "symbol" in data.columns else self.symbols[0]
        
        # 1. Calculate SMA
        close = data["close"]
        if len(close) < self.slow_period:
            return signals
            
        sma_fast = close.rolling(self.fast_period).mean()
        sma_slow = close.rolling(self.slow_period).mean()
        
        # 2. Check Crossover
        # Golden Cross: Fast crosses above Slow
        golden_cross = (sma_fast.iloc[-1] > sma_slow.iloc[-1]) and (sma_fast.iloc[-2] <= sma_slow.iloc[-2])
        death_cross = (sma_fast.iloc[-1] < sma_slow.iloc[-1]) and (sma_fast.iloc[-2] >= sma_slow.iloc[-2])
        
        # 3. ML Filter
        if golden_cross:
            # Check ML
            # We pass the WHOLE dataframe to ML Engine, it handles feature generation
            pred = self.ml_engine.get_prediction(self._model_key(symbol), data)
            
            if "error" in pred:
                logger.warning(f"ML Error: {pred['error']}")
                # Fallback? Or skip? Skip is safer.
                return signals
                
            prob = pred.get("probability", 0.0)
            logger.info(f"Golden Cross detected on {symbol}. ML Prob: {prob:.2f}")
            
            if prob > self.threshold:
                # Generate BUY Signal
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    timestamp=datetime.now(),
                    price=close.iloc[-1],
                    quantity=100, # Should use position sizing logic
                    confidence=prob,
                    metadata={"ml_prob": prob, "type": "golden_cross"}
                )
                signals.append(signal)
                
        elif death_cross:
            # Exit long
            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.CLOSE_LONG,
                timestamp=datetime.now(),
                price=close.iloc[-1],
                metadata={"reason": "death_cross"}
            )
            signals.append(signal)
             
        # Manage Exits via ML confidence? 
        # User said: "Take profit basado en ML confidence"
        # If we are long, and ML prob drops below 0.5?
        if self.has_position(symbol):
            pos = self.get_position(symbol)
            if pos.side == PositionSide.LONG:
                # Check current ML confidence
                pred = self.ml_engine.get_prediction(self._model_key(symbol), data)
                prob = pred.get("probability", 0.5)
                if prob < 0.4: # Exit if confidence drops
                     signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_LONG,
                        timestamp=datetime.now(),
                        price=close.iloc[-1],
                        metadata={"reason": "low_confidence", "prob": prob}
                    )
                     if not any(s.signal_type == SignalType.CLOSE_LONG for s in signals):
                         signals.append(signal)

        return signals

    def on_bar(self, symbol: str, bar: Dict[str, Any]) -> Optional[Signal]:
        # Reuse BaseStrategy logic which calls calculate_signals if we update data
        # But BaseStrategy on_bar implementation in base.py is abstract.
        # MovingAverageCrossover implements it by updating data and calling calculate_signals.
        
        # Copy-paste logic from MovingAverageCrossover roughly
        if not self.is_running: 
            return None
            
        # Update data logic
        # ... (Assuming BaseStrategy doesn't do it automatically, we must do it)
        # Convert bar dict to DataFrame row
        
        # Simplified:
        # We need to maintain history.
        # Ideally we fetch history from engine state?
        # But here we are independent.
        
        # Let's assume `bar` has enough info or we accumulate it.
        # For this implementation, I'll rely on `update_data`
        
        df = self.get_data(symbol)
        new_row = pd.DataFrame([bar])
        if df is None:
            df = new_row
        else:
            df = pd.concat([df, new_row], ignore_index=True)
            
        # Limit history
        if len(df) > self.slow_period + 50:
            df = df.tail(self.slow_period + 50)
            
        self.update_data(symbol, df)
        
        signals = self.calculate_signals(df)
        
        for s in signals:
            self.process_signal(s)
            
        return signals[0] if signals else None
