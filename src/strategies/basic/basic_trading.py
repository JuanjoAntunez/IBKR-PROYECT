"""
Basic trading module with simple technical strategies.

Includes:
- SMA Crossover
- RSI Mean Reversion
- Range Breakout

Also provides a simple runner that can load historical data from
IB (ib_insync) or CSV (test mode), then periodically calls
generate_signal() and execute_trade() using print-only orders.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd

from ta.momentum import RSIIndicator

from src.strategies.base import (
    BaseStrategy,
    StrategyConfig,
    Signal,
    SignalType,
)
from src.strategies.runner import StrategyRunner, StrategyRunnerConfig


class SimpleTechnicalStrategy(BaseStrategy):
    """
    Base class for simple strategies with shared on_bar logic.
    """

    def __init__(self, config: StrategyConfig, max_bars: int = 300):
        super().__init__(config)
        self.max_bars = max_bars
        self._last_signal_type: Dict[str, SignalType] = {}

    def on_bar(self, symbol: str, bar: Dict[str, Any]) -> Optional[Signal]:
        if not self.is_running:
            return None

        if symbol not in self.symbols:
            return None

        df = self.get_data(symbol)
        new_row = pd.DataFrame([bar])
        if df is None:
            df = new_row
        else:
            df = pd.concat([df, new_row], ignore_index=True)

        if len(df) > self.max_bars:
            df = df.tail(self.max_bars).reset_index(drop=True)

        df["symbol"] = symbol
        self.update_data(symbol, df)

        signals = self.calculate_signals(df)
        for signal in signals:
            self.process_signal(signal)
            self._last_signal_type[signal.symbol] = signal.signal_type

        return signals[0] if signals else None

    def last_signal_type(self, symbol: str) -> Optional[SignalType]:
        return self._last_signal_type.get(symbol)


class SmaCrossoverStrategy(SimpleTechnicalStrategy):
    """
    SMA Crossover:
    - BUY when short SMA crosses above long SMA
    - SELL when short SMA crosses below long SMA
    Avoids repeated signals by only acting on real cross events.
    """

    def __init__(
        self,
        config: StrategyConfig,
        sma_short: int = 10,
        sma_long: int = 30,
        allow_short: bool = False,
    ):
        if sma_short >= sma_long:
            raise ValueError("sma_short must be < sma_long")
        super().__init__(config, max_bars=max(200, sma_long * 3))
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.allow_short = allow_short

    def calculate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals: List[Signal] = []
        if data.empty or "close" not in data.columns:
            return signals

        symbol = data["symbol"].iloc[0] if "symbol" in data.columns else self.symbols[0]
        close = data["close"]
        if len(close) < self.sma_long + 1:
            return signals

        sma_fast = close.rolling(self.sma_short).mean()
        sma_slow = close.rolling(self.sma_long).mean()

        fast_prev, fast_now = sma_fast.iloc[-2], sma_fast.iloc[-1]
        slow_prev, slow_now = sma_slow.iloc[-2], sma_slow.iloc[-1]

        if pd.isna([fast_prev, fast_now, slow_prev, slow_now]).any():
            return signals

        cross_up = fast_prev <= slow_prev and fast_now > slow_now
        cross_down = fast_prev >= slow_prev and fast_now < slow_now

        current_price = close.iloc[-1]
        current_time = data["date"].iloc[-1] if "date" in data.columns else datetime.now()

        if cross_up:
            if not self.has_position(symbol) and self.last_signal_type(symbol) != SignalType.BUY:
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        timestamp=current_time if isinstance(current_time, datetime) else datetime.now(),
                        price=float(current_price),
                        quantity=self.calculate_position_size(symbol, float(current_price)),
                        metadata={"strategy": "sma_crossover", "event": "cross_up"},
                    )
                )
        elif cross_down:
            if self.has_position(symbol) and self.last_signal_type(symbol) != SignalType.CLOSE_LONG:
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_LONG,
                        timestamp=current_time if isinstance(current_time, datetime) else datetime.now(),
                        price=float(current_price),
                        metadata={"strategy": "sma_crossover", "event": "cross_down"},
                    )
                )
            elif self.allow_short and not self.has_position(symbol) and self.last_signal_type(symbol) != SignalType.SELL:
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        timestamp=current_time if isinstance(current_time, datetime) else datetime.now(),
                        price=float(current_price),
                        quantity=self.calculate_position_size(symbol, float(current_price)),
                        metadata={"strategy": "sma_crossover", "event": "cross_down_short"},
                    )
                )

        return signals


class RSIMeanReversionStrategy(SimpleTechnicalStrategy):
    """
    RSI Mean Reversion:
    - BUY when RSI < oversold (default 30)
    - SELL when RSI > overbought (default 70)
    """

    def __init__(
        self,
        config: StrategyConfig,
        rsi_length: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        allow_short: bool = False,
    ):
        super().__init__(config, max_bars=max(200, rsi_length * 5))
        self.rsi_length = rsi_length
        self.oversold = oversold
        self.overbought = overbought
        self.allow_short = allow_short

    def calculate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals: List[Signal] = []
        if data.empty or "close" not in data.columns:
            return signals

        symbol = data["symbol"].iloc[0] if "symbol" in data.columns else self.symbols[0]
        close = data["close"]
        if len(close) < self.rsi_length + 2:
            return signals

        rsi = RSIIndicator(close=close, window=self.rsi_length).rsi()
        prev_rsi, curr_rsi = rsi.iloc[-2], rsi.iloc[-1]
        if pd.isna([prev_rsi, curr_rsi]).any():
            return signals

        current_price = close.iloc[-1]
        current_time = data["date"].iloc[-1] if "date" in data.columns else datetime.now()

        cross_oversold = prev_rsi >= self.oversold and curr_rsi < self.oversold
        cross_overbought = prev_rsi <= self.overbought and curr_rsi > self.overbought

        if cross_oversold:
            if not self.has_position(symbol) and self.last_signal_type(symbol) != SignalType.BUY:
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        timestamp=current_time if isinstance(current_time, datetime) else datetime.now(),
                        price=float(current_price),
                        quantity=self.calculate_position_size(symbol, float(current_price)),
                        metadata={"strategy": "rsi_mean_reversion", "rsi": float(curr_rsi)},
                    )
                )
        elif cross_overbought:
            if self.has_position(symbol) and self.last_signal_type(symbol) != SignalType.CLOSE_LONG:
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_LONG,
                        timestamp=current_time if isinstance(current_time, datetime) else datetime.now(),
                        price=float(current_price),
                        metadata={"strategy": "rsi_mean_reversion", "rsi": float(curr_rsi)},
                    )
                )
            elif self.allow_short and not self.has_position(symbol) and self.last_signal_type(symbol) != SignalType.SELL:
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        timestamp=current_time if isinstance(current_time, datetime) else datetime.now(),
                        price=float(current_price),
                        quantity=self.calculate_position_size(symbol, float(current_price)),
                        metadata={"strategy": "rsi_mean_reversion", "rsi": float(curr_rsi)},
                    )
                )

        return signals


class RangeBreakoutStrategy(SimpleTechnicalStrategy):
    """
    Range Breakout:
    - BUY when price breaks above max of last N bars
    - SELL when price breaks below min of last N bars
    """

    def __init__(
        self,
        config: StrategyConfig,
        lookback: int = 20,
        allow_short: bool = False,
    ):
        super().__init__(config, max_bars=max(200, lookback * 5))
        self.lookback = lookback
        self.allow_short = allow_short

    def calculate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals: List[Signal] = []
        if data.empty or "close" not in data.columns:
            return signals

        symbol = data["symbol"].iloc[0] if "symbol" in data.columns else self.symbols[0]
        if len(data) < self.lookback + 2:
            return signals

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Exclude current bar from range
        range_high = high.shift(1).rolling(self.lookback).max()
        range_low = low.shift(1).rolling(self.lookback).min()

        prev_close, curr_close = close.iloc[-2], close.iloc[-1]
        prev_high, curr_high = range_high.iloc[-2], range_high.iloc[-1]
        prev_low, curr_low = range_low.iloc[-2], range_low.iloc[-1]

        if pd.isna([prev_high, curr_high, prev_low, curr_low]).any():
            return signals

        breakout_up = prev_close <= prev_high and curr_close > curr_high
        breakout_down = prev_close >= prev_low and curr_close < curr_low

        current_time = data["date"].iloc[-1] if "date" in data.columns else datetime.now()

        if breakout_up:
            if not self.has_position(symbol) and self.last_signal_type(symbol) != SignalType.BUY:
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        timestamp=current_time if isinstance(current_time, datetime) else datetime.now(),
                        price=float(curr_close),
                        quantity=self.calculate_position_size(symbol, float(curr_close)),
                        metadata={"strategy": "range_breakout", "event": "breakout_up"},
                    )
                )
        elif breakout_down:
            if self.has_position(symbol) and self.last_signal_type(symbol) != SignalType.CLOSE_LONG:
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_LONG,
                        timestamp=current_time if isinstance(current_time, datetime) else datetime.now(),
                        price=float(curr_close),
                        metadata={"strategy": "range_breakout", "event": "breakout_down"},
                    )
                )
            elif self.allow_short and not self.has_position(symbol) and self.last_signal_type(symbol) != SignalType.SELL:
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        timestamp=current_time if isinstance(current_time, datetime) else datetime.now(),
                        price=float(curr_close),
                        quantity=self.calculate_position_size(symbol, float(curr_close)),
                        metadata={"strategy": "range_breakout", "event": "breakout_down_short"},
                    )
                )

        return signals

BasicTradingRunnerConfig = StrategyRunnerConfig


class BasicTradingRunner(StrategyRunner):
    """Alias del runner común para estrategias básicas."""
