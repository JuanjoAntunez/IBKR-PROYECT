import pandas as pd
from datetime import datetime, timedelta

from src.strategies.moving_average_crossover import MovingAverageCrossover, MAType
from src.strategies.base import StrategyConfig, SignalType


def test_ma_crossover_replay_buy_signal():
    # Build deterministic price series that triggers a bullish crossover
    base_time = datetime(2026, 2, 4, 12, 0, 0)
    closes = [10, 10, 10, 10, 12]
    data = pd.DataFrame({
        "date": [base_time + timedelta(minutes=i) for i in range(len(closes))],
        "close": closes,
        "symbol": ["AAPL"] * len(closes),
    })

    config = StrategyConfig(name="ma_test", symbols=["AAPL"])
    strat = MovingAverageCrossover(config=config, fast_period=2, slow_period=3, ma_type=MAType.SMA)

    signals = strat.calculate_signals(data)
    assert len(signals) >= 1
    assert signals[-1].signal_type == SignalType.BUY
