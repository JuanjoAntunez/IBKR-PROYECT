import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from src.strategies.moving_average_crossover import MovingAverageCrossover, MAType
from src.strategies.base import StrategyConfig


def test_ma_crossover_golden_signals():
    data_path = Path(__file__).parent / "data" / "ma_crossover_golden.json"
    payload = json.loads(data_path.read_text())

    base_time = datetime.fromisoformat(payload["base_time"])
    closes = payload["closes"]
    symbol = payload["symbol"]
    strategy_cfg = payload["strategy"]

    df = pd.DataFrame(
        {
            "date": [base_time + timedelta(minutes=i) for i in range(len(closes))],
            "close": closes,
            "symbol": [symbol] * len(closes),
        }
    )

    config = StrategyConfig(name="ma_golden", symbols=[symbol])
    strategy = MovingAverageCrossover(
        config=config,
        fast_period=strategy_cfg["fast_period"],
        slow_period=strategy_cfg["slow_period"],
        ma_type=MAType[strategy_cfg["ma_type"]],
    )

    signals = strategy.calculate_signals(df)

    actual = [
        {
            "symbol": s.symbol,
            "signal_type": s.signal_type.value,
            "timestamp": s.timestamp.isoformat(),
            "price": s.price,
            "quantity": s.quantity,
        }
        for s in signals
    ]

    assert actual == payload["expected_signals"]
