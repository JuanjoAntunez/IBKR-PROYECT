# Este archivo hace que Python trate este directorio como un paquete

from src.strategies.base import (
    BaseStrategy,
    StrategyConfig,
    StrategyState,
    Signal,
    SignalType,
    Position,
    PositionSide,
)

from src.strategies.moving_average_crossover import (
    MovingAverageCrossover,
    MAType,
    CrossoverDirection,
)

from src.strategies.basic import (
    SmaCrossoverStrategy,
    RSIMeanReversionStrategy,
    RangeBreakoutStrategy,
    BasicTradingRunner,
    BasicTradingRunnerConfig,
)
from src.strategies.runner import StrategyRunner, StrategyRunnerConfig

__all__ = [
    # Base
    "BaseStrategy",
    "StrategyConfig",
    "StrategyState",
    "Signal",
    "SignalType",
    "Position",
    "PositionSide",
    # Estrategias concretas
    "MovingAverageCrossover",
    "MAType",
    "CrossoverDirection",
    # Basic strategies
    "SmaCrossoverStrategy",
    "RSIMeanReversionStrategy",
    "RangeBreakoutStrategy",
    "BasicTradingRunner",
    "BasicTradingRunnerConfig",
    "StrategyRunner",
    "StrategyRunnerConfig",
]
