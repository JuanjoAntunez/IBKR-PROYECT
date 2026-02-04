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
]
