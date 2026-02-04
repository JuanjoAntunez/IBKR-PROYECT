"""
Clase base abstracta para estrategias de trading.

Todas las estrategias concretas deben heredar de BaseStrategy
e implementar los métodos abstractos definidos aquí.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any

import pandas as pd

from src.utils.logger import logger


class StrategyState(Enum):
    """Estados posibles de una estrategia."""

    INITIALIZED = auto()  # Estrategia creada pero no iniciada
    RUNNING = auto()      # Estrategia activa y procesando
    PAUSED = auto()       # Estrategia pausada temporalmente
    STOPPED = auto()      # Estrategia detenida
    ERROR = auto()        # Estrategia en estado de error


class SignalType(Enum):
    """Tipos de señales de trading."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    CLOSE_ALL = "CLOSE_ALL"


class PositionSide(Enum):
    """Lado de una posición."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Signal:
    """Representa una señal de trading generada por una estrategia."""

    symbol: str
    signal_type: SignalType
    timestamp: datetime
    price: Optional[float] = None
    quantity: Optional[int] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validación después de inicialización."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence debe estar entre 0 y 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la señal a diccionario."""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "quantity": self.quantity,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class Position:
    """Representa una posición en un instrumento."""

    symbol: str
    side: PositionSide
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        """Indica si la posición está abierta."""
        return self.side != PositionSide.FLAT and self.quantity != 0

    @property
    def unrealized_pnl(self) -> Optional[float]:
        """Calcula el P&L no realizado."""
        if self.current_price is None:
            return None

        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            return (self.entry_price - self.current_price) * self.quantity
        return 0.0

    @property
    def unrealized_pnl_percent(self) -> Optional[float]:
        """Calcula el P&L no realizado en porcentaje."""
        if self.current_price is None or self.entry_price == 0:
            return None

        if self.side == PositionSide.LONG:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        elif self.side == PositionSide.SHORT:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la posición a diccionario."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_percent": self.unrealized_pnl_percent,
            "metadata": self.metadata,
        }


@dataclass
class StrategyConfig:
    """Configuración base para estrategias."""

    name: str
    symbols: List[str]
    max_position_size: float = 10000.0
    max_positions: int = 5
    risk_per_trade: float = 0.02  # 2% del capital
    use_stop_loss: bool = True
    use_take_profit: bool = True
    stop_loss_percent: float = 0.02  # 2%
    take_profit_percent: float = 0.04  # 4%
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Clase base abstracta para todas las estrategias de trading.

    Las estrategias concretas deben implementar:
    - calculate_signals(): Lógica para generar señales
    - on_bar(): Procesamiento de cada nueva barra
    - on_tick(): Procesamiento de cada tick (opcional)

    Ejemplo de implementación:
        class MovingAverageCrossover(BaseStrategy):
            def __init__(self, config: StrategyConfig, fast_period: int, slow_period: int):
                super().__init__(config)
                self.fast_period = fast_period
                self.slow_period = slow_period

            def calculate_signals(self, data: pd.DataFrame) -> List[Signal]:
                # Implementar lógica de cruce de medias
                ...
    """

    def __init__(self, config: StrategyConfig):
        """
        Inicializa la estrategia base.

        Args:
            config: Configuración de la estrategia
        """
        self.config = config
        self.name = config.name
        self.symbols = config.symbols

        self._state = StrategyState.INITIALIZED
        self._positions: Dict[str, Position] = {}
        self._signals_history: List[Signal] = []
        self._data: Dict[str, pd.DataFrame] = {}
        self._last_update: Optional[datetime] = None

        logger.info(f"Estrategia '{self.name}' inicializada con {len(self.symbols)} símbolos")

    @property
    def state(self) -> StrategyState:
        """Estado actual de la estrategia."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Indica si la estrategia está activa."""
        return self._state == StrategyState.RUNNING

    @property
    def positions(self) -> Dict[str, Position]:
        """Posiciones actuales de la estrategia."""
        return self._positions.copy()

    @property
    def open_positions(self) -> List[Position]:
        """Lista de posiciones abiertas."""
        return [p for p in self._positions.values() if p.is_open]

    @property
    def signals_history(self) -> List[Signal]:
        """Historial de señales generadas."""
        return self._signals_history.copy()

    # ==================== Métodos abstractos ====================

    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Calcula señales de trading basándose en los datos.

        Este es el método principal donde se implementa la lógica
        de la estrategia.

        Args:
            data: DataFrame con datos OHLCV

        Returns:
            Lista de señales generadas
        """
        pass

    @abstractmethod
    def on_bar(self, symbol: str, bar: Dict[str, Any]) -> Optional[Signal]:
        """
        Procesa una nueva barra de datos.

        Se llama cada vez que se completa una barra (ej: cada minuto).

        Args:
            symbol: Símbolo del instrumento
            bar: Datos de la barra (open, high, low, close, volume)

        Returns:
            Señal si se genera, None si no
        """
        pass

    # ==================== Métodos opcionales para sobrescribir ====================

    def on_tick(self, symbol: str, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        Procesa un nuevo tick de datos.

        Método opcional para estrategias que necesitan
        granularidad de tick.

        Args:
            symbol: Símbolo del instrumento
            tick: Datos del tick

        Returns:
            Señal si se genera, None si no
        """
        return None

    def on_signal(self, signal: Signal) -> None:
        """
        Callback cuando se genera una señal.

        Útil para logging adicional o procesamiento.

        Args:
            signal: Señal generada
        """
        logger.info(
            f"[{self.name}] Señal generada: {signal.signal_type.value} "
            f"{signal.symbol} @ {signal.price}"
        )

    def on_position_opened(self, position: Position) -> None:
        """
        Callback cuando se abre una posición.

        Args:
            position: Posición abierta
        """
        logger.info(
            f"[{self.name}] Posición abierta: {position.side.value} "
            f"{position.quantity} {position.symbol} @ {position.entry_price}"
        )

    def on_position_closed(self, position: Position, exit_price: float) -> None:
        """
        Callback cuando se cierra una posición.

        Args:
            position: Posición cerrada
            exit_price: Precio de salida
        """
        pnl = (exit_price - position.entry_price) * position.quantity
        if position.side == PositionSide.SHORT:
            pnl = -pnl

        logger.info(
            f"[{self.name}] Posición cerrada: {position.symbol} "
            f"P&L: ${pnl:.2f}"
        )

    def validate_signal(self, signal: Signal) -> bool:
        """
        Valida una señal antes de actuar.

        Sobrescribir para añadir validaciones personalizadas.

        Args:
            signal: Señal a validar

        Returns:
            True si la señal es válida
        """
        # Verificar que el símbolo está en la lista
        if signal.symbol not in self.symbols:
            logger.warning(f"Señal rechazada: {signal.symbol} no está en la lista")
            return False

        # Verificar máximo de posiciones
        if len(self.open_positions) >= self.config.max_positions:
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                logger.warning("Señal rechazada: máximo de posiciones alcanzado")
                return False

        return True

    # ==================== Métodos de gestión ====================

    def start(self) -> None:
        """Inicia la estrategia."""
        if self._state == StrategyState.RUNNING:
            logger.warning(f"Estrategia '{self.name}' ya está corriendo")
            return

        self._state = StrategyState.RUNNING
        logger.info(f"Estrategia '{self.name}' iniciada")

    def stop(self) -> None:
        """Detiene la estrategia."""
        self._state = StrategyState.STOPPED
        logger.info(f"Estrategia '{self.name}' detenida")

    def pause(self) -> None:
        """Pausa la estrategia."""
        if self._state != StrategyState.RUNNING:
            logger.warning(f"Estrategia '{self.name}' no está corriendo")
            return

        self._state = StrategyState.PAUSED
        logger.info(f"Estrategia '{self.name}' pausada")

    def resume(self) -> None:
        """Reanuda la estrategia."""
        if self._state != StrategyState.PAUSED:
            logger.warning(f"Estrategia '{self.name}' no está pausada")
            return

        self._state = StrategyState.RUNNING
        logger.info(f"Estrategia '{self.name}' reanudada")

    def reset(self) -> None:
        """Reinicia la estrategia a su estado inicial."""
        self._state = StrategyState.INITIALIZED
        self._positions.clear()
        self._signals_history.clear()
        self._data.clear()
        self._last_update = None
        logger.info(f"Estrategia '{self.name}' reiniciada")

    # ==================== Gestión de datos ====================

    def update_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Actualiza los datos para un símbolo.

        Args:
            symbol: Símbolo del instrumento
            data: DataFrame con datos OHLCV
        """
        self._data[symbol] = data
        self._last_update = datetime.now()
        logger.debug(f"[{self.name}] Datos actualizados para {symbol}: {len(data)} barras")

    def get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Obtiene los datos de un símbolo.

        Args:
            symbol: Símbolo del instrumento

        Returns:
            DataFrame con datos o None si no hay
        """
        return self._data.get(symbol)

    # ==================== Gestión de posiciones ====================

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Obtiene la posición para un símbolo.

        Args:
            symbol: Símbolo del instrumento

        Returns:
            Position o None si no hay posición
        """
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """
        Verifica si hay posición abierta en un símbolo.

        Args:
            symbol: Símbolo del instrumento

        Returns:
            True si hay posición abierta
        """
        pos = self._positions.get(symbol)
        return pos is not None and pos.is_open

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: int,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Position:
        """
        Registra la apertura de una posición.

        Args:
            symbol: Símbolo del instrumento
            side: LONG o SHORT
            quantity: Cantidad
            price: Precio de entrada
            stop_loss: Nivel de stop loss
            take_profit: Nivel de take profit

        Returns:
            Position creada
        """
        # Calcular stop/take profit si no se especifican
        if stop_loss is None and self.config.use_stop_loss:
            if side == PositionSide.LONG:
                stop_loss = price * (1 - self.config.stop_loss_percent)
            else:
                stop_loss = price * (1 + self.config.stop_loss_percent)

        if take_profit is None and self.config.use_take_profit:
            if side == PositionSide.LONG:
                take_profit = price * (1 + self.config.take_profit_percent)
            else:
                take_profit = price * (1 - self.config.take_profit_percent)

        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            current_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self._positions[symbol] = position
        self.on_position_opened(position)

        return position

    def close_position(self, symbol: str, price: float) -> Optional[Position]:
        """
        Registra el cierre de una posición.

        Args:
            symbol: Símbolo del instrumento
            price: Precio de salida

        Returns:
            Position cerrada o None si no existía
        """
        if symbol not in self._positions:
            return None

        position = self._positions[symbol]
        self.on_position_closed(position, price)

        # Marcar como cerrada
        position.side = PositionSide.FLAT
        position.quantity = 0
        position.current_price = price

        return position

    def update_position_price(self, symbol: str, price: float) -> None:
        """
        Actualiza el precio actual de una posición.

        Args:
            symbol: Símbolo del instrumento
            price: Precio actual
        """
        if symbol in self._positions:
            self._positions[symbol].current_price = price

    # ==================== Procesamiento de señales ====================

    def process_signal(self, signal: Signal) -> bool:
        """
        Procesa una señal generada.

        Args:
            signal: Señal a procesar

        Returns:
            True si la señal fue procesada correctamente
        """
        if not self.validate_signal(signal):
            return False

        self._signals_history.append(signal)
        self.on_signal(signal)

        return True

    # ==================== Utilidades ====================

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        stop_loss: Optional[float] = None,
    ) -> int:
        """
        Calcula el tamaño de posición basado en el riesgo.

        Args:
            symbol: Símbolo del instrumento
            price: Precio actual
            stop_loss: Nivel de stop loss

        Returns:
            Cantidad de acciones/contratos
        """
        max_value = self.config.max_position_size

        if stop_loss is not None:
            # Position sizing basado en riesgo
            risk_amount = max_value * self.config.risk_per_trade
            risk_per_share = abs(price - stop_loss)

            if risk_per_share > 0:
                quantity = int(risk_amount / risk_per_share)
            else:
                quantity = int(max_value / price)
        else:
            quantity = int(max_value / price)

        return max(1, quantity)

    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen del estado de la estrategia.

        Returns:
            Diccionario con información de la estrategia
        """
        total_pnl = sum(
            p.unrealized_pnl or 0
            for p in self._positions.values()
            if p.is_open
        )

        return {
            "name": self.name,
            "state": self._state.name,
            "symbols": self.symbols,
            "open_positions": len(self.open_positions),
            "total_signals": len(self._signals_history),
            "unrealized_pnl": total_pnl,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "config": {
                "max_position_size": self.config.max_position_size,
                "max_positions": self.config.max_positions,
                "risk_per_trade": self.config.risk_per_trade,
            },
        }

    def __repr__(self) -> str:
        """Representación string de la estrategia."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"state={self._state.name}, "
            f"positions={len(self.open_positions)})"
        )
