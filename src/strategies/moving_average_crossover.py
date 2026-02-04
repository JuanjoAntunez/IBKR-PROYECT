"""
Estrategia de cruce de medias móviles.

Genera señales de compra cuando la media rápida cruza por encima
de la media lenta, y señales de venta en el cruce contrario.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

from src.strategies.base import (
    BaseStrategy,
    StrategyConfig,
    Signal,
    SignalType,
    Position,
    PositionSide,
)
from src.utils.logger import logger


class MAType(Enum):
    """Tipos de media móvil disponibles."""

    SMA = "SMA"  # Simple Moving Average
    EMA = "EMA"  # Exponential Moving Average
    WMA = "WMA"  # Weighted Moving Average


class CrossoverDirection(Enum):
    """Dirección del cruce de medias."""

    BULLISH = "BULLISH"  # Fast cruza por encima de slow
    BEARISH = "BEARISH"  # Fast cruza por debajo de slow
    NONE = "NONE"        # Sin cruce


class MovingAverageCrossover(BaseStrategy):
    """
    Estrategia de cruce de medias móviles.

    Señales:
    - BUY: Media rápida cruza por encima de media lenta (golden cross)
    - SELL: Media rápida cruza por debajo de media lenta (death cross)

    Parámetros configurables:
    - fast_period: Período de la media rápida
    - slow_period: Período de la media lenta
    - ma_type: Tipo de media móvil (SMA, EMA, WMA)
    - confirmation_bars: Barras de confirmación antes de señal
    - min_distance_percent: Distancia mínima entre medias para señal

    Ejemplo:
        config = StrategyConfig(
            name="MA Crossover AAPL",
            symbols=["AAPL"],
        )
        strategy = MovingAverageCrossover(
            config=config,
            fast_period=10,
            slow_period=30,
            ma_type=MAType.EMA,
        )
    """

    def __init__(
        self,
        config: StrategyConfig,
        fast_period: int = 10,
        slow_period: int = 30,
        ma_type: MAType = MAType.SMA,
        confirmation_bars: int = 1,
        min_distance_percent: float = 0.0,
        allow_short: bool = False,
    ):
        """
        Inicializa la estrategia de cruce de medias.

        Args:
            config: Configuración base de la estrategia
            fast_period: Período de la media rápida
            slow_period: Período de la media lenta
            ma_type: Tipo de media móvil a usar
            confirmation_bars: Número de barras para confirmar el cruce
            min_distance_percent: Distancia mínima entre medias (%)
            allow_short: Permitir posiciones cortas
        """
        super().__init__(config)

        if fast_period >= slow_period:
            raise ValueError("fast_period debe ser menor que slow_period")

        if fast_period < 2:
            raise ValueError("fast_period debe ser al menos 2")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type
        self.confirmation_bars = confirmation_bars
        self.min_distance_percent = min_distance_percent
        self.allow_short = allow_short

        # Estado interno por símbolo
        self._ma_data: Dict[str, Dict[str, Any]] = {}
        self._pending_signals: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"MovingAverageCrossover inicializada: "
            f"fast={fast_period}, slow={slow_period}, type={ma_type.value}"
        )

    def _calculate_ma(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calcula la media móvil según el tipo configurado.

        Args:
            series: Serie de precios
            period: Período de la media

        Returns:
            Serie con la media móvil calculada
        """
        if self.ma_type == MAType.SMA:
            return series.rolling(window=period).mean()
        elif self.ma_type == MAType.EMA:
            return series.ewm(span=period, adjust=False).mean()
        elif self.ma_type == MAType.WMA:
            weights = np.arange(1, period + 1)
            return series.rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(),
                raw=True
            )
        else:
            raise ValueError(f"Tipo de MA no soportado: {self.ma_type}")

    def _detect_crossover(
        self,
        fast_ma: pd.Series,
        slow_ma: pd.Series,
    ) -> CrossoverDirection:
        """
        Detecta si hay un cruce de medias.

        Args:
            fast_ma: Serie de media rápida
            slow_ma: Serie de media lenta

        Returns:
            Dirección del cruce o NONE
        """
        if len(fast_ma) < 2 or len(slow_ma) < 2:
            return CrossoverDirection.NONE

        # Valores actuales y anteriores
        fast_current = fast_ma.iloc[-1]
        fast_prev = fast_ma.iloc[-2]
        slow_current = slow_ma.iloc[-1]
        slow_prev = slow_ma.iloc[-2]

        # Verificar NaN
        if any(pd.isna([fast_current, fast_prev, slow_current, slow_prev])):
            return CrossoverDirection.NONE

        # Detectar cruce alcista (golden cross)
        if fast_prev <= slow_prev and fast_current > slow_current:
            return CrossoverDirection.BULLISH

        # Detectar cruce bajista (death cross)
        if fast_prev >= slow_prev and fast_current < slow_current:
            return CrossoverDirection.BEARISH

        return CrossoverDirection.NONE

    def _check_min_distance(
        self,
        fast_ma: float,
        slow_ma: float,
    ) -> bool:
        """
        Verifica que la distancia entre medias sea suficiente.

        Args:
            fast_ma: Valor de media rápida
            slow_ma: Valor de media lenta

        Returns:
            True si la distancia es suficiente
        """
        if self.min_distance_percent <= 0:
            return True

        distance_percent = abs(fast_ma - slow_ma) / slow_ma * 100
        return distance_percent >= self.min_distance_percent

    def calculate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Calcula señales de trading basándose en cruces de medias.

        Args:
            data: DataFrame con columnas: date, open, high, low, close, volume
                  Debe incluir columna 'symbol' o procesará como símbolo único

        Returns:
            Lista de señales generadas
        """
        signals: List[Signal] = []

        if data.empty:
            return signals

        # Verificar columnas requeridas
        required_cols = ["close"]
        if not all(col in data.columns for col in required_cols):
            logger.error(f"DataFrame debe contener columnas: {required_cols}")
            return signals

        # Determinar símbolo
        symbol = data["symbol"].iloc[0] if "symbol" in data.columns else self.symbols[0]

        # Verificar datos suficientes
        if len(data) < self.slow_period + 1:
            logger.debug(
                f"Datos insuficientes para {symbol}: "
                f"{len(data)} < {self.slow_period + 1}"
            )
            return signals

        # Calcular medias móviles
        close = data["close"]
        fast_ma = self._calculate_ma(close, self.fast_period)
        slow_ma = self._calculate_ma(close, self.slow_period)

        # Guardar datos de MA para referencia
        self._ma_data[symbol] = {
            "fast_ma": fast_ma.iloc[-1],
            "slow_ma": slow_ma.iloc[-1],
            "fast_ma_prev": fast_ma.iloc[-2] if len(fast_ma) > 1 else None,
            "slow_ma_prev": slow_ma.iloc[-2] if len(slow_ma) > 1 else None,
        }

        # Detectar cruce
        crossover = self._detect_crossover(fast_ma, slow_ma)

        if crossover == CrossoverDirection.NONE:
            return signals

        # Verificar distancia mínima
        if not self._check_min_distance(fast_ma.iloc[-1], slow_ma.iloc[-1]):
            logger.debug(f"Cruce ignorado en {symbol}: distancia insuficiente")
            return signals

        # Obtener precio actual
        current_price = close.iloc[-1]
        current_time = (
            data["date"].iloc[-1]
            if "date" in data.columns
            else datetime.now()
        )

        # Generar señal según dirección del cruce
        if crossover == CrossoverDirection.BULLISH:
            # Verificar si ya tenemos posición long
            if self.has_position(symbol):
                pos = self.get_position(symbol)
                if pos and pos.side == PositionSide.LONG:
                    logger.debug(f"Cruce alcista ignorado: ya hay posición long en {symbol}")
                    return signals

            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                timestamp=current_time if isinstance(current_time, datetime) else datetime.now(),
                price=current_price,
                quantity=self.calculate_position_size(symbol, current_price),
                confidence=self._calculate_confidence(fast_ma, slow_ma),
                metadata={
                    "crossover": "golden_cross",
                    "fast_ma": fast_ma.iloc[-1],
                    "slow_ma": slow_ma.iloc[-1],
                    "fast_period": self.fast_period,
                    "slow_period": self.slow_period,
                    "ma_type": self.ma_type.value,
                },
            )
            signals.append(signal)

            # Si hay posición short, cerrarla primero
            if self.has_position(symbol):
                pos = self.get_position(symbol)
                if pos and pos.side == PositionSide.SHORT:
                    close_signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE_SHORT,
                        timestamp=signal.timestamp,
                        price=current_price,
                        metadata={"reason": "crossover_reversal"},
                    )
                    signals.insert(0, close_signal)

        elif crossover == CrossoverDirection.BEARISH:
            if self.allow_short:
                # Señal de venta en corto
                if self.has_position(symbol):
                    pos = self.get_position(symbol)
                    if pos and pos.side == PositionSide.SHORT:
                        logger.debug(f"Cruce bajista ignorado: ya hay posición short en {symbol}")
                        return signals

                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    timestamp=current_time if isinstance(current_time, datetime) else datetime.now(),
                    price=current_price,
                    quantity=self.calculate_position_size(symbol, current_price),
                    confidence=self._calculate_confidence(fast_ma, slow_ma),
                    metadata={
                        "crossover": "death_cross",
                        "fast_ma": fast_ma.iloc[-1],
                        "slow_ma": slow_ma.iloc[-1],
                        "fast_period": self.fast_period,
                        "slow_period": self.slow_period,
                        "ma_type": self.ma_type.value,
                    },
                )
                signals.append(signal)

                # Si hay posición long, cerrarla primero
                if self.has_position(symbol):
                    pos = self.get_position(symbol)
                    if pos and pos.side == PositionSide.LONG:
                        close_signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.CLOSE_LONG,
                            timestamp=signal.timestamp,
                            price=current_price,
                            metadata={"reason": "crossover_reversal"},
                        )
                        signals.insert(0, close_signal)
            else:
                # Solo cerrar posición long si existe
                if self.has_position(symbol):
                    pos = self.get_position(symbol)
                    if pos and pos.side == PositionSide.LONG:
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.CLOSE_LONG,
                            timestamp=current_time if isinstance(current_time, datetime) else datetime.now(),
                            price=current_price,
                            confidence=self._calculate_confidence(fast_ma, slow_ma),
                            metadata={
                                "crossover": "death_cross",
                                "fast_ma": fast_ma.iloc[-1],
                                "slow_ma": slow_ma.iloc[-1],
                                "reason": "exit_signal",
                            },
                        )
                        signals.append(signal)

        # Procesar señales
        for signal in signals:
            self.process_signal(signal)

        return signals

    def _calculate_confidence(
        self,
        fast_ma: pd.Series,
        slow_ma: pd.Series,
    ) -> float:
        """
        Calcula la confianza de la señal basada en la fuerza del cruce.

        Args:
            fast_ma: Serie de media rápida
            slow_ma: Serie de media lenta

        Returns:
            Valor de confianza entre 0 y 1
        """
        # Distancia porcentual entre medias
        distance = abs(fast_ma.iloc[-1] - slow_ma.iloc[-1]) / slow_ma.iloc[-1]

        # Pendiente de la media rápida (momentum)
        if len(fast_ma) >= 3:
            slope = (fast_ma.iloc[-1] - fast_ma.iloc[-3]) / fast_ma.iloc[-3]
        else:
            slope = 0

        # Combinar factores
        confidence = min(1.0, 0.5 + distance * 10 + abs(slope) * 5)
        return round(confidence, 2)

    def on_bar(self, symbol: str, bar: Dict[str, Any]) -> Optional[Signal]:
        """
        Procesa una nueva barra y genera señal si corresponde.

        Args:
            symbol: Símbolo del instrumento
            bar: Datos de la barra (open, high, low, close, volume, date)

        Returns:
            Señal generada o None
        """
        if not self.is_running:
            return None

        if symbol not in self.symbols:
            return None

        # Obtener datos existentes
        data = self.get_data(symbol)

        if data is None:
            # Crear DataFrame inicial
            data = pd.DataFrame([bar])
        else:
            # Añadir nueva barra
            new_row = pd.DataFrame([bar])
            data = pd.concat([data, new_row], ignore_index=True)

        # Mantener solo las barras necesarias
        max_bars = self.slow_period * 2
        if len(data) > max_bars:
            data = data.tail(max_bars).reset_index(drop=True)

        # Actualizar datos
        data["symbol"] = symbol
        self.update_data(symbol, data)

        # Calcular señales
        signals = self.calculate_signals(data)

        # Actualizar precio de posiciones existentes
        if "close" in bar:
            self.update_position_price(symbol, bar["close"])

        # Verificar stop loss / take profit
        sl_tp_signal = self._check_stop_loss_take_profit(symbol, bar.get("close"))
        if sl_tp_signal:
            signals.append(sl_tp_signal)

        return signals[0] if signals else None

    def _check_stop_loss_take_profit(
        self,
        symbol: str,
        current_price: Optional[float],
    ) -> Optional[Signal]:
        """
        Verifica si se ha alcanzado stop loss o take profit.

        Args:
            symbol: Símbolo del instrumento
            current_price: Precio actual

        Returns:
            Señal de cierre si corresponde
        """
        if current_price is None:
            return None

        position = self.get_position(symbol)
        if not position or not position.is_open:
            return None

        signal_type = None
        reason = None

        if position.side == PositionSide.LONG:
            if position.stop_loss and current_price <= position.stop_loss:
                signal_type = SignalType.CLOSE_LONG
                reason = "stop_loss"
            elif position.take_profit and current_price >= position.take_profit:
                signal_type = SignalType.CLOSE_LONG
                reason = "take_profit"

        elif position.side == PositionSide.SHORT:
            if position.stop_loss and current_price >= position.stop_loss:
                signal_type = SignalType.CLOSE_SHORT
                reason = "stop_loss"
            elif position.take_profit and current_price <= position.take_profit:
                signal_type = SignalType.CLOSE_SHORT
                reason = "take_profit"

        if signal_type:
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                timestamp=datetime.now(),
                price=current_price,
                metadata={"reason": reason},
            )
            self.process_signal(signal)
            return signal

        return None

    def on_tick(self, symbol: str, tick: Dict[str, Any]) -> Optional[Signal]:
        """
        Procesa un tick (solo verifica stop loss/take profit).

        Args:
            symbol: Símbolo del instrumento
            tick: Datos del tick

        Returns:
            Señal si se activa SL/TP
        """
        if not self.is_running:
            return None

        price = tick.get("last") or tick.get("close")
        if price:
            self.update_position_price(symbol, price)
            return self._check_stop_loss_take_profit(symbol, price)

        return None

    def get_ma_values(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Obtiene los valores actuales de las medias móviles.

        Args:
            symbol: Símbolo del instrumento

        Returns:
            Diccionario con valores de MA o None
        """
        return self._ma_data.get(symbol)

    def get_strategy_params(self) -> Dict[str, Any]:
        """
        Obtiene los parámetros de la estrategia.

        Returns:
            Diccionario con parámetros
        """
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "ma_type": self.ma_type.value,
            "confirmation_bars": self.confirmation_bars,
            "min_distance_percent": self.min_distance_percent,
            "allow_short": self.allow_short,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Obtiene resumen extendido de la estrategia."""
        base_summary = super().get_summary()
        base_summary["parameters"] = self.get_strategy_params()
        base_summary["ma_values"] = self._ma_data
        return base_summary
