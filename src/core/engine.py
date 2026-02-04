"""
Trading engine (live runtime) that connects data -> strategies -> signals.

This is a minimal, pluggable skeleton to run strategies on live data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol

from src.connection.ib_client import IBClient
from src.data.stream import MarketDataStream, BarAggregator, TickData
from src.strategies.base import BaseStrategy, Signal
from src.utils.logger import logger


class SignalHandler(Protocol):
    """Callable interface for handling emitted signals."""

    def __call__(self, signal: Signal, strategy: BaseStrategy) -> object:  # pragma: no cover
        ...


@dataclass
class EngineConfig:
    """Configuration for the trading engine."""

    symbols: List[str]
    bar_interval_seconds: int = 60
    auto_connect: bool = True
    auto_subscribe: bool = True
    stream_generic_tick_list: str = ""


@dataclass
class _StrategyState:
    """Internal tracking for each strategy."""

    strategy: BaseStrategy
    last_signal_index: int = 0


class TradingEngine:
    """
    Live trading engine skeleton.

    Responsibilities:
    - Connect to IB (optional)
    - Subscribe to live market data
    - Aggregate ticks into bars
    - Feed bars to strategies
    - Emit generated signals to handlers
    """

    def __init__(
        self,
        config: EngineConfig,
        ib_client: Optional[IBClient] = None,
        stream: Optional[MarketDataStream] = None,
    ) -> None:
        self.config = config
        self.ib_client = ib_client
        self.stream = stream

        self._running = False
        self._strategies: Dict[str, _StrategyState] = {}
        self._signal_handlers: List[SignalHandler] = []

        # One bar aggregator for all symbols
        self._bar_aggregator = BarAggregator(
            interval_seconds=config.bar_interval_seconds,
            on_bar=self._on_bar,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Registers a strategy with the engine."""
        if strategy.name in self._strategies:
            raise ValueError(f"Estrategia ya registrada: {strategy.name}")

        self._strategies[strategy.name] = _StrategyState(strategy=strategy)
        logger.info(f"Estrategia registrada: {strategy.name}")

    def remove_strategy(self, name: str) -> bool:
        """Removes a strategy by name."""
        if name in self._strategies:
            del self._strategies[name]
            logger.info(f"Estrategia removida: {name}")
            return True
        return False

    def add_signal_handler(self, handler: SignalHandler) -> None:
        """Adds a signal handler callback."""
        self._signal_handlers.append(handler)

    async def start(self) -> None:
        """Starts the engine: connect, subscribe, and listen to data."""
        if self._running:
            logger.warning("Engine ya está corriendo")
            return

        if self.config.auto_connect:
            await self._connect_if_needed()

        if not self.stream:
            if not self.ib_client:
                raise ValueError("Se requiere ib_client o stream para iniciar el engine")
            self.stream = MarketDataStream(self.ib_client.ib)

        if self.config.auto_subscribe:
            await self._subscribe_symbols(self.config.symbols)

        self._running = True
        logger.info("Engine iniciado")

    async def stop(self) -> None:
        """Stops the engine and unsubscribes from market data."""
        if not self._running:
            return

        self._running = False

        if self.stream:
            await self.stream.unsubscribe_all()

        if self.ib_client and self.ib_client.is_connected():
            await self.ib_client.disconnect()

        logger.info("Engine detenido")

    async def _connect_if_needed(self) -> None:
        if not self.ib_client:
            raise ValueError("ib_client requerido para conexión automática")

        if not self.ib_client.is_connected():
            await self.ib_client.connect()

    async def _subscribe_symbols(self, symbols: Iterable[str]) -> None:
        if not self.stream:
            raise ValueError("No hay stream disponible para suscribirse")

        symbols_list = list({s.strip().upper() for s in symbols if s.strip()})
        if not symbols_list:
            raise ValueError("Lista de símbolos vacía")

        logger.info(f"Suscribiendo a {len(symbols_list)} símbolos")
        await self.stream.subscribe_multiple(
            symbols_list,
            callback=self._on_tick,
            generic_tick_list=self.config.stream_generic_tick_list,
        )

    def _on_tick(self, tick: TickData) -> None:
        """Tick callback from MarketDataStream."""
        if not self._running:
            return
        try:
            self._bar_aggregator.process_tick(tick)
        except Exception as exc:
            logger.error(f"Error procesando tick {tick.symbol}: {exc}")

    def _on_bar(self, symbol: str, bar: Dict[str, object]) -> None:
        """Bar callback from BarAggregator."""
        if not self._running:
            return

        # Normalize bar fields for strategies
        normalized = dict(bar)
        if "date" not in normalized:
            normalized["date"] = normalized.get("end_time")

        for state in self._strategies.values():
            strategy = state.strategy

            if not strategy.is_running:
                continue

            if symbol not in strategy.symbols:
                continue

            try:
                strategy.on_bar(symbol, normalized)
                self._emit_new_signals(state)
            except Exception as exc:
                logger.error(f"Error en estrategia {strategy.name}: {exc}")

    def _emit_new_signals(self, state: _StrategyState) -> None:
        """Emit signals created since last check."""
        history = state.strategy.signals_history
        if state.last_signal_index >= len(history):
            return

        new_signals = history[state.last_signal_index :]
        state.last_signal_index = len(history)

        for signal in new_signals:
            self._dispatch_signal(signal, state.strategy)

    def _dispatch_signal(self, signal: Signal, strategy: BaseStrategy) -> None:
        if not self._signal_handlers:
            logger.info(
                f"Señal emitida (sin handler): {signal.signal_type.value} "
                f"{signal.symbol} @ {signal.price}"
            )
            return

        for handler in self._signal_handlers:
            try:
                result = handler(signal, strategy)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as exc:
                logger.error(f"Error en signal handler: {exc}")
