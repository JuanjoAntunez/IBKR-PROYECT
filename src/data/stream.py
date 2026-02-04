"""
Módulo para streaming de datos en tiempo real de Interactive Brokers.

Proporciona suscripción a datos de mercado en tiempo real,
incluyendo precios, volumen y datos de nivel 2.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Callable, Any, Set

from ib_insync import IB, Stock, Contract, Ticker

from src.utils.logger import logger


class TickType(Enum):
    """Tipos de datos de tick disponibles."""

    LAST = "last"
    BID = "bid"
    ASK = "ask"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    BID_SIZE = "bidSize"
    ASK_SIZE = "askSize"
    LAST_SIZE = "lastSize"


@dataclass
class TickData:
    """Datos de un tick de mercado."""

    symbol: str
    timestamp: datetime
    last: Optional[float] = None
    last_size: Optional[int] = None
    bid: Optional[float] = None
    bid_size: Optional[int] = None
    ask: Optional[float] = None
    ask_size: Optional[int] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    vwap: Optional[float] = None

    @property
    def mid(self) -> Optional[float]:
        """Precio medio entre bid y ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Spread entre bid y ask."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "last": self.last,
            "last_size": self.last_size,
            "bid": self.bid,
            "bid_size": self.bid_size,
            "ask": self.ask,
            "ask_size": self.ask_size,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "mid": self.mid,
            "spread": self.spread,
        }


@dataclass
class StreamSubscription:
    """Representa una suscripción activa a datos de mercado."""

    symbol: str
    contract: Contract
    ticker: Optional[Ticker] = None
    callbacks: List[Callable[[TickData], None]] = field(default_factory=list)
    is_active: bool = False


# Type alias para callbacks
TickCallback = Callable[[TickData], None]


class MarketDataStream:
    """
    Gestor de streaming de datos de mercado en tiempo real.

    Permite suscribirse a múltiples símbolos y recibir actualizaciones
    de precios en tiempo real mediante callbacks.

    Ejemplo de uso:
        async with IBClient() as client:
            stream = MarketDataStream(client.ib)

            def on_tick(data: TickData):
                print(f"{data.symbol}: {data.last}")

            await stream.subscribe("AAPL", callback=on_tick)
            await asyncio.sleep(60)  # Escuchar por 1 minuto
            await stream.unsubscribe("AAPL")
    """

    def __init__(self, ib: IB):
        """
        Inicializa el gestor de streaming.

        Args:
            ib: Instancia conectada de IB
        """
        self.ib = ib
        self._subscriptions: Dict[str, StreamSubscription] = {}
        self._global_callbacks: List[TickCallback] = []
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

        logger.info("MarketDataStream inicializado")

    async def subscribe(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
        callback: Optional[TickCallback] = None,
        generic_tick_list: str = "",
    ) -> bool:
        """
        Suscribe a datos de mercado en tiempo real para un símbolo.

        Args:
            symbol: Símbolo del instrumento (ej: "AAPL")
            exchange: Exchange (SMART para routing inteligente)
            currency: Moneda del instrumento
            callback: Función a llamar cuando llegan nuevos datos
            generic_tick_list: Lista de ticks genéricos adicionales

        Returns:
            True si la suscripción fue exitosa

        Raises:
            ConnectionError: Si no hay conexión con IB
            ValueError: Si el símbolo no es válido
        """
        if not self.ib.isConnected():
            raise ConnectionError("No hay conexión activa con IB")

        # Verificar si ya existe suscripción
        if symbol in self._subscriptions:
            sub = self._subscriptions[symbol]
            if callback and callback not in sub.callbacks:
                sub.callbacks.append(callback)
                logger.debug(f"Callback añadido a suscripción existente: {symbol}")
            return True

        # Crear contrato
        contract = Stock(symbol, exchange, currency)

        # Validar contrato
        logger.debug(f"Validando contrato para streaming: {symbol}...")
        qualified = await self.ib.qualifyContractsAsync(contract)

        if not qualified:
            raise ValueError(
                f"No se encontró definición para {symbol} en {exchange}/{currency}"
            )

        contract = qualified[0]

        # Crear suscripción
        subscription = StreamSubscription(
            symbol=symbol,
            contract=contract,
            callbacks=[callback] if callback else [],
        )

        # Solicitar datos de mercado
        logger.info(f"Suscribiendo a datos de mercado: {symbol}")

        try:
            ticker = self.ib.reqMktData(
                contract=contract,
                genericTickList=generic_tick_list,
                snapshot=False,
                regulatorySnapshot=False,
            )

            subscription.ticker = ticker
            subscription.is_active = True
            self._subscriptions[symbol] = subscription

            # Configurar callback de actualización
            ticker.updateEvent += lambda t: self._on_ticker_update(symbol, t)

            logger.info(f"Suscripción activa: {symbol}")
            return True

        except Exception as e:
            logger.error(f"Error suscribiendo a {symbol}: {e}")
            raise

    async def subscribe_multiple(
        self,
        symbols: List[str],
        callback: Optional[TickCallback] = None,
        **kwargs,
    ) -> Dict[str, bool]:
        """
        Suscribe a múltiples símbolos.

        Args:
            symbols: Lista de símbolos
            callback: Callback común para todos los símbolos
            **kwargs: Parámetros adicionales para subscribe

        Returns:
            Diccionario {símbolo: éxito}
        """
        results: Dict[str, bool] = {}

        logger.info(f"Suscribiendo a {len(symbols)} símbolos...")

        for symbol in symbols:
            try:
                success = await self.subscribe(symbol, callback=callback, **kwargs)
                results[symbol] = success
            except Exception as e:
                logger.error(f"Error suscribiendo a {symbol}: {e}")
                results[symbol] = False

        successful = sum(1 for v in results.values() if v)
        logger.info(f"Suscripciones completadas: {successful}/{len(symbols)}")

        return results

    async def unsubscribe(self, symbol: str) -> bool:
        """
        Cancela la suscripción a un símbolo.

        Args:
            symbol: Símbolo a desuscribir

        Returns:
            True si se canceló correctamente
        """
        if symbol not in self._subscriptions:
            logger.warning(f"No existe suscripción para {symbol}")
            return False

        subscription = self._subscriptions[symbol]

        try:
            if subscription.ticker:
                self.ib.cancelMktData(subscription.contract)

            subscription.is_active = False
            del self._subscriptions[symbol]

            logger.info(f"Suscripción cancelada: {symbol}")
            return True

        except Exception as e:
            logger.error(f"Error cancelando suscripción de {symbol}: {e}")
            return False

    async def unsubscribe_all(self) -> int:
        """
        Cancela todas las suscripciones activas.

        Returns:
            Número de suscripciones canceladas
        """
        symbols = list(self._subscriptions.keys())
        count = 0

        for symbol in symbols:
            if await self.unsubscribe(symbol):
                count += 1

        logger.info(f"Canceladas {count} suscripciones")
        return count

    def add_global_callback(self, callback: TickCallback) -> None:
        """
        Añade un callback que recibe datos de todos los símbolos.

        Args:
            callback: Función a llamar con cada actualización
        """
        if callback not in self._global_callbacks:
            self._global_callbacks.append(callback)
            logger.debug("Callback global añadido")

    def remove_global_callback(self, callback: TickCallback) -> bool:
        """
        Elimina un callback global.

        Args:
            callback: Función a eliminar

        Returns:
            True si se eliminó correctamente
        """
        if callback in self._global_callbacks:
            self._global_callbacks.remove(callback)
            logger.debug("Callback global eliminado")
            return True
        return False

    def _on_ticker_update(self, symbol: str, ticker: Ticker) -> None:
        """
        Callback interno cuando se actualiza un ticker.

        Args:
            symbol: Símbolo actualizado
            ticker: Objeto Ticker con datos actualizados
        """
        if symbol not in self._subscriptions:
            return

        subscription = self._subscriptions[symbol]

        # Crear objeto TickData
        tick_data = TickData(
            symbol=symbol,
            timestamp=datetime.now(),
            last=ticker.last if ticker.last != -1 else None,
            last_size=ticker.lastSize if ticker.lastSize != -1 else None,
            bid=ticker.bid if ticker.bid != -1 else None,
            bid_size=ticker.bidSize if ticker.bidSize != -1 else None,
            ask=ticker.ask if ticker.ask != -1 else None,
            ask_size=ticker.askSize if ticker.askSize != -1 else None,
            open=ticker.open if ticker.open != -1 else None,
            high=ticker.high if ticker.high != -1 else None,
            low=ticker.low if ticker.low != -1 else None,
            close=ticker.close if ticker.close != -1 else None,
            volume=ticker.volume if ticker.volume != -1 else None,
            vwap=ticker.vwap if ticker.vwap != -1 else None,
        )

        # Llamar callbacks específicos del símbolo
        for callback in subscription.callbacks:
            try:
                callback(tick_data)
            except Exception as e:
                logger.error(f"Error en callback de {symbol}: {e}")

        # Llamar callbacks globales
        for callback in self._global_callbacks:
            try:
                callback(tick_data)
            except Exception as e:
                logger.error(f"Error en callback global: {e}")

    def get_latest(self, symbol: str) -> Optional[TickData]:
        """
        Obtiene el último tick recibido para un símbolo.

        Args:
            symbol: Símbolo a consultar

        Returns:
            TickData con últimos datos o None si no hay suscripción
        """
        if symbol not in self._subscriptions:
            return None

        subscription = self._subscriptions[symbol]
        ticker = subscription.ticker

        if not ticker:
            return None

        return TickData(
            symbol=symbol,
            timestamp=datetime.now(),
            last=ticker.last if ticker.last != -1 else None,
            last_size=ticker.lastSize if ticker.lastSize != -1 else None,
            bid=ticker.bid if ticker.bid != -1 else None,
            bid_size=ticker.bidSize if ticker.bidSize != -1 else None,
            ask=ticker.ask if ticker.ask != -1 else None,
            ask_size=ticker.askSize if ticker.askSize != -1 else None,
            open=ticker.open if ticker.open != -1 else None,
            high=ticker.high if ticker.high != -1 else None,
            low=ticker.low if ticker.low != -1 else None,
            close=ticker.close if ticker.close != -1 else None,
            volume=ticker.volume if ticker.volume != -1 else None,
            vwap=ticker.vwap if ticker.vwap != -1 else None,
        )

    def get_all_latest(self) -> Dict[str, TickData]:
        """
        Obtiene los últimos ticks de todos los símbolos suscritos.

        Returns:
            Diccionario {símbolo: TickData}
        """
        results: Dict[str, TickData] = {}

        for symbol in self._subscriptions:
            tick = self.get_latest(symbol)
            if tick:
                results[symbol] = tick

        return results

    @property
    def subscribed_symbols(self) -> Set[str]:
        """Conjunto de símbolos con suscripción activa."""
        return set(self._subscriptions.keys())

    @property
    def subscription_count(self) -> int:
        """Número de suscripciones activas."""
        return len(self._subscriptions)

    async def snapshot(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
        timeout: float = 5.0,
    ) -> Optional[TickData]:
        """
        Obtiene un snapshot de datos de mercado (sin suscripción continua).

        Útil para obtener precio actual sin mantener suscripción abierta.

        Args:
            symbol: Símbolo del instrumento
            exchange: Exchange
            currency: Moneda
            timeout: Tiempo máximo de espera en segundos

        Returns:
            TickData con datos actuales o None si falla
        """
        if not self.ib.isConnected():
            raise ConnectionError("No hay conexión activa con IB")

        contract = Stock(symbol, exchange, currency)

        # Validar contrato
        qualified = await self.ib.qualifyContractsAsync(contract)
        if not qualified:
            raise ValueError(f"No se encontró definición para {symbol}")

        contract = qualified[0]

        logger.debug(f"Solicitando snapshot de {symbol}...")

        try:
            ticker = self.ib.reqMktData(
                contract=contract,
                genericTickList="",
                snapshot=True,
                regulatorySnapshot=False,
            )

            # Esperar a que lleguen datos
            start_time = asyncio.get_event_loop().time()
            while ticker.last == -1 and ticker.bid == -1:
                await asyncio.sleep(0.1)
                if asyncio.get_event_loop().time() - start_time > timeout:
                    logger.warning(f"Timeout esperando snapshot de {symbol}")
                    break

            tick_data = TickData(
                symbol=symbol,
                timestamp=datetime.now(),
                last=ticker.last if ticker.last != -1 else None,
                last_size=ticker.lastSize if ticker.lastSize != -1 else None,
                bid=ticker.bid if ticker.bid != -1 else None,
                bid_size=ticker.bidSize if ticker.bidSize != -1 else None,
                ask=ticker.ask if ticker.ask != -1 else None,
                ask_size=ticker.askSize if ticker.askSize != -1 else None,
                open=ticker.open if ticker.open != -1 else None,
                high=ticker.high if ticker.high != -1 else None,
                low=ticker.low if ticker.low != -1 else None,
                close=ticker.close if ticker.close != -1 else None,
                volume=ticker.volume if ticker.volume != -1 else None,
            )

            logger.debug(f"Snapshot obtenido: {symbol} @ {tick_data.last}")
            return tick_data

        except Exception as e:
            logger.error(f"Error obteniendo snapshot de {symbol}: {e}")
            return None


class BarAggregator:
    """
    Agrega ticks en barras de tiempo configurable.

    Útil para crear barras de 1 minuto, 5 minutos, etc.
    a partir del stream de ticks.
    """

    def __init__(
        self,
        interval_seconds: int = 60,
        on_bar: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        Inicializa el agregador de barras.

        Args:
            interval_seconds: Intervalo de cada barra en segundos
            on_bar: Callback cuando se completa una barra
        """
        self.interval = interval_seconds
        self.on_bar = on_bar
        self._bars: Dict[str, Dict[str, Any]] = {}
        self._last_bar_time: Dict[str, datetime] = {}

        logger.debug(f"BarAggregator inicializado (intervalo={interval_seconds}s)")

    def process_tick(self, tick: TickData) -> Optional[Dict[str, Any]]:
        """
        Procesa un tick y lo agrega a la barra actual.

        Args:
            tick: Datos del tick

        Returns:
            Barra completada si el intervalo terminó, None si no
        """
        symbol = tick.symbol
        price = tick.last

        if price is None:
            return None

        now = tick.timestamp
        current_bar_time = self._get_bar_time(now)

        # Verificar si es nueva barra
        if symbol not in self._last_bar_time:
            self._last_bar_time[symbol] = current_bar_time
            self._bars[symbol] = self._create_new_bar(symbol, price, tick.volume or 0)
            return None

        # Si cambió el intervalo, cerrar barra anterior
        if current_bar_time > self._last_bar_time[symbol]:
            completed_bar = self._bars[symbol].copy()
            completed_bar["close"] = price
            completed_bar["end_time"] = self._last_bar_time[symbol]

            # Callback si está configurado
            if self.on_bar:
                try:
                    self.on_bar(symbol, completed_bar)
                except Exception as e:
                    logger.error(f"Error en callback on_bar: {e}")

            # Iniciar nueva barra
            self._last_bar_time[symbol] = current_bar_time
            self._bars[symbol] = self._create_new_bar(symbol, price, tick.volume or 0)

            return completed_bar

        # Actualizar barra actual
        bar = self._bars[symbol]
        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        bar["volume"] += tick.last_size or 0
        bar["tick_count"] += 1

        return None

    def _get_bar_time(self, dt: datetime) -> datetime:
        """Calcula el inicio del intervalo de barra."""
        timestamp = dt.timestamp()
        bar_timestamp = (timestamp // self.interval) * self.interval
        return datetime.fromtimestamp(bar_timestamp)

    def _create_new_bar(
        self,
        symbol: str,
        price: float,
        volume: int,
    ) -> Dict[str, Any]:
        """Crea una nueva barra."""
        return {
            "symbol": symbol,
            "start_time": self._last_bar_time.get(symbol, datetime.now()),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume,
            "tick_count": 1,
        }

    def get_current_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtiene la barra actual en construcción."""
        return self._bars.get(symbol)
