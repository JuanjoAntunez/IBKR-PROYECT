"""
Módulo para obtención de datos históricos de Interactive Brokers.

Implementa rate limiting y caché para respetar los límites de IB
y optimizar el rendimiento.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from ib_insync import IB, Stock, Contract, BarData

from src.utils.logger import logger

try:
    from config.settings import data_config, ib_config
except ImportError:
    from config.settings import DataConfig, IBConfig
    data_config = DataConfig()
    ib_config = IBConfig()


@dataclass
class HistoricalBarsRequest:
    """Parámetros para solicitar datos históricos."""

    symbol: str
    exchange: str = "SMART"
    currency: str = "USD"
    duration: str = "1 M"
    bar_size: str = "1 day"
    what_to_show: str = "TRADES"
    use_rth: bool = True
    end_datetime: str = ""


class RateLimiter:
    """
    Controlador de rate limiting para respetar límites de IB.

    IB permite aproximadamente 60 requests de datos históricos
    cada 10 minutos. Esta clase mantiene un registro de requests
    y bloquea si se excede el límite.
    """

    def __init__(
        self,
        max_requests: int = 50,
        period_seconds: int = 600,
    ):
        """
        Inicializa el rate limiter.

        Args:
            max_requests: Máximo de requests permitidos por periodo
            period_seconds: Duración del periodo en segundos
        """
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self._request_times: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Adquiere permiso para hacer una request.

        Espera si es necesario para no exceder el rate limit.
        """
        async with self._lock:
            now = time.time()
            cutoff = now - self.period_seconds

            # Limpiar requests antiguas
            self._request_times = [
                t for t in self._request_times if t > cutoff
            ]

            if len(self._request_times) >= self.max_requests:
                # Calcular tiempo de espera
                oldest = self._request_times[0]
                wait_time = oldest + self.period_seconds - now + 1

                logger.warning(
                    f"Rate limit alcanzado. Esperando {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)

                # Limpiar de nuevo después de esperar
                now = time.time()
                cutoff = now - self.period_seconds
                self._request_times = [
                    t for t in self._request_times if t > cutoff
                ]

            self._request_times.append(now)
            logger.debug(
                f"Rate limiter: {len(self._request_times)}/{self.max_requests} "
                f"requests en los últimos {self.period_seconds}s"
            )

    @property
    def remaining_requests(self) -> int:
        """Número de requests disponibles en el periodo actual."""
        now = time.time()
        cutoff = now - self.period_seconds
        active_requests = sum(1 for t in self._request_times if t > cutoff)
        return max(0, self.max_requests - active_requests)


class DataCache:
    """
    Sistema de caché en disco para datos históricos.

    Almacena datos en formato Parquet para eficiencia y
    evita requests repetitivas a IB.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Inicializa el sistema de caché.

        Args:
            cache_dir: Directorio para almacenar archivos de caché
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = data_config.cache_enabled
        logger.debug(f"DataCache inicializado en {self.cache_dir}")

    def _get_cache_key(self, request: HistoricalBarsRequest) -> str:
        """Genera una clave única para el request."""
        key_data = f"{request.symbol}_{request.exchange}_{request.currency}_" \
                   f"{request.duration}_{request.bar_size}_{request.what_to_show}_" \
                   f"{request.use_rth}_{request.end_datetime}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Obtiene la ruta del archivo de caché."""
        return self.cache_dir / f"{key}.parquet"

    def get(self, request: HistoricalBarsRequest) -> Optional[pd.DataFrame]:
        """
        Obtiene datos del caché si existen.

        Args:
            request: Parámetros de la solicitud

        Returns:
            DataFrame con datos o None si no hay caché
        """
        if not self._enabled:
            return None

        key = self._get_cache_key(request)
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logger.debug(f"Datos obtenidos del caché: {request.symbol}")
                return df
            except Exception as e:
                logger.warning(f"Error leyendo caché: {e}")
                return None

        return None

    def set(
        self,
        request: HistoricalBarsRequest,
        data: pd.DataFrame,
    ) -> None:
        """
        Guarda datos en el caché.

        Args:
            request: Parámetros de la solicitud
            data: DataFrame con los datos a guardar
        """
        if not self._enabled:
            return

        key = self._get_cache_key(request)
        cache_path = self._get_cache_path(key)

        try:
            data.to_parquet(cache_path, index=False)
            logger.debug(f"Datos guardados en caché: {request.symbol}")
        except Exception as e:
            logger.warning(f"Error guardando caché: {e}")

    def clear(self, symbol: Optional[str] = None) -> int:
        """
        Limpia el caché.

        Args:
            symbol: Si se especifica, solo limpia caché de ese símbolo

        Returns:
            Número de archivos eliminados
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Error eliminando {cache_file}: {e}")

        logger.info(f"Caché limpiado: {count} archivos eliminados")
        return count


class HistoricalDataFetcher:
    """
    Clase principal para obtener datos históricos de IB.

    Ejemplo de uso:
        async with IBClient() as client:
            fetcher = HistoricalDataFetcher(client.ib)
            df = await fetcher.get_stock_bars("AAPL")
            print(df)
    """

    def __init__(
        self,
        ib: IB,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Inicializa el fetcher de datos históricos.

        Args:
            ib: Instancia conectada de IB
            use_cache: Si usar caché para evitar requests repetitivas
            cache_dir: Directorio de caché (usa default si None)
        """
        self.ib = ib
        self.rate_limiter = RateLimiter(
            max_requests=ib_config.max_requests_per_period,
            period_seconds=ib_config.rate_limit_period,
        )
        self.cache = DataCache(
            cache_dir=cache_dir or data_config.cache_dir
        ) if use_cache else None

        logger.info("HistoricalDataFetcher inicializado")

    async def get_stock_bars(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
        duration: str = "1 M",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        end_datetime: str = "",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Obtiene barras históricas para una acción.

        Args:
            symbol: Símbolo del instrumento (ej: "AAPL", "MSFT")
            exchange: Exchange (SMART para routing inteligente)
            currency: Moneda del instrumento
            duration: Período de datos (ej: "1 D", "1 W", "1 M", "1 Y")
            bar_size: Tamaño de barra (ej: "1 min", "5 mins", "1 hour", "1 day")
            what_to_show: Tipo de datos (TRADES, MIDPOINT, BID, ASK, etc)
            use_rth: Solo horario regular de trading
            end_datetime: Fecha final (vacío = ahora)
            use_cache: Si usar caché para esta request

        Returns:
            DataFrame con columnas: date, open, high, low, close, volume, average, barCount

        Raises:
            ValueError: Si los parámetros son inválidos
            ConnectionError: Si no hay conexión con IB
        """
        if not self.ib.isConnected():
            raise ConnectionError("No hay conexión activa con IB")

        request = HistoricalBarsRequest(
            symbol=symbol,
            exchange=exchange,
            currency=currency,
            duration=duration,
            bar_size=bar_size,
            what_to_show=what_to_show,
            use_rth=use_rth,
            end_datetime=end_datetime,
        )

        # Intentar obtener del caché
        if use_cache and self.cache:
            cached_data = self.cache.get(request)
            if cached_data is not None:
                logger.info(f"Datos de {symbol} obtenidos del caché")
                return cached_data

        # Crear contrato de stock
        contract = Stock(symbol, exchange, currency)

        # Validar contrato
        logger.debug(f"Validando contrato para {symbol}...")
        qualified = await self.ib.qualifyContractsAsync(contract)

        if not qualified:
            raise ValueError(
                f"No se encontró definición para {symbol} en {exchange}/{currency}"
            )

        contract = qualified[0]
        logger.debug(f"Contrato validado: {contract}")

        # Esperar si es necesario por rate limiting
        await self.rate_limiter.acquire()

        # Solicitar datos históricos
        logger.info(
            f"Solicitando datos históricos: {symbol} "
            f"(duration={duration}, bar_size={bar_size})"
        )

        try:
            bars: List[BarData] = await self.ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime=end_datetime,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=2,  # Unix timestamp
            )
        except Exception as e:
            logger.error(f"Error obteniendo datos históricos de {symbol}: {e}")
            raise

        if not bars:
            logger.warning(f"No se obtuvieron datos para {symbol}")
            return pd.DataFrame()

        # Convertir a DataFrame
        df = self._bars_to_dataframe(bars)

        logger.info(f"Obtenidas {len(df)} barras para {symbol}")

        # Guardar en caché
        if use_cache and self.cache and not df.empty:
            self.cache.set(request, df)

        return df

    async def get_multiple_stocks(
        self,
        symbols: List[str],
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos históricos para múltiples acciones.

        Args:
            symbols: Lista de símbolos
            **kwargs: Parámetros adicionales para get_stock_bars

        Returns:
            Diccionario {símbolo: DataFrame}
        """
        results: Dict[str, pd.DataFrame] = {}

        logger.info(f"Obteniendo datos para {len(symbols)} símbolos...")

        for symbol in symbols:
            try:
                df = await self.get_stock_bars(symbol, **kwargs)
                results[symbol] = df
            except Exception as e:
                logger.error(f"Error obteniendo {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        successful = sum(1 for df in results.values() if not df.empty)
        logger.info(
            f"Completado: {successful}/{len(symbols)} símbolos obtenidos"
        )

        return results

    def _bars_to_dataframe(self, bars: List[BarData]) -> pd.DataFrame:
        """
        Convierte lista de BarData a DataFrame.

        Args:
            bars: Lista de objetos BarData de IB

        Returns:
            DataFrame con datos de las barras
        """
        if not bars:
            return pd.DataFrame()

        data = []
        for bar in bars:
            data.append({
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "average": bar.average,
                "barCount": bar.barCount,
            })

        df = pd.DataFrame(data)

        # Convertir fecha si es string
        if df["date"].dtype == object:
            df["date"] = pd.to_datetime(df["date"])

        return df

    @property
    def remaining_requests(self) -> int:
        """Número de requests disponibles antes de rate limiting."""
        return self.rate_limiter.remaining_requests


# Valores válidos para referencia
VALID_BAR_SIZES = [
    "1 secs", "5 secs", "10 secs", "15 secs", "30 secs",
    "1 min", "2 mins", "3 mins", "5 mins", "10 mins", "15 mins", "20 mins", "30 mins",
    "1 hour", "2 hours", "3 hours", "4 hours", "8 hours",
    "1 day", "1 week", "1 month",
]

VALID_WHAT_TO_SHOW = [
    "TRADES", "MIDPOINT", "BID", "ASK",
    "BID_ASK", "ADJUSTED_LAST", "HISTORICAL_VOLATILITY",
    "OPTION_IMPLIED_VOLATILITY", "REBATE_RATE", "FEE_RATE",
    "YIELD_BID", "YIELD_ASK", "YIELD_BID_ASK", "YIELD_LAST",
]

VALID_DURATIONS = [
    "60 S", "120 S", "1800 S",  # Segundos
    "1 D", "2 D", "1 W", "2 W",  # Días/Semanas
    "1 M", "2 M", "3 M", "6 M",  # Meses
    "1 Y", "2 Y",  # Años
]
