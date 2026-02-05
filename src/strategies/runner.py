from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import time

import pandas as pd

from src.utils.logger import logger
from src.utils.market_data import normalize_ohlcv
from src.strategies.base import BaseStrategy, Signal, SignalType, PositionSide


TIMEFRAME_TO_IB = {
    "1m": "1 min",
    "1min": "1 min",
    "5m": "5 mins",
    "5min": "5 mins",
    "15m": "15 mins",
    "15min": "15 mins",
    "30m": "30 mins",
    "30min": "30 mins",
    "1h": "1 hour",
    "1d": "1 day",
}


@dataclass
class StrategyRunnerConfig:
    symbol: str = "SPY"
    symbols: Optional[List[str]] = None
    timeframe: str = "5m"
    duration: str = "2 D"
    csv_path: Optional[str] = None
    use_csv: bool = False
    use_engine: bool = True
    use_market_cache: bool = True
    cache_stale_seconds: int = 90
    cache_miss_limit: int = 3
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    mode: str = "paper"
    confirm_live: bool = False
    timeout: int = 15
    exchange: str = "SMART"
    currency: str = "USD"
    poll_seconds: int = 30
    use_rth: bool = True
    disconnect_on_exit: bool = False


class StrategyRunner:
    """
    Shared runner for Basic and ML strategies.
    Calls generate_signal() and execute_trade() periodically.
    """

    def __init__(self, strategy: BaseStrategy, config: StrategyRunnerConfig):
        self.strategy = strategy
        self.config = config
        self.symbols = config.symbols or [config.symbol]
        self._ib = None
        self._contract = None
        self._adapter = None
        self._last_bar_time: Dict[str, pd.Timestamp] = {}
        self._engine_connected = False
        self._cache_miss_count: Dict[str, int] = {sym: 0 for sym in self.symbols}
        self._current_bucket_start: Dict[str, pd.Timestamp] = {}
        self._current_bar: Dict[str, Dict[str, Any]] = {}
        self._last_tick_time: Dict[str, pd.Timestamp] = {}

    def connect_ib(self):
        if self.config.use_csv:
            return
        from ib_insync import IB, Stock
        self._ib = IB()
        self._ib.connect(self.config.host, self.config.port, clientId=self.config.client_id)
        self._contract = Stock(self.config.symbol, self.config.exchange, self.config.currency)

    def disconnect_ib(self):
        if self._ib:
            self._ib.disconnect()
            self._ib = None

    def connect_engine(self):
        if self.config.use_csv:
            return
        from src.engine.frontend_adapter import get_adapter
        self._adapter = get_adapter()
        self._adapter.ensure_running()
        ok, err, _info = self._adapter.connect(
            host=self.config.host,
            port=self.config.port,
            client_id=self.config.client_id,
            mode=self.config.mode,
            timeout=self.config.timeout,
            confirm_live=self.config.confirm_live,
        )
        if not ok:
            raise RuntimeError(f"Engine connect failed: {err}")
        self._engine_connected = True

        if self.config.use_market_cache:
            subscribed = 0
            for sym in self.symbols:
                ok, err, _ = self._adapter.subscribe_market_data(sym)
                if ok:
                    subscribed += 1
                else:
                    logger.warning(f"Market data subscription failed for {sym}: {err}")
            if subscribed == 0:
                self.config.use_market_cache = False

    def disconnect_engine(self):
        if self._adapter and self.config.disconnect_on_exit and self._engine_connected:
            self._adapter.disconnect()
        self._adapter = None
        self._engine_connected = False

    def load_csv_history(self) -> pd.DataFrame:
        if not self.config.csv_path:
            raise ValueError("csv_path is required when use_csv=True")
        df = pd.read_csv(self.config.csv_path)
        df = normalize_ohlcv(df, schema="lower", set_index=False)
        return df

    def fetch_ib_history(self) -> pd.DataFrame:
        from ib_insync import util
        if not self._ib or not self._contract:
            raise RuntimeError("IB not connected")
        bar_size = TIMEFRAME_TO_IB.get(self.config.timeframe, self.config.timeframe)
        bars = self._ib.reqHistoricalData(
            self._contract,
            endDateTime="",
            durationStr=self.config.duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=self.config.use_rth,
            formatDate=1,
        )
        df = util.df(bars)
        df = normalize_ohlcv(df, schema="lower", set_index=False)
        return df

    def fetch_engine_history(self, symbol: Optional[str] = None) -> pd.DataFrame:
        if not self._adapter:
            raise RuntimeError("Engine adapter not connected")
        sym = symbol or self.config.symbol
        bar_size = TIMEFRAME_TO_IB.get(self.config.timeframe, self.config.timeframe)
        ok, err, df = self._adapter.fetch_historical_data(
            symbol=sym,
            duration=self.config.duration,
            bar_size=bar_size,
            timeout=self.config.timeout + 5,
        )
        if not ok or df is None:
            raise RuntimeError(f"Engine fetch failed: {err}")
        df = normalize_ohlcv(df, schema="lower", set_index=False)
        return df

    def fetch_engine_market_cache(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self._adapter:
            return None
        cache = self._adapter.get_cached_market_data() or {}
        return cache.get(symbol.upper())

    def _bar_from_market_cache(self, symbol: str, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not snapshot:
            return None

        price = (
            snapshot.get("last")
            or snapshot.get("close")
            or snapshot.get("mid")
            or snapshot.get("bid")
            or snapshot.get("ask")
        )
        if price is None:
            return None

        ts = snapshot.get("timestamp")
        dt = pd.to_datetime(ts) if ts else pd.Timestamp.utcnow()
        bucket_start = self._bucket_start(dt)

        if symbol not in self._current_bucket_start:
            self._current_bucket_start[symbol] = bucket_start
            self._current_bar[symbol] = self._init_bar(symbol, price, snapshot, bucket_start)
            return None

        # Same bucket: update current bar
        if bucket_start == self._current_bucket_start[symbol]:
            self._update_bar(symbol, price, snapshot)
            return None

        # New bucket: close previous bar and start a new one
        closed_bar = self._current_bar.get(symbol)
        self._current_bucket_start[symbol] = bucket_start
        self._current_bar[symbol] = self._init_bar(symbol, price, snapshot, bucket_start)
        return closed_bar

    def _bucket_start(self, dt: pd.Timestamp) -> pd.Timestamp:
        seconds = self._timeframe_seconds(self.config.timeframe)
        if seconds >= 86400:
            return dt.normalize()
        ns = int(seconds * 1_000_000_000)
        return pd.Timestamp((dt.value // ns) * ns, tz=dt.tz)

    def _timeframe_seconds(self, tf: str) -> int:
        tf = tf.strip().lower().replace(" ", "")
        if tf.endswith("min"):
            tf = tf.replace("min", "m")
        if tf.endswith("mins"):
            tf = tf.replace("mins", "m")
        if tf.endswith("hour"):
            tf = tf.replace("hour", "h")
        if tf.endswith("hours"):
            tf = tf.replace("hours", "h")
        if tf.endswith("day"):
            tf = tf.replace("day", "d")
        if tf.endswith("days"):
            tf = tf.replace("days", "d")

        num = "".join(ch for ch in tf if ch.isdigit())
        unit = "".join(ch for ch in tf if ch.isalpha())
        val = int(num) if num else 1

        if unit in ("m", "min"):
            return val * 60
        if unit in ("h", "hour"):
            return val * 3600
        if unit in ("d", "day"):
            return val * 86400
        # Fallback: 1 minute
        return 60

    def _init_bar(self, symbol: str, price: float, snapshot: Dict[str, Any], bucket_start: pd.Timestamp) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "date": bucket_start,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": snapshot.get("last_size") or 0,
        }

    def _update_bar(self, symbol: str, price: float, snapshot: Dict[str, Any]) -> None:
        if symbol not in self._current_bar:
            return
        bar = self._current_bar[symbol]
        bar["close"] = price
        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["volume"] += snapshot.get("last_size") or 0

    def _seed_history(self, symbol: str, df: pd.DataFrame):
        if df is None or df.empty:
            return
        df = normalize_ohlcv(df, schema="lower", set_index=False)
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        max_bars = getattr(self.strategy, "max_bars", None)
        if max_bars:
            df = df.tail(max_bars).reset_index(drop=True)
        self.strategy.update_data(symbol, df)
        if "date" in df.columns:
            self._last_bar_time[symbol] = df["date"].iloc[-1]

    def _bar_from_row(self, row: pd.Series, symbol: Optional[str] = None) -> Dict[str, Any]:
        bar = row.to_dict()
        if "symbol" not in bar:
            bar["symbol"] = symbol or self.config.symbol
        return bar

    def generate_signal(self, symbol: str, bar: Dict[str, Any]) -> Optional[Signal]:
        return self.strategy.on_bar(symbol, bar)

    def _is_snapshot_stale(self, timestamp: Optional[pd.Timestamp]) -> bool:
        if timestamp is None:
            return True
        if timestamp.tzinfo is None:
            now = pd.Timestamp.utcnow().tz_localize(None)
        else:
            now = pd.Timestamp.now(tz=timestamp.tz)
        return (now - timestamp).total_seconds() > self.config.cache_stale_seconds

    def execute_trade(self, signal: Optional[Signal]):
        if not signal:
            return

        price = signal.price or 0.0
        symbol = signal.symbol

        if signal.signal_type == SignalType.BUY:
            qty = signal.quantity or self.strategy.calculate_position_size(symbol, price)
            self.strategy.open_position(symbol, PositionSide.LONG, qty, price)
            print(f"ORDER: BUY {qty} {symbol} @ {price:.2f}")

        elif signal.signal_type in (SignalType.SELL, SignalType.CLOSE_LONG):
            pos = self.strategy.get_position(symbol)
            if pos and pos.is_open:
                self.strategy.close_position(symbol, price)
            print(f"ORDER: SELL {symbol} @ {price:.2f}")

        elif signal.signal_type in (SignalType.CLOSE_SHORT,):
            pos = self.strategy.get_position(symbol)
            if pos and pos.is_open:
                self.strategy.close_position(symbol, price)
            print(f"ORDER: CLOSE SHORT {symbol} @ {price:.2f}")

    def run_loop(self, iterations: Optional[int] = None):
        self.strategy.start()
        logger.info("StrategyRunner started")

        if self.config.use_csv:
            if len(self.symbols) > 1:
                logger.warning("CSV mode supports a single symbol; using the first one only.")
                self.symbols = [self.symbols[0]]
            df = self.load_csv_history()
            for i, row in df.iterrows():
                bar = self._bar_from_row(row, self.symbols[0])
                signal = self.generate_signal(self.symbols[0], bar)
                self.execute_trade(signal)
                if iterations is not None and i + 1 >= iterations:
                    break
                time.sleep(self.config.poll_seconds)
            return
        if self.config.use_engine:
            self.connect_engine()
            try:
                if self.config.use_market_cache:
                    # Warm-up with historical data once
                    for sym in self.symbols:
                        try:
                            hist = self.fetch_engine_history(sym)
                            self._seed_history(sym, hist)
                        except Exception as exc:
                            logger.warning(f"History warm-up failed for {sym}: {exc}")

                while True:
                    if self.config.use_market_cache:
                        for sym in self.symbols:
                            snapshot = self.fetch_engine_market_cache(sym)
                            ts = None
                            if snapshot:
                                ts_raw = snapshot.get("timestamp")
                                ts = pd.to_datetime(ts_raw) if ts_raw else None
                                if ts:
                                    self._last_tick_time[sym] = ts

                            if snapshot and not self._is_snapshot_stale(ts):
                                bar = self._bar_from_market_cache(sym, snapshot)
                                if bar:
                                    bar_time = bar.get("date")
                                    last_time = self._last_bar_time.get(sym)
                                    if bar_time is not None and last_time is not None:
                                        if pd.to_datetime(bar_time) <= pd.to_datetime(last_time):
                                            continue
                                    self._last_bar_time[sym] = bar_time
                                    signal = self.generate_signal(sym, bar)
                                    self.execute_trade(signal)
                                    self._cache_miss_count[sym] = 0
                                else:
                                    # data flowing but no closed bar yet
                                    self._cache_miss_count[sym] = 0
                            else:
                                self._cache_miss_count[sym] = self._cache_miss_count.get(sym, 0) + 1
                                if self._cache_miss_count[sym] >= self.config.cache_miss_limit:
                                    try:
                                        df = self.fetch_engine_history(sym)
                                        if not df.empty:
                                            last = df.iloc[-1]
                                            bar_time = last.get("date")
                                            last_time = self._last_bar_time.get(sym)
                                            if bar_time is None or last_time is None or pd.to_datetime(bar_time) > pd.to_datetime(last_time):
                                                self._last_bar_time[sym] = bar_time
                                                bar = self._bar_from_row(last, sym)
                                                signal = self.generate_signal(sym, bar)
                                                self.execute_trade(signal)
                                        self._cache_miss_count[sym] = 0
                                    except Exception as exc:
                                        logger.warning(f"Engine history fallback failed for {sym}: {exc}")
                    else:
                        for sym in self.symbols:
                            df = self.fetch_engine_history(sym)
                            if df.empty:
                                continue

                            last = df.iloc[-1]
                            bar_time = last.get("date")
                            last_time = self._last_bar_time.get(sym)
                            if bar_time is not None and last_time is not None:
                                if pd.to_datetime(bar_time) <= pd.to_datetime(last_time):
                                    continue

                            self._last_bar_time[sym] = bar_time
                            bar = self._bar_from_row(last, sym)
                            signal = self.generate_signal(sym, bar)
                            self.execute_trade(signal)

                    if iterations is not None:
                        iterations -= 1
                        if iterations <= 0:
                            break

                    time.sleep(self.config.poll_seconds)
            finally:
                self.disconnect_engine()
            return

        self.connect_ib()
        try:
            if len(self.symbols) > 1:
                logger.warning("IB direct mode supports a single symbol; using the first one only.")
                self.symbols = [self.symbols[0]]
            while True:
                df = self.fetch_ib_history()
                if df.empty:
                    time.sleep(self.config.poll_seconds)
                    continue

                last = df.iloc[-1]
                bar_time = last.get("date")
                last_time = self._last_bar_time.get(self.symbols[0])
                if bar_time is not None and last_time is not None:
                    if pd.to_datetime(bar_time) <= pd.to_datetime(last_time):
                        time.sleep(self.config.poll_seconds)
                        continue

                self._last_bar_time[self.symbols[0]] = bar_time
                bar = self._bar_from_row(last, self.symbols[0])
                signal = self.generate_signal(self.symbols[0], bar)
                self.execute_trade(signal)

                if iterations is not None:
                    iterations -= 1
                    if iterations <= 0:
                        break

                time.sleep(self.config.poll_seconds)
        finally:
            self.disconnect_ib()
