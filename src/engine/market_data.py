"""
Market Data Service
===================
Manages real-time market data subscriptions and cache updates.

Design:
- Open subscriptions once via reqMktData (no snapshots).
- Update EngineState cache on ticker callbacks.
- Engine/business logic polls the cache, not IBKR.
"""

from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Dict, Optional, Callable, Any

from ib_insync import IB, Stock

from .state import EngineState


@dataclass
class Subscription:
    symbol: str
    contract: Any
    ticker: Any
    callback: Callable


class MarketDataService:
    def __init__(self, state: EngineState, ib: Optional[IB] = None):
        self._state = state
        self._ib: Optional[IB] = ib
        self._lock = RLock()
        self._subs: Dict[str, Subscription] = {}

    def set_ib(self, ib: Optional[IB]) -> None:
        with self._lock:
            self._ib = ib

    def _on_ticker_update(self, symbol: str, ticker) -> None:
        def _val(value):
            if value is None or value == -1:
                return None
            return value

        bid = _val(getattr(ticker, "bid", None))
        ask = _val(getattr(ticker, "ask", None))
        data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "last": _val(getattr(ticker, "last", None)),
            "last_size": _val(getattr(ticker, "lastSize", None)),
            "bid": bid,
            "bid_size": _val(getattr(ticker, "bidSize", None)),
            "ask": ask,
            "ask_size": _val(getattr(ticker, "askSize", None)),
            "open": _val(getattr(ticker, "open", None)),
            "high": _val(getattr(ticker, "high", None)),
            "low": _val(getattr(ticker, "low", None)),
            "close": _val(getattr(ticker, "close", None)),
            "volume": _val(getattr(ticker, "volume", None)),
            "vwap": _val(getattr(ticker, "vwap", None)),
        }
        if bid is not None and ask is not None:
            data["mid"] = (bid + ask) / 2
            data["spread"] = ask - bid
        else:
            data["mid"] = None
            data["spread"] = None

        self._state.set_market_data(symbol, data)

    def subscribe(self, symbol: str) -> dict:
        if not self._ib or not self._ib.isConnected():
            return {"status": "error", "error": "Not connected"}

        sym = symbol.upper().strip()
        if not sym:
            return {"status": "error", "error": "Invalid symbol"}

        with self._lock:
            if sym in self._subs:
                return {"status": "already_subscribed", "symbol": sym}

        contract = Stock(sym, "SMART", "USD")
        qualified = self._ib.qualifyContracts(contract)
        if not qualified:
            return {"status": "error", "error": f"Contract not found: {sym}"}
        contract = qualified[0]

        ticker = self._ib.reqMktData(
            contract,
            genericTickList="",
            snapshot=False,
            regulatorySnapshot=False,
        )
        callback = lambda t, s=sym: self._on_ticker_update(s, t)
        ticker.updateEvent += callback

        with self._lock:
            self._subs[sym] = Subscription(symbol=sym, contract=contract, ticker=ticker, callback=callback)

        return {"status": "subscribed", "symbol": sym}

    def unsubscribe(self, symbol: str) -> dict:
        if not self._ib or not self._ib.isConnected():
            return {"status": "error", "error": "Not connected"}

        sym = symbol.upper().strip()
        with self._lock:
            sub = self._subs.get(sym)

        if not sub:
            return {"status": "not_subscribed", "symbol": sym}

        try:
            try:
                sub.ticker.updateEvent -= sub.callback
            except Exception:
                pass
            self._ib.cancelMktData(sub.contract)
        finally:
            with self._lock:
                self._subs.pop(sym, None)

        return {"status": "unsubscribed", "symbol": sym}

    def stop_all(self) -> None:
        if not self._ib:
            return
        with self._lock:
            symbols = list(self._subs.keys())
        for sym in symbols:
            try:
                self.unsubscribe(sym)
            except Exception:
                pass

    def resubscribe_all(self) -> None:
        """Resubscribe to all symbols after reconnect."""
        with self._lock:
            symbols = list(self._subs.keys())
            # Clear old subscriptions (tickers/handlers invalid after reconnect)
            self._subs.clear()
        for sym in symbols:
            try:
                self.subscribe(sym)
            except Exception:
                pass

    def get_subscriptions(self) -> Dict[str, dict]:
        with self._lock:
            return {sym: {"symbol": sub.symbol} for sym, sub in self._subs.items()}
