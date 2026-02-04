"""
Engine State Management
=======================
Centralized state for the Trading Engine with thread-safe access.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from threading import RLock
from copy import deepcopy
import pandas as pd


class ConnectionStatus(Enum):
    """Connection status with IB."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class Position:
    """Represents a position in the portfolio."""
    symbol: str
    quantity: int
    avg_cost: float
    market_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    account: str = ""
    currency: str = "USD"
    sec_type: str = "STK"
    exchange: str = "SMART"
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def pnl_percent(self) -> float:
        """Calculate P&L percentage."""
        if self.avg_cost > 0:
            return ((self.market_price - self.avg_cost) / self.avg_cost) * 100
        return 0.0


@dataclass
class Order:
    """Represents an order."""
    order_id: int
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    order_type: str  # MKT, LMT, STP, etc.
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_key: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class AccountSummary:
    """Account summary information."""
    account_id: str = ""
    net_liquidation: float = 0.0
    total_cash: float = 0.0
    buying_power: float = 0.0
    available_funds: float = 0.0
    excess_liquidity: float = 0.0
    init_margin_req: float = 0.0
    maint_margin_req: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    cushion: float = 0.0
    currency: str = "USD"
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class HistoricalData:
    """Historical market data."""
    symbol: str
    data: pd.DataFrame
    duration: str
    bar_size: str
    fetched_at: datetime = field(default_factory=datetime.now)


class EngineState:
    """
    Thread-safe centralized state for the Trading Engine.

    All state modifications happen through the Single Writer (engine thread).
    Reads can happen from any thread through snapshot methods.
    """

    def __init__(self):
        self._lock = RLock()

        # Connection state
        self._connection_status: ConnectionStatus = ConnectionStatus.DISCONNECTED
        self._connected_host: Optional[str] = None
        self._connected_port: Optional[int] = None
        self._trading_mode: str = "paper"
        self._platform: str = "TWS"
        self._accounts: List[str] = []

        # Account state
        self._account_summary: AccountSummary = AccountSummary()

        # Portfolio state
        self._positions: Dict[str, Position] = {}  # symbol -> Position

        # Orders state
        self._orders: Dict[int, Order] = {}  # order_id -> Order
        self._next_order_id: int = 1

        # Market data cache
        self._historical_data: Dict[str, HistoricalData] = {}  # symbol -> HistoricalData
        self._market_data: Dict[str, Dict[str, Any]] = {}  # symbol -> latest tick dict

        # Debug/logging
        self._messages: List[str] = []
        self._max_messages: int = 1000

    # =========================================================================
    # Connection State
    # =========================================================================

    @property
    def connection_status(self) -> ConnectionStatus:
        with self._lock:
            return self._connection_status

    def set_connection_status(self, status: ConnectionStatus):
        with self._lock:
            self._connection_status = status
            self._log(f"Connection status: {status.value}")

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connection_status == ConnectionStatus.CONNECTED

    def set_connected(self, host: str, port: int, mode: str, platform: str, accounts: List[str]):
        with self._lock:
            self._connection_status = ConnectionStatus.CONNECTED
            self._connected_host = host
            self._connected_port = port
            self._trading_mode = mode
            self._platform = platform
            self._accounts = accounts
            self._log(f"Connected to {platform} {mode} at {host}:{port}")

    def set_disconnected(self):
        with self._lock:
            self._connection_status = ConnectionStatus.DISCONNECTED
            self._connected_host = None
            self._connected_port = None
            self._log("Disconnected")

    def get_connection_info(self) -> dict:
        with self._lock:
            return {
                "status": self._connection_status.value,
                "host": self._connected_host,
                "port": self._connected_port,
                "mode": self._trading_mode,
                "platform": self._platform,
                "accounts": list(self._accounts),
            }

    # =========================================================================
    # Account State
    # =========================================================================

    def update_account_summary(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._account_summary, key):
                    setattr(self._account_summary, key, value)
            self._account_summary.last_update = datetime.now()

    def get_account_summary(self) -> AccountSummary:
        """Return a snapshot of account summary."""
        with self._lock:
            return deepcopy(self._account_summary)

    # =========================================================================
    # Portfolio State
    # =========================================================================

    def update_position(self, position: Position):
        with self._lock:
            self._positions[position.symbol] = position
            self._log(f"Position updated: {position.symbol} qty={position.quantity}")

    def remove_position(self, symbol: str):
        with self._lock:
            if symbol in self._positions:
                del self._positions[symbol]
                self._log(f"Position removed: {symbol}")

    def get_positions(self) -> Dict[str, Position]:
        """Return a snapshot of all positions."""
        with self._lock:
            return deepcopy(self._positions)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return a snapshot of a specific position."""
        with self._lock:
            pos = self._positions.get(symbol)
            return deepcopy(pos) if pos else None

    def clear_positions(self):
        with self._lock:
            self._positions.clear()

    # =========================================================================
    # Orders State
    # =========================================================================

    def get_next_order_id(self) -> int:
        with self._lock:
            order_id = self._next_order_id
            self._next_order_id += 1
            return order_id

    def set_next_order_id(self, order_id: int):
        with self._lock:
            self._next_order_id = max(self._next_order_id, order_id)

    def add_order(self, order: Order):
        with self._lock:
            self._orders[order.order_id] = order
            self._log(f"Order added: {order.order_id} {order.action} {order.quantity} {order.symbol}")

    def update_order(self, order_id: int, **kwargs):
        with self._lock:
            if order_id in self._orders:
                order = self._orders[order_id]
                for key, value in kwargs.items():
                    if hasattr(order, key):
                        setattr(order, key, value)
                order.updated_at = datetime.now()

    def get_orders(self) -> Dict[int, Order]:
        """Return a snapshot of all orders."""
        with self._lock:
            return deepcopy(self._orders)

    def get_order(self, order_id: int) -> Optional[Order]:
        """Return a snapshot of a specific order."""
        with self._lock:
            order = self._orders.get(order_id)
            return deepcopy(order) if order else None

    def get_active_orders(self) -> List[Order]:
        """Return orders that are still active (not filled/cancelled)."""
        with self._lock:
            active_statuses = {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}
            return [deepcopy(o) for o in self._orders.values() if o.status in active_statuses]

    # =========================================================================
    # Historical Data Cache
    # =========================================================================

    def set_historical_data(self, symbol: str, data: pd.DataFrame, duration: str, bar_size: str):
        with self._lock:
            self._historical_data[symbol] = HistoricalData(
                symbol=symbol,
                data=data.copy(),
                duration=duration,
                bar_size=bar_size,
            )

    def get_historical_data(self, symbol: str) -> Optional[HistoricalData]:
        with self._lock:
            hd = self._historical_data.get(symbol)
            if hd:
                return HistoricalData(
                    symbol=hd.symbol,
                    data=hd.data.copy(),
                    duration=hd.duration,
                    bar_size=hd.bar_size,
                    fetched_at=hd.fetched_at,
                )
            return None

    # =========================================================================
    # Market Data Cache
    # =========================================================================

    def set_market_data(self, symbol: str, data: Dict[str, Any]):
        """Store latest market data snapshot for a symbol."""
        with self._lock:
            self._market_data[symbol] = deepcopy(data)

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest market data for a symbol."""
        with self._lock:
            data = self._market_data.get(symbol)
            return deepcopy(data) if data else None

    def get_all_market_data(self) -> Dict[str, Dict[str, Any]]:
        """Get latest market data for all symbols."""
        with self._lock:
            return deepcopy(self._market_data)

    # =========================================================================
    # Messages/Logging
    # =========================================================================

    def _log(self, message: str):
        """Internal logging."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        self._messages.append(full_message)
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]
        print(f"[ENGINE] {message}")

    def add_message(self, message: str):
        with self._lock:
            self._log(message)

    def get_messages(self, last_n: int = 100) -> List[str]:
        with self._lock:
            return list(self._messages[-last_n:])

    def clear_messages(self):
        with self._lock:
            self._messages.clear()

    # =========================================================================
    # Full State Snapshot
    # =========================================================================

    def get_snapshot(self) -> dict:
        """Return a complete snapshot of the engine state."""
        with self._lock:
            return {
                "connection": self.get_connection_info(),
                "account": {
                    "account_id": self._account_summary.account_id,
                    "net_liquidation": self._account_summary.net_liquidation,
                    "total_cash": self._account_summary.total_cash,
                    "buying_power": self._account_summary.buying_power,
                    "unrealized_pnl": self._account_summary.unrealized_pnl,
                    "realized_pnl": self._account_summary.realized_pnl,
                },
                "positions_count": len(self._positions),
                "active_orders_count": len([o for o in self._orders.values()
                                           if o.status not in {OrderStatus.FILLED, OrderStatus.CANCELLED}]),
                "market_data_count": len(self._market_data),
                "messages_count": len(self._messages),
            }
