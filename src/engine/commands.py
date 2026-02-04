"""
Command Pattern for Trading Engine
===================================
All commands that can be submitted to the Trading Engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from enum import Enum
import uuid
from datetime import datetime


class CommandStatus(Enum):
    """Status of a command in the queue."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CommandType(Enum):
    """Types of commands supported by the engine."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    FETCH_HISTORICAL_DATA = "fetch_historical_data"
    GET_ACCOUNT = "get_account"
    GET_POSITIONS = "get_positions"
    GET_ORDERS = "get_orders"
    PLACE_ORDER = "place_order"
    CANCEL_ORDER = "cancel_order"
    GET_GUARDRAILS_STATUS = "get_guardrails_status"
    ACTIVATE_KILL_SWITCH = "activate_kill_switch"
    GET_MARKET_SUBSCRIPTIONS = "get_market_subscriptions"
    SUBSCRIBE_MARKET_DATA = "subscribe_market_data"
    UNSUBSCRIBE_MARKET_DATA = "unsubscribe_market_data"


@dataclass
class CommandResult:
    """Result of a command execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Command(ABC):
    """
    Base class for all commands.

    Commands are immutable instructions that get queued for execution
    by the Trading Engine's single writer thread.
    """
    command_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    status: CommandStatus = field(default=CommandStatus.PENDING)
    result: Optional[CommandResult] = None
    callback: Optional[Callable[[CommandResult], None]] = None
    priority: int = 0  # Higher = more urgent

    @property
    @abstractmethod
    def command_type(self) -> CommandType:
        """Return the type of this command."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.command_id}, status={self.status.value})"


# =============================================================================
# Connection Commands
# =============================================================================

@dataclass
class ConnectCommand(Command):
    """Command to establish connection with IB."""
    host: str = "127.0.0.1"
    port: int = 7497  # Paper trading by default
    client_id: int = 1
    timeout: int = 10
    mode: str = "paper"  # "paper" or "live"
    confirm_live: bool = False  # Require explicit confirmation for live mode

    @property
    def command_type(self) -> CommandType:
        return CommandType.CONNECT


@dataclass
class DisconnectCommand(Command):
    """Command to disconnect from IB."""

    @property
    def command_type(self) -> CommandType:
        return CommandType.DISCONNECT


# =============================================================================
# Data Commands
# =============================================================================

@dataclass
class FetchHistoricalDataCommand(Command):
    """Command to fetch historical market data."""
    symbol: str = ""
    duration: str = "1 M"  # IB format: "1 D", "5 D", "1 M", etc.
    bar_size: str = "1 day"  # IB format: "1 min", "5 mins", "1 hour", "1 day"
    what_to_show: str = "TRADES"
    use_rth: bool = True  # Regular trading hours only

    @property
    def command_type(self) -> CommandType:
        return CommandType.FETCH_HISTORICAL_DATA


@dataclass
class SubscribeMarketDataCommand(Command):
    """Command to subscribe to real-time market data."""
    symbol: str = ""
    data_type: str = "TRADES"  # TRADES, MIDPOINT, BID, ASK

    @property
    def command_type(self) -> CommandType:
        return CommandType.SUBSCRIBE_MARKET_DATA


@dataclass
class UnsubscribeMarketDataCommand(Command):
    """Command to unsubscribe from market data."""
    symbol: str = ""
    req_id: int = 0

    @property
    def command_type(self) -> CommandType:
        return CommandType.UNSUBSCRIBE_MARKET_DATA


# =============================================================================
# Account Commands
# =============================================================================

@dataclass
class GetAccountCommand(Command):
    """Command to get account summary."""
    tags: str = "NetLiquidation,TotalCashValue,BuyingPower,AvailableFunds,UnrealizedPnL,RealizedPnL"

    @property
    def command_type(self) -> CommandType:
        return CommandType.GET_ACCOUNT


@dataclass
class GetPositionsCommand(Command):
    """Command to get current positions."""

    @property
    def command_type(self) -> CommandType:
        return CommandType.GET_POSITIONS


@dataclass
class GetOrdersCommand(Command):
    """Command to get active orders."""

    @property
    def command_type(self) -> CommandType:
        return CommandType.GET_ORDERS


@dataclass
class GetMarketSubscriptionsCommand(Command):
    """Command to get active market data subscriptions."""

    @property
    def command_type(self) -> CommandType:
        return CommandType.GET_MARKET_SUBSCRIPTIONS


@dataclass
class GetGuardrailsStatusCommand(Command):
    """Command to get guardrails status."""

    @property
    def command_type(self) -> CommandType:
        return CommandType.GET_GUARDRAILS_STATUS


@dataclass
class ActivateKillSwitchCommand(Command):
    """Command to activate kill switch manually."""
    reason: str = "Manual activation"

    @property
    def command_type(self) -> CommandType:
        return CommandType.ACTIVATE_KILL_SWITCH


# =============================================================================
# Order Commands
# =============================================================================

class OrderType(Enum):
    """Order types supported."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


class OrderAction(Enum):
    """Order actions."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class PlaceOrderCommand(Command):
    """Command to place a new order."""
    symbol: str = ""
    action: OrderAction = OrderAction.BUY
    quantity: int = 0
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    client_order_key: Optional[str] = None  # Idempotency key from client
    max_retries: int = 0  # Controlled retries on submit failure

    @property
    def command_type(self) -> CommandType:
        return CommandType.PLACE_ORDER

    def validate(self) -> bool:
        """Validate the order before submission."""
        if not self.symbol:
            return False
        if self.quantity <= 0:
            return False
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            return False
        if self.order_type == OrderType.STOP and self.stop_price is None:
            return False
        if self.client_order_key is not None and not str(self.client_order_key).strip():
            return False
        return True


@dataclass
class CancelOrderCommand(Command):
    """Command to cancel an existing order."""
    order_id: int = 0

    @property
    def command_type(self) -> CommandType:
        return CommandType.CANCEL_ORDER
