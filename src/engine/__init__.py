"""
Trading Engine Module
=====================
Single Writer pattern implementation for IB trading with safety guardrails.
"""

from .commands import (
    Command,
    CommandResult,
    CommandStatus,
    CommandType,
    ConnectCommand,
    DisconnectCommand,
    FetchHistoricalDataCommand,
    GetAccountCommand,
    GetPositionsCommand,
    GetOrdersCommand,
    PlaceOrderCommand,
    CancelOrderCommand,
    OrderAction,
    OrderType,
)
from .state import (
    EngineState,
    ConnectionStatus,
    Position,
    Order,
    OrderStatus,
    AccountSummary,
)
from .trading_engine import TradingEngine, get_engine, reset_engine
from .trading_mode import (
    TradingMode,
    TradingModeManager,
    get_mode_manager,
    is_live_mode,
    is_paper_mode,
)
from .guardrails import (
    Guardrails,
    TradingLimits,
    LimitType,
    LimitViolation,
    ViolationSeverity,
    OrderRequest,
    ValidationResult,
)
from .audit_log import AuditLog, AuditEntry, get_audit_log
from .notifications import (
    NotificationManager,
    NotificationChannel,
    NotificationPriority,
    get_notification_manager,
)

__all__ = [
    # Engine
    "TradingEngine",
    "get_engine",
    "reset_engine",
    # Commands
    "Command",
    "CommandResult",
    "CommandStatus",
    "CommandType",
    "ConnectCommand",
    "DisconnectCommand",
    "FetchHistoricalDataCommand",
    "GetAccountCommand",
    "GetPositionsCommand",
    "GetOrdersCommand",
    "PlaceOrderCommand",
    "CancelOrderCommand",
    "OrderAction",
    "OrderType",
    # State
    "EngineState",
    "ConnectionStatus",
    "Position",
    "Order",
    "OrderStatus",
    "AccountSummary",
    # Trading Mode
    "TradingMode",
    "TradingModeManager",
    "get_mode_manager",
    "is_live_mode",
    "is_paper_mode",
    # Guardrails
    "Guardrails",
    "TradingLimits",
    "LimitType",
    "LimitViolation",
    "ViolationSeverity",
    "OrderRequest",
    "ValidationResult",
    # Audit
    "AuditLog",
    "AuditEntry",
    "get_audit_log",
    # Notifications
    "NotificationManager",
    "NotificationChannel",
    "NotificationPriority",
    "get_notification_manager",
]
