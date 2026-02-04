"""
Trading Engine Module
=====================
Single Writer pattern implementation for IB trading.
"""

from .commands import (
    Command,
    ConnectCommand,
    DisconnectCommand,
    FetchHistoricalDataCommand,
    GetAccountCommand,
    GetPositionsCommand,
    GetOrdersCommand,
    PlaceOrderCommand,
    CancelOrderCommand,
)
from .state import EngineState, Position, Order, AccountSummary
from .trading_engine import TradingEngine

__all__ = [
    # Engine
    "TradingEngine",
    # Commands
    "Command",
    "ConnectCommand",
    "DisconnectCommand",
    "FetchHistoricalDataCommand",
    "GetAccountCommand",
    "GetPositionsCommand",
    "GetOrdersCommand",
    "PlaceOrderCommand",
    "CancelOrderCommand",
    # State
    "EngineState",
    "Position",
    "Order",
    "AccountSummary",
]
