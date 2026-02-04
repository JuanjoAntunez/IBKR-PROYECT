"""
Frontend Adapter for Trading Engine
====================================
Bridge between Streamlit frontend and Trading Engine.
Handles async communication and state synchronization.
"""

import streamlit as st
from typing import Optional, Any, Callable
from datetime import datetime
import time
import threading

from .trading_engine import TradingEngine, get_engine
from .commands import (
    Command,
    CommandResult,
    CommandStatus,
    ConnectCommand,
    DisconnectCommand,
    FetchHistoricalDataCommand,
    GetAccountCommand,
    GetPositionsCommand,
    PlaceOrderCommand,
    CancelOrderCommand,
    OrderAction,
    OrderType,
)
from .state import EngineState, ConnectionStatus


class EngineAdapter:
    """
    Adapter for using TradingEngine from Streamlit.

    Provides synchronous-looking API that internally uses
    the command queue pattern.
    """

    def __init__(self):
        self._engine: TradingEngine = get_engine()
        self._pending_commands: dict = {}

    @property
    def engine(self) -> TradingEngine:
        """Get the underlying engine."""
        return self._engine

    @property
    def state(self) -> EngineState:
        """Get engine state."""
        return self._engine.state

    def ensure_running(self):
        """Ensure engine is running."""
        if not self._engine._running:
            self._engine.start()

    # =========================================================================
    # Synchronous API (blocks until command completes)
    # =========================================================================

    def connect(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        mode: str = "paper",
        timeout: int = 15
    ) -> tuple[bool, Optional[str], Optional[dict]]:
        """
        Connect to IB synchronously.

        Returns:
            tuple: (success, error_message, connection_info)
        """
        self.ensure_running()

        cmd = ConnectCommand(
            host=host,
            port=port,
            client_id=client_id,
            timeout=timeout,
            mode=mode,
        )

        result = self._execute_sync(cmd, timeout=timeout + 5)

        if result.success:
            return True, None, result.data
        return False, result.error, None

    def disconnect(self) -> tuple[bool, Optional[str]]:
        """
        Disconnect from IB synchronously.

        Returns:
            tuple: (success, error_message)
        """
        cmd = DisconnectCommand()
        result = self._execute_sync(cmd)

        return result.success, result.error

    def fetch_historical_data(
        self,
        symbol: str,
        duration: str = "1 M",
        bar_size: str = "1 day",
        timeout: int = 30
    ) -> tuple[bool, Optional[str], Any]:
        """
        Fetch historical data synchronously.

        Returns:
            tuple: (success, error_message, dataframe)
        """
        self.ensure_running()

        cmd = FetchHistoricalDataCommand(
            symbol=symbol,
            duration=duration,
            bar_size=bar_size,
        )

        result = self._execute_sync(cmd, timeout=timeout)

        if result.success:
            return True, None, result.data
        return False, result.error, None

    def get_account(self, timeout: int = 15) -> tuple[bool, Optional[str], Any]:
        """
        Get account summary synchronously.

        Returns:
            tuple: (success, error_message, account_summary)
        """
        self.ensure_running()

        cmd = GetAccountCommand()
        result = self._execute_sync(cmd, timeout=timeout)

        if result.success:
            return True, None, result.data
        return False, result.error, None

    def get_positions(self, timeout: int = 15) -> tuple[bool, Optional[str], dict]:
        """
        Get positions synchronously.

        Returns:
            tuple: (success, error_message, positions_dict)
        """
        self.ensure_running()

        cmd = GetPositionsCommand()
        result = self._execute_sync(cmd, timeout=timeout)

        if result.success:
            return True, None, result.data
        return False, result.error, None

    def place_order(
        self,
        symbol: str,
        action: str,  # "BUY" or "SELL"
        quantity: int,
        order_type: str = "MKT",  # "MKT", "LMT", "STP"
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        timeout: int = 10
    ) -> tuple[bool, Optional[str], Optional[int]]:
        """
        Place an order synchronously.

        Returns:
            tuple: (success, error_message, order_id)
        """
        self.ensure_running()

        # Map string action to enum
        action_enum = OrderAction.BUY if action.upper() == "BUY" else OrderAction.SELL

        # Map string order type to enum
        type_map = {
            "MKT": OrderType.MARKET,
            "LMT": OrderType.LIMIT,
            "STP": OrderType.STOP,
            "STP LMT": OrderType.STOP_LIMIT,
        }
        order_type_enum = type_map.get(order_type.upper(), OrderType.MARKET)

        cmd = PlaceOrderCommand(
            symbol=symbol,
            action=action_enum,
            quantity=quantity,
            order_type=order_type_enum,
            limit_price=limit_price,
            stop_price=stop_price,
        )

        result = self._execute_sync(cmd, timeout=timeout)

        if result.success:
            return True, None, result.data.get("order_id")
        return False, result.error, None

    def cancel_order(self, order_id: int, timeout: int = 10) -> tuple[bool, Optional[str]]:
        """
        Cancel an order synchronously.

        Returns:
            tuple: (success, error_message)
        """
        cmd = CancelOrderCommand(order_id=order_id)
        result = self._execute_sync(cmd, timeout=timeout)

        return result.success, result.error

    # =========================================================================
    # State Access (Read-only, always available)
    # =========================================================================

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._engine.state.is_connected

    def get_connection_info(self) -> dict:
        """Get current connection info."""
        return self._engine.state.get_connection_info()

    def get_cached_positions(self) -> dict:
        """Get cached positions (no IB request)."""
        return self._engine.get_positions()

    def get_cached_account(self):
        """Get cached account summary (no IB request)."""
        return self._engine.get_account_summary()

    def get_cached_orders(self) -> dict:
        """Get cached orders (no IB request)."""
        return self._engine.get_orders()

    def get_active_orders(self) -> list:
        """Get active orders."""
        return self._engine.get_active_orders()

    def get_cached_historical_data(self, symbol: str):
        """Get cached historical data (no IB request)."""
        return self._engine.get_historical_data(symbol)

    def get_messages(self, last_n: int = 100) -> list:
        """Get recent log messages."""
        return self._engine.get_messages(last_n)

    def get_status(self) -> dict:
        """Get engine status."""
        return self._engine.get_status()

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _execute_sync(self, command: Command, timeout: int = 30) -> CommandResult:
        """Execute a command and wait for result."""
        # Create completion event
        completed = threading.Event()
        result_holder = {"result": None}

        def on_complete(result: CommandResult):
            result_holder["result"] = result
            completed.set()

        command.callback = on_complete

        # Submit command
        self._engine.submit_command(command)

        # Wait for completion
        if completed.wait(timeout=timeout):
            return result_holder["result"]

        # Timeout
        return CommandResult(
            success=False,
            error=f"Command timeout after {timeout}s"
        )


# =============================================================================
# Streamlit Session State Integration
# =============================================================================

def get_adapter() -> EngineAdapter:
    """
    Get or create EngineAdapter in Streamlit session state.

    This ensures a single adapter instance per session.
    """
    if 'engine_adapter' not in st.session_state:
        st.session_state.engine_adapter = EngineAdapter()
    return st.session_state.engine_adapter


def init_engine_state():
    """Initialize engine-related session state variables."""
    defaults = {
        'engine_initialized': False,
        'last_fetch_symbol': None,
        'last_fetch_data': None,
        'engine_messages': [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# Helper Functions for Dashboard
# =============================================================================

def get_port_for_mode(platform: str, mode: str) -> int:
    """Get port number based on platform and mode."""
    port_map = {
        ("TWS", "paper"): 7497,
        ("TWS", "live"): 7496,
        ("Gateway", "paper"): 4002,
        ("Gateway", "live"): 4001,
    }
    return port_map.get((platform, mode), 7497)


def format_duration(duration: str) -> str:
    """Convert user-friendly duration to IB format."""
    duration_map = {
        "1D": "1 D",
        "5D": "5 D",
        "1M": "1 M",
        "3M": "3 M",
        "6M": "6 M",
        "1Y": "1 Y",
    }
    return duration_map.get(duration, duration)


def format_bar_size(interval: str) -> str:
    """Convert user-friendly interval to IB format."""
    bar_map = {
        "1min": "1 min",
        "5min": "5 mins",
        "15min": "15 mins",
        "1h": "1 hour",
        "1d": "1 day",
    }
    return bar_map.get(interval, interval)
