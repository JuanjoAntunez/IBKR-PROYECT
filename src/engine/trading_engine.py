"""
Trading Engine - Single Writer Pattern
======================================
Maintains a single persistent connection to IB and processes
commands from a thread-safe queue.
"""

import threading
import queue
import time
from typing import Optional, Callable, List
from datetime import datetime
import pandas as pd

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order as IBOrder

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


class IBWrapper(EWrapper):
    """IB API Wrapper that updates EngineState."""

    def __init__(self, state: EngineState, engine: 'TradingEngine'):
        EWrapper.__init__(self)
        self.state = state
        self.engine = engine

        # Temporary storage for async responses
        self._historical_data_buffer: List[dict] = []
        self._data_end_event = threading.Event()
        self._account_end_event = threading.Event()
        self._positions_end_event = threading.Event()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Handle errors from IB."""
        if errorCode in [2104, 2106, 2158]:  # Info messages
            self.state.add_message(f"Info: {errorString}")
        elif errorCode == 162:  # No data
            self.state.add_message(f"No data: {errorString}")
            self._data_end_event.set()
        elif errorCode >= 2000:  # Warnings
            self.state.add_message(f"Warning [{errorCode}]: {errorString}")
        else:
            self.state.add_message(f"Error [{errorCode}]: {errorString}")

    def connectAck(self):
        """Connection acknowledged."""
        self.state.add_message("Connection acknowledged")

    def connectionClosed(self):
        """Connection closed."""
        self.state.set_disconnected()

    def managedAccounts(self, accountsList):
        """Received list of managed accounts."""
        accounts = [a.strip() for a in accountsList.split(',') if a.strip()]
        self.state.add_message(f"Managed accounts: {accounts}")
        # Store temporarily for connection completion
        self.engine._temp_accounts = accounts

    def nextValidId(self, orderId):
        """Next valid order ID."""
        self.state.set_next_order_id(orderId)
        self.state.add_message(f"Next valid order ID: {orderId}")

    # =========================================================================
    # Account Updates
    # =========================================================================

    def accountSummary(self, reqId, account, tag, value, currency):
        """Receive account summary values."""
        try:
            val = float(value)
        except (ValueError, TypeError):
            val = 0.0

        tag_mapping = {
            "NetLiquidation": "net_liquidation",
            "TotalCashValue": "total_cash",
            "BuyingPower": "buying_power",
            "AvailableFunds": "available_funds",
            "ExcessLiquidity": "excess_liquidity",
            "InitMarginReq": "init_margin_req",
            "MaintMarginReq": "maint_margin_req",
            "UnrealizedPnL": "unrealized_pnl",
            "RealizedPnL": "realized_pnl",
            "Cushion": "cushion",
        }

        if tag in tag_mapping:
            self.state.update_account_summary(
                account_id=account,
                currency=currency,
                **{tag_mapping[tag]: val}
            )

    def accountSummaryEnd(self, reqId):
        """Account summary complete."""
        self.state.add_message("Account summary received")
        self._account_end_event.set()

    # =========================================================================
    # Portfolio Updates
    # =========================================================================

    def updatePortfolio(self, contract, position, marketPrice, marketValue,
                        averageCost, unrealizedPNL, realizedPNL, accountName):
        """Receive portfolio position updates."""
        if position != 0:  # Only track non-zero positions
            pos = Position(
                symbol=contract.symbol,
                quantity=int(position),
                avg_cost=averageCost,
                market_price=marketPrice,
                market_value=marketValue,
                unrealized_pnl=unrealizedPNL,
                realized_pnl=realizedPNL,
                account=accountName,
                currency=contract.currency,
                sec_type=contract.secType,
                exchange=contract.exchange,
            )
            self.state.update_position(pos)
        else:
            self.state.remove_position(contract.symbol)

    def accountDownloadEnd(self, accountName):
        """Account/portfolio download complete."""
        self.state.add_message(f"Portfolio download complete for {accountName}")
        self._positions_end_event.set()

    # =========================================================================
    # Historical Data
    # =========================================================================

    def historicalData(self, reqId, bar):
        """Receive historical data bar."""
        self._historical_data_buffer.append({
            'Date': bar.date,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume,
        })

    def historicalDataEnd(self, reqId, start, end):
        """Historical data complete."""
        self.state.add_message(f"Historical data received: {len(self._historical_data_buffer)} bars")
        self._data_end_event.set()

    # =========================================================================
    # Order Updates
    # =========================================================================

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                    permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        """Receive order status update."""
        status_mapping = {
            "PendingSubmit": OrderStatus.PENDING,
            "PendingCancel": OrderStatus.PENDING,
            "PreSubmitted": OrderStatus.PENDING,
            "Submitted": OrderStatus.SUBMITTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.ERROR,
        }

        order_status = status_mapping.get(status, OrderStatus.PENDING)

        self.state.update_order(
            orderId,
            status=order_status,
            filled_quantity=int(filled),
            avg_fill_price=avgFillPrice,
        )

    def openOrder(self, orderId, contract, order, orderState):
        """Receive open order info."""
        self.state.add_message(f"Open order: {orderId} {order.action} {order.totalQuantity} {contract.symbol}")

    def execDetails(self, reqId, contract, execution):
        """Receive execution details."""
        self.state.add_message(
            f"Execution: {execution.orderId} {execution.side} {execution.shares} {contract.symbol} @ {execution.price}"
        )

    def commissionReport(self, commissionReport):
        """Receive commission report."""
        self.state.update_order(
            commissionReport.execId,
            commission=commissionReport.commission,
        )


class TradingEngine:
    """
    Single Writer Trading Engine.

    Maintains a single persistent connection to IB and processes
    all commands through a dedicated thread.
    """

    def __init__(self):
        self.state = EngineState()

        # IB API client
        self._wrapper: Optional[IBWrapper] = None
        self._client: Optional[EClient] = None
        self._api_thread: Optional[threading.Thread] = None

        # Command processing
        self._command_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._engine_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # Temporary storage
        self._temp_accounts: List[str] = []

        # Request ID counter
        self._next_req_id = 1
        self._req_id_lock = threading.Lock()

    def _get_req_id(self) -> int:
        """Get next request ID (thread-safe)."""
        with self._req_id_lock:
            req_id = self._next_req_id
            self._next_req_id += 1
            return req_id

    # =========================================================================
    # Public API (Thread-Safe)
    # =========================================================================

    def start(self):
        """Start the trading engine."""
        if self._running:
            self.state.add_message("Engine already running")
            return

        self._running = True
        self._stop_event.clear()

        # Start command processing thread
        self._engine_thread = threading.Thread(
            target=self._engine_loop,
            name="TradingEngine",
            daemon=True
        )
        self._engine_thread.start()

        self.state.add_message("Trading Engine started")

    def stop(self):
        """Stop the trading engine gracefully."""
        if not self._running:
            return

        self.state.add_message("Stopping Trading Engine...")

        # Signal stop
        self._running = False
        self._stop_event.set()

        # Disconnect if connected
        if self._client and self._client.isConnected():
            self._client.disconnect()

        # Wait for engine thread
        if self._engine_thread and self._engine_thread.is_alive():
            self._engine_thread.join(timeout=5)

        self.state.add_message("Trading Engine stopped")

    def submit_command(self, command: Command) -> str:
        """
        Submit a command to the engine queue.

        Returns the command ID for tracking.
        """
        if not self._running:
            command.status = CommandStatus.FAILED
            command.result = CommandResult(
                success=False,
                error="Engine not running"
            )
            return command.command_id

        # Add to priority queue (lower priority number = higher priority)
        # Using negative priority so higher priority values come first
        self._command_queue.put((-command.priority, command.created_at, command))
        self.state.add_message(f"Command queued: {command}")

        return command.command_id

    def get_status(self) -> dict:
        """Get current engine status snapshot."""
        return {
            "running": self._running,
            "connected": self.state.is_connected,
            "state": self.state.get_snapshot(),
            "queue_size": self._command_queue.qsize(),
        }

    def get_positions(self) -> dict:
        """Get snapshot of current positions."""
        return self.state.get_positions()

    def get_account_summary(self) -> AccountSummary:
        """Get snapshot of account summary."""
        return self.state.get_account_summary()

    def get_orders(self) -> dict:
        """Get snapshot of orders."""
        return self.state.get_orders()

    def get_active_orders(self) -> list:
        """Get list of active orders."""
        return self.state.get_active_orders()

    def get_historical_data(self, symbol: str):
        """Get cached historical data for symbol."""
        return self.state.get_historical_data(symbol)

    def get_messages(self, last_n: int = 100) -> list:
        """Get recent messages/logs."""
        return self.state.get_messages(last_n)

    # =========================================================================
    # Engine Loop (Single Writer)
    # =========================================================================

    def _engine_loop(self):
        """Main engine loop - processes commands from queue."""
        self.state.add_message("Engine loop started")

        while self._running:
            try:
                # Wait for command with timeout
                try:
                    _, _, command = self._command_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Process command
                self._process_command(command)
                self._command_queue.task_done()

            except Exception as e:
                self.state.add_message(f"Engine loop error: {e}")

        self.state.add_message("Engine loop stopped")

    def _process_command(self, command: Command):
        """Process a single command."""
        command.status = CommandStatus.PROCESSING
        self.state.add_message(f"Processing: {command}")

        try:
            result = self._execute_command(command)
            command.result = result
            command.status = CommandStatus.COMPLETED if result.success else CommandStatus.FAILED
        except Exception as e:
            command.result = CommandResult(success=False, error=str(e))
            command.status = CommandStatus.FAILED
            self.state.add_message(f"Command failed: {e}")

        # Call callback if provided
        if command.callback and command.result:
            try:
                command.callback(command.result)
            except Exception as e:
                self.state.add_message(f"Callback error: {e}")

    def _execute_command(self, command: Command) -> CommandResult:
        """Execute a command and return result."""
        cmd_type = command.command_type

        if cmd_type == CommandType.CONNECT:
            return self._execute_connect(command)
        elif cmd_type == CommandType.DISCONNECT:
            return self._execute_disconnect(command)
        elif cmd_type == CommandType.FETCH_HISTORICAL_DATA:
            return self._execute_fetch_historical(command)
        elif cmd_type == CommandType.GET_ACCOUNT:
            return self._execute_get_account(command)
        elif cmd_type == CommandType.GET_POSITIONS:
            return self._execute_get_positions(command)
        elif cmd_type == CommandType.PLACE_ORDER:
            return self._execute_place_order(command)
        elif cmd_type == CommandType.CANCEL_ORDER:
            return self._execute_cancel_order(command)
        else:
            return CommandResult(success=False, error=f"Unknown command type: {cmd_type}")

    # =========================================================================
    # Command Executors
    # =========================================================================

    def _execute_connect(self, cmd: ConnectCommand) -> CommandResult:
        """Execute connect command."""
        if self.state.is_connected:
            return CommandResult(success=True, data="Already connected")

        try:
            self.state.set_connection_status(ConnectionStatus.CONNECTING)

            # Create wrapper and client
            self._wrapper = IBWrapper(self.state, self)
            self._client = EClient(self._wrapper)

            # Connect
            self._client.connect(cmd.host, cmd.port, cmd.client_id)

            # Start API thread
            self._api_thread = threading.Thread(
                target=self._client.run,
                name="IBApiThread",
                daemon=True
            )
            self._api_thread.start()

            # Wait for connection
            start_time = time.time()
            while not self._client.isConnected() and (time.time() - start_time) < cmd.timeout:
                time.sleep(0.1)

            if not self._client.isConnected():
                self.state.set_connection_status(ConnectionStatus.ERROR)
                return CommandResult(success=False, error="Connection timeout")

            # Wait for accounts
            time.sleep(2)

            # Update state
            self.state.set_connected(
                host=cmd.host,
                port=cmd.port,
                mode=cmd.mode,
                platform="TWS" if cmd.port in [7496, 7497] else "Gateway",
                accounts=self._temp_accounts,
            )

            return CommandResult(
                success=True,
                data={
                    "host": cmd.host,
                    "port": cmd.port,
                    "mode": cmd.mode,
                    "accounts": self._temp_accounts,
                }
            )

        except Exception as e:
            self.state.set_connection_status(ConnectionStatus.ERROR)
            return CommandResult(success=False, error=str(e))

    def _execute_disconnect(self, cmd: DisconnectCommand) -> CommandResult:
        """Execute disconnect command."""
        if not self.state.is_connected:
            return CommandResult(success=True, data="Already disconnected")

        try:
            if self._client:
                self._client.disconnect()
            self.state.set_disconnected()
            return CommandResult(success=True)
        except Exception as e:
            return CommandResult(success=False, error=str(e))

    def _execute_fetch_historical(self, cmd: FetchHistoricalDataCommand) -> CommandResult:
        """Execute fetch historical data command."""
        if not self.state.is_connected:
            return CommandResult(success=False, error="Not connected")

        try:
            # Create contract
            contract = Contract()
            contract.symbol = cmd.symbol.upper()
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"

            # Clear buffer and reset event
            self._wrapper._historical_data_buffer = []
            self._wrapper._data_end_event.clear()

            # Request data
            req_id = self._get_req_id()
            self._client.reqHistoricalData(
                reqId=req_id,
                contract=contract,
                endDateTime="",
                durationStr=cmd.duration,
                barSizeSetting=cmd.bar_size,
                whatToShow=cmd.what_to_show,
                useRTH=1 if cmd.use_rth else 0,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[],
            )

            # Wait for data
            if not self._wrapper._data_end_event.wait(timeout=30):
                return CommandResult(success=False, error="Timeout waiting for data")

            if not self._wrapper._historical_data_buffer:
                return CommandResult(success=False, error="No data received")

            # Create DataFrame
            df = pd.DataFrame(self._wrapper._historical_data_buffer)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')

            # Cache data
            self.state.set_historical_data(cmd.symbol, df, cmd.duration, cmd.bar_size)

            return CommandResult(success=True, data=df)

        except Exception as e:
            return CommandResult(success=False, error=str(e))

    def _execute_get_account(self, cmd: GetAccountCommand) -> CommandResult:
        """Execute get account command."""
        if not self.state.is_connected:
            return CommandResult(success=False, error="Not connected")

        try:
            self._wrapper._account_end_event.clear()

            req_id = self._get_req_id()
            self._client.reqAccountSummary(req_id, "All", cmd.tags)

            if not self._wrapper._account_end_event.wait(timeout=10):
                self._client.cancelAccountSummary(req_id)
                return CommandResult(success=False, error="Timeout getting account")

            self._client.cancelAccountSummary(req_id)

            return CommandResult(
                success=True,
                data=self.state.get_account_summary()
            )

        except Exception as e:
            return CommandResult(success=False, error=str(e))

    def _execute_get_positions(self, cmd: GetPositionsCommand) -> CommandResult:
        """Execute get positions command."""
        if not self.state.is_connected:
            return CommandResult(success=False, error="Not connected")

        try:
            self._wrapper._positions_end_event.clear()
            self.state.clear_positions()

            # Request portfolio updates
            accounts = self.state.get_connection_info().get("accounts", [])
            if accounts:
                self._client.reqAccountUpdates(True, accounts[0])

                if not self._wrapper._positions_end_event.wait(timeout=10):
                    self._client.reqAccountUpdates(False, accounts[0])
                    return CommandResult(success=False, error="Timeout getting positions")

                self._client.reqAccountUpdates(False, accounts[0])

            return CommandResult(
                success=True,
                data=self.state.get_positions()
            )

        except Exception as e:
            return CommandResult(success=False, error=str(e))

    def _execute_place_order(self, cmd: PlaceOrderCommand) -> CommandResult:
        """Execute place order command."""
        if not self.state.is_connected:
            return CommandResult(success=False, error="Not connected")

        if not cmd.validate():
            return CommandResult(success=False, error="Invalid order parameters")

        try:
            # Create contract
            contract = Contract()
            contract.symbol = cmd.symbol.upper()
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"

            # Create order
            order = IBOrder()
            order.action = cmd.action.value
            order.totalQuantity = cmd.quantity
            order.orderType = cmd.order_type.value
            order.tif = cmd.time_in_force

            if cmd.limit_price is not None:
                order.lmtPrice = cmd.limit_price
            if cmd.stop_price is not None:
                order.auxPrice = cmd.stop_price

            # Get order ID and place order
            order_id = self.state.get_next_order_id()

            # Track order in state
            tracked_order = Order(
                order_id=order_id,
                symbol=cmd.symbol,
                action=cmd.action.value,
                quantity=cmd.quantity,
                order_type=cmd.order_type.value,
                limit_price=cmd.limit_price,
                stop_price=cmd.stop_price,
                status=OrderStatus.PENDING,
            )
            self.state.add_order(tracked_order)

            # Place order
            self._client.placeOrder(order_id, contract, order)

            return CommandResult(
                success=True,
                data={"order_id": order_id}
            )

        except Exception as e:
            return CommandResult(success=False, error=str(e))

    def _execute_cancel_order(self, cmd: CancelOrderCommand) -> CommandResult:
        """Execute cancel order command."""
        if not self.state.is_connected:
            return CommandResult(success=False, error="Not connected")

        try:
            self._client.cancelOrder(cmd.order_id, "")
            self.state.update_order(cmd.order_id, status=OrderStatus.CANCELLED)

            return CommandResult(success=True, data={"cancelled_order_id": cmd.order_id})

        except Exception as e:
            return CommandResult(success=False, error=str(e))


# =============================================================================
# Singleton Pattern for Global Engine Instance
# =============================================================================

_engine_instance: Optional[TradingEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> TradingEngine:
    """Get or create the global TradingEngine instance."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = TradingEngine()
        return _engine_instance


def reset_engine():
    """Reset the global engine instance (for testing)."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance:
            _engine_instance.stop()
        _engine_instance = None
