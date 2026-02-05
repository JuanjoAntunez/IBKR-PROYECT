"""
Trading Engine - Single Writer Pattern (ib_insync)
==================================================
Maintains a single persistent connection to IB and processes
commands from a thread-safe queue using ib_insync.
"""

import queue
import threading
import asyncio
import time
import os
from datetime import datetime
from typing import Optional, Dict

# Ensure an event loop exists in the importing thread (required by ib_insync/eventkit)
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Stock, Order, LimitOrder, MarketOrder, StopOrder, util, Trade
import ib_insync

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
    GetGuardrailsStatusCommand,
    ActivateKillSwitchCommand,
    GetMarketSubscriptionsCommand,
    PlaceOrderCommand,
    CancelOrderCommand,
    SubscribeMarketDataCommand,
    UnsubscribeMarketDataCommand,
    OrderType,
)
from .state import (
    EngineState,
    ConnectionStatus,
    Position,
    OrderStatus,
    AccountSummary,
)
from .guardrails import Guardrails, OrderRequest, TradingLimits
from .trading_mode import TradingMode, get_mode_manager
from .audit_log import get_audit_log
from .order_manager import OrderManager
from .market_data import MarketDataService
from .config_loader import resolve_connection_params, load_credentials


class TradingEngine:
    """
    Single Writer Trading Engine (ib_insync).

    Maintains a single persistent connection to IB and processes
    all commands through a dedicated thread.
    """

    def __init__(self):
        self.state = EngineState()

        # IB client (ib_insync)
        self._ib: Optional[IB] = None

        # Command processing
        self._command_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._engine_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # Safety systems
        self._mode_manager = get_mode_manager()
        self._guardrails: Optional[Guardrails] = None
        self._audit_log = get_audit_log()

        # Order tracking
        self._order_requests: Dict[int, OrderRequest] = {}
        self._market_data = MarketDataService(self.state)
        self._order_manager = OrderManager(self.state)

        # Reconciliation + resilience
        self._reconcile_interval = int(os.getenv("IB_RECONCILE_INTERVAL", "10"))
        self._heartbeat_interval = int(os.getenv("IB_HEARTBEAT_INTERVAL", "10"))
        self._last_reconcile_time = 0.0
        self._last_heartbeat_time = 0.0
        self._auto_reconnect_config = os.getenv("IB_AUTO_RECONNECT", "1") != "0"
        self._auto_reconnect = self._auto_reconnect_config
        self._reconnect_attempts = 0
        self._next_reconnect_time: Optional[float] = None
        self._last_connect_params: Optional[dict] = None

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
            daemon=True,
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
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()

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
                error="Engine not running",
            )
            return command.command_id

        # Add to priority queue (lower priority number = higher priority)
        self._command_queue.put((-command.priority, command.created_at, command))
        self.state.add_message(f"Command queued: {command}")

        return command.command_id

    def get_status(self) -> dict:
        """Get current engine status snapshot."""
        def _ts(value: Optional[float]) -> Optional[str]:
            if not value:
                return None
            try:
                return datetime.fromtimestamp(value).isoformat()
            except Exception:
                return None

        next_reconnect = _ts(self._next_reconnect_time) if self._next_reconnect_time else None
        return {
            "running": self._running,
            "connected": self.state.is_connected,
            "ib_connected": self._ib.isConnected() if self._ib else False,
            "state": self.state.get_snapshot(),
            "queue_size": self._command_queue.qsize(),
            "heartbeat": {
                "interval_sec": self._heartbeat_interval,
                "last": _ts(self._last_heartbeat_time),
            },
            "reconcile": {
                "interval_sec": self._reconcile_interval,
                "last": _ts(self._last_reconcile_time),
            },
            "reconnect": {
                "auto": self._auto_reconnect,
                "attempts": self._reconnect_attempts,
                "next_attempt": next_reconnect,
            },
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

        # Avoid patching asyncio by default; nest_asyncio can break on newer
        # Python versions (e.g., 3.14) during IB connect timeouts.
        if os.getenv("IB_PATCH_ASYNCIO", "0") == "1":
            util.patchAsyncio()
            self.state.add_message("Asyncio patched via IB_PATCH_ASYNCIO=1")

        # Now create/get the event loop for this thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        while self._running:
            try:
                # Wait for command with timeout
                try:
                    _, _, command = self._command_queue.get(timeout=0.1)
                except queue.Empty:
                    command = None

                # Process command
                if command is not None:
                    self._process_command(command)
                    self._command_queue.task_done()

            except Exception as e:
                self.state.add_message(f"Engine loop error: {e}")

            # Periodic tasks (reconcile, heartbeat, reconnect)
            self._tick()

        self.state.add_message("Engine loop stopped")

    def _tick(self):
        """Periodic engine maintenance tasks."""
        now = time.time()

        # Process ib_insync events (required for callbacks to work)
        if self._ib:
            try:
                self._ib.sleep(0.01)  # Small sleep to process pending events
            except Exception:
                pass

        # Heartbeat + reconcile when connected
        if self.state.is_connected and self._ib and self._ib.isConnected():
            if now - self._last_heartbeat_time >= self._heartbeat_interval:
                self._last_heartbeat_time = now
                try:
                    self._ib.reqCurrentTime()
                except Exception:
                    self._handle_disconnect("Heartbeat failed")
            if now - self._last_reconcile_time >= self._reconcile_interval:
                self._last_reconcile_time = now
                self._reconcile_state()

        # Reconnect if needed
        if not self.state.is_connected and self._auto_reconnect:
            self._maybe_reconnect(now)

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
        if cmd_type == CommandType.DISCONNECT:
            return self._execute_disconnect(command)
        if cmd_type == CommandType.FETCH_HISTORICAL_DATA:
            return self._execute_fetch_historical(command)
        if cmd_type == CommandType.GET_ACCOUNT:
            return self._execute_get_account(command)
        if cmd_type == CommandType.GET_POSITIONS:
            return self._execute_get_positions(command)
        if cmd_type == CommandType.GET_ORDERS:
            return self._execute_get_orders(command)
        if cmd_type == CommandType.GET_GUARDRAILS_STATUS:
            return self._execute_get_guardrails_status(command)
        if cmd_type == CommandType.ACTIVATE_KILL_SWITCH:
            return self._execute_activate_kill_switch(command)
        if cmd_type == CommandType.GET_MARKET_SUBSCRIPTIONS:
            return self._execute_get_market_subscriptions(command)
        if cmd_type == CommandType.SUBSCRIBE_MARKET_DATA:
            return self._execute_subscribe_market_data(command)
        if cmd_type == CommandType.UNSUBSCRIBE_MARKET_DATA:
            return self._execute_unsubscribe_market_data(command)
        if cmd_type == CommandType.PLACE_ORDER:
            return self._execute_place_order(command)
        if cmd_type == CommandType.CANCEL_ORDER:
            return self._execute_cancel_order(command)
        return CommandResult(success=False, error=f"Unknown command type: {cmd_type}")

    # =========================================================================
    # IB Event Handlers
    # =========================================================================

    def _setup_ib_handlers(self):
        if not self._ib:
            return
        self._ib.orderStatusEvent += self._on_order_status
        self._ib.execDetailsEvent += self._on_execution
        self._ib.errorEvent += self._on_error
        self._ib.disconnectedEvent += self._on_disconnected
        self._ib.timeoutEvent += self._on_timeout

    def _on_order_status(self, trade: Trade):
        """Handle order status updates."""
        order_id = trade.order.orderId
        self._order_manager.handle_order_status(trade)

        updated = self.state.get_order(order_id)
        if updated and updated.status == OrderStatus.FILLED:
            order_request = self._order_requests.get(order_id)
            if order_request and self._guardrails:
                self._guardrails.record_order_executed(order_request)
            self._audit_log.log_order_executed(
                mode=self.state.get_connection_info().get("mode", "paper"),
                order_id=order_id,
                symbol=trade.contract.symbol,
                action=trade.order.action,
                quantity=int(trade.orderStatus.filled),
                fill_price=trade.orderStatus.avgFillPrice,
                guardrails_state=self._guardrails.get_status() if self._guardrails else None,
            )
        if updated and updated.status == OrderStatus.REJECTED:
            self._audit_log.log_order_rejected(
                mode=self.state.get_connection_info().get("mode", "paper"),
                symbol=trade.contract.symbol,
                action=trade.order.action,
                quantity=int(trade.order.totalQuantity),
                reason=trade.orderStatus.status,
                violations=[],
                guardrails_state=self._guardrails.get_status() if self._guardrails else None,
            )

    def _on_execution(self, trade: Trade, fill):
        """Handle execution details (logging only)."""
        self.state.add_message(
            f"Execution: {trade.order.orderId} {fill.execution.side} "
            f"{fill.execution.shares} {trade.contract.symbol} @ {fill.execution.price}"
        )

    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle IB errors."""
        if errorCode in [2104, 2106, 2158]:
            self.state.add_message(f"Info: {errorString}")
        elif errorCode >= 2000:
            self.state.add_message(f"Warning [{errorCode}]: {errorString}")
        else:
            self.state.add_message(f"Error [{errorCode}]: {errorString}")

    def _on_disconnected(self):
        """Handle IB disconnection event."""
        self._handle_disconnect("IB disconnected")

    def _on_timeout(self, idle_seconds: float):
        """Handle IB timeout event (no data received)."""
        self._handle_disconnect(f"IB timeout after {idle_seconds:.1f}s idle")

    # =========================================================================
    # Command Executors
    # =========================================================================

    def _execute_connect(self, cmd: ConnectCommand) -> CommandResult:
        """Execute connect command."""
        if self.state.is_connected:
            return CommandResult(success=True, data="Already connected")

        try:
            host, port, client_id, config, creds = resolve_connection_params(
                mode=cmd.mode,
                host=cmd.host,
                port=cmd.port,
                client_id=cmd.client_id,
            )

            requested_mode = TradingMode.LIVE if cmd.mode.lower() == "live" else TradingMode.PAPER
            validation = self._mode_manager.validate_mode(requested_mode)

            if requested_mode == TradingMode.LIVE:
                if not validation.valid:
                    return CommandResult(success=False, error="Live mode validation failed")
                if not self._mode_manager.require_double_confirmation(auto_confirm=cmd.confirm_live):
                    return CommandResult(success=False, error="Live mode confirmation required")

            self.state.set_connection_status(ConnectionStatus.CONNECTING)

            self._ib = IB()

            # Log connection attempt
            self.state.add_message(
                f"Connecting to IB at {host}:{port} (client {client_id}, mode {cmd.mode})..."
            )

            try:
                self._ib.connect(host, port, clientId=client_id, readonly=False)
            except Exception as conn_err:
                self.state.add_message(f"Connection error: {conn_err}")
                raise

            # Give connection time to establish without blocking engine loop
            if self._ib:
                self._ib.sleep(1)

            max_wait = 5.0
            start = time.time()
            while time.time() - start < max_wait:
                if self._ib.isConnected():
                    self.state.add_message("IB connection confirmed")
                    break
                if self._ib:
                    self._ib.sleep(0.2)  # Poll interval
            else:
                self.state.add_message("Connection timeout - not confirmed")

            self._market_data.set_ib(self._ib)

            if not self._ib.isConnected():
                self.state.set_connection_status(ConnectionStatus.ERROR)
                return CommandResult(success=False, error="Connection failed")

            self._setup_ib_handlers()
            try:
                self._ib.setTimeout(self._heartbeat_interval * 2)
            except Exception:
                pass

            accounts = self._ib.managedAccounts()
            platform = "TWS" if cmd.port in [7496, 7497] else "Gateway"

            self.state.set_connected(
                host=host,
                port=port,
                mode=cmd.mode,
                platform=platform,
                accounts=accounts,
            )
            if creds.get("IB_ACCOUNT_ID"):
                self.state.update_account_summary(account_id=str(creds["IB_ACCOUNT_ID"]))

            # Initialize guardrails based on mode (config overrides)
            limits_cfg = (config or {}).get("limits") if isinstance(config, dict) else None
            if limits_cfg:
                limits = TradingLimits.from_dict(limits_cfg)
            else:
                if requested_mode == TradingMode.LIVE:
                    limits = TradingLimits.live_strict_limits()
                else:
                    limits = TradingLimits.paper_limits()

            self._guardrails = Guardrails(mode=requested_mode, limits=limits)
            # Apply account id if provided in credentials
            account_id = creds.get("IB_ACCOUNT_ID")
            if account_id:
                self.state.update_account_summary(account_id=account_id)
            self._auto_reconnect = self._auto_reconnect_config
            self._reconnect_attempts = 0
            self._next_reconnect_time = None
            self._last_connect_params = {
                "host": host,
                "port": port,
                "client_id": client_id,
                "mode": cmd.mode,
                "confirm_live": cmd.confirm_live,
            }

            # Reconcile open orders on connect
            try:
                self._order_manager.reconcile_open_orders(self._ib)
            except Exception:
                pass

            self._audit_log.log_connection(
                mode=cmd.mode,
                host=host,
                port=port,
                success=True,
            )

            return CommandResult(
                success=True,
                data={
                    "host": host,
                    "port": port,
                    "mode": cmd.mode,
                    "accounts": accounts,
                },
            )

        except Exception as e:
            self.state.set_connection_status(ConnectionStatus.ERROR)
            self._audit_log.log_connection(
                mode=cmd.mode,
                host=host if 'host' in locals() else cmd.host,
                port=port if 'port' in locals() else cmd.port,
                success=False,
                error=str(e),
            )
            return CommandResult(success=False, error=str(e))

    def _execute_disconnect(self, cmd: DisconnectCommand) -> CommandResult:
        """Execute disconnect command."""
        if not self.state.is_connected:
            return CommandResult(success=True, data="Already disconnected")

        try:
            # User-initiated disconnect disables auto-reconnect
            self._auto_reconnect = False
            self._last_connect_params = None
            self._reconnect_attempts = 0
            self._next_reconnect_time = None
            # Cancel market data subscriptions
            self._market_data.stop_all()
            if self._ib:
                self._ib.disconnect()
            self.state.set_disconnected()
            return CommandResult(success=True)
        except Exception as e:
            return CommandResult(success=False, error=str(e))

    def _execute_fetch_historical(self, cmd: FetchHistoricalDataCommand) -> CommandResult:
        """Execute fetch historical data command."""
        if not self.state.is_connected or not self._ib:
            return CommandResult(success=False, error="Not connected")

        try:
            contract = Stock(cmd.symbol.upper(), "SMART", "USD")

            bars = self._ib.reqHistoricalData(
                contract=contract,
                endDateTime="",
                durationStr=cmd.duration,
                barSizeSetting=cmd.bar_size,
                whatToShow=cmd.what_to_show,
                useRTH=1 if cmd.use_rth else 0,
                formatDate=1,
                keepUpToDate=False,
            )

            if not bars:
                return CommandResult(success=False, error="No data received")

            df = util.df(bars)
            if df is None or df.empty:
                return CommandResult(success=False, error="No data received")

            # Normalize column names to match dashboard expectations
            df = df.rename(
                columns={
                    "date": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )

            self.state.set_historical_data(cmd.symbol, df, cmd.duration, cmd.bar_size)

            return CommandResult(success=True, data=df)

        except Exception as e:
            return CommandResult(success=False, error=str(e))

    def _execute_get_account(self, cmd: GetAccountCommand) -> CommandResult:
        """Execute get account command."""
        if not self.state.is_connected or not self._ib:
            return CommandResult(success=False, error="Not connected")

        try:
            creds = load_credentials(self.state.get_connection_info().get("mode", "paper"))
            values = self._ib.accountValues()

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

            account_id = ""
            currency = "USD"
            for av in values:
                if av.tag in tag_mapping:
                    try:
                        val = float(av.value)
                    except (ValueError, TypeError):
                        val = 0.0
                    self.state.update_account_summary(**{tag_mapping[av.tag]: val})
                    account_id = av.account or account_id
                    currency = av.currency or currency

            self.state.update_account_summary(account_id=account_id, currency=currency)
            if not account_id and creds.get("IB_ACCOUNT_ID"):
                self.state.update_account_summary(account_id=str(creds["IB_ACCOUNT_ID"]))

            summary = self.state.get_account_summary()
            if self._guardrails:
                account_value = summary.net_liquidation or summary.total_cash
                if account_value:
                    self._guardrails.update_account_value(account_value)
                daily_pnl = summary.realized_pnl + summary.unrealized_pnl
                self._guardrails.update_daily_pnl(daily_pnl)

            return CommandResult(success=True, data=summary)

        except Exception as e:
            return CommandResult(success=False, error=str(e))

    def _execute_get_positions(self, cmd: GetPositionsCommand) -> CommandResult:
        """Execute get positions command."""
        if not self.state.is_connected or not self._ib:
            return CommandResult(success=False, error="Not connected")

        try:
            self.state.clear_positions()
            positions = self._ib.positions()

            total_exposure = 0.0
            for pos in positions:
                contract = pos.contract
                if pos.position == 0:
                    continue
                position = Position(
                    symbol=contract.symbol,
                    quantity=int(pos.position),
                    avg_cost=pos.avgCost,
                    market_price=getattr(pos, "marketPrice", 0.0),
                    market_value=getattr(pos, "marketValue", 0.0),
                    unrealized_pnl=getattr(pos, "unrealizedPNL", 0.0),
                    realized_pnl=getattr(pos, "realizedPNL", 0.0),
                    account=pos.account,
                    currency=contract.currency,
                    sec_type=contract.secType,
                    exchange=contract.exchange,
                )
                self.state.update_position(position)
                mv = position.market_value or (position.avg_cost * position.quantity)
                total_exposure += abs(mv)

            if self._guardrails:
                self._guardrails.update_portfolio_exposure(total_exposure)

            return CommandResult(success=True, data=self.state.get_positions())

        except Exception as e:
            return CommandResult(success=False, error=str(e))

    def _execute_get_orders(self, cmd: GetOrdersCommand) -> CommandResult:
        """Execute get orders command (state snapshot)."""
        if self.state.is_connected and self._ib:
            try:
                self._order_manager.reconcile_open_orders(self._ib)
            except Exception:
                pass
        return CommandResult(success=True, data=self.state.get_orders())

    def _execute_get_market_subscriptions(self, cmd: GetMarketSubscriptionsCommand) -> CommandResult:
        """Get active market data subscriptions."""
        return CommandResult(success=True, data=self._market_data.get_subscriptions())

    def _execute_get_guardrails_status(self, cmd: GetGuardrailsStatusCommand) -> CommandResult:
        """Get guardrails status."""
        if not self._guardrails:
            return CommandResult(success=False, error="Guardrails not initialized")
        return CommandResult(success=True, data=self._guardrails.get_status())

    def _execute_activate_kill_switch(self, cmd: ActivateKillSwitchCommand) -> CommandResult:
        """Activate kill switch manually."""
        if not self._guardrails:
            return CommandResult(success=False, error="Guardrails not initialized")
        self._guardrails.activate_kill_switch_manual(cmd.reason)
        self._audit_log.log(
            event_type="kill_switch",
            mode=self.state.get_connection_info().get("mode", "paper"),
            success=True,
            details={"reason": cmd.reason},
            guardrails_state=self._guardrails.get_status(),
        )
        return CommandResult(success=True, data=self._guardrails.get_kill_switch_status())

    def _execute_subscribe_market_data(self, cmd: SubscribeMarketDataCommand) -> CommandResult:
        """Subscribe to real-time market data for a symbol."""
        if not self.state.is_connected or not self._ib:
            return CommandResult(success=False, error="Not connected")

        try:
            result = self._market_data.subscribe(cmd.symbol)
            if result.get("status") == "error":
                return CommandResult(success=False, error=result.get("error"))
            return CommandResult(success=True, data=result)
        except Exception as e:
            return CommandResult(success=False, error=str(e))

    def _execute_unsubscribe_market_data(self, cmd: UnsubscribeMarketDataCommand) -> CommandResult:
        """Unsubscribe from real-time market data for a symbol."""
        if not self.state.is_connected or not self._ib:
            return CommandResult(success=False, error="Not connected")

        try:
            result = self._market_data.unsubscribe(cmd.symbol)
            if result.get("status") == "error":
                return CommandResult(success=False, error=result.get("error"))
            return CommandResult(success=True, data=result)
        except Exception as e:
            return CommandResult(success=False, error=str(e))

    # =========================================================================
    # Reconciliation + Reconnect
    # =========================================================================

    def _reconcile_state(self):
        """Reconcile open orders, positions, and account summary."""
        if not self._ib or not self._ib.isConnected():
            return
        try:
            self._ib.reqAllOpenOrders()
        except Exception:
            pass
        try:
            self._order_manager.reconcile_open_orders(self._ib)
        except Exception:
            pass
        try:
            fills = self._ib.reqExecutions()
            for fill in fills:
                self._order_manager.handle_execution(fill)
        except Exception:
            pass
        try:
            self._execute_get_positions(GetPositionsCommand())
        except Exception:
            pass
        try:
            self._execute_get_account(GetAccountCommand())
        except Exception:
            pass

    def _handle_disconnect(self, reason: str):
        """Handle disconnects and schedule reconnection."""
        self.state.add_message(f"Disconnected: {reason}")
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self.state.set_connection_status(ConnectionStatus.ERROR)
        self.state.set_disconnected()
        if self._auto_reconnect and self._last_connect_params:
            self._schedule_reconnect()

    def _schedule_reconnect(self):
        """Schedule next reconnect attempt with backoff."""
        base = 2.0
        max_backoff = 60.0
        backoff = min(base * (2 ** self._reconnect_attempts), max_backoff)
        self._next_reconnect_time = time.time() + backoff
        self._reconnect_attempts += 1
        self.state.add_message(f"Reconnect scheduled in {int(backoff)}s")

    def _maybe_reconnect(self, now: float):
        """Attempt reconnect if scheduled."""
        if not self._last_connect_params:
            return
        if self._next_reconnect_time is None or now < self._next_reconnect_time:
            return

        params = dict(self._last_connect_params)
        host = params["host"]
        port = params["port"]
        client_id = params["client_id"]
        mode = params.get("mode", "paper")
        confirm_live = params.get("confirm_live", False)

        try:
            requested_mode = TradingMode.LIVE if mode == "live" else TradingMode.PAPER
            if requested_mode == TradingMode.LIVE and not self._mode_manager.is_live:
                validation = self._mode_manager.validate_mode(requested_mode)
                if not validation.valid:
                    self._schedule_reconnect()
                    return
                if not self._mode_manager.require_double_confirmation(auto_confirm=confirm_live):
                    self._schedule_reconnect()
                    return

            self.state.set_connection_status(ConnectionStatus.CONNECTING)
            self._ib = IB()
            self._ib.connect(host, port, clientId=client_id, readonly=False)

            # Wait for connection without blocking engine loop
            if self._ib:
                self._ib.sleep(1)
            max_wait = 5.0
            start = time.time()
            while time.time() - start < max_wait:
                if self._ib.isConnected():
                    break
                if self._ib:
                    self._ib.sleep(0.2)

            self._market_data.set_ib(self._ib)

            if not self._ib.isConnected():
                self._handle_disconnect("Reconnect failed")
                return

            self._setup_ib_handlers()
            try:
                self._ib.setTimeout(self._heartbeat_interval * 2)
            except Exception:
                pass

            accounts = self._ib.managedAccounts()
            platform = "TWS" if port in [7496, 7497] else "Gateway"
            self.state.set_connected(
                host=host,
                port=port,
                mode=mode,
                platform=platform,
                accounts=accounts,
            )

            # Resubscribe market data
            try:
                self._market_data.resubscribe_all()
            except Exception:
                pass

            # Full resync
            self._reconcile_state()

            self._reconnect_attempts = 0
            self._next_reconnect_time = None
            self.state.add_message("Reconnected successfully")
        except Exception:
            self._schedule_reconnect()

    def _create_ib_order(self, cmd: PlaceOrderCommand) -> Order:
        """Create an ib_insync Order from PlaceOrderCommand."""
        if cmd.order_type == OrderType.MARKET:
            order = MarketOrder(action=cmd.action.value, totalQuantity=cmd.quantity)
            order.tif = cmd.time_in_force
            return order
        if cmd.order_type == OrderType.LIMIT:
            order = LimitOrder(action=cmd.action.value, totalQuantity=cmd.quantity, lmtPrice=cmd.limit_price)
            order.tif = cmd.time_in_force
            return order
        if cmd.order_type == OrderType.STOP:
            order = StopOrder(action=cmd.action.value, totalQuantity=cmd.quantity, stopPrice=cmd.stop_price)
            order.tif = cmd.time_in_force
            return order
        if cmd.order_type == OrderType.STOP_LIMIT:
            order = Order(
                action=cmd.action.value,
                totalQuantity=cmd.quantity,
                orderType="STP LMT",
                lmtPrice=cmd.limit_price,
                auxPrice=cmd.stop_price,
            )
            order.tif = cmd.time_in_force
            return order
        order = Order(action=cmd.action.value, totalQuantity=cmd.quantity, orderType=cmd.order_type.value)
        order.tif = cmd.time_in_force
        return order

    def _execute_place_order(self, cmd: PlaceOrderCommand) -> CommandResult:
        """Execute place order command."""
        if not self.state.is_connected or not self._ib:
            return CommandResult(success=False, error="Not connected")

        if not cmd.validate():
            return CommandResult(success=False, error="Invalid order parameters")

        existing_order, is_idempotent = self._order_manager.resolve_idempotent(cmd)
        if is_idempotent:
            if not existing_order:
                return CommandResult(success=False, error="Idempotency key already used but order not found")
            return CommandResult(
                success=True,
                data={
                    "order_id": existing_order.order_id,
                    "idempotent": True,
                    "status": existing_order.status.value,
                },
            )

        try:
            # Guardrails validation
            order_request = OrderRequest(
                symbol=cmd.symbol,
                action=cmd.action.value,
                quantity=cmd.quantity,
                order_type=cmd.order_type.value,
                limit_price=cmd.limit_price,
                estimated_value=(cmd.limit_price * cmd.quantity) if cmd.limit_price else None,
                account_id=self.state.get_account_summary().account_id,
            )

            if self._guardrails:
                validation = self._guardrails.validate_order(order_request)
                self._audit_log.log_order_attempt(
                    mode=self.state.get_connection_info().get("mode", "paper"),
                    symbol=cmd.symbol,
                    action=cmd.action.value,
                    quantity=cmd.quantity,
                    order_type=cmd.order_type.value,
                    validation_result=validation.to_dict(),
                    guardrails_state=self._guardrails.get_status(),
                )
                if not validation.approved:
                    self._audit_log.log_order_rejected(
                        mode=self.state.get_connection_info().get("mode", "paper"),
                        symbol=cmd.symbol,
                        action=cmd.action.value,
                        quantity=cmd.quantity,
                        reason=validation.message,
                        violations=validation.to_dict().get("violations", []),
                        guardrails_state=self._guardrails.get_status(),
                    )
                    return CommandResult(success=False, error=validation.message)

            contract = Stock(cmd.symbol.upper(), "SMART", "USD")
            ib_order = self._create_ib_order(cmd)

            # Allocate order id from nextValidId (ib_insync)
            order_id = self._order_manager.allocate_order_id(self._ib)
            ib_order.orderId = order_id

            # Register idempotency key + created state
            self._order_manager.register_key(order_id, cmd.client_order_key)
            self._order_manager.add_created_order(cmd, order_id)

            # Submit with controlled retries
            last_error = None
            for attempt in range(cmd.max_retries + 1):
                try:
                    self._ib.placeOrder(contract, ib_order)
                    self._order_manager.mark_submitted(order_id)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt >= cmd.max_retries:
                        break
                    # brief backoff to avoid hammering IB
                    time.sleep(0.25 * (attempt + 1))

            if last_error:
                self._order_manager.mark_rejected(order_id, str(last_error))
                return CommandResult(success=False, error=str(last_error))

            # Track for guardrails/audit
            self._order_requests[order_id] = order_request

            return CommandResult(success=True, data={"order_id": order_id, "idempotent": False})

        except Exception as e:
            return CommandResult(success=False, error=str(e))

    def _execute_cancel_order(self, cmd: CancelOrderCommand) -> CommandResult:
        """Execute cancel order command."""
        if not self.state.is_connected or not self._ib:
            return CommandResult(success=False, error="Not connected")

        try:
            order = self.state.get_order(cmd.order_id)
            if not order:
                return CommandResult(success=False, error="Order not found")
            self._ib.cancelOrder(cmd.order_id)
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
