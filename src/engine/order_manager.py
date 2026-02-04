"""
Order Manager for Trading Engine
================================
Centralized order lifecycle management with idempotency.

Implements a simple state machine:
Created -> Submitted -> PartiallyFilled -> Filled / Cancelled / Rejected
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Tuple, List
from threading import RLock

from ib_insync import Trade, IB

from .state import EngineState, OrderStatus, Order as EngineOrder
from .commands import PlaceOrderCommand


@dataclass
class OrderRecord:
    """Internal order record for idempotency tracking."""
    order_id: int
    client_order_key: Optional[str]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class OrderManager:
    """
    Manages orders lifecycle and idempotency inside the engine.
    """

    def __init__(self, state: EngineState):
        self._state = state
        self._lock = RLock()
        self._client_key_index: Dict[str, int] = {}
        self._records: Dict[int, OrderRecord] = {}

    # ---------------------------------------------------------------------
    # Idempotency
    # ---------------------------------------------------------------------

    def resolve_idempotent(self, cmd: PlaceOrderCommand) -> Tuple[Optional[EngineOrder], bool]:
        """Return existing order if client_order_key was already used."""
        key = cmd.client_order_key
        if not key:
            return None, False

        with self._lock:
            order_id = self._client_key_index.get(key)
            if not order_id:
                return None, False
        return self._state.get_order(order_id), True

    def register_key(self, order_id: int, client_order_key: Optional[str]) -> None:
        if not client_order_key:
            return
        with self._lock:
            self._client_key_index[client_order_key] = order_id
            self._records[order_id] = OrderRecord(
                order_id=order_id,
                client_order_key=client_order_key,
            )

    # ---------------------------------------------------------------------
    # Order ID allocation
    # ---------------------------------------------------------------------

    def allocate_order_id(self, ib: IB) -> int:
        """
        Allocate order id using IB nextValidId (via client.getReqId()).
        """
        order_id = ib.client.getReqId()
        self._state.set_next_order_id(order_id + 1)
        return order_id

    # ---------------------------------------------------------------------
    # State updates
    # ---------------------------------------------------------------------

    def add_created_order(self, cmd: PlaceOrderCommand, order_id: int) -> None:
        """Add a created order to state."""
        order = EngineOrder(
            order_id=order_id,
            symbol=cmd.symbol.upper(),
            action=cmd.action.value,
            quantity=cmd.quantity,
            order_type=cmd.order_type.value,
            limit_price=cmd.limit_price,
            stop_price=cmd.stop_price,
            client_order_key=cmd.client_order_key,
            status=OrderStatus.PENDING,
        )
        self._state.add_order(order)

    def mark_submitted(self, order_id: int) -> None:
        self._state.update_order(order_id, status=OrderStatus.SUBMITTED)

    def mark_rejected(self, order_id: int, error_message: str) -> None:
        self._state.update_order(order_id, status=OrderStatus.REJECTED, error_message=error_message)

    def handle_order_status(self, trade: Trade) -> None:
        """Map IB orderStatus to engine state."""
        order_id = trade.order.orderId

        status_mapping = {
            "PendingSubmit": OrderStatus.PENDING,
            "PendingCancel": OrderStatus.PENDING,
            "PreSubmitted": OrderStatus.SUBMITTED,
            "Submitted": OrderStatus.SUBMITTED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "ApiCancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.REJECTED,
        }

        new_status = status_mapping.get(trade.orderStatus.status, OrderStatus.PENDING)

        self._state.update_order(
            order_id,
            status=new_status,
            filled_quantity=int(trade.orderStatus.filled),
            avg_fill_price=trade.orderStatus.avgFillPrice,
        )

    def reconcile_open_orders(self, ib: IB) -> Dict[int, str]:
        """
        Reconcile open orders with IB.
        Returns dict of order_id -> status for reconciled orders.
        """
        reconciled: Dict[int, str] = {}
        try:
            trades: List[Trade] = ib.openTrades()
        except Exception:
            trades = []

        for trade in trades:
            order_id = trade.order.orderId
            if self._state.get_order(order_id) is None:
                # Unknown order; register a minimal record
                order = EngineOrder(
                    order_id=order_id,
                    symbol=trade.contract.symbol,
                    action=trade.order.action,
                    quantity=int(trade.order.totalQuantity),
                    order_type=getattr(trade.order, "orderType", ""),
                    limit_price=getattr(trade.order, "lmtPrice", None),
                    stop_price=getattr(trade.order, "auxPrice", None),
                    status=OrderStatus.PENDING,
                )
                self._state.add_order(order)

            self.handle_order_status(trade)
            status = self._state.get_order(order_id)
            if status:
                reconciled[order_id] = status.status.value

        return reconciled
