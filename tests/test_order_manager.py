import pytest

from src.engine.order_manager import OrderManager
from src.engine.state import EngineState, OrderStatus
from src.engine.commands import PlaceOrderCommand, OrderAction, OrderType


class DummyOrderStatus:
    def __init__(self, status: str, filled: int = 0, avg_fill: float = 0.0):
        self.status = status
        self.filled = filled
        self.avgFillPrice = avg_fill


class DummyOrder:
    def __init__(self, order_id: int, total_qty: int = 1, action: str = "BUY"):
        self.orderId = order_id
        self.totalQuantity = total_qty
        self.action = action


class DummyContract:
    def __init__(self, symbol: str):
        self.symbol = symbol


class DummyTrade:
    def __init__(self, order_id: int, status: str, filled: int = 0, avg_fill: float = 0.0):
        self.order = DummyOrder(order_id=order_id)
        self.orderStatus = DummyOrderStatus(status=status, filled=filled, avg_fill=avg_fill)
        self.contract = DummyContract("AAPL")


class DummyIB:
    def __init__(self, trades):
        self._trades = trades

    def openTrades(self):
        return self._trades


@pytest.fixture(autouse=True)
def _idempotency_file(tmp_path, monkeypatch):
    monkeypatch.setenv("IB_IDEMPOTENCY_FILE", str(tmp_path / "idempotency.jsonl"))


def test_idempotency_resolve():
    state = EngineState()
    om = OrderManager(state)

    cmd = PlaceOrderCommand(
        symbol="AAPL",
        action=OrderAction.BUY,
        quantity=1,
        order_type=OrderType.MARKET,
        client_order_key="intent-1",
    )
    om.add_created_order(cmd, order_id=101)
    om.register_key(101, "intent-1")

    order, is_idem = om.resolve_idempotent(cmd)
    assert is_idem is True
    assert order is not None
    assert order.order_id == 101


def test_order_status_mapping():
    state = EngineState()
    om = OrderManager(state)

    cmd = PlaceOrderCommand(
        symbol="AAPL",
        action=OrderAction.BUY,
        quantity=1,
        order_type=OrderType.MARKET,
    )
    om.add_created_order(cmd, order_id=200)

    trade = DummyTrade(order_id=200, status="Filled", filled=1, avg_fill=123.45)
    om.handle_order_status(trade)

    updated = state.get_order(200)
    assert updated is not None
    assert updated.status == OrderStatus.FILLED
    assert updated.filled_quantity == 1
    assert updated.avg_fill_price == 123.45


def test_order_status_partial_and_reject():
    state = EngineState()
    om = OrderManager(state)

    cmd = PlaceOrderCommand(
        symbol="AAPL",
        action=OrderAction.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
    )
    om.add_created_order(cmd, order_id=201)

    trade_partial = DummyTrade(order_id=201, status="PartiallyFilled", filled=4, avg_fill=120.0)
    om.handle_order_status(trade_partial)
    updated = state.get_order(201)
    assert updated.status == OrderStatus.PARTIALLY_FILLED
    assert updated.filled_quantity == 4

    trade_reject = DummyTrade(order_id=201, status="Inactive", filled=0, avg_fill=0.0)
    om.handle_order_status(trade_reject)
    updated = state.get_order(201)
    assert updated.status == OrderStatus.REJECTED


def test_reconcile_adds_unknown_orders():
    state = EngineState()
    om = OrderManager(state)

    trade = DummyTrade(order_id=300, status="Submitted", filled=0, avg_fill=0.0)
    ib = DummyIB(trades=[trade])

    reconciled = om.reconcile_open_orders(ib)
    assert 300 in reconciled
    assert state.get_order(300) is not None
