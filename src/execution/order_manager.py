"""
Módulo de gestión de órdenes para Interactive Brokers.

Maneja la creación, envío, seguimiento y cancelación de órdenes,
incluyendo validación y gestión de riesgos.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable

from ib_insync import IB, Order, Trade, Stock, Contract, LimitOrder, MarketOrder, StopOrder

from src.utils.logger import logger, log_trade

try:
    from config.settings import trading_config, ib_config
except ImportError:
    from config.settings import TradingConfig, IBConfig
    trading_config = TradingConfig()
    ib_config = IBConfig()


class OrderStatus(Enum):
    """Estados posibles de una orden."""

    PENDING = auto()       # Orden creada pero no enviada
    SUBMITTED = auto()     # Orden enviada a IB
    ACCEPTED = auto()      # Orden aceptada por el exchange
    PARTIALLY_FILLED = auto()  # Orden parcialmente ejecutada
    FILLED = auto()        # Orden completamente ejecutada
    CANCELLED = auto()     # Orden cancelada
    REJECTED = auto()      # Orden rechazada
    ERROR = auto()         # Error en la orden


class OrderType(Enum):
    """Tipos de orden soportados."""

    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    TRAILING_STOP = "TRAIL"
    MOC = "MOC"  # Market On Close
    LOC = "LOC"  # Limit On Close


class OrderAction(Enum):
    """Acción de la orden."""

    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """Tiempo de validez de la orden."""

    DAY = "DAY"       # Válida durante el día
    GTC = "GTC"       # Good Till Cancelled
    IOC = "IOC"       # Immediate Or Cancel
    FOK = "FOK"       # Fill Or Kill
    OPG = "OPG"       # At the Opening
    GTD = "GTD"       # Good Till Date


@dataclass
class OrderRequest:
    """Solicitud de orden antes de validación."""

    symbol: str
    action: OrderAction
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    exchange: str = "SMART"
    currency: str = "USD"
    account: Optional[str] = None
    parent_id: Optional[int] = None
    transmit: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validación básica después de inicialización."""
        if self.quantity <= 0:
            raise ValueError("Quantity debe ser positivo")

        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit price requerido para orden LIMIT")

        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError("Stop price requerido para orden STOP")


@dataclass
class ManagedOrder:
    """Orden gestionada con estado y tracking."""

    order_id: int
    request: OrderRequest
    contract: Contract
    ib_order: Order
    trade: Optional[Trade] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    error_message: Optional[str] = None
    fills: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        """Indica si la orden está activa."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
        ]

    @property
    def is_complete(self) -> bool:
        """Indica si la orden está completa (filled o cancelada)."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        ]

    @property
    def remaining_quantity(self) -> int:
        """Cantidad pendiente de ejecutar."""
        return self.request.quantity - self.filled_quantity

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        return {
            "order_id": self.order_id,
            "symbol": self.request.symbol,
            "action": self.request.action.value,
            "quantity": self.request.quantity,
            "order_type": self.request.order_type.value,
            "limit_price": self.request.limit_price,
            "stop_price": self.request.stop_price,
            "status": self.status.name,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "remaining": self.remaining_quantity,
            "commission": self.commission,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
        }


# Type alias para callbacks
OrderCallback = Callable[[ManagedOrder], None]


class OrderValidator:
    """Validador de órdenes antes de envío."""

    def __init__(
        self,
        max_order_value: float = 50000.0,
        max_position_size: float = 100000.0,
        max_daily_orders: int = 100,
        allowed_symbols: Optional[List[str]] = None,
        blocked_symbols: Optional[List[str]] = None,
    ):
        """
        Inicializa el validador de órdenes.

        Args:
            max_order_value: Valor máximo por orden en USD
            max_position_size: Tamaño máximo de posición en USD
            max_daily_orders: Máximo de órdenes por día
            allowed_symbols: Lista blanca de símbolos (None = todos permitidos)
            blocked_symbols: Lista negra de símbolos
        """
        self.max_order_value = max_order_value
        self.max_position_size = max_position_size
        self.max_daily_orders = max_daily_orders
        self.allowed_symbols = set(allowed_symbols) if allowed_symbols else None
        self.blocked_symbols = set(blocked_symbols) if blocked_symbols else set()

        self._daily_order_count = 0
        self._last_reset_date = datetime.now().date()

    def validate(
        self,
        request: OrderRequest,
        current_price: Optional[float] = None,
        current_position: int = 0,
    ) -> tuple[bool, Optional[str]]:
        """
        Valida una solicitud de orden.

        Args:
            request: Solicitud de orden a validar
            current_price: Precio actual del instrumento
            current_position: Posición actual en el instrumento

        Returns:
            Tupla (válido, mensaje_error)
        """
        # Reset contador diario si es nuevo día
        today = datetime.now().date()
        if today > self._last_reset_date:
            self._daily_order_count = 0
            self._last_reset_date = today

        # Validar símbolo
        if request.symbol in self.blocked_symbols:
            return False, f"Símbolo bloqueado: {request.symbol}"

        if self.allowed_symbols and request.symbol not in self.allowed_symbols:
            return False, f"Símbolo no permitido: {request.symbol}"

        # Validar cantidad
        if request.quantity <= 0:
            return False, "Cantidad debe ser positiva"

        # Validar valor de la orden
        if current_price:
            order_value = request.quantity * current_price

            if order_value > self.max_order_value:
                return False, f"Valor de orden (${order_value:.2f}) excede máximo (${self.max_order_value:.2f})"

            # Validar tamaño de posición resultante
            if request.action == OrderAction.BUY:
                new_position = current_position + request.quantity
            else:
                new_position = current_position - request.quantity

            position_value = abs(new_position) * current_price
            if position_value > self.max_position_size:
                return False, f"Posición resultante (${position_value:.2f}) excede máximo (${self.max_position_size:.2f})"

        # Validar límite diario
        if self._daily_order_count >= self.max_daily_orders:
            return False, f"Límite diario de órdenes alcanzado ({self.max_daily_orders})"

        # Validar precio límite
        if request.order_type == OrderType.LIMIT:
            if request.limit_price is None or request.limit_price <= 0:
                return False, "Precio límite inválido"

            if current_price:
                # Advertir si el precio límite está muy lejos del mercado
                diff_percent = abs(request.limit_price - current_price) / current_price * 100
                if diff_percent > 10:
                    logger.warning(
                        f"Precio límite {diff_percent:.1f}% lejos del mercado: "
                        f"limit={request.limit_price}, market={current_price}"
                    )

        return True, None

    def increment_order_count(self) -> None:
        """Incrementa el contador de órdenes diarias."""
        self._daily_order_count += 1


class OrderManager:
    """
    Gestor principal de órdenes.

    Maneja el ciclo de vida completo de las órdenes:
    creación, validación, envío, tracking y cancelación.

    Ejemplo de uso:
        async with IBClient() as client:
            manager = OrderManager(client.ib)

            # Crear y enviar orden
            order = await manager.submit_order(
                OrderRequest(
                    symbol="AAPL",
                    action=OrderAction.BUY,
                    quantity=10,
                    order_type=OrderType.LIMIT,
                    limit_price=150.00,
                )
            )

            # Esperar fill
            await manager.wait_for_fill(order.order_id, timeout=60)

            # O cancelar
            await manager.cancel_order(order.order_id)
    """

    def __init__(
        self,
        ib: IB,
        validator: Optional[OrderValidator] = None,
        auto_qualify_contracts: bool = True,
    ):
        """
        Inicializa el gestor de órdenes.

        Args:
            ib: Instancia conectada de IB
            validator: Validador de órdenes (opcional)
            auto_qualify_contracts: Validar contratos automáticamente
        """
        self.ib = ib
        self.validator = validator or OrderValidator(
            max_order_value=trading_config.max_position_size,
            max_position_size=trading_config.max_portfolio_exposure,
        )
        self.auto_qualify_contracts = auto_qualify_contracts

        self._orders: Dict[int, ManagedOrder] = {}
        self._next_order_id = 1
        self._callbacks: Dict[str, List[OrderCallback]] = {
            "on_submitted": [],
            "on_filled": [],
            "on_partially_filled": [],
            "on_cancelled": [],
            "on_rejected": [],
            "on_error": [],
        }

        # Configurar event handlers de IB
        self._setup_event_handlers()

        logger.info("OrderManager inicializado")

    def _setup_event_handlers(self) -> None:
        """Configura handlers para eventos de IB."""
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_execution
        self.ib.commissionReportEvent += self._on_commission
        self.ib.errorEvent += self._on_error

    def _on_order_status(
        self,
        trade: Trade,
    ) -> None:
        """Handler para cambios de estado de órdenes."""
        order_id = trade.order.orderId

        if order_id not in self._orders:
            return

        managed_order = self._orders[order_id]
        managed_order.trade = trade

        old_status = managed_order.status
        new_status = self._map_ib_status(trade.orderStatus.status)

        if new_status == old_status:
            return

        managed_order.status = new_status
        managed_order.filled_quantity = int(trade.orderStatus.filled)
        managed_order.avg_fill_price = trade.orderStatus.avgFillPrice

        logger.debug(
            f"Orden {order_id} status: {old_status.name} -> {new_status.name} "
            f"(filled={managed_order.filled_quantity}/{managed_order.request.quantity})"
        )

        # Actualizar timestamps
        if new_status == OrderStatus.SUBMITTED:
            managed_order.submitted_at = datetime.now()
            self._trigger_callbacks("on_submitted", managed_order)

        elif new_status == OrderStatus.FILLED:
            managed_order.filled_at = datetime.now()
            self._trigger_callbacks("on_filled", managed_order)

        elif new_status == OrderStatus.PARTIALLY_FILLED:
            self._trigger_callbacks("on_partially_filled", managed_order)

        elif new_status == OrderStatus.CANCELLED:
            managed_order.cancelled_at = datetime.now()
            self._trigger_callbacks("on_cancelled", managed_order)

        elif new_status == OrderStatus.REJECTED:
            self._trigger_callbacks("on_rejected", managed_order)

    def _on_execution(self, trade: Trade, fill) -> None:
        """Handler para ejecuciones de órdenes."""
        order_id = trade.order.orderId

        if order_id not in self._orders:
            return

        managed_order = self._orders[order_id]

        fill_data = {
            "exec_id": fill.execution.execId,
            "time": fill.execution.time,
            "shares": fill.execution.shares,
            "price": fill.execution.price,
            "side": fill.execution.side,
            "exchange": fill.execution.exchange,
        }

        managed_order.fills.append(fill_data)

        logger.info(
            f"Ejecución orden {order_id}: {fill.execution.shares} @ {fill.execution.price}"
        )

    def _on_commission(self, trade: Trade, fill, report) -> None:
        """Handler para reportes de comisiones."""
        order_id = trade.order.orderId

        if order_id not in self._orders:
            return

        managed_order = self._orders[order_id]
        managed_order.commission += report.commission

        logger.debug(
            f"Comisión orden {order_id}: ${report.commission:.2f} "
            f"(total: ${managed_order.commission:.2f})"
        )

    def _on_error(self, reqId, errorCode, errorString, contract) -> None:
        """Handler para errores de órdenes."""
        if reqId not in self._orders:
            return

        managed_order = self._orders[reqId]

        # Códigos de error que indican rechazo
        rejection_codes = [201, 202, 203, 321, 322]

        if errorCode in rejection_codes:
            managed_order.status = OrderStatus.REJECTED
            managed_order.error_message = errorString
            self._trigger_callbacks("on_rejected", managed_order)
        else:
            managed_order.error_message = errorString
            self._trigger_callbacks("on_error", managed_order)

        logger.error(
            f"Error orden {reqId} [{errorCode}]: {errorString}"
        )

    def _map_ib_status(self, ib_status: str) -> OrderStatus:
        """Mapea status de IB a OrderStatus."""
        status_map = {
            "PendingSubmit": OrderStatus.PENDING,
            "PendingCancel": OrderStatus.PENDING,
            "PreSubmitted": OrderStatus.SUBMITTED,
            "Submitted": OrderStatus.SUBMITTED,
            "ApiCancelled": OrderStatus.CANCELLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Filled": OrderStatus.FILLED,
            "Inactive": OrderStatus.REJECTED,
        }
        return status_map.get(ib_status, OrderStatus.PENDING)

    def _trigger_callbacks(self, event: str, order: ManagedOrder) -> None:
        """Ejecuta callbacks para un evento."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error en callback {event}: {e}")

    async def submit_order(
        self,
        request: OrderRequest,
        validate: bool = True,
    ) -> ManagedOrder:
        """
        Envía una orden a IB.

        Args:
            request: Solicitud de orden
            validate: Si validar la orden antes de enviar

        Returns:
            ManagedOrder con tracking de la orden

        Raises:
            ValueError: Si la validación falla
            ConnectionError: Si no hay conexión con IB
        """
        if not self.ib.isConnected():
            raise ConnectionError("No hay conexión activa con IB")

        with log_trade(f"Orden {request.action.value} {request.quantity} {request.symbol}"):
            # Crear contrato
            contract = Stock(
                request.symbol,
                request.exchange,
                request.currency,
            )

            # Validar contrato si está habilitado
            if self.auto_qualify_contracts:
                qualified = await self.ib.qualifyContractsAsync(contract)
                if not qualified:
                    raise ValueError(f"No se pudo validar contrato: {request.symbol}")
                contract = qualified[0]

            # Obtener precio actual para validación
            current_price = None
            if validate:
                ticker = self.ib.reqMktData(contract, snapshot=True)
                await asyncio.sleep(1)
                current_price = ticker.last if ticker.last > 0 else ticker.close
                self.ib.cancelMktData(contract)

            # Validar orden
            if validate:
                # Obtener posición actual
                positions = self.ib.positions()
                current_position = 0
                for pos in positions:
                    if pos.contract.symbol == request.symbol:
                        current_position = int(pos.position)
                        break

                valid, error = self.validator.validate(
                    request,
                    current_price=current_price,
                    current_position=current_position,
                )

                if not valid:
                    raise ValueError(f"Validación fallida: {error}")

            # Crear orden de IB
            ib_order = self._create_ib_order(request)

            # Obtener ID de orden
            order_id = self.ib.client.getReqId()

            # Crear orden gestionada
            managed_order = ManagedOrder(
                order_id=order_id,
                request=request,
                contract=contract,
                ib_order=ib_order,
            )

            self._orders[order_id] = managed_order

            # Enviar orden
            logger.info(
                f"Enviando orden {order_id}: {request.action.value} "
                f"{request.quantity} {request.symbol} @ "
                f"{request.limit_price or 'MKT'}"
            )

            trade = self.ib.placeOrder(contract, ib_order)
            managed_order.trade = trade
            managed_order.status = OrderStatus.SUBMITTED
            managed_order.submitted_at = datetime.now()

            # Incrementar contador de órdenes
            self.validator.increment_order_count()

            return managed_order

    def _create_ib_order(self, request: OrderRequest) -> Order:
        """Crea objeto Order de IB desde OrderRequest."""
        if request.order_type == OrderType.MARKET:
            order = MarketOrder(
                action=request.action.value,
                totalQuantity=request.quantity,
            )

        elif request.order_type == OrderType.LIMIT:
            order = LimitOrder(
                action=request.action.value,
                totalQuantity=request.quantity,
                lmtPrice=request.limit_price,
            )

        elif request.order_type == OrderType.STOP:
            order = StopOrder(
                action=request.action.value,
                totalQuantity=request.quantity,
                stopPrice=request.stop_price,
            )

        elif request.order_type == OrderType.STOP_LIMIT:
            order = Order(
                action=request.action.value,
                totalQuantity=request.quantity,
                orderType="STP LMT",
                lmtPrice=request.limit_price,
                auxPrice=request.stop_price,
            )

        else:
            order = Order(
                action=request.action.value,
                totalQuantity=request.quantity,
                orderType=request.order_type.value,
            )

        # Configurar tiempo de validez
        order.tif = request.time_in_force.value

        # Configurar cuenta si se especifica
        if request.account:
            order.account = request.account

        # Configurar transmisión
        order.transmit = request.transmit

        # Parent order para brackets
        if request.parent_id:
            order.parentId = request.parent_id

        return order

    async def submit_bracket_order(
        self,
        symbol: str,
        action: OrderAction,
        quantity: int,
        entry_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        **kwargs,
    ) -> List[ManagedOrder]:
        """
        Envía una orden bracket (entry + take profit + stop loss).

        Args:
            symbol: Símbolo del instrumento
            action: BUY o SELL
            quantity: Cantidad
            entry_price: Precio de entrada (limit)
            take_profit_price: Precio de take profit
            stop_loss_price: Precio de stop loss
            **kwargs: Parámetros adicionales

        Returns:
            Lista con las tres órdenes [entry, take_profit, stop_loss]
        """
        # Determinar acción de salida
        exit_action = OrderAction.SELL if action == OrderAction.BUY else OrderAction.BUY

        # Orden de entrada
        entry_request = OrderRequest(
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=entry_price,
            transmit=False,  # No transmitir aún
            **kwargs,
        )

        entry_order = await self.submit_order(entry_request, validate=True)
        parent_id = entry_order.order_id

        # Orden de take profit
        tp_request = OrderRequest(
            symbol=symbol,
            action=exit_action,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=take_profit_price,
            parent_id=parent_id,
            transmit=False,
            **kwargs,
        )

        tp_order = await self.submit_order(tp_request, validate=False)

        # Orden de stop loss
        sl_request = OrderRequest(
            symbol=symbol,
            action=exit_action,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_loss_price,
            parent_id=parent_id,
            transmit=True,  # Transmitir todo el bracket
            **kwargs,
        )

        sl_order = await self.submit_order(sl_request, validate=False)

        logger.info(
            f"Bracket order enviada: entry={entry_order.order_id}, "
            f"tp={tp_order.order_id}, sl={sl_order.order_id}"
        )

        return [entry_order, tp_order, sl_order]

    async def cancel_order(self, order_id: int) -> bool:
        """
        Cancela una orden.

        Args:
            order_id: ID de la orden a cancelar

        Returns:
            True si se envió la cancelación
        """
        if order_id not in self._orders:
            logger.warning(f"Orden {order_id} no encontrada")
            return False

        managed_order = self._orders[order_id]

        if not managed_order.is_active:
            logger.warning(f"Orden {order_id} no está activa: {managed_order.status.name}")
            return False

        if managed_order.trade:
            self.ib.cancelOrder(managed_order.trade.order)
            logger.info(f"Cancelación enviada para orden {order_id}")
            return True

        return False

    async def cancel_all_orders(self) -> int:
        """
        Cancela todas las órdenes activas.

        Returns:
            Número de órdenes canceladas
        """
        count = 0
        for order_id, order in self._orders.items():
            if order.is_active:
                if await self.cancel_order(order_id):
                    count += 1

        logger.info(f"Canceladas {count} órdenes")
        return count

    async def modify_order(
        self,
        order_id: int,
        new_quantity: Optional[int] = None,
        new_limit_price: Optional[float] = None,
        new_stop_price: Optional[float] = None,
    ) -> bool:
        """
        Modifica una orden activa.

        Args:
            order_id: ID de la orden a modificar
            new_quantity: Nueva cantidad (opcional)
            new_limit_price: Nuevo precio límite (opcional)
            new_stop_price: Nuevo precio stop (opcional)

        Returns:
            True si se envió la modificación
        """
        if order_id not in self._orders:
            logger.warning(f"Orden {order_id} no encontrada")
            return False

        managed_order = self._orders[order_id]

        if not managed_order.is_active:
            logger.warning(f"Orden {order_id} no está activa")
            return False

        ib_order = managed_order.ib_order

        if new_quantity is not None:
            ib_order.totalQuantity = new_quantity

        if new_limit_price is not None:
            ib_order.lmtPrice = new_limit_price

        if new_stop_price is not None:
            ib_order.auxPrice = new_stop_price

        self.ib.placeOrder(managed_order.contract, ib_order)

        logger.info(
            f"Orden {order_id} modificada: qty={new_quantity}, "
            f"limit={new_limit_price}, stop={new_stop_price}"
        )

        return True

    async def wait_for_fill(
        self,
        order_id: int,
        timeout: float = 60.0,
    ) -> bool:
        """
        Espera a que una orden se ejecute.

        Args:
            order_id: ID de la orden
            timeout: Tiempo máximo de espera en segundos

        Returns:
            True si la orden se ejecutó completamente
        """
        if order_id not in self._orders:
            return False

        start_time = asyncio.get_event_loop().time()

        while True:
            order = self._orders[order_id]

            if order.status == OrderStatus.FILLED:
                return True

            if order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                return False

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                logger.warning(f"Timeout esperando fill de orden {order_id}")
                return False

            await asyncio.sleep(0.1)

    def get_order(self, order_id: int) -> Optional[ManagedOrder]:
        """Obtiene una orden por ID."""
        return self._orders.get(order_id)

    def get_active_orders(self) -> List[ManagedOrder]:
        """Obtiene todas las órdenes activas."""
        return [o for o in self._orders.values() if o.is_active]

    def get_filled_orders(self) -> List[ManagedOrder]:
        """Obtiene todas las órdenes ejecutadas."""
        return [o for o in self._orders.values() if o.status == OrderStatus.FILLED]

    def get_orders_by_symbol(self, symbol: str) -> List[ManagedOrder]:
        """Obtiene órdenes de un símbolo específico."""
        return [o for o in self._orders.values() if o.request.symbol == symbol]

    def on_filled(self, callback: OrderCallback) -> None:
        """Registra callback para órdenes ejecutadas."""
        self._callbacks["on_filled"].append(callback)

    def on_cancelled(self, callback: OrderCallback) -> None:
        """Registra callback para órdenes canceladas."""
        self._callbacks["on_cancelled"].append(callback)

    def on_rejected(self, callback: OrderCallback) -> None:
        """Registra callback para órdenes rechazadas."""
        self._callbacks["on_rejected"].append(callback)

    def on_error(self, callback: OrderCallback) -> None:
        """Registra callback para errores."""
        self._callbacks["on_error"].append(callback)

    def get_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del estado del manager."""
        orders = list(self._orders.values())

        return {
            "total_orders": len(orders),
            "active_orders": len([o for o in orders if o.is_active]),
            "filled_orders": len([o for o in orders if o.status == OrderStatus.FILLED]),
            "cancelled_orders": len([o for o in orders if o.status == OrderStatus.CANCELLED]),
            "rejected_orders": len([o for o in orders if o.status == OrderStatus.REJECTED]),
            "total_commission": sum(o.commission for o in orders),
            "daily_order_count": self.validator._daily_order_count,
        }
