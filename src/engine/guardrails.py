"""
Trading Guardrails - Safety Limits System
==========================================
Hard limits to prevent excessive losses in live trading.

CRITICAL: These limits are ENFORCED in live mode.
In paper mode, violations generate warnings but orders proceed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import threading
from collections import deque

from .trading_mode import TradingMode, is_live_mode
from src.utils.logger import logger


class LimitType(Enum):
    """Types of limits that can be violated."""
    DAILY_LOSS = "daily_loss"
    POSITION_LOSS = "position_loss"
    ORDER_SIZE = "order_size"
    ORDER_VALUE = "order_value"
    ORDERS_PER_SESSION = "orders_per_session"
    ORDERS_PER_MINUTE = "orders_per_minute"
    PORTFOLIO_EXPOSURE = "portfolio_exposure"
    POSITION_CONCENTRATION = "position_concentration"
    EMERGENCY_STOP_LOSS = "emergency_stop_loss"


class ViolationSeverity(Enum):
    """Severity levels for limit violations."""
    WARNING = "warning"  # Approaching limit
    SOFT_LIMIT = "soft_limit"  # Exceeded soft limit (warning in paper, block in live)
    HARD_LIMIT = "hard_limit"  # Exceeded hard limit (always block)
    CRITICAL = "critical"  # Triggers kill switch


@dataclass
class LimitViolation:
    """Record of a limit violation."""
    timestamp: datetime
    limit_type: LimitType
    severity: ViolationSeverity
    current_value: float
    limit_value: float
    message: str
    order_details: Optional[Dict] = None
    blocked: bool = False


@dataclass
class TradingLimits:
    """
    Trading limits configuration.

    HARD-CODED DEFAULTS - These are the strictest safe limits.
    Can be relaxed only via config files, never exceeded.
    """
    # Loss limits
    max_daily_loss: float = 1000.0  # USD
    max_position_loss: float = 500.0  # USD per position
    emergency_stop_loss: float = 0.05  # 5% of account

    # Order limits
    max_order_size: int = 100  # shares
    max_order_value: float = 5000.0  # USD

    # Rate limits
    max_orders_per_session: int = 50
    max_orders_per_minute: int = 5

    # Exposure limits
    max_portfolio_exposure: float = 50000.0  # USD total
    max_position_concentration: float = 0.20  # 20% of portfolio

    # Warning thresholds (percentage of limit)
    warning_threshold: float = 0.80  # Warn at 80%

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "max_daily_loss": self.max_daily_loss,
            "max_position_loss": self.max_position_loss,
            "emergency_stop_loss": self.emergency_stop_loss,
            "max_order_size": self.max_order_size,
            "max_order_value": self.max_order_value,
            "max_orders_per_session": self.max_orders_per_session,
            "max_orders_per_minute": self.max_orders_per_minute,
            "max_portfolio_exposure": self.max_portfolio_exposure,
            "max_position_concentration": self.max_position_concentration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TradingLimits':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    @classmethod
    def paper_limits(cls) -> 'TradingLimits':
        """Relaxed limits for paper trading."""
        return cls(
            max_daily_loss=100000.0,
            max_position_loss=50000.0,
            max_order_size=10000,
            max_order_value=1000000.0,
            max_orders_per_session=1000,
            max_orders_per_minute=60,
            max_portfolio_exposure=10000000.0,
            max_position_concentration=1.0,
            emergency_stop_loss=1.0,
        )

    @classmethod
    def live_strict_limits(cls) -> 'TradingLimits':
        """Very strict limits for live trading."""
        return cls(
            max_daily_loss=500.0,
            max_position_loss=200.0,
            max_order_size=50,
            max_order_value=2500.0,
            max_orders_per_session=20,
            max_orders_per_minute=3,
            max_portfolio_exposure=25000.0,
            max_position_concentration=0.10,
            emergency_stop_loss=0.03,
        )


@dataclass
class OrderRequest:
    """Order request to be validated."""
    symbol: str
    action: str  # BUY or SELL
    quantity: int
    order_type: str
    limit_price: Optional[float] = None
    estimated_value: Optional[float] = None
    account_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    """Result of order validation."""
    approved: bool
    violations: List[LimitViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "violations": [
                {
                    "type": v.limit_type.value,
                    "severity": v.severity.value,
                    "current": v.current_value,
                    "limit": v.limit_value,
                    "message": v.message,
                    "blocked": v.blocked,
                }
                for v in self.violations
            ],
            "warnings": self.warnings,
            "message": self.message,
        }


class Guardrails:
    """
    Trading guardrails that enforce safety limits.

    CRITICAL: In live mode, orders that violate limits are REJECTED.
    In paper mode, violations generate warnings but orders proceed.
    """

    def __init__(
        self,
        mode: TradingMode = TradingMode.PAPER,
        limits: Optional[TradingLimits] = None,
        account_value: float = 100000.0,
    ):
        self._mode = mode
        self._limits = limits or (
            TradingLimits.paper_limits() if mode == TradingMode.PAPER
            else TradingLimits()
        )
        self._account_value = account_value

        # Tracking state
        self._lock = threading.RLock()
        self._daily_pnl: float = 0.0
        self._session_orders: int = 0
        self._order_timestamps: deque = deque(maxlen=1000)
        self._position_pnl: Dict[str, float] = {}
        self._portfolio_exposure: float = 0.0

        # Kill switch
        self._kill_switch_active: bool = False
        self._kill_switch_reason: Optional[str] = None
        self._kill_switch_time: Optional[datetime] = None

        # Violation history
        self._violations: List[LimitViolation] = []
        self._max_violations_stored = 1000

        # Callbacks
        self._on_kill_switch: Optional[Callable[[str], None]] = None
        self._on_violation: Optional[Callable[[Any], None]] = None

        self._log(f"Guardrails initialized in {mode.value} mode")

    def _log(self, message: str, level: str = "INFO"):
        """Internal logging."""
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[GUARDRAILS] {message}")

    @property
    def limits(self) -> TradingLimits:
        """Get current limits."""
        return self._limits

    @property
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self._kill_switch_active

    @property
    def mode(self) -> TradingMode:
        """Get current mode."""
        return self._mode

    # =========================================================================
    # Order Validation
    # =========================================================================

    def validate_order(self, order: OrderRequest) -> ValidationResult:
        """
        Validate an order against all limits.

        In LIVE mode: Violations block the order.
        In PAPER mode: Violations generate warnings only.

        Returns ValidationResult with approval status and any violations.
        """
        with self._lock:
            violations = []
            warnings = []

            # Check kill switch first
            if self._kill_switch_active:
                return ValidationResult(
                    approved=False,
                    message=f"KILL SWITCH ACTIVE: {self._kill_switch_reason}",
                    violations=[
                        LimitViolation(
                            timestamp=datetime.now(),
                            limit_type=LimitType.EMERGENCY_STOP_LOSS,
                            severity=ViolationSeverity.CRITICAL,
                            current_value=0,
                            limit_value=0,
                            message="Kill switch is active",
                            blocked=True,
                        )
                    ],
                )

            # Validate each limit
            violations.extend(self._check_order_size(order))
            violations.extend(self._check_order_value(order))
            violations.extend(self._check_rate_limits(order))
            violations.extend(self._check_daily_loss())
            violations.extend(self._check_portfolio_exposure(order))
            violations.extend(self._check_position_concentration(order))
            violations.extend(self._check_emergency_stop())

            # Determine if order should be blocked
            blocked_violations = [v for v in violations if v.severity in
                                  {ViolationSeverity.HARD_LIMIT, ViolationSeverity.CRITICAL}]

            soft_violations = [v for v in violations if v.severity == ViolationSeverity.SOFT_LIMIT]

            # In live mode, soft limits also block
            if self._mode == TradingMode.LIVE and soft_violations:
                for v in soft_violations:
                    v.blocked = True
                blocked_violations.extend(soft_violations)

            # Generate warnings for approaching limits
            warnings.extend(self._check_warning_thresholds())

            # Store violations
            self._violations.extend(violations)
            if len(self._violations) > self._max_violations_stored:
                self._violations = self._violations[-self._max_violations_stored:]

            # Call violation callback
            if violations and self._on_violation:
                for v in violations:
                    self._on_violation(v)

            # Check for critical violations that trigger kill switch
            critical = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
            if critical:
                self._activate_kill_switch(critical[0].message)

            # Determine approval
            if blocked_violations:
                return ValidationResult(
                    approved=False,
                    violations=violations,
                    warnings=warnings,
                    message=f"Order blocked: {blocked_violations[0].message}",
                )

            return ValidationResult(
                approved=True,
                violations=violations,
                warnings=warnings,
                message="Order approved" + (" with warnings" if warnings else ""),
            )

    def _check_order_size(self, order: OrderRequest) -> List[LimitViolation]:
        """Check order size limit."""
        violations = []
        if order.quantity > self._limits.max_order_size:
            violations.append(LimitViolation(
                timestamp=datetime.now(),
                limit_type=LimitType.ORDER_SIZE,
                severity=ViolationSeverity.HARD_LIMIT,
                current_value=order.quantity,
                limit_value=self._limits.max_order_size,
                message=f"Order size {order.quantity} exceeds limit {self._limits.max_order_size}",
                order_details={"symbol": order.symbol, "quantity": order.quantity},
                blocked=True,
            ))
        return violations

    def _check_order_value(self, order: OrderRequest) -> List[LimitViolation]:
        """Check order value limit."""
        violations = []
        estimated_value = order.estimated_value or (
            order.quantity * (order.limit_price or 100)
        )

        if estimated_value > self._limits.max_order_value:
            violations.append(LimitViolation(
                timestamp=datetime.now(),
                limit_type=LimitType.ORDER_VALUE,
                severity=ViolationSeverity.HARD_LIMIT,
                current_value=estimated_value,
                limit_value=self._limits.max_order_value,
                message=f"Order value ${estimated_value:.2f} exceeds limit ${self._limits.max_order_value:.2f}",
                blocked=True,
            ))
        return violations

    def _check_rate_limits(self, order: OrderRequest) -> List[LimitViolation]:
        """Check rate limits (orders per session and per minute)."""
        violations = []
        now = datetime.now()

        # Orders per session
        if self._session_orders >= self._limits.max_orders_per_session:
            violations.append(LimitViolation(
                timestamp=now,
                limit_type=LimitType.ORDERS_PER_SESSION,
                severity=ViolationSeverity.HARD_LIMIT,
                current_value=self._session_orders,
                limit_value=self._limits.max_orders_per_session,
                message=f"Session orders {self._session_orders} reached limit {self._limits.max_orders_per_session}",
                blocked=True,
            ))

        # Orders per minute
        one_minute_ago = now - timedelta(minutes=1)
        recent_orders = sum(1 for ts in self._order_timestamps if ts > one_minute_ago)

        if recent_orders >= self._limits.max_orders_per_minute:
            violations.append(LimitViolation(
                timestamp=now,
                limit_type=LimitType.ORDERS_PER_MINUTE,
                severity=ViolationSeverity.SOFT_LIMIT,
                current_value=recent_orders,
                limit_value=self._limits.max_orders_per_minute,
                message=f"Rate limit: {recent_orders} orders in last minute (limit: {self._limits.max_orders_per_minute})",
                blocked=self._mode == TradingMode.LIVE,
            ))

        return violations

    def _check_daily_loss(self) -> List[LimitViolation]:
        """Check daily loss limit."""
        violations = []

        if self._daily_pnl < 0 and abs(self._daily_pnl) >= self._limits.max_daily_loss:
            violations.append(LimitViolation(
                timestamp=datetime.now(),
                limit_type=LimitType.DAILY_LOSS,
                severity=ViolationSeverity.CRITICAL,
                current_value=abs(self._daily_pnl),
                limit_value=self._limits.max_daily_loss,
                message=f"Daily loss ${abs(self._daily_pnl):.2f} exceeds limit ${self._limits.max_daily_loss:.2f}",
                blocked=True,
            ))

        return violations

    def _check_portfolio_exposure(self, order: OrderRequest) -> List[LimitViolation]:
        """Check portfolio exposure limit."""
        violations = []
        estimated_value = order.estimated_value or (order.quantity * (order.limit_price or 100))

        new_exposure = self._portfolio_exposure
        if order.action.upper() == "BUY":
            new_exposure += estimated_value

        if new_exposure > self._limits.max_portfolio_exposure:
            violations.append(LimitViolation(
                timestamp=datetime.now(),
                limit_type=LimitType.PORTFOLIO_EXPOSURE,
                severity=ViolationSeverity.HARD_LIMIT,
                current_value=new_exposure,
                limit_value=self._limits.max_portfolio_exposure,
                message=f"Portfolio exposure ${new_exposure:.2f} would exceed limit ${self._limits.max_portfolio_exposure:.2f}",
                blocked=True,
            ))

        return violations

    def _check_position_concentration(self, order: OrderRequest) -> List[LimitViolation]:
        """Check position concentration limit."""
        violations = []
        estimated_value = order.estimated_value or (order.quantity * (order.limit_price or 100))

        if self._account_value > 0:
            concentration = estimated_value / self._account_value
            if concentration > self._limits.max_position_concentration:
                violations.append(LimitViolation(
                    timestamp=datetime.now(),
                    limit_type=LimitType.POSITION_CONCENTRATION,
                    severity=ViolationSeverity.SOFT_LIMIT,
                    current_value=concentration,
                    limit_value=self._limits.max_position_concentration,
                    message=f"Position concentration {concentration:.1%} exceeds limit {self._limits.max_position_concentration:.1%}",
                    blocked=self._mode == TradingMode.LIVE,
                ))

        return violations

    def _check_emergency_stop(self) -> List[LimitViolation]:
        """Check emergency stop loss (total account drawdown)."""
        violations = []

        if self._account_value > 0 and self._daily_pnl < 0:
            drawdown = abs(self._daily_pnl) / self._account_value
            if drawdown >= self._limits.emergency_stop_loss:
                violations.append(LimitViolation(
                    timestamp=datetime.now(),
                    limit_type=LimitType.EMERGENCY_STOP_LOSS,
                    severity=ViolationSeverity.CRITICAL,
                    current_value=drawdown,
                    limit_value=self._limits.emergency_stop_loss,
                    message=f"EMERGENCY: Account drawdown {drawdown:.1%} exceeds {self._limits.emergency_stop_loss:.1%}",
                    blocked=True,
                ))

        return violations

    def _check_warning_thresholds(self) -> List[str]:
        """Check if approaching any limits (for warnings)."""
        warnings = []
        threshold = self._limits.warning_threshold

        # Daily loss warning
        if self._daily_pnl < 0:
            loss_pct = abs(self._daily_pnl) / self._limits.max_daily_loss
            if loss_pct >= threshold:
                warnings.append(
                    f"Approaching daily loss limit: ${abs(self._daily_pnl):.2f} / ${self._limits.max_daily_loss:.2f}"
                )

        # Session orders warning
        order_pct = self._session_orders / self._limits.max_orders_per_session
        if order_pct >= threshold:
            warnings.append(
                f"Approaching session order limit: {self._session_orders} / {self._limits.max_orders_per_session}"
            )

        return warnings

    # =========================================================================
    # Kill Switch
    # =========================================================================

    def _activate_kill_switch(self, reason: str):
        """Activate the kill switch - stops all trading."""
        with self._lock:
            if self._kill_switch_active:
                return

            self._kill_switch_active = True
            self._kill_switch_reason = reason
            self._kill_switch_time = datetime.now()

            self._log(f"🚨 KILL SWITCH ACTIVATED: {reason}", "CRITICAL")

            if self._on_kill_switch:
                self._on_kill_switch(reason)

    def activate_kill_switch_manual(self, reason: str = "Manual activation"):
        """Manually activate kill switch."""
        self._activate_kill_switch(f"MANUAL: {reason}")

    def get_kill_switch_status(self) -> dict:
        """Get kill switch status."""
        with self._lock:
            return {
                "active": self._kill_switch_active,
                "reason": self._kill_switch_reason,
                "activated_at": self._kill_switch_time.isoformat() if self._kill_switch_time else None,
            }

    # =========================================================================
    # State Updates
    # =========================================================================

    def record_order_executed(self, order: OrderRequest):
        """Record that an order was executed."""
        with self._lock:
            self._session_orders += 1
            self._order_timestamps.append(datetime.now())

            if order.estimated_value and order.action.upper() == "BUY":
                self._portfolio_exposure += order.estimated_value

    def update_daily_pnl(self, pnl: float):
        """Update daily P&L."""
        with self._lock:
            self._daily_pnl = pnl
            # Check if this triggers emergency stop
            violations = self._check_daily_loss()
            violations.extend(self._check_emergency_stop())
            for v in violations:
                if v.severity == ViolationSeverity.CRITICAL:
                    self._activate_kill_switch(v.message)

    def update_position_pnl(self, symbol: str, pnl: float):
        """Update P&L for a specific position."""
        with self._lock:
            self._position_pnl[symbol] = pnl
            if pnl < 0 and abs(pnl) > self._limits.max_position_loss:
                self._log(f"Position {symbol} loss ${abs(pnl):.2f} exceeds limit", "WARNING")

    def update_portfolio_exposure(self, exposure: float):
        """Update total portfolio exposure."""
        with self._lock:
            self._portfolio_exposure = exposure

    def update_account_value(self, value: float):
        """Update account value."""
        with self._lock:
            self._account_value = value

    def reset_daily_stats(self):
        """Reset daily statistics (call at start of trading day)."""
        with self._lock:
            self._daily_pnl = 0.0
            self._session_orders = 0
            self._order_timestamps.clear()
            self._log("Daily stats reset")

    # =========================================================================
    # Status and Reporting
    # =========================================================================

    def get_status(self) -> dict:
        """Get current guardrails status."""
        with self._lock:
            return {
                "mode": self._mode.value,
                "kill_switch": self.get_kill_switch_status(),
                "limits": self._limits.to_dict(),
                "current": {
                    "daily_pnl": self._daily_pnl,
                    "session_orders": self._session_orders,
                    "portfolio_exposure": self._portfolio_exposure,
                    "account_value": self._account_value,
                },
                "utilization": {
                    "daily_loss": abs(self._daily_pnl) / self._limits.max_daily_loss if self._daily_pnl < 0 else 0,
                    "session_orders": self._session_orders / self._limits.max_orders_per_session,
                    "portfolio_exposure": self._portfolio_exposure / self._limits.max_portfolio_exposure,
                },
                "violations_count": len(self._violations),
            }

    def get_violations(self, last_n: int = 50) -> List[dict]:
        """Get recent violations."""
        with self._lock:
            return [
                {
                    "timestamp": v.timestamp.isoformat(),
                    "type": v.limit_type.value,
                    "severity": v.severity.value,
                    "current": v.current_value,
                    "limit": v.limit_value,
                    "message": v.message,
                    "blocked": v.blocked,
                }
                for v in self._violations[-last_n:]
            ]

    def set_callbacks(
        self,
        on_kill_switch: Optional[callable] = None,
        on_violation: Optional[callable] = None
    ):
        """Set callbacks for events."""
        self._on_kill_switch = on_kill_switch
        self._on_violation = on_violation
