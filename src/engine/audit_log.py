"""
Audit Log for Trading Operations
=================================
Immutable log of all trading attempts for compliance and debugging.

Logs are stored in JSON Lines format (.jsonl) for easy parsing.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from threading import Lock
import hashlib


@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: str
    event_type: str  # order_attempt, order_executed, order_rejected, kill_switch, etc.
    mode: str  # paper or live
    success: bool
    details: Dict[str, Any]
    guardrails_state: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    checksum: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLog:
    """
    Audit logging for trading operations.

    Features:
    - JSON Lines format for easy parsing
    - Checksums for integrity verification
    - Thread-safe writing
    - No rotation (complete history maintained)
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        filename: str = "audit_trades.jsonl"
    ):
        self._log_dir = log_dir or Path(__file__).parent.parent.parent / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / filename
        self._lock = Lock()
        self._entry_count = 0
        self._last_checksum: Optional[str] = None

        # Initialize log file with header if new
        if not self._log_file.exists():
            self._write_header()

    def _write_header(self):
        """Write header comment to new log file."""
        header = {
            "type": "audit_log_header",
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "description": "Trading audit log - DO NOT MODIFY",
        }
        with open(self._log_file, 'w') as f:
            f.write(json.dumps(header) + '\n')

    def _compute_checksum(self, data: str) -> str:
        """Compute checksum for integrity."""
        combined = f"{self._last_checksum or ''}{data}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def log(
        self,
        event_type: str,
        mode: str,
        success: bool,
        details: Dict[str, Any],
        guardrails_state: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None
    ):
        """
        Log an audit entry.

        Args:
            event_type: Type of event (order_attempt, order_executed, etc.)
            mode: Trading mode (paper/live)
            success: Whether the operation succeeded
            details: Event-specific details
            guardrails_state: Current state of guardrails
            reason: Reason for failure (if applicable)
        """
        with self._lock:
            entry = AuditEntry(
                timestamp=datetime.now().isoformat(),
                event_type=event_type,
                mode=mode,
                success=success,
                details=details,
                guardrails_state=guardrails_state,
                reason=reason,
            )

            # Compute checksum
            json_str = entry.to_json()
            entry.checksum = self._compute_checksum(json_str)
            self._last_checksum = entry.checksum

            # Write to file
            with open(self._log_file, 'a') as f:
                f.write(entry.to_json() + '\n')

            self._entry_count += 1

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def log_order_attempt(
        self,
        mode: str,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str,
        validation_result: Dict,
        guardrails_state: Optional[Dict] = None
    ):
        """Log an order attempt (before execution)."""
        self.log(
            event_type="order_attempt",
            mode=mode,
            success=validation_result.get("approved", False),
            details={
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "order_type": order_type,
                "validation": validation_result,
            },
            guardrails_state=guardrails_state,
            reason=validation_result.get("message") if not validation_result.get("approved") else None,
        )

    def log_order_executed(
        self,
        mode: str,
        order_id: int,
        symbol: str,
        action: str,
        quantity: int,
        fill_price: float,
        guardrails_state: Optional[Dict] = None
    ):
        """Log a successfully executed order."""
        self.log(
            event_type="order_executed",
            mode=mode,
            success=True,
            details={
                "order_id": order_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "fill_price": fill_price,
            },
            guardrails_state=guardrails_state,
        )

    def log_order_rejected(
        self,
        mode: str,
        symbol: str,
        action: str,
        quantity: int,
        reason: str,
        violations: List[Dict],
        guardrails_state: Optional[Dict] = None
    ):
        """Log a rejected order."""
        self.log(
            event_type="order_rejected",
            mode=mode,
            success=False,
            details={
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "violations": violations,
            },
            guardrails_state=guardrails_state,
            reason=reason,
        )

    def log_kill_switch(
        self,
        mode: str,
        reason: str,
        triggered_by: str,
        guardrails_state: Optional[Dict] = None
    ):
        """Log kill switch activation."""
        self.log(
            event_type="kill_switch_activated",
            mode=mode,
            success=True,  # Kill switch activation is "successful" in the sense it worked
            details={
                "triggered_by": triggered_by,
            },
            guardrails_state=guardrails_state,
            reason=reason,
        )

    def log_mode_change(
        self,
        old_mode: str,
        new_mode: str,
        validation_passed: bool,
        reason: Optional[str] = None
    ):
        """Log mode change."""
        self.log(
            event_type="mode_change",
            mode=new_mode,
            success=validation_passed,
            details={
                "old_mode": old_mode,
                "new_mode": new_mode,
            },
            reason=reason,
        )

    def log_connection(
        self,
        mode: str,
        host: str,
        port: int,
        success: bool,
        error: Optional[str] = None
    ):
        """Log connection attempt."""
        self.log(
            event_type="connection",
            mode=mode,
            success=success,
            details={
                "host": host,
                "port": port,
            },
            reason=error,
        )

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_entries(
        self,
        last_n: Optional[int] = None,
        event_type: Optional[str] = None,
        mode: Optional[str] = None,
        success: Optional[bool] = None
    ) -> List[Dict]:
        """
        Read and filter audit entries.

        Args:
            last_n: Return only last N entries
            event_type: Filter by event type
            mode: Filter by mode
            success: Filter by success status

        Returns:
            List of matching entries
        """
        entries = []

        with self._lock:
            if not self._log_file.exists():
                return entries

            with open(self._log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("type") == "audit_log_header":
                            continue

                        # Apply filters
                        if event_type and entry.get("event_type") != event_type:
                            continue
                        if mode and entry.get("mode") != mode:
                            continue
                        if success is not None and entry.get("success") != success:
                            continue

                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue

        if last_n:
            entries = entries[-last_n:]

        return entries

    def get_order_history(self, last_n: int = 100) -> List[Dict]:
        """Get recent order history."""
        return self.get_entries(
            last_n=last_n,
            event_type=None  # Get both attempts and executions
        )

    def get_violations(self, last_n: int = 50) -> List[Dict]:
        """Get recent violations."""
        return [
            e for e in self.get_entries(last_n=last_n * 2)
            if e.get("event_type") == "order_rejected"
        ][-last_n:]

    def get_stats(self) -> Dict:
        """Get audit log statistics."""
        entries = self.get_entries()

        stats = {
            "total_entries": len(entries),
            "orders_attempted": 0,
            "orders_executed": 0,
            "orders_rejected": 0,
            "kill_switch_activations": 0,
            "by_mode": {"paper": 0, "live": 0},
        }

        for entry in entries:
            event_type = entry.get("event_type")
            mode = entry.get("mode", "paper")

            if event_type == "order_attempt":
                stats["orders_attempted"] += 1
            elif event_type == "order_executed":
                stats["orders_executed"] += 1
            elif event_type == "order_rejected":
                stats["orders_rejected"] += 1
            elif event_type == "kill_switch_activated":
                stats["kill_switch_activations"] += 1

            if mode in stats["by_mode"]:
                stats["by_mode"][mode] += 1

        return stats

    @property
    def log_file_path(self) -> Path:
        """Get log file path."""
        return self._log_file


# =============================================================================
# Module-level singleton
# =============================================================================

_audit_log: Optional[AuditLog] = None


def get_audit_log() -> AuditLog:
    """Get or create global audit log."""
    global _audit_log
    if _audit_log is None:
        _audit_log = AuditLog()
    return _audit_log
