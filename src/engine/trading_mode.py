"""
Trading Mode Management
=======================
Triple validation system for Live Trading access.

SECURITY: Live trading requires ALL THREE validations:
1. Environment variable TRADING_MODE=live
2. Secret token LIVE_TRADING_TOKEN
3. File .live_trading_enabled exists

This prevents accidental live trading activation.
"""

import os
import hashlib
import getpass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


class TradingMode(Enum):
    """Trading modes available."""
    PAPER = "paper"
    LIVE = "live"


@dataclass
class ModeValidationResult:
    """Result of mode validation."""
    valid: bool
    mode: TradingMode
    errors: list
    warnings: list


class TradingModeManager:
    """
    Manages trading mode with triple validation for live trading.

    CRITICAL: Live mode requires:
    1. TRADING_MODE=live environment variable
    2. LIVE_TRADING_TOKEN with correct secret
    3. .live_trading_enabled file in project root

    All three must be present to enable live trading.
    """

    # File that must exist to enable live trading
    LIVE_ENABLE_FILE = ".live_trading_enabled"

    # Environment variables
    ENV_MODE = "TRADING_MODE"
    ENV_TOKEN = "LIVE_TRADING_TOKEN"

    # Token validation
    TOKEN_MIN_LENGTH = 32

    def __init__(self, project_root: Optional[Path] = None):
        self._project_root = project_root or Path(__file__).parent.parent.parent
        self._mode: TradingMode = TradingMode.PAPER
        self._validated = False
        self._validation_errors: list = []
        self._live_confirmed = False

        # Logging
        self._log_entries: list = []

    def _log(self, message: str, level: str = "INFO"):
        """Internal logging."""
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}] [{level}] {message}"
        self._log_entries.append(entry)
        print(f"[MODE] {message}")

    @property
    def mode(self) -> TradingMode:
        """Get current trading mode."""
        return self._mode

    @property
    def is_live(self) -> bool:
        """Check if in live mode."""
        return self._mode == TradingMode.LIVE and self._validated and self._live_confirmed

    @property
    def is_paper(self) -> bool:
        """Check if in paper mode."""
        return self._mode == TradingMode.PAPER

    def get_port(self, platform: str = "TWS") -> int:
        """Get port based on mode and platform."""
        ports = {
            (TradingMode.PAPER, "TWS"): 7497,
            (TradingMode.PAPER, "Gateway"): 4002,
            (TradingMode.LIVE, "TWS"): 7496,
            (TradingMode.LIVE, "Gateway"): 4001,
        }
        return ports.get((self._mode, platform), 7497)

    # =========================================================================
    # Triple Validation System
    # =========================================================================

    def validate_mode(self, requested_mode: TradingMode = TradingMode.PAPER) -> ModeValidationResult:
        """
        Validate and set trading mode.

        For PAPER mode: Always succeeds.
        For LIVE mode: Requires triple validation.

        Returns ModeValidationResult with details.
        """
        errors = []
        warnings = []

        if requested_mode == TradingMode.PAPER:
            self._mode = TradingMode.PAPER
            self._validated = True
            self._log("Paper trading mode activated")
            return ModeValidationResult(
                valid=True,
                mode=TradingMode.PAPER,
                errors=[],
                warnings=[]
            )

        # Live mode requires triple validation
        self._log("Attempting live mode activation - starting triple validation")

        # Validation 1: Environment variable
        env_mode = os.environ.get(self.ENV_MODE, "").lower()
        if env_mode != "live":
            errors.append(
                f"Environment variable {self.ENV_MODE} must be 'live' (got: '{env_mode}')"
            )
            self._log(f"FAIL: Environment variable check failed", "ERROR")
        else:
            self._log("PASS: Environment variable check")

        # Validation 2: Secret token
        token = os.environ.get(self.ENV_TOKEN, "")
        if not token:
            errors.append(f"Environment variable {self.ENV_TOKEN} not set")
            self._log("FAIL: Token not found", "ERROR")
        elif len(token) < self.TOKEN_MIN_LENGTH:
            errors.append(
                f"Token too short (min {self.TOKEN_MIN_LENGTH} chars)"
            )
            self._log("FAIL: Token too short", "ERROR")
        else:
            # Verify token matches stored hash (if exists)
            if self._verify_token(token):
                self._log("PASS: Token verification")
            else:
                errors.append("Token verification failed")
                self._log("FAIL: Token verification failed", "ERROR")

        # Validation 3: Enable file exists
        enable_file = self._project_root / self.LIVE_ENABLE_FILE
        if not enable_file.exists():
            errors.append(
                f"File '{self.LIVE_ENABLE_FILE}' not found. "
                "Run scripts/enable_live_trading.py to create it."
            )
            self._log(f"FAIL: Enable file not found at {enable_file}", "ERROR")
        else:
            # Verify file content
            content = enable_file.read_text().strip()
            if "LIVE_TRADING_ENABLED" not in content:
                errors.append("Enable file has invalid content")
                self._log("FAIL: Enable file content invalid", "ERROR")
            else:
                self._log("PASS: Enable file check")

        # Result
        if errors:
            self._mode = TradingMode.PAPER
            self._validated = False
            warnings.append("Falling back to PAPER mode due to validation errors")
            self._log("Live mode validation FAILED - using Paper mode", "WARNING")
            return ModeValidationResult(
                valid=False,
                mode=TradingMode.PAPER,
                errors=errors,
                warnings=warnings
            )

        # All validations passed - but still need double confirmation
        self._mode = TradingMode.LIVE
        self._validated = True
        self._live_confirmed = False  # Still needs confirmation
        self._log("Live mode validation PASSED - awaiting confirmation")

        return ModeValidationResult(
            valid=True,
            mode=TradingMode.LIVE,
            errors=[],
            warnings=["Live mode validated. Call require_double_confirmation() to activate."]
        )

    def _verify_token(self, token: str) -> bool:
        """Verify token against stored hash."""
        token_file = self._project_root / ".live_token_hash"
        if not token_file.exists():
            # No stored hash - accept any valid-length token
            return len(token) >= self.TOKEN_MIN_LENGTH

        stored_hash = token_file.read_text().strip()
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return token_hash == stored_hash

    def require_double_confirmation(self, auto_confirm: bool = False) -> bool:
        """
        Require user to type 'CONFIRM LIVE' to activate live trading.

        This is the final safety check before live trading is enabled.

        Args:
            auto_confirm: If True, skip interactive confirmation (for testing only)

        Returns:
            True if confirmed, False otherwise
        """
        if self._mode != TradingMode.LIVE or not self._validated:
            self._log("Cannot confirm - mode not validated for live", "ERROR")
            return False

        if self._live_confirmed:
            return True

        if auto_confirm:
            self._log("WARNING: Auto-confirm used - for testing only!", "WARNING")
            self._live_confirmed = True
            return True

        # Interactive confirmation
        print("\n" + "=" * 60)
        print("⚠️  LIVE TRADING CONFIRMATION REQUIRED  ⚠️")
        print("=" * 60)
        print("\nYou are about to enable LIVE TRADING.")
        print("This will use REAL MONEY from your account.")
        print("\nTo confirm, type exactly: CONFIRM LIVE")
        print("Or press Enter to cancel.\n")

        try:
            response = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            response = ""

        if response == "CONFIRM LIVE":
            self._live_confirmed = True
            self._log("LIVE TRADING CONFIRMED BY USER", "CRITICAL")
            print("\n✅ Live trading enabled. Trade carefully!")
            return True

        self._mode = TradingMode.PAPER
        self._validated = False
        self._log("Live trading confirmation REJECTED - using Paper mode", "WARNING")
        print("\n❌ Live trading NOT enabled. Using Paper mode.")
        return False

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_status(self) -> dict:
        """Get current mode status."""
        return {
            "mode": self._mode.value,
            "is_live": self.is_live,
            "is_paper": self.is_paper,
            "validated": self._validated,
            "live_confirmed": self._live_confirmed,
            "validation_errors": self._validation_errors,
        }

    def force_paper_mode(self):
        """Force switch to paper mode (emergency use)."""
        self._mode = TradingMode.PAPER
        self._validated = True
        self._live_confirmed = False
        self._log("FORCED switch to Paper mode", "WARNING")

    def get_log(self) -> list:
        """Get mode manager logs."""
        return list(self._log_entries)


# =============================================================================
# Module-level helpers
# =============================================================================

_mode_manager: Optional[TradingModeManager] = None


def get_mode_manager() -> TradingModeManager:
    """Get or create global mode manager."""
    global _mode_manager
    if _mode_manager is None:
        _mode_manager = TradingModeManager()
    return _mode_manager


def is_live_mode() -> bool:
    """Quick check if in live mode."""
    return get_mode_manager().is_live


def is_paper_mode() -> bool:
    """Quick check if in paper mode."""
    return get_mode_manager().is_paper


def require_paper_mode():
    """Decorator/assertion that requires paper mode."""
    if is_live_mode():
        raise RuntimeError("This operation is not allowed in live mode")
