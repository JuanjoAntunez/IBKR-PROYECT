"""
Engine Config Loader
====================
Loads paper/live config from config/*.yaml for connection and limits.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml
import importlib.util
import os


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_mode_config(mode: str) -> Dict[str, Any]:
    """Load config for paper/live mode."""
    root = Path(__file__).resolve().parents[2]
    config_dir = root / "config"

    mode_lower = (mode or "paper").lower()
    if mode_lower == "live":
        live_path = config_dir / "live.yaml"
        data = _read_yaml(live_path)
        if not data:
            # Fall back to template (best effort)
            data = _read_yaml(config_dir / "live.yaml.template")
        return data

    return _read_yaml(config_dir / "paper.yaml")


def _load_module_from_path(module_name: str, path: Path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def load_credentials(mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Load credentials from config.

    Order of precedence:
    - config/credentials_live.py (if mode == live)
    - config/credentials_paper.py (if mode == paper)
    - config/credentials.py (fallback)
    """
    root = Path(__file__).resolve().parents[2]
    config_dir = root / "config"
    mode_lower = (mode or "").lower()

    candidates = []
    if mode_lower == "live":
        candidates = [config_dir / "credentials_live.py", config_dir / "credentials.py"]
    elif mode_lower == "paper":
        candidates = [config_dir / "credentials_paper.py", config_dir / "credentials.py"]
    else:
        candidates = [config_dir / "credentials.py"]

    module = None
    for path in candidates:
        if path.exists():
            module = _load_module_from_path("ib_credentials", path)
            if module:
                break
    if module is None:
        return {}

    def _get(name, default=None):
        return getattr(module, name, os.getenv(name, default))

    return {
        "IB_ACCOUNT_ID": _get("IB_ACCOUNT_ID"),
        "CLIENT_ID": _get("CLIENT_ID"),
        "API_TOKEN": _get("API_TOKEN"),
        "ACCOUNT_TYPE": _get("ACCOUNT_TYPE"),
    }


def _to_int(value, fallback: int) -> int:
    try:
        if value is None:
            return fallback
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if text == "":
            return fallback
        return int(text)
    except Exception:
        return fallback


def resolve_connection_params(
    mode: str,
    host: str,
    port: int,
    client_id: int,
) -> Tuple[str, int, int, Dict[str, Any], Dict[str, Any]]:
    """
    Resolve connection params with precedence:
    config/*.yaml -> passed args -> env overrides.
    Returns (host, port, client_id, config, creds).
    """
    mode_lower = (mode or "paper").lower()
    config = load_mode_config(mode_lower)
    cfg_conn = (config or {}).get("connection", {}) if isinstance(config, dict) else {}
    creds = load_credentials(mode_lower)

    default_host = "127.0.0.1"
    default_client_id = 1
    default_port = 7496 if mode_lower == "live" else 7497

    # Base from config, fallback to passed args
    resolved_host = cfg_conn.get("host", host)
    resolved_port = _to_int(cfg_conn.get("port", port), port)
    resolved_client_id = _to_int(
        cfg_conn.get("client_id", creds.get("CLIENT_ID") or client_id),
        client_id,
    )

    # If user explicitly changed from defaults, prefer passed args
    if host != default_host:
        resolved_host = host
    if port != default_port:
        resolved_port = _to_int(port, resolved_port)
    if client_id != default_client_id:
        resolved_client_id = _to_int(client_id, resolved_client_id)

    # Mode-specific environment overrides (highest precedence)
    mode_key = "LIVE" if mode_lower == "live" else "PAPER"
    resolved_host = os.getenv(f"IB_{mode_key}_HOST", resolved_host)
    resolved_port = _to_int(os.getenv(f"IB_{mode_key}_PORT", resolved_port), resolved_port)
    resolved_client_id = _to_int(
        os.getenv(f"IB_{mode_key}_CLIENT_ID", resolved_client_id),
        resolved_client_id,
    )

    return resolved_host, resolved_port, resolved_client_id, config, creds
