"""
End-to-end paper test (nightly).

This test is skipped by default. Enable only in CI/nightly when TWS/Gateway
is running and paper account is available.
"""

import pytest
from src.engine.frontend_adapter import EngineAdapter


@pytest.mark.skipif(
    True,  # Enable in nightly CI with TWS/Gateway running
    reason="Requires live TWS/Gateway paper session"
)
def test_e2e_paper_connect():
    adapter = EngineAdapter()
    ok, err, _info = adapter.connect(host="127.0.0.1", port=7497, client_id=777, mode="paper", timeout=10)
    assert ok, err
    adapter.disconnect()
