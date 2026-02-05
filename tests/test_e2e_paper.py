"""
End-to-end paper test (nightly).

This test is skipped by default. Enable only in CI/nightly when TWS/Gateway
is running and paper account is available.
"""

import os
import pytest
from src.engine.frontend_adapter import EngineAdapter


RUN_E2E = os.getenv("IB_E2E_PAPER") == "1"

@pytest.mark.skipif(
    not RUN_E2E,
    reason="Set IB_E2E_PAPER=1 and ensure TWS/Gateway paper is running",
)
def test_e2e_paper_connect():
    adapter = EngineAdapter()
    ok, err, _info = adapter.connect(host="127.0.0.1", port=7497, client_id=777, mode="paper", timeout=10)
    assert ok, err
    adapter.disconnect()
