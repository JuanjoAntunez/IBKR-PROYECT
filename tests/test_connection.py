"""
Tests de conexión (single-writer Trading Engine).

Los tests de integración con TWS/Gateway deben estar explícitamente habilitados.
"""

import pytest
from src.engine.frontend_adapter import EngineAdapter


def test_engine_adapter_initialization():
    """Test que el adapter se inicializa correctamente."""
    adapter = EngineAdapter()
    assert adapter is not None
    assert adapter.engine is not None
    assert adapter.is_connected() is False


@pytest.mark.skipif(
    True,  # Cambiar a False cuando TWS/Gateway esté corriendo
    reason="Requiere TWS/Gateway activo"
)
def test_engine_connect_paper():
    """Test de conexión real con IB (paper)."""
    adapter = EngineAdapter()
    ok, err, _info = adapter.connect(host="127.0.0.1", port=7497, client_id=999, mode="paper", timeout=10)
    assert ok, err
    adapter.disconnect()
