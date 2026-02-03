"""
Tests de conexión con Interactive Brokers.

Estos tests requieren que TWS/Gateway esté activo.
"""

import pytest
from src.connection.ib_client import IBClient


@pytest.mark.asyncio
class TestIBConnection:
    """Tests del cliente de IB."""
    
    async def test_client_initialization(self):
        """Test que el cliente se inicializa correctamente."""
        client = IBClient(
            host="127.0.0.1",
            port=7497,
            client_id=999,  # ID único para tests
            readonly=True,
        )
        
        assert client.host == "127.0.0.1"
        assert client.port == 7497
        assert client.client_id == 999
        assert client.readonly is True
        assert not client.is_connected()
    
    @pytest.mark.skipif(
        True,  # Cambiar a False cuando TWS esté corriendo
        reason="Requiere TWS/Gateway activo"
    )
    async def test_connection(self):
        """
        Test de conexión real con IB.
        
        NOTA: Este test solo funciona si TWS/Gateway está activo.
        Cambiar el decorator skipif a False para ejecutarlo.
        """
        client = IBClient()
        
        try:
            await client.connect(timeout=10)
            assert client.is_connected()
            
            # Test obtener datos de cuenta
            summary = await client.get_account_summary()
            assert isinstance(summary, list)
            assert len(summary) > 0
            
        finally:
            await client.disconnect()
            assert not client.is_connected()
    
    async def test_context_manager(self):
        """Test del context manager."""
        # Este test no conecta realmente, solo verifica la estructura
        client = IBClient()
        
        # Verificar que tiene los métodos necesarios
        assert hasattr(client, '__aenter__')
        assert hasattr(client, '__aexit__')


@pytest.mark.asyncio
async def test_client_disconnect_when_not_connected():
    """Test que disconnect no falla si no hay conexión."""
    client = IBClient()
    
    # No debe lanzar error
    await client.disconnect()
    assert not client.is_connected()
