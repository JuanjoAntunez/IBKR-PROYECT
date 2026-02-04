"""
Cliente de conexión con Interactive Brokers usando ib_insync.

Maneja la conexión, reconexión automática, y provee una interfaz
limpia para interactuar con la API de IB.
"""

import asyncio
from typing import Optional, List
from ib_insync import IB, Contract, util
from src.utils.logger import logger


class IBClient:
    """
    Cliente principal para interactuar con Interactive Brokers.

    Uso:
        client = IBClient()
        await client.connect()
        # ... usar el cliente
        await client.disconnect()

    O con context manager:
        async with IBClient() as client:
            # ... usar el cliente
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7496,
        client_id: int = 1,
        readonly: bool = True,
    ):
        """
        Inicializa el cliente IB.

        Args:
            host: Dirección del servidor TWS/Gateway
            port: Puerto de conexión (7497=TWS Paper, 4002=Gateway Paper)
            client_id: ID único del cliente
            readonly: Si True, solo lectura (más seguro para empezar)
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.readonly = readonly

        self.ib = IB()
        self._connected = False
        self._setup_event_handlers()

        logger.info(
            f"IBClient inicializado: {host}:{port} (client_id={client_id}, "
            f"readonly={readonly})"
        )

    def _setup_event_handlers(self):
        """Configura handlers para eventos de IB."""
        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error

    def _on_connected(self):
        """Callback cuando se establece conexión."""
        self._connected = True
        logger.info(f"✓ Conectado a IB en {self.host}:{self.port}")

    def _on_disconnected(self):
        """Callback cuando se pierde conexión."""
        self._connected = False
        logger.warning(f"✗ Desconectado de IB")

    def _on_error(self, reqId, errorCode, errorString, contract):
        """Callback para errores de IB."""
        # Filtrar warnings que no son errores reales
        if errorCode in [2104, 2106, 2158]:  # Market data warnings
            logger.debug(f"IB Info [{errorCode}]: {errorString}")
        elif errorCode >= 2000:  # Warnings generales
            logger.warning(f"IB Warning [{errorCode}]: {errorString}")
        else:  # Errores reales
            logger.error(
                f"IB Error [{errorCode}]: {errorString} "
                f"(reqId={reqId}, contract={contract})"
            )

    async def connect(self, timeout: int = 10) -> bool:
        """
        Conecta con TWS/Gateway.

        Args:
            timeout: Tiempo máximo de espera en segundos

        Returns:
            True si la conexión fue exitosa

        Raises:
            ConnectionError: Si no se puede conectar
        """
        if self._connected:
            logger.warning("Ya hay una conexión activa")
            return True

        try:
            logger.info(f"Intentando conectar a {self.host}:{self.port}...")

            await asyncio.wait_for(
                self.ib.connectAsync(
                    host=self.host,
                    port=self.port,
                    clientId=self.client_id,
                    readonly=self.readonly,
                ),
                timeout=timeout,
            )

            # Esperar a que se complete la conexión
            await asyncio.sleep(0.5)

            if not self.ib.isConnected():
                raise ConnectionError("No se pudo establecer conexión con IB")

            logger.info(f"✓ Conexión exitosa (Account: {self.ib.wrapper.accounts})")
            return True

        except asyncio.TimeoutError:
            logger.error(f"Timeout conectando a IB después de {timeout}s")
            raise ConnectionError(f"Timeout al conectar ({timeout}s)")
        except Exception as e:
            logger.error(f"Error conectando a IB: {e}")
            raise

    async def disconnect(self):
        """Desconecta de TWS/Gateway de forma segura."""
        if not self._connected:
            logger.debug("No hay conexión activa para desconectar")
            return

        logger.info("Desconectando de IB...")
        self.ib.disconnect()
        await asyncio.sleep(0.5)
        logger.info("✓ Desconectado correctamente")

    def is_connected(self) -> bool:
        """Verifica si hay conexión activa."""
        return self._connected and self.ib.isConnected()

    async def get_account_summary(self) -> List[dict]:
        """
        Obtiene resumen de la cuenta.

        Returns:
            Lista de diccionarios con valores de la cuenta
        """
        if not self.is_connected():
            raise ConnectionError("No hay conexión activa con IB")

        logger.info("Obteniendo resumen de cuenta...")

        account_values = self.ib.accountValues()

        summary = [
            {
                "account": av.account,
                "tag": av.tag,
                "value": av.value,
                "currency": av.currency,
            }
            for av in account_values
        ]

        logger.info(f"✓ Obtenidos {len(summary)} valores de cuenta")
        return summary

    async def get_positions(self) -> List[dict]:
        """
        Obtiene posiciones actuales.

        Returns:
            Lista de posiciones con detalles
        """
        if not self.is_connected():
            raise ConnectionError("No hay conexión activa con IB")

        logger.info("Obteniendo posiciones...")

        positions = self.ib.positions()

        result = [
            {
                "account": pos.account,
                "contract": pos.contract,
                "position": pos.position,
                "avgCost": pos.avgCost,
            }
            for pos in positions
        ]

        logger.info(f"✓ Obtenidas {len(result)} posiciones")
        return result

    async def qualify_contract(self, contract: Contract) -> Optional[Contract]:
        """
        Valida y completa los detalles de un contrato.

        Args:
            contract: Contrato a validar

        Returns:
            Contrato completo si es válido, None si no
        """
        if not self.is_connected():
            raise ConnectionError("No hay conexión activa con IB")

        try:
            qualified = await self.ib.qualifyContractsAsync(contract)
            if qualified:
                logger.debug(f"✓ Contrato validado: {qualified[0]}")
                return qualified[0]
            else:
                logger.warning(f"No se pudo validar el contrato: {contract}")
                return None
        except Exception as e:
            logger.error(f"Error validando contrato: {e}")
            return None

    # Context manager support
    async def __aenter__(self):
        """Soporte para 'async with'."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Limpieza al salir del context manager."""
        await self.disconnect()
