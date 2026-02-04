"""
Script principal del proyecto IB Trading - Con fix para macOS.
"""

import nest_asyncio

nest_asyncio.apply()

from ib_insync import IB, Stock, util
from src.utils.logger import logger


def main():
    """
    Funcion principal del programa.

    Demuestra:
    1. Conexion con IB
    2. Obtener informacion de cuenta
    3. Validar un contrato (simbolo)
    4. Desconexion limpia
    """
    logger.info("=== Iniciando IB Trading System ===")

    # Crear conexion
    ib = IB()

    try:
        # Conectar a TWS Paper Trading
        logger.info("Intentando conectar a IB...")
        ib.connect(
            host="127.0.0.1", port=7497, clientId=1, readonly=True  # ← Esta línea
        )

        logger.info("Conexion exitosa con IB")

        # 1. Obtener resumen de cuenta
        logger.info("\n--- Resumen de Cuenta ---")
        account_values = ib.accountValues()

        # Mostrar valores clave
        key_metrics = ["NetLiquidation", "TotalCashValue", "BuyingPower"]
        for metric in key_metrics:
            values = [av for av in account_values if av.tag == metric]
            if values:
                logger.info(f"{metric}: {values[0].value} {values[0].currency}")

        # 2. Obtener posiciones actuales
        logger.info("\n--- Posiciones Actuales ---")
        positions = ib.positions()

        if positions:
            for pos in positions:
                logger.info(
                    f"{pos.contract.symbol}: {pos.position} @ {pos.avgCost:.2f}"
                )
        else:
            logger.info("No hay posiciones abiertas")

        # 3. Validar un contrato de ejemplo (AAPL)
        logger.info("\n--- Validando Contrato ---")
        apple_stock = Stock("AAPL", "SMART", "USD")

        qualified = ib.qualifyContracts(apple_stock)
        if qualified:
            logger.info(
                f"Contrato validado: {qualified[0].symbol} - {qualified[0].primaryExchange}"
            )
        else:
            logger.warning("No se pudo validar el contrato AAPL")

        logger.info("\n=== Test completado exitosamente ===")

    except ConnectionRefusedError:
        logger.error("No se pudo conectar a IB")
        logger.info("\nAsegurate de que TWS/Gateway este corriendo:")
        logger.info("  1. Abre TWS o IB Gateway")
        logger.info("  2. Verifica el puerto: 7497 (TWS Paper)")
        logger.info(
            "  3. Habilita API: File -> Global Configuration -> API -> Settings"
        )

    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        import traceback

        logger.debug(traceback.format_exc())

    finally:
        # Desconectar siempre
        if ib.isConnected():
            ib.disconnect()
            logger.info("Desconectado de IB")

        logger.info("=== Programa finalizado ===")


if __name__ == "__main__":
    # Iniciar event loop
    util.startLoop()
    main()
