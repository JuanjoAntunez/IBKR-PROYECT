"""
Script principal del proyecto IB Trading (single-writer).
"""

from src.utils.logger import logger
from src.engine.frontend_adapter import EngineAdapter


def main():
    """
    Funcion principal del programa.

    Demuestra:
    1. Conexion con IB via Trading Engine (single-writer)
    2. Obtener informacion de cuenta
    3. Obtener posiciones
    4. Desconexion limpia
    """
    logger.info("=== Iniciando IB Trading System ===")

    adapter = EngineAdapter()
    try:
        logger.info("Intentando conectar a IB (paper)...")
        ok, err, info = adapter.connect(
            host="127.0.0.1",
            port=7497,
            client_id=1,
            mode="paper",
            timeout=10,
        )

        if not ok:
            raise ConnectionRefusedError(err or "Connection failed")

        logger.info("Conexion exitosa con IB")

        # 1. Obtener resumen de cuenta
        logger.info("\n--- Resumen de Cuenta ---")
        acc_ok, acc_err, summary = adapter.get_account()
        if acc_ok and summary:
            key_metrics = {
                "NetLiquidation": summary.net_liquidation,
                "TotalCashValue": summary.total_cash,
                "BuyingPower": summary.buying_power,
            }
            for metric, value in key_metrics.items():
                logger.info(f"{metric}: {value} {summary.currency}")
        else:
            logger.warning(f"No se pudo obtener resumen: {acc_err}")

        # 2. Obtener posiciones actuales
        logger.info("\n--- Posiciones Actuales ---")
        pos_ok, pos_err, positions = adapter.get_positions()
        if pos_ok and positions:
            for symbol, pos in positions.items():
                logger.info(f"{symbol}: {pos.quantity} @ {pos.avg_cost:.2f}")
        else:
            logger.info("No hay posiciones abiertas")

        logger.info("\n=== Test completado exitosamente ===")

    except ConnectionRefusedError:
        logger.error("No se pudo conectar a IB")
        logger.info("\nAsegurate de que TWS/Gateway este corriendo:")
        logger.info("  1. Abre TWS o IB Gateway")
        logger.info("  2. Verifica el puerto: 7497 (TWS Paper)")
        logger.info("  3. Habilita API: File -> Global Configuration -> API -> Settings")

    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        import traceback

        logger.debug(traceback.format_exc())

    finally:
        adapter.disconnect()
        logger.info("=== Programa finalizado ===")


if __name__ == "__main__":
    main()
