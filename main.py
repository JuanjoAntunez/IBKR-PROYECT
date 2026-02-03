"""
Script principal del proyecto IB Trading.

Este es un ejemplo básico que demuestra cómo usar el IBClient.
"""

import asyncio
from ib_insync import Stock
from src.connection.ib_client import IBClient
from src.utils.logger import logger


async def main():
    """
    Función principal del programa.
    
    Demuestra:
    1. Conexión con IB
    2. Obtener información de cuenta
    3. Validar un contrato (símbolo)
    4. Desconexión limpia
    """
    logger.info("=== Iniciando IB Trading System ===")
    
    # Crear cliente (Paper Trading por defecto)
    client = IBClient(
        host="127.0.0.1",
        port=7497,  # TWS Paper Trading
        client_id=1,
        readonly=True,  # Solo lectura para seguridad
    )
    
    try:
        # Conectar
        await client.connect(timeout=10)
        
        # Verificar conexión
        if not client.is_connected():
            logger.error("No se pudo establecer conexión")
            return
        
        logger.info("✓ Conexión exitosa con IB")
        
        # 1. Obtener resumen de cuenta
        logger.info("\n--- Resumen de Cuenta ---")
        account_summary = await client.get_account_summary()
        
        # Mostrar algunos valores clave
        key_metrics = ["NetLiquidation", "TotalCashValue", "BuyingPower"]
        for metric in key_metrics:
            values = [av for av in account_summary if av["tag"] == metric]
            if values:
                logger.info(f"{metric}: {values[0]['value']} {values[0]['currency']}")
        
        # 2. Obtener posiciones actuales
        logger.info("\n--- Posiciones Actuales ---")
        positions = await client.get_positions()
        
        if positions:
            for pos in positions:
                logger.info(
                    f"{pos['contract'].symbol}: {pos['position']} @ "
                    f"{pos['avgCost']:.2f}"
                )
        else:
            logger.info("No hay posiciones abiertas")
        
        # 3. Validar un contrato de ejemplo (AAPL)
        logger.info("\n--- Validando Contrato ---")
        apple_stock = Stock("AAPL", "SMART", "USD")
        
        qualified = await client.qualify_contract(apple_stock)
        if qualified:
            logger.info(f"✓ Contrato validado: {qualified.symbol} - {qualified.primaryExchange}")
        else:
            logger.warning("No se pudo validar el contrato AAPL")
        
        logger.info("\n=== Test completado exitosamente ===")
        
    except ConnectionError as e:
        logger.error(f"Error de conexión: {e}")
        logger.info("Asegúrate de que TWS/Gateway esté corriendo:")
        logger.info("  - TWS Paper: Puerto 7497")
        logger.info("  - Gateway Paper: Puerto 4002")
        logger.info("  - Habilitar API en: File → Global Configuration → API")
        
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        
    finally:
        # Desconectar siempre
        await client.disconnect()
        logger.info("=== Programa finalizado ===")


if __name__ == "__main__":
    # Ejecutar el programa
    asyncio.run(main())
