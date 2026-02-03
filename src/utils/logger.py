"""
Configuración centralizada de logging usando loguru.

Uso:
    from src.utils.logger import logger
    logger.info("Mensaje informativo")
    logger.error("Mensaje de error")
"""

import sys
from pathlib import Path
from loguru import logger

# Importar configuración
try:
    from config.settings import log_config
except ImportError:
    # Si no existe config, usar defaults
    class DefaultLogConfig:
        log_dir = "logs"
        console_level = "INFO"
        file_level = "DEBUG"
        rotation = "1 day"
        retention = "30 days"
        format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    log_config = DefaultLogConfig()


def setup_logger():
    """
    Configura el logger global de la aplicación.
    
    - Console: nivel INFO, con colores
    - File: nivel DEBUG, rotación diaria
    """
    # Crear directorio de logs si no existe
    log_dir = Path(log_config.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Remover handler por defecto de loguru
    logger.remove()
    
    # Handler para consola (colorizado)
    logger.add(
        sys.stderr,
        format=log_config.format,
        level=log_config.console_level,
        colorize=True,
    )
    
    # Handler para archivo (todas las apps)
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format=log_config.format,
        level=log_config.file_level,
        rotation=log_config.rotation,
        retention=log_config.retention,
        compression="zip",
    )
    
    # Handler específico para eventos de trading
    logger.add(
        log_dir / "trading_{time:YYYY-MM-DD}.log",
        format=log_config.format,
        level="INFO",
        filter=lambda record: "trading" in record["extra"],
        rotation=log_config.rotation,
        retention="90 days",  # Guardar trades por más tiempo
        compression="zip",
    )
    
    # Handler para errores críticos
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        format=log_config.format,
        level="ERROR",
        rotation=log_config.rotation,
        retention="90 days",
        compression="zip",
    )
    
    logger.info("Logger configurado correctamente")


# Configurar automáticamente al importar
setup_logger()


# Context managers útiles para logging de trading
class log_trade:
    """Context manager para loggear operaciones de trading."""
    
    def __init__(self, operation: str):
        self.operation = operation
        self.trade_logger = logger.bind(trading=True)
    
    def __enter__(self):
        self.trade_logger.info(f"Iniciando: {self.operation}")
        return self.trade_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.trade_logger.error(
                f"Error en {self.operation}: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
        else:
            self.trade_logger.info(f"Completado: {self.operation}")
