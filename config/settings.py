"""
Configuración centralizada del proyecto IB Trading.

Este módulo contiene todas las configuraciones no sensibles del proyecto.
Para credenciales y datos sensibles, usar config/credentials.py
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IBConfig:
    """Configuración de conexión a Interactive Brokers."""
    
    # Conexión
    host: str = "127.0.0.1"
    port: int = 7497  # TWS Paper Trading por defecto
    client_id: int = 1
    readonly: bool = True  # Modo solo lectura para seguridad inicial
    
    # Timeouts (en segundos)
    connect_timeout: int = 10
    request_timeout: int = 30
    
    # Rate limiting (IB permite ~60 requests cada 10 minutos)
    max_requests_per_period: int = 50  # Margen de seguridad
    rate_limit_period: int = 600  # 10 minutos en segundos
    
    # Reconexión automática
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_delay: int = 5  # segundos entre intentos


@dataclass
class DataConfig:
    """Configuración para obtención de datos."""
    
    # Datos históricos
    default_duration: str = "1 M"  # 1 mes por defecto
    default_bar_size: str = "1 day"  # Barras diarias por defecto
    
    # Cacheo
    cache_enabled: bool = True
    cache_dir: str = "data/cache"
    
    # Formatos
    use_rth: bool = True  # Regular Trading Hours only
    date_format: int = 2  # 2 = Unix timestamp


@dataclass
class LogConfig:
    """Configuración de logging."""
    
    # Directorios
    log_dir: str = "logs"
    
    # Niveles
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Rotación
    rotation: str = "1 day"
    retention: str = "30 days"
    
    # Formato
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )


@dataclass
class TradingConfig:
    """Configuración de trading y ejecución."""
    
    # Cuenta
    account: Optional[str] = None  # Se debe configurar en credentials.py
    
    # Órdenes
    default_order_type: str = "LMT"  # Limit orders por defecto
    use_adaptive_algo: bool = False
    
    # Gestión de riesgo
    max_position_size: float = 10000.0  # USD por posición
    max_portfolio_exposure: float = 100000.0  # USD total en el mercado
    
    # Slippage y comisiones (para backtesting)
    slippage_bps: float = 5.0  # 5 basis points
    commission_per_share: float = 0.005  # USD por acción


# Instancias globales
ib_config = IBConfig()
data_config = DataConfig()
log_config = LogConfig()
trading_config = TradingConfig()


# Puertos comunes de IB para referencia rápida
IB_PORTS = {
    "TWS_PAPER": 7497,
    "TWS_LIVE": 7496,
    "GATEWAY_PAPER": 4002,
    "GATEWAY_LIVE": 4001,
}
