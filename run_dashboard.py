#!/usr/bin/env python3
"""
Script para ejecutar el dashboard web de trading.

Uso:
    python run_dashboard.py
    python run_dashboard.py --port 8080
    python run_dashboard.py --reload  # Para desarrollo
"""

import argparse
import uvicorn

from src.utils.logger import logger


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(description="IB Trading Dashboard")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host para el servidor (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Puerto para el servidor (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Habilitar auto-reload para desarrollo",
    )

    args = parser.parse_args()

    logger.info(f"Iniciando dashboard en http://{args.host}:{args.port}")
    logger.info("Presiona Ctrl+C para detener")

    uvicorn.run(
        "src.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
