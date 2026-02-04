# Este archivo hace que Python trate este directorio como un paquete

from src.web.app import create_app

__all__ = ["create_app"]
