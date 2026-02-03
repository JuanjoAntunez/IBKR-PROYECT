"""
Plantilla de credenciales para IB Trading.

IMPORTANTE: 
1. Copiar este archivo a 'credentials.py' 
2. Rellenar con tus datos reales
3. NUNCA subir credentials.py a Git (está en .gitignore)
"""

# Account ID de Interactive Brokers
# Ejemplo: "DU1234567" para paper trading
IB_ACCOUNT_ID = "YOUR_ACCOUNT_ID_HERE"

# Client ID (puede ser cualquier número único)
# Si tienes múltiples instancias corriendo, usa IDs diferentes
CLIENT_ID = 1

# API Token (si usas autenticación adicional)
# La mayoría de conexiones locales no lo necesitan
API_TOKEN = None

# Configuración de cuenta
ACCOUNT_TYPE = "PAPER"  # "PAPER" o "LIVE"
