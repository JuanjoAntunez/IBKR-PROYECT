# IB Trading Project

Sistema de trading automatizado usando Interactive Brokers API con Python.

## 🎯 Características

- ✅ Conexión robusta con TWS/Gateway
- ✅ Manejo automático de reconexión
- ✅ Logging completo con Loguru
- ✅ Arquitectura modular y escalable
- ✅ Type hints completos
- ✅ Async/await nativo
- ✅ Preparado para Claude Code
- ✅ Trading Engine single-writer con guardrails de seguridad

## 🧠 Arquitectura (Single Writer)

El sistema opera con un **Trading Engine** centralizado (`src/engine`) que es el único
responsable de conectar a IB, enviar órdenes y reconciliar estado. Los frontends
(Streamlit y FastAPI) solo envían comandos al engine.

Notas:
- El stack antiguo fue eliminado; todo el flujo pasa por `src/engine`.
- El ejemplo principal (`main.py`) ya usa el engine para evitar accesos directos a IB.

## 📋 Requisitos Previos

1. **Python 3.10+**
2. **Interactive Brokers TWS o Gateway**
   - [Descargar TWS](https://www.interactivebrokers.com/en/trading/tws.php)
   - [Descargar Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
3. **Cuenta de IB** (Paper Trading recomendado para empezar)

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/JuanjoAntunez/IBKR-PROYECT.git
cd IBKR-PROYECT
```

### 2. Crear entorno virtual

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar credenciales

```bash
# Copiar plantilla
cp config/credentials.py.example config/credentials.py

# Editar con tus datos
nano config/credentials.py  # o tu editor preferido
```

## ⚙️ Configuración de TWS/Gateway

### Habilitar API

1. Abrir TWS o Gateway
2. Ir a: **File → Global Configuration → API → Settings**
3. Activar:
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Allow connections from localhost only
   - ✅ Read-Only API (para empezar)
4. **Puerto TWS Paper**: 7497
5. **Puerto Gateway Paper**: 4002

### Verificar conexión

```bash
# Asegurarse de que TWS/Gateway esté corriendo
python main.py
```

Si ves `✓ Conexión exitosa con IB`, ¡todo funciona!

## 📁 Estructura del Proyecto

```
IBKR-PROJECT/
│
├── config/
│   ├── __init__.py
│   ├── settings.py              # Configuración general
│   └── credentials.py           # Credenciales (NO SUBIR A GIT)
│
├── src/
│   ├── __init__.py
│   ├── connection/
│   │   ├── __init__.py
│   │   └── ib_client.py         # Cliente principal de IB
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py           # Datos históricos
│   │   └── stream.py            # Datos en tiempo real
│   │
│   ├── strategies/
│   │   ├── __init__.py
│   │   └── base.py              # Clase base para estrategias
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   └── order_manager.py     # Gestión de órdenes
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Logging configurado
│
├── tests/
│   ├── __init__.py
│   ├── test_connection.py
│   └── test_data_fetcher.py
│
├── logs/                         # Logs de la aplicación
├── data/                         # Datos descargados/cache
│
├── main.py                       # Script principal
├── requirements.txt              # Dependencias
├── CLAUDE.md                     # Config para Claude Code
├── .gitignore
└── README.md
```

## 🤖 Trabajar con Claude Code

Este proyecto está optimizado para trabajar con **Claude Code**, el agente de coding de Anthropic.

### Instalación de Claude Code

```bash
# macOS/Linux con Homebrew
brew install anthropic/claude/claude

# O descarga directa
# https://claude.ai/download
```

### Uso básico

```bash
# Desde la raíz del proyecto
cd IBKR-PROYECT
claude

# Claude Code leerá automáticamente CLAUDE.md
# y entenderá la estructura del proyecto
```

### Prompts útiles para Claude Code

```
"Añade un módulo de data fetching para obtener datos históricos de SPY"

"Crea una estrategia de cruce de medias móviles que herede de BaseStrategy"

"Implementa rate limiting en el fetcher para respetar límites de IB"

"Añade tests para verificar la conexión con IB"

"Documenta todas las funciones de ib_client.py con docstrings Google style"
```

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Test con cobertura
pytest --cov=src

# Test específico
pytest tests/test_connection.py
```

## 📊 Uso Básico

### Conectar y obtener datos de cuenta

```python
import asyncio
from src.connection.ib_client import IBClient

async def ejemplo():
    async with IBClient() as client:
        # Resumen de cuenta
        summary = await client.get_account_summary()
        
        # Posiciones
        positions = await client.get_positions()
        
        print(f"Posiciones: {len(positions)}")

asyncio.run(ejemplo())
```

### Validar un símbolo

```python
from ib_insync import Stock

async def validar_simbolo():
    async with IBClient() as client:
        # Crear contrato
        aapl = Stock("AAPL", "SMART", "USD")
        
        # Validar
        qualified = await client.qualify_contract(aapl)
        
        if qualified:
            print(f"✓ {qualified.symbol} es válido")

asyncio.run(validar_simbolo())
```

## 🔒 Seguridad

- ✅ **Nunca** subas `config/credentials.py` a Git
- ✅ Usa Paper Trading para probar
- ✅ Modo `readonly=True` por defecto
- ✅ Valida todas las órdenes antes de enviar
- ✅ Mantén logs detallados

## 📚 Recursos

- [ib_insync Documentación](https://ib-insync.readthedocs.io/)
- [IB API Reference](https://interactivebrokers.github.io/tws-api/)
- [TWS API Guide](https://www.interactivebrokers.com/en/software/api/apiguide.htm)

## 🐛 Problemas Comunes

### "Connection refused"
→ TWS/Gateway no está corriendo o puerto incorrecto

### "Error validating request: Pacing violation"
→ Demasiadas requests a IB, implementar rate limiting

### "No security definition found"
→ Símbolo incorrecto o contrato mal especificado

### El logger no funciona
→ Verificar que existe el directorio `logs/`

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-feature`)
3. Commit tus cambios (`git commit -am 'Añade nueva feature'`)
4. Push a la rama (`git push origin feature/nueva-feature`)
5. Crea un Pull Request

## 📝 Licencia

Este proyecto es de uso personal/educativo.

## ⚠️ Disclaimer

Este software es para propósitos educativos. El trading implica riesgos financieros. No nos hacemos responsables de pérdidas incurridas usando este código.

---

**Hecho con ❤️ y Claude Code**
