# 🚀 Setup Rápido - IB Trading Project

## Instrucciones de Instalación

### 1️⃣ Preparar el proyecto local

```bash
# Navegar a tu carpeta de proyectos
cd ~/Documents  # o donde tengas IBKR-PROYECT

# Asegúrate de estar en la carpeta del proyecto
cd IBKR-PROYECT
```

### 2️⃣ Copiar los archivos descargados

Los archivos que descargaste tienen nombres específicos para evitar conflictos. 
Debes renombrarlos así:

```bash
# En la raíz del proyecto:
requirements.txt → requirements.txt
.gitignore → .gitignore
CLAUDE.md → CLAUDE.md
main.py → main.py
README.md → README.md
__init__.py → (crear en cada carpeta de módulo)

# Crear estructura de carpetas
mkdir -p config src/connection src/data src/strategies src/execution src/utils tests logs data

# Archivos de config/
config_settings.py → config/settings.py
config_credentials.py.example → config/credentials.py.example
__init__.py → config/__init__.py

# Archivos de src/utils/
src_utils_logger.py → src/utils/logger.py
__init__.py → src/utils/__init__.py

# Archivos de src/connection/
src_connection_ib_client.py → src/connection/ib_client.py
__init__.py → src/connection/__init__.py

# Archivos de tests/
test_connection.py → tests/test_connection.py
__init__.py → tests/__init__.py

# Crear __init__.py en carpetas restantes
touch src/__init__.py
touch src/data/__init__.py
touch src/strategies/__init__.py
touch src/execution/__init__.py
```

### 3️⃣ Instalar dependencias

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate  # macOS/Linux

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

**NOTA sobre TA-Lib**: Si `ta-lib` falla al instalar (requiere compilación C):

```bash
# macOS
brew install ta-lib
pip install ta-lib

# O usar alternativa sin dependencias C:
pip install pandas-ta
```

### 4️⃣ Configurar credenciales

```bash
# Copiar plantilla de credenciales
cp config/credentials.py.example config/credentials.py

# Editar con tus datos (usa VS Code, nano, vim...)
code config/credentials.py

# Contenido a modificar:
# IB_ACCOUNT_ID = "DU1234567"  # Tu account ID de IB Paper
# CLIENT_ID = 1
# ACCOUNT_TYPE = "PAPER"
```

### 5️⃣ Configurar TWS/Gateway

1. **Abrir TWS o IB Gateway**
2. **Habilitar API**:
   - File → Global Configuration → API → Settings
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Allow connections from localhost only
   - ✅ Read-Only API (para empezar)
   
3. **Verificar puerto**:
   - TWS Paper Trading: **7497**
   - Gateway Paper Trading: **4002**

### 6️⃣ Probar la conexión

```bash
# Asegúrate de que TWS/Gateway esté corriendo
python main.py
```

**Output esperado:**
```
✓ Conexión exitosa con IB
--- Resumen de Cuenta ---
NetLiquidation: 1000000.00 USD
TotalCashValue: 1000000.00 USD
...
=== Test completado exitosamente ===
```

### 7️⃣ Ejecutar tests

```bash
# Tests que no requieren conexión
pytest tests/test_connection.py -v

# Para ejecutar tests que SÍ requieren TWS activo:
# Editar tests/test_connection.py y cambiar skipif(True) a skipif(False)
```

---

## 🤖 Usar con Claude Code

### Primera vez

```bash
# Desde la raíz del proyecto
cd IBKR-PROYECT

# Iniciar Claude Code
claude
```

Claude Code leerá automáticamente `CLAUDE.md` y entenderá:
- La estructura del proyecto
- Reglas de estilo de código
- Configuración de IB
- Mejores prácticas

### Prompts útiles para empezar

```
"Lee el proyecto y explícame la arquitectura actual"

"Añade logging a todas las funciones que falten"

"Crea un módulo data/fetcher.py para obtener datos históricos de IB"

"Implementa rate limiting para respetar los límites de IB"

"Crea una estrategia simple de cruce de medias móviles"
```

---

## 🔍 Verificación del Setup

### Checklist

- [ ] Entorno virtual activado (`venv/`)
- [ ] Dependencias instaladas (`pip list`)
- [ ] Carpetas creadas (`config/`, `src/`, `tests/`, `logs/`)
- [ ] Archivos copiados y renombrados correctamente
- [ ] `config/credentials.py` configurado
- [ ] TWS/Gateway corriendo
- [ ] API habilitada en TWS
- [ ] `python main.py` funciona sin errores
- [ ] Logs generados en `logs/`

### Solución de problemas comunes

**"ModuleNotFoundError: No module named 'ib_insync'"**
→ Entorno virtual no activado o dependencias no instaladas
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**"ConnectionError: Timeout al conectar"**
→ TWS/Gateway no está corriendo o puerto incorrecto
```bash
# Verificar que TWS esté abierto
# Verificar puerto en config/settings.py (debe ser 7497 para TWS Paper)
```

**"ImportError: cannot import name 'logger'"**
→ Estructura de carpetas incorrecta o faltan `__init__.py`
```bash
# Verificar que existan todos los __init__.py
find . -name "__init__.py"
```

---

## 📂 Estructura Final Verificada

```
IBKR-PROYECT/
├── config/
│   ├── __init__.py ✅
│   ├── settings.py ✅
│   ├── credentials.py ✅
│   └── credentials.py.example ✅
├── src/
│   ├── __init__.py ✅
│   ├── connection/
│   │   ├── __init__.py ✅
│   │   └── ib_client.py ✅
│   ├── data/
│   │   └── __init__.py ✅
│   ├── strategies/
│   │   └── __init__.py ✅
│   ├── execution/
│   │   └── __init__.py ✅
│   └── utils/
│       ├── __init__.py ✅
│       └── logger.py ✅
├── tests/
│   ├── __init__.py ✅
│   └── test_connection.py ✅
├── logs/ (se crea automáticamente)
├── data/ (se crea automáticamente)
├── venv/
├── main.py ✅
├── requirements.txt ✅
├── CLAUDE.md ✅
├── README.md ✅
└── .gitignore ✅
```

---

## ✅ Todo listo

Si llegaste hasta aquí y todo funciona, ya tienes:

1. ✅ Proyecto estructurado profesionalmente
2. ✅ Conexión con IB funcionando
3. ✅ Logging configurado
4. ✅ Tests básicos
5. ✅ Preparado para Claude Code

**Próximo paso**: Pedir a Claude Code que añada el módulo de data fetching o la primera estrategia.

```bash
claude
# "Ahora que tengo la base, crea el módulo data/fetcher.py para obtener 
#  datos históricos de IB respetando rate limits"
```

🎉 **¡A programar!**
