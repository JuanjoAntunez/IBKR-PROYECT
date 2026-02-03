# CLAUDE.md - Configuración del Proyecto IB Trading

## 📋 Contexto del Proyecto
Este es un proyecto de trading automatizado usando Interactive Brokers API.
- **Lenguaje**: Python 3.10+
- **API**: ib_insync (wrapper moderno de IBAPI)
- **Objetivo**: Sistema modular para conexión, obtención de datos, estrategias y ejecución

---

## 🎯 Principios de Código

### Estructura
- **Modular**: Cada componente (conexión, datos, estrategias, órdenes) en su propio módulo
- **Async-first**: Usar async/await cuando sea posible (ib_insync es async nativo)
- **Type hints**: Siempre usar anotaciones de tipos
- **Logging**: Usar loguru para todo el logging, nunca prints

### Estilo
- **PEP 8** estricto
- **Docstrings**: Google style para todas las funciones y clases
- **Nombres**: snake_case para funciones/variables, PascalCase para clases
- **Imports**: Ordenados (stdlib, third-party, local) y agrupados

### Testing
- **pytest** para todos los tests
- **Tests unitarios** para lógica de negocio
- **Tests de integración** para conexión con IB (requieren TWS/Gateway activo)

---

## 🔧 Configuración de IB

### Conexión por defecto
- **Host**: localhost (127.0.0.1)
- **Puerto TWS Paper**: 7497
- **Puerto TWS Live**: 7496
- **Puerto Gateway Paper**: 4002
- **Puerto Gateway Live**: 4001
- **Client ID**: 1 (por defecto, puede cambiar)

### Importante
- Siempre verificar que TWS/Gateway esté activo antes de ejecutar
- Habilitar API en TWS: File → Global Configuration → API → Settings
  - Enable ActiveX and Socket Clients ✓
  - Allow connections from localhost only ✓
  - Read-Only API ✓ (para testing inicial)

---

## 📁 Estructura de Archivos

### config/
- `settings.py`: Configuración global (puertos, timeouts, etc)
- `credentials.py`: ⚠️ NO SUBIR A GIT - Contiene account IDs, tokens

### src/connection/
- `ib_client.py`: Cliente principal de conexión con IB
  - Clase IBClient con métodos connect(), disconnect()
  - Manejo automático de reconexión
  - Logging de eventos de conexión

### src/data/
- `fetcher.py`: Obtención de datos históricos
  - Respetar rate limits de IB (60 requests por 10 minutos)
  - Cachear datos cuando sea posible
- `stream.py`: Streaming de datos en tiempo real
  - Usar reqMktData de ib_insync

### src/strategies/
- `base.py`: Clase abstracta BaseStrategy
  - Métodos: calculate_signals(), get_positions(), etc
- Cada estrategia concreta hereda de BaseStrategy

### src/execution/
- `order_manager.py`: Gestión de órdenes
  - Validación antes de enviar
  - Tracking de órdenes activas
  - Manejo de fills y cancelaciones

### src/utils/
- `logger.py`: Configuración de loguru
  - Rotar logs diariamente
  - Nivel DEBUG en desarrollo, INFO en producción

---

## 🚫 Qué NO hacer

❌ No usar `print()` → Usar `logger.info()`, `logger.debug()`, etc
❌ No hardcodear credenciales en el código
❌ No bloquear el event loop con operaciones síncronas pesadas
❌ No ignorar errores de IB (siempre loggear y manejar)
❌ No hacer requests masivos a IB sin rate limiting

---

## ✅ Qué SÍ hacer

✅ Usar context managers para conexiones (`async with ib.connect()`)
✅ Validar datos antes de procesarlos (checks de None, tipos, rangos)
✅ Documentar decisiones de diseño en docstrings
✅ Escribir tests para lógica crítica
✅ Usar enums para estados (OrderStatus, StrategyState, etc)

---

## 🔄 Workflow con Claude Code

### Para nuevas features:
1. Claude Code lee este CLAUDE.md primero
2. Analiza archivos existentes relevantes
3. Genera código siguiendo los principios
4. Ejecuta tests para validar
5. Muestra cambios antes de guardar

### Prompts útiles:
- "Añade logging a todas las funciones de ib_client.py"
- "Crea una estrategia de cruce de medias móviles que herede de BaseStrategy"
- "Implementa rate limiting en fetcher.py para respetar límites de IB"
- "Añade tests para verificar que la conexión con IB funciona"

---

## 📚 Referencias

- [ib_insync docs](https://ib-insync.readthedocs.io/)
- [IB API docs](https://interactivebrokers.github.io/tws-api/)
- [TWS API release notes](https://www.interactivebrokers.com/en/software/api/apiguide.htm)

---

## 🐛 Debugging común

### "Connection refused"
→ TWS/Gateway no está corriendo o puerto incorrecto

### "Error validating request: Pacing violation"
→ Demasiadas requests, implementar rate limiting

### "No security definition found"
→ Símbolo incorrecto o contrato mal especificado

---

## 📝 Notas adicionales

- Este proyecto usa **Paper Trading por defecto** para evitar riesgos
- Antes de pasar a Live, revisar TODO el código de ejecución
- Mantener logs detallados de todas las órdenes ejecutadas
