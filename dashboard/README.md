# IB Trading Dashboard

Dashboard web interactivo para visualizar datos de mercado de Interactive Brokers.

## Instalación

```bash
# Desde el directorio raíz del proyecto
pip install -r dashboard/requirements.txt
```

## Ejecución

```bash
# Desde el directorio raíz del proyecto
streamlit run dashboard/app.py

# O especificando el tema oscuro
streamlit run dashboard/app.py --theme.base="dark"
```

El dashboard estará disponible en: **http://localhost:8501**

## Requisitos previos

1. **TWS o IB Gateway** debe estar ejecutándose y logueado
2. **API habilitada** en TWS:
   - File → Global Configuration → API → Settings
   - ☑️ Enable ActiveX and Socket Clients
   - ☑️ Allow connections from localhost only

## Puertos de conexión

| Aplicación | Paper Trading | Live Trading |
|------------|---------------|--------------|
| TWS        | 7497          | 7496         |
| IB Gateway | 4002          | 4001         |

## Funcionalidades

- **Selector de símbolo**: Cualquier acción US (AAPL, MSFT, GOOGL, etc.)
- **Duración**: 1D, 5D, 1M, 3M, 6M, 1Y
- **Intervalos**: 1min, 5min, 15min, 1h, 1d
- **Gráfico de velas** con volumen
- **Métricas**: Precio actual, cambio %, máximo, mínimo
- **Test de conexión**: Verifica cuenta e info de balance
- **Descarga CSV**: Exporta datos históricos
