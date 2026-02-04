"""
Dashboard de Trading con Interactive Brokers
============================================
Dashboard web interactivo usando Streamlit para visualizar datos de mercado.

Ejecutar con: streamlit run dashboard/app.py

IMPORTANTE: Usa la misma lógica de conexión que test_ibapi.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
from datetime import datetime


# =============================================================================
# Configuración de página (DEBE ser lo primero)
# =============================================================================
st.set_page_config(
    page_title="IB Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .status-connected {
        color: #00c853;
        font-weight: bold;
        padding: 5px 10px;
        border-radius: 5px;
        background-color: rgba(0, 200, 83, 0.1);
    }
    .status-disconnected {
        color: #ff1744;
        font-weight: bold;
        padding: 5px 10px;
        border-radius: 5px;
        background-color: rgba(255, 23, 68, 0.1);
    }
    .debug-box {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 12px;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Clase IBApp - EXACTA de test_ibapi.py + Portfolio
# =============================================================================
class IBApp(EWrapper, EClient):
    """Aplicacion simple de IB API - MISMA que test_ibapi.py + Portfolio."""

    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.accounts = []
        self.net_liquidation = None
        self.account_info = {}
        self.historical_data = []
        self.data_end = False
        self.debug_messages = []
        # Portfolio
        self.portfolio_items = []
        self.portfolio_end = False
        self.account_values = {}
        self.account_update_end = False

    def _log(self, msg):
        """Log interno para debug."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.debug_messages.append(f"[{timestamp}] {msg}")
        print(f"[IB] {msg}")  # También a consola

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Manejo de errores - MISMA FIRMA que test_ibapi.py (sin type hints)."""
        if errorCode in [2104, 2106, 2158]:
            self._log(f"Info: {errorString}")
        elif errorCode == 162:
            self._log(f"Sin datos: {errorString}")
            self.data_end = True
        elif errorCode >= 2000:
            self._log(f"Warning [{errorCode}]: {errorString}")
        else:
            self._log(f"Error [{errorCode}]: {errorString}")

    def connectAck(self):
        """Confirmacion de conexion - MISMA que test_ibapi.py."""
        self._log("✓ Conexion establecida (connectAck)")
        self.connected = True

    def managedAccounts(self, accountsList):
        """Recibe la lista de cuentas - MISMA que test_ibapi.py."""
        self.accounts = accountsList.split(',')
        self._log(f"Cuentas disponibles: {self.accounts}")

    def accountSummary(self, reqId, account, tag, value, currency):
        """Recibe valores de resumen de cuenta."""
        if tag == "NetLiquidation":
            self.net_liquidation = f"{value} {currency}"
        self.account_info[tag] = {"value": value, "currency": currency, "account": account}
        self._log(f"Account {tag}: {value} {currency}")

    def accountSummaryEnd(self, reqId):
        """Fin del resumen de cuenta."""
        self._log("accountSummaryEnd recibido")
        self.account_update_end = True

    def historicalData(self, reqId, bar):
        """Recibe datos históricos."""
        self.historical_data.append({
            'Date': bar.date,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        """Fin de datos históricos."""
        self._log(f"historicalDataEnd: {len(self.historical_data)} barras recibidas")
        self.data_end = True

    # =========================================================================
    # Portfolio callbacks
    # =========================================================================
    def updatePortfolio(self, contract, position, marketPrice, marketValue,
                        averageCost, unrealizedPNL, realizedPNL, accountName):
        """Recibe actualizaciones del portfolio."""
        self.portfolio_items.append({
            'symbol': contract.symbol,
            'secType': contract.secType,
            'exchange': contract.exchange,
            'currency': contract.currency,
            'position': position,
            'marketPrice': marketPrice,
            'marketValue': marketValue,
            'averageCost': averageCost,
            'unrealizedPNL': unrealizedPNL,
            'realizedPNL': realizedPNL,
            'account': accountName
        })
        self._log(f"Portfolio: {contract.symbol} pos={position} mktVal={marketValue:.2f} unrealPNL={unrealizedPNL:.2f}")

    def updateAccountValue(self, key, val, currency, accountName):
        """Recibe valores de cuenta (más detallados que accountSummary)."""
        self.account_values[key] = {
            'value': val,
            'currency': currency,
            'account': accountName
        }

    def updateAccountTime(self, timeStamp):
        """Timestamp de la última actualización de cuenta."""
        self._log(f"Account update time: {timeStamp}")

    def accountDownloadEnd(self, accountName):
        """Fin de descarga de datos de cuenta."""
        self._log(f"accountDownloadEnd para {accountName}")
        self.portfolio_end = True


# =============================================================================
# Funciones de conexión - MISMA LÓGICA que test_ibapi.py
# =============================================================================
def connect_to_ib(host, port, client_id):
    """
    Conecta a IB usando EXACTAMENTE la misma lógica que test_ibapi.py.

    Returns:
        tuple: (app, error_message, debug_messages)
    """
    debug = []
    debug.append(f"Iniciando conexión a {host}:{port} con client_id={client_id}")

    # Crear aplicacion - IGUAL que test_ibapi.py
    app = IBApp()

    try:
        debug.append(f"Llamando app.connect({host}, {port}, {client_id})...")

        # Conectar - IGUAL que test_ibapi.py
        app.connect(host, port, client_id)

        debug.append("Conexión socket iniciada, creando thread...")

        # Iniciar thread de la API - IGUAL que test_ibapi.py
        api_thread = threading.Thread(target=app.run, daemon=True)
        api_thread.start()

        debug.append("Thread iniciado, esperando connectAck...")

        # Esperar a que se establezca la conexion - IGUAL que test_ibapi.py
        timeout = 10
        start_time = time.time()

        while not app.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        elapsed = time.time() - start_time
        debug.append(f"Espera terminada después de {elapsed:.1f}s, connected={app.connected}")

        if not app.connected:
            debug.append("✗ TIMEOUT: connectAck no recibido")
            debug.extend(app.debug_messages)
            return None, "Timeout: No se recibió confirmación de conexión", debug

        debug.append("✓ Conexión confirmada!")

        # Esperar a recibir cuentas - IGUAL que test_ibapi.py
        time.sleep(2)
        debug.append(f"Cuentas recibidas: {app.accounts}")
        debug.extend(app.debug_messages)

        return app, None, debug

    except Exception as e:
        debug.append(f"✗ EXCEPCIÓN: {type(e).__name__}: {e}")
        debug.extend(app.debug_messages)
        return None, str(e), debug


def get_account_summary(app, timeout=5):
    """Solicita resumen de cuenta."""
    if app.accounts:
        app.reqAccountSummary(1, "All", "NetLiquidation,TotalCashValue,AvailableFunds")
        time.sleep(timeout)
    return app.account_info


def fetch_portfolio(host, port, client_id):
    """
    Obtiene el portfolio completo y resumen de cuenta.

    Returns:
        tuple: (portfolio_data, account_data, error_message, debug_messages)
    """
    debug = []
    debug.append(f"Solicitando portfolio de {host}:{port}")

    # Conectar
    app, error, conn_debug = connect_to_ib(host, port, client_id)
    debug.extend(conn_debug)

    if error:
        return None, None, error, debug

    try:
        # Limpiar datos anteriores
        app.portfolio_items = []
        app.account_values = {}
        app.account_info = {}
        app.portfolio_end = False
        app.account_update_end = False

        # Solicitar Account Summary con todos los tags importantes
        account_tags = (
            "NetLiquidation,TotalCashValue,SettledCash,"
            "AccruedCash,BuyingPower,EquityWithLoanValue,"
            "GrossPositionValue,RegTEquity,RegTMargin,"
            "InitMarginReq,MaintMarginReq,AvailableFunds,"
            "ExcessLiquidity,Cushion,FullInitMarginReq,"
            "FullMaintMarginReq,FullAvailableFunds,FullExcessLiquidity,"
            "LookAheadNextChange,LookAheadInitMarginReq,LookAheadMaintMarginReq,"
            "LookAheadAvailableFunds,LookAheadExcessLiquidity,"
            "HighestSeverity,DayTradesRemaining,Leverage,"
            "RealizedPnL,UnrealizedPnL"
        )

        debug.append("Solicitando reqAccountSummary...")
        app.reqAccountSummary(1, "All", account_tags)

        # Solicitar portfolio (posiciones)
        if app.accounts:
            account = app.accounts[0].strip()
            debug.append(f"Solicitando reqAccountUpdates para {account}...")
            app.reqAccountUpdates(True, account)

        # Esperar datos
        timeout = 15
        start_time = time.time()
        while not app.portfolio_end and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        # Dar tiempo extra para account summary
        time.sleep(2)

        elapsed = time.time() - start_time
        debug.append(f"Espera terminada después de {elapsed:.1f}s")
        debug.append(f"Portfolio items: {len(app.portfolio_items)}")
        debug.append(f"Account values: {len(app.account_values)}")
        debug.append(f"Account info: {len(app.account_info)}")
        debug.extend(app.debug_messages)

        # Cancelar suscripciones
        app.reqAccountUpdates(False, "")
        app.cancelAccountSummary(1)

        # Preparar datos
        portfolio_data = app.portfolio_items if app.portfolio_items else []

        # Combinar account_info y account_values
        account_data = {**app.account_info, **app.account_values}

        return portfolio_data, account_data, None, debug

    except Exception as e:
        debug.append(f"✗ Error: {e}")
        return None, None, str(e), debug

    finally:
        if app and app.isConnected():
            app.disconnect()
            debug.append("Desconectado")


def create_contract(symbol):
    """Crea un contrato de acciones."""
    contract = Contract()
    contract.symbol = symbol.upper()
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def get_duration_string(duration):
    """Convierte duración legible a formato IB."""
    duration_map = {
        "1D": "1 D",
        "5D": "5 D",
        "1M": "1 M",
        "3M": "3 M",
        "6M": "6 M",
        "1Y": "1 Y"
    }
    return duration_map.get(duration, "1 M")


def get_bar_size(interval):
    """Convierte intervalo legible a formato IB."""
    bar_map = {
        "1min": "1 min",
        "5min": "5 mins",
        "15min": "15 mins",
        "1h": "1 hour",
        "1d": "1 day"
    }
    return bar_map.get(interval, "1 day")


def fetch_historical_data(host, port, client_id, symbol, duration, bar_size):
    """
    Obtiene datos históricos usando la misma lógica de conexión.

    Returns:
        tuple: (dataframe, error_message, debug_messages)
    """
    debug = []
    debug.append(f"Solicitando datos para {symbol}, duración={duration}, barras={bar_size}")

    # Conectar
    app, error, conn_debug = connect_to_ib(host, port, client_id)
    debug.extend(conn_debug)

    if error:
        return None, error, debug

    try:
        # Crear contrato
        contract = create_contract(symbol)
        debug.append(f"Contrato creado: {symbol} STK SMART USD")

        # Limpiar datos anteriores
        app.historical_data = []
        app.data_end = False

        # Solicitar datos
        end_datetime = ""  # Cadena vacía = ahora

        debug.append(f"Solicitando reqHistoricalData...")
        app.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime=end_datetime,
            durationStr=get_duration_string(duration),
            barSizeSetting=get_bar_size(bar_size),
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

        # Esperar datos
        timeout = 30
        start_time = time.time()
        while not app.data_end and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        elapsed = time.time() - start_time
        debug.append(f"Espera de datos terminada después de {elapsed:.1f}s")
        debug.extend(app.debug_messages)

        if not app.historical_data:
            return None, "No se recibieron datos históricos", debug

        # Crear DataFrame
        df = pd.DataFrame(app.historical_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        debug.append(f"✓ Datos procesados: {len(df)} barras")

        return df, None, debug

    except Exception as e:
        debug.append(f"✗ Error: {e}")
        return None, str(e), debug

    finally:
        if app and app.isConnected():
            app.disconnect()
            debug.append("Desconectado")


def create_candlestick_chart(df, symbol):
    """Crea un gráfico de velas con volumen."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} - Precio', 'Volumen'),
        row_heights=[0.7, 0.3]
    )

    colors = ['#ff1744' if row['Close'] < row['Open'] else '#00c853'
              for _, row in df.iterrows()]

    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            increasing_line_color='#00c853',
            decreasing_line_color='#ff1744',
            increasing_fillcolor='#00c853',
            decreasing_fillcolor='#ff1744'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volumen',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )

    fig.update_layout(
        template='plotly_dark',
        showlegend=False,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

    return fig


# =============================================================================
# Interfaz del Dashboard
# =============================================================================
def main():
    """Función principal del dashboard."""

    # Inicializar estado de sesión
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = False
    if 'connection_info' not in st.session_state:
        st.session_state.connection_info = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'last_symbol' not in st.session_state:
        st.session_state.last_symbol = None
    if 'debug_log' not in st.session_state:
        st.session_state.debug_log = []

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.title("📊 IB Dashboard")
        st.markdown("---")

        # Configuración de conexión
        st.subheader("⚙️ Conexión")

        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host", value="127.0.0.1", key="host")
        with col2:
            # Puerto 7496 por defecto (como test_ibapi.py)
            port = st.number_input(
                "Puerto",
                value=7496,
                min_value=1,
                max_value=65535,
                help="7496=Live TWS (default), 7497=Paper TWS"
            )

        # Client ID 1 por defecto (como test_ibapi.py)
        client_id = st.number_input("Client ID", value=1, min_value=1, max_value=999)

        st.markdown("---")

        # Selector de símbolo
        st.subheader("📈 Datos de Mercado")
        symbol = st.text_input(
            "Símbolo",
            value="AAPL",
            help="Símbolo de la acción (ej: AAPL, MSFT, GOOGL)"
        ).upper()

        # Selector de duración
        duration = st.selectbox(
            "Duración",
            options=["1D", "5D", "1M", "3M", "6M", "1Y"],
            index=2,
            help="Período de tiempo"
        )

        # Selector de intervalo
        interval = st.selectbox(
            "Intervalo",
            options=["1min", "5min", "15min", "1h", "1d"],
            index=4,
            help="Tamaño de cada barra"
        )

        st.markdown("---")

        # Botones de acción
        col1, col2 = st.columns(2)

        with col1:
            connect_btn = st.button(
                "🔌 Test Conexión",
                use_container_width=True,
                type="secondary"
            )

        with col2:
            fetch_btn = st.button(
                "📥 Obtener Datos",
                use_container_width=True,
                type="primary"
            )

        # Estado de conexión en sidebar
        st.markdown("---")
        if st.session_state.connection_status:
            st.success("✓ Última conexión exitosa")
        else:
            st.warning("⚠️ Sin conexión verificada")

        # Info
        st.markdown("---")
        st.caption(f"**Config:** {host}:{port} (ID:{client_id})")
        st.caption("💡 Asegúrate de que TWS esté abierto")

    # =========================================================================
    # ÁREA PRINCIPAL
    # =========================================================================
    st.title(f"📊 Dashboard de Trading - {symbol}")

    # Estado de conexión
    if st.session_state.connection_status:
        st.markdown('<p class="status-connected">✓ Conectado a Interactive Brokers</p>',
                   unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-disconnected">✗ Haz clic en "Test Conexión" para verificar</p>',
                   unsafe_allow_html=True)

    st.markdown("---")

    # Pestañas
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Datos Históricos", "💼 Portfolio", "🔧 Test Conexión", "🐛 Debug"])

    # =========================================================================
    # TAB 1: Datos Históricos
    # =========================================================================
    with tab1:
        if fetch_btn:
            with st.spinner(f"📡 Conectando y obteniendo datos de {symbol}..."):
                # Usar client_id + 1 para no colisionar con test de conexión
                df, error, debug = fetch_historical_data(
                    host=host,
                    port=port,
                    client_id=client_id + 1,
                    symbol=symbol,
                    duration=duration,
                    bar_size=interval
                )

                st.session_state.debug_log = debug

                if error:
                    st.error(f"❌ Error: {error}")
                    with st.expander("🐛 Ver debug log"):
                        for msg in debug:
                            st.text(msg)
                    st.info("""
                    **Verifica:**
                    1. TWS está abierto y logueado
                    2. Puerto correcto (7496 Live, 7497 Paper)
                    3. API habilitada en TWS
                    4. El símbolo es válido
                    """)
                elif df is not None and not df.empty:
                    st.session_state.data = df
                    st.session_state.last_symbol = symbol
                    st.session_state.connection_status = True
                    st.success(f"✓ {len(df)} barras obtenidas")
                else:
                    st.warning("⚠️ No se encontraron datos")

        # Mostrar datos si existen
        if st.session_state.data is not None and not st.session_state.data.empty:
            df = st.session_state.data

            # Métricas
            st.subheader("💰 Métricas")

            current_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else df['Open'].iloc[0]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            high = df['High'].max()
            low = df['Low'].min()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="💵 Precio Actual",
                    value=f"${current_price:.2f}",
                    delta=f"{change:.2f} ({change_pct:+.2f}%)"
                )

            with col2:
                st.metric(label="📈 Máximo", value=f"${high:.2f}")

            with col3:
                st.metric(label="📉 Mínimo", value=f"${low:.2f}")

            with col4:
                avg_volume = df['Volume'].mean()
                st.metric(label="📊 Vol. Promedio", value=f"{avg_volume:,.0f}")

            st.markdown("---")

            # Gráfico de velas
            st.subheader("🕯️ Gráfico de Velas")
            chart = create_candlestick_chart(df, st.session_state.last_symbol or symbol)
            st.plotly_chart(chart, use_container_width=True)

            st.markdown("---")

            # Tabla de datos
            st.subheader("📋 Últimos 20 Registros")

            display_df = df.tail(20).copy()
            display_df = display_df.sort_values('Date', ascending=False)
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
            display_df = display_df.round(2)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.TextColumn("Fecha"),
                    "Open": st.column_config.NumberColumn("Apertura", format="$%.2f"),
                    "High": st.column_config.NumberColumn("Máximo", format="$%.2f"),
                    "Low": st.column_config.NumberColumn("Mínimo", format="$%.2f"),
                    "Close": st.column_config.NumberColumn("Cierre", format="$%.2f"),
                    "Volume": st.column_config.NumberColumn("Volumen", format="%d")
                }
            )

            # Botón de descarga
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name=f"{symbol}_historical_data.csv",
                mime="text/csv"
            )
        else:
            st.info("👆 Ingresa un símbolo y haz clic en 'Obtener Datos' para comenzar")

    # =========================================================================
    # TAB 2: Portfolio
    # =========================================================================
    with tab2:
        st.subheader("💼 Portfolio y Resumen de Cuenta")

        # Inicializar estado de portfolio
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = None
        if 'account_data' not in st.session_state:
            st.session_state.account_data = None

        # Botón para cargar portfolio
        if st.button("🔄 Actualizar Portfolio", type="primary", key="refresh_portfolio"):
            with st.spinner("📡 Obteniendo portfolio y datos de cuenta..."):
                portfolio, account, error, debug = fetch_portfolio(
                    host=host,
                    port=port,
                    client_id=client_id + 2
                )

                st.session_state.debug_log = debug

                if error:
                    st.error(f"❌ Error: {error}")
                    with st.expander("🐛 Ver debug log"):
                        for msg in debug:
                            st.text(msg)
                else:
                    st.session_state.portfolio_data = portfolio
                    st.session_state.account_data = account
                    st.session_state.connection_status = True
                    st.success("✓ Portfolio actualizado")

        # Mostrar datos si existen
        if st.session_state.account_data:
            account = st.session_state.account_data

            # =================================================================
            # RESUMEN GENERAL - Métricas destacadas
            # =================================================================
            st.markdown("---")
            st.subheader("📊 Resumen General")

            # Función helper para obtener valor
            def get_value(key, default=0):
                if key in account:
                    try:
                        return float(account[key].get('value', default))
                    except (ValueError, TypeError):
                        return default
                return default

            def get_currency(key):
                if key in account:
                    return account[key].get('currency', 'USD')
                return 'USD'

            # Fila 1: Métricas principales
            col1, col2, col3 = st.columns(3)

            with col1:
                net_liq = get_value('NetLiquidation')
                st.metric(
                    label="💰 Net Liquidation",
                    value=f"${net_liq:,.2f}",
                    help="Valor total de la cuenta"
                )

            with col2:
                cash = get_value('TotalCashValue')
                st.metric(
                    label="💵 Total Cash",
                    value=f"${cash:,.2f}",
                    help="Efectivo disponible"
                )

            with col3:
                stock_value = get_value('GrossPositionValue')
                if stock_value == 0:
                    stock_value = get_value('StockMarketValue')
                st.metric(
                    label="📈 Stock Market Value",
                    value=f"${stock_value:,.2f}",
                    help="Valor en acciones"
                )

            # Fila 2: Más métricas
            col1, col2, col3 = st.columns(3)

            with col1:
                buying_power = get_value('BuyingPower')
                st.metric(
                    label="🛒 Buying Power",
                    value=f"${buying_power:,.2f}",
                    help="Poder de compra"
                )

            with col2:
                unrealized = get_value('UnrealizedPnL')
                delta_color = "normal" if unrealized >= 0 else "inverse"
                st.metric(
                    label="📊 Unrealized P&L",
                    value=f"${unrealized:,.2f}",
                    delta=f"{'↑' if unrealized >= 0 else '↓'} {'Ganancia' if unrealized >= 0 else 'Pérdida'}",
                    delta_color=delta_color,
                    help="Ganancias/pérdidas no realizadas"
                )

            with col3:
                realized = get_value('RealizedPnL')
                delta_color = "normal" if realized >= 0 else "inverse"
                st.metric(
                    label="✅ Realized P&L",
                    value=f"${realized:,.2f}",
                    delta=f"{'↑' if realized >= 0 else '↓'} {'Ganancia' if realized >= 0 else 'Pérdida'}",
                    delta_color=delta_color,
                    help="Ganancias/pérdidas realizadas"
                )

            # Fila 3: Márgenes
            st.markdown("---")
            st.subheader("📋 Márgenes y Liquidez")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avail_funds = get_value('AvailableFunds')
                st.metric(
                    label="✅ Available Funds",
                    value=f"${avail_funds:,.2f}"
                )

            with col2:
                excess_liq = get_value('ExcessLiquidity')
                st.metric(
                    label="💧 Excess Liquidity",
                    value=f"${excess_liq:,.2f}"
                )

            with col3:
                init_margin = get_value('InitMarginReq')
                st.metric(
                    label="📌 Init Margin Req",
                    value=f"${init_margin:,.2f}"
                )

            with col4:
                maint_margin = get_value('MaintMarginReq')
                st.metric(
                    label="🔒 Maint Margin Req",
                    value=f"${maint_margin:,.2f}"
                )

            # Cushion (indicador de salud de la cuenta)
            cushion = get_value('Cushion')
            if cushion > 0:
                st.markdown("---")
                cushion_pct = cushion * 100
                color = "green" if cushion_pct > 25 else "orange" if cushion_pct > 10 else "red"
                st.markdown(f"**🛡️ Cushion (Margin Safety):** :{color}[{cushion_pct:.1f}%]")
                st.progress(min(cushion, 1.0))

            # =================================================================
            # POSICIONES DEL PORTFOLIO
            # =================================================================
            if st.session_state.portfolio_data:
                st.markdown("---")
                st.subheader("📦 Posiciones Actuales")

                portfolio_df = pd.DataFrame(st.session_state.portfolio_data)

                if not portfolio_df.empty:
                    # Calcular P&L total
                    total_unrealized = portfolio_df['unrealizedPNL'].sum()
                    total_realized = portfolio_df['realizedPNL'].sum()
                    total_market_value = portfolio_df['marketValue'].sum()

                    # Métricas de portfolio
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Posiciones", len(portfolio_df))
                    with col2:
                        st.metric("💰 Valor Total", f"${total_market_value:,.2f}")
                    with col3:
                        color = "normal" if total_unrealized >= 0 else "inverse"
                        st.metric(
                            "📈 P&L Total",
                            f"${total_unrealized:,.2f}",
                            delta_color=color
                        )

                    # Tabla de posiciones
                    st.markdown("---")

                    # Formatear DataFrame para mostrar
                    display_portfolio = portfolio_df[[
                        'symbol', 'position', 'marketPrice', 'marketValue',
                        'averageCost', 'unrealizedPNL', 'realizedPNL'
                    ]].copy()

                    display_portfolio.columns = [
                        'Símbolo', 'Posición', 'Precio', 'Valor Mercado',
                        'Costo Promedio', 'P&L No Realizado', 'P&L Realizado'
                    ]

                    # Calcular % P&L
                    display_portfolio['% P&L'] = (
                        (display_portfolio['Precio'] - display_portfolio['Costo Promedio'])
                        / display_portfolio['Costo Promedio'] * 100
                    ).round(2)

                    st.dataframe(
                        display_portfolio,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Símbolo": st.column_config.TextColumn("Símbolo"),
                            "Posición": st.column_config.NumberColumn("Posición", format="%d"),
                            "Precio": st.column_config.NumberColumn("Precio", format="$%.2f"),
                            "Valor Mercado": st.column_config.NumberColumn("Valor", format="$%.2f"),
                            "Costo Promedio": st.column_config.NumberColumn("Costo Prom.", format="$%.2f"),
                            "P&L No Realizado": st.column_config.NumberColumn("P&L NR", format="$%.2f"),
                            "P&L Realizado": st.column_config.NumberColumn("P&L R", format="$%.2f"),
                            "% P&L": st.column_config.NumberColumn("% P&L", format="%.2f%%")
                        }
                    )

                    # Gráfico de composición del portfolio
                    if len(portfolio_df) > 1:
                        st.markdown("---")
                        st.subheader("🥧 Composición del Portfolio")

                        fig = go.Figure(data=[go.Pie(
                            labels=portfolio_df['symbol'],
                            values=portfolio_df['marketValue'].abs(),
                            hole=0.4,
                            textinfo='label+percent',
                            marker=dict(colors=['#00c853', '#2196f3', '#ff9800', '#9c27b0', '#f44336'])
                        )])
                        fig.update_layout(
                            template='plotly_dark',
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("📭 No hay posiciones abiertas en el portfolio")
            else:
                st.info("📭 No hay posiciones en el portfolio")

            # =================================================================
            # TODOS LOS VALORES DE CUENTA (expandible)
            # =================================================================
            with st.expander("📋 Ver todos los valores de cuenta"):
                account_df = pd.DataFrame([
                    {
                        'Campo': key,
                        'Valor': data.get('value', 'N/A'),
                        'Moneda': data.get('currency', 'N/A')
                    }
                    for key, data in sorted(account.items())
                ])
                st.dataframe(account_df, use_container_width=True, hide_index=True)

        else:
            st.info("👆 Haz clic en 'Actualizar Portfolio' para cargar los datos de tu cuenta")

    # =========================================================================
    # TAB 3: Test Conexión
    # =========================================================================
    with tab3:
        st.subheader("🔧 Test de Conexión con IB")
        st.caption("Usa la misma lógica que test_ibapi.py")

        if connect_btn:
            with st.spinner("🔄 Conectando a Interactive Brokers..."):
                app, error, debug = connect_to_ib(host, port, client_id)

                st.session_state.debug_log = debug

                if error:
                    st.session_state.connection_status = False
                    st.session_state.connection_info = {"error": error, "debug": debug}
                else:
                    # Solicitar info de cuenta
                    st.write("Solicitando resumen de cuenta...")
                    account_info = get_account_summary(app)

                    st.session_state.connection_status = True
                    st.session_state.connection_info = {
                        "connected": True,
                        "accounts": app.accounts,
                        "account_info": account_info,
                        "net_liquidation": app.net_liquidation,
                        "debug": debug + app.debug_messages
                    }

                    # Desconectar
                    if app.isConnected():
                        app.disconnect()

        # Mostrar resultados
        if st.session_state.connection_info:
            info = st.session_state.connection_info

            if info.get("connected"):
                st.success("✓ CONEXIÓN EXITOSA - Igual que test_ibapi.py")

                # Información de cuentas
                st.subheader("👤 Cuentas")
                if info.get("accounts"):
                    for acc in info["accounts"]:
                        st.code(acc)
                else:
                    st.info("Esperando cuentas...")

                # Net Liquidation
                if info.get("net_liquidation"):
                    st.subheader("💰 Valor de Cuenta")
                    st.metric("Net Liquidation", info["net_liquidation"])

                # Información de cuenta detallada
                if info.get("account_info"):
                    st.subheader("💼 Resumen de Cuenta")

                    labels = {
                        "NetLiquidation": "💰 Valor Neto",
                        "TotalCashValue": "💵 Efectivo Total",
                        "AvailableFunds": "✅ Fondos Disponibles"
                    }

                    cols = st.columns(3)
                    for i, (key, data) in enumerate(info["account_info"].items()):
                        with cols[i % 3]:
                            label = labels.get(key, key)
                            try:
                                value = float(data["value"])
                                currency = data["currency"]
                                st.metric(label=label, value=f"{value:,.2f} {currency}")
                            except (ValueError, TypeError):
                                st.metric(label=label, value=str(data.get("value", "N/A")))

                # Debug log
                with st.expander("📝 Log de conexión"):
                    for msg in info.get("debug", []):
                        st.text(msg)

            elif info.get("error"):
                st.error(f"❌ Error: {info['error']}")

                with st.expander("🐛 Debug log"):
                    for msg in info.get("debug", []):
                        st.text(msg)

                st.warning("""
                **Verifica (igual que test_ibapi.py):**

                1. ✅ TWS está abierto y logueado
                2. ✅ Puerto correcto:
                   - **7496** - TWS Live (default en test_ibapi.py)
                   - **7497** - TWS Paper Trading
                3. ✅ API habilitada en TWS:
                   - File → Global Configuration → API → Settings
                   - ☑️ Enable ActiveX and Socket Clients
                   - ☑️ Allow connections from localhost only
                """)
        else:
            st.info("👆 Haz clic en 'Test Conexión' para probar")

        # Información de configuración
        with st.expander("ℹ️ Configuración actual"):
            st.json({
                "Host": host,
                "Puerto": port,
                "Client ID": client_id,
                "Tipo": "Paper Trading" if port in [7497, 4002] else "Live Trading",
                "Nota": "test_ibapi.py usa puerto 7496 y client_id 1"
            })

    # =========================================================================
    # TAB 4: Debug
    # =========================================================================
    with tab4:
        st.subheader("🐛 Debug Log")

        if st.session_state.debug_log:
            st.markdown('<div class="debug-box">', unsafe_allow_html=True)
            for msg in st.session_state.debug_log:
                st.text(msg)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🗑️ Limpiar log"):
                st.session_state.debug_log = []
                st.rerun()
        else:
            st.info("No hay mensajes de debug. Ejecuta una conexión o solicitud de datos.")

        st.markdown("---")
        st.subheader("📋 Estado de Session")
        st.json({
            "connection_status": st.session_state.connection_status,
            "last_symbol": st.session_state.last_symbol,
            "data_loaded": st.session_state.data is not None,
            "data_rows": len(st.session_state.data) if st.session_state.data is not None else 0
        })


if __name__ == "__main__":
    main()
