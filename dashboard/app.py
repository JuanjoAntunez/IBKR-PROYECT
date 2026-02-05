"""
Dashboard de Trading con Interactive Brokers
============================================
Dashboard web interactivo usando Streamlit para visualizar datos de mercado.

Ejecutar con: streamlit run dashboard/app.py

IMPORTANTE: Usa el Trading Engine (single-writer) del proyecto
"""

import sys
import asyncio
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is on sys.path before importing src modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import logger


# Ensure an event loop exists for ib_insync/eventkit when running in Streamlit thread
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


from src.engine.frontend_adapter import get_adapter, get_port_for_mode
from src.ml.ml_engine import MLEngine


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
    /* Modo Paper - Verde */
    .mode-paper {
        background: linear-gradient(90deg, #00c853 0%, #69f0ae 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 10px;
        box-shadow: 0 2px 10px rgba(0, 200, 83, 0.3);
    }
    .mode-badge {
        padding: 6px 10px;
        font-size: 12px;
        border-radius: 8px;
        margin: 6px 0 8px 0;
        box-shadow: none;
    }
    /* Modo Live - Rojo */
    .mode-live {
        background: linear-gradient(90deg, #ff1744 0%, #ff5252 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 10px;
        box-shadow: 0 2px 10px rgba(255, 23, 68, 0.3);
        animation: pulse-live 2s infinite;
    }
    .health-bar.sidebar {
        flex-wrap: wrap;
    }
    .health-bar.sidebar .health-pill {
        font-size: 11px;
        padding: 4px 8px;
    }
    @keyframes pulse-live {
        0% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 23, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0); }
    }
    /* Badge de modo */
    .badge-paper {
        background-color: #00c853;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }
    .badge-live {
        background-color: #ff1744;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }
    /* Warning box para Live */
    .live-warning {
        background-color: rgba(255, 23, 68, 0.1);
        border-left: 4px solid #ff1744;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .live-warning h4 {
        color: #ff1744;
        margin: 0 0 10px 0;
    }
    /* Conexión status mejorado */
    .connection-status {
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 13px;
        margin: 5px 0;
    }
    .conn-paper {
        background-color: rgba(0, 200, 83, 0.15);
        border: 1px solid #00c853;
        color: #00c853;
    }
    .conn-live {
        background-color: rgba(255, 23, 68, 0.15);
        border: 1px solid #ff1744;
        color: #ff1744;
    }
    .conn-disconnected {
        background-color: rgba(158, 158, 158, 0.15);
        border: 1px solid #9e9e9e;
        color: #9e9e9e;
    }
    .sidebar-hint {
        margin-top: 6px;
        padding: 6px 8px;
        border-radius: 8px;
        background: rgba(255, 23, 68, 0.08);
        color: #ff6b7a;
        font-size: 12px;
        font-weight: 600;
    }
    /* Botón Test Conexión - Gris intenso */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #4a4a4a !important;
        color: white !important;
        border: 1px solid #5a5a5a !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #5a5a5a !important;
        border-color: #6a6a6a !important;
    }
    /* Selector de modo tipo pestañas */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        flex-direction: row !important;
        gap: 0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background-color: #2a2a2a;
        padding: 8px 16px !important;
        border: 1px solid #3a3a3a;
        cursor: pointer;
        margin: 0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:first-of-type {
        border-radius: 8px 0 0 8px;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:last-of-type {
        border-radius: 0 8px 8px 0;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #4a4a4a;
        border-color: #5a5a5a;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Configuración por defecto
# =============================================================================
DEFAULT_CONFIG = {
    "mode": "paper",
    "platform": "TWS",
    "host": "127.0.0.1",
    "client_id": 1,
    "timeout": 10,
}


def log_mode_change(old_mode, new_mode, platform, port):
    """Log de cambio de modo para auditoría."""
    logger.info(f"MODE CHANGE: {old_mode} → {new_mode} | Platform: {platform} | Port: {port}")


def _auto_refresh(interval_seconds: int, key: str):
    """Auto-refresh helper with fallback."""
    if interval_seconds <= 0:
        return
    refresh_fn = getattr(st, "autorefresh", None)
    if callable(refresh_fn):
        refresh_fn(interval=interval_seconds * 1000, key=key)
        return
    # If autorefresh is not available, do nothing to avoid blocking UI.
    return


# =============================================================================
# Conexión via Trading Engine (ib_insync single-writer)
# =============================================================================

def connect_to_ib(host, port, client_id, timeout=10, mode="paper", confirm_live=False):
    """
    Conecta al Trading Engine (single-writer). No conecta a IB directamente.

    Returns:
        tuple: (success, error_message, connection_info, debug_messages)
    """
    adapter = get_adapter()
    success, error, info = adapter.connect(
        host=host,
        port=port,
        client_id=client_id,
        mode=mode,
        timeout=timeout,
        confirm_live=confirm_live,
    )
    debug = adapter.get_messages()
    return success, error, info, debug


def _build_account_data(summary) -> dict:
    """Adapt Engine account summary to dashboard schema."""
    if summary is None:
        return {}

    return {
        "NetLiquidation": {"value": summary.net_liquidation, "currency": summary.currency},
        "TotalCashValue": {"value": summary.total_cash, "currency": summary.currency},
        "BuyingPower": {"value": summary.buying_power, "currency": summary.currency},
        "AvailableFunds": {"value": summary.available_funds, "currency": summary.currency},
        "ExcessLiquidity": {"value": summary.excess_liquidity, "currency": summary.currency},
        "InitMarginReq": {"value": summary.init_margin_req, "currency": summary.currency},
        "MaintMarginReq": {"value": summary.maint_margin_req, "currency": summary.currency},
        "UnrealizedPnL": {"value": summary.unrealized_pnl, "currency": summary.currency},
        "RealizedPnL": {"value": summary.realized_pnl, "currency": summary.currency},
    }


def fetch_portfolio(host, port, client_id, mode="paper", confirm_live=False, timeout=15):
    """
    Obtiene el portfolio completo y resumen de cuenta vía Trading Engine.

    Returns:
        tuple: (portfolio_data, account_data, error_message, debug_messages)
    """
    adapter = get_adapter()

    try:
        resolved = adapter.resolve_connection(
            mode=mode,
            host=host,
            port=port,
            client_id=client_id,
        )
    except Exception:
        resolved = {"host": host, "port": port, "client_id": client_id}

    success, error, _, debug = connect_to_ib(
        host=resolved.get("host", host),
        port=resolved.get("port", port),
        client_id=resolved.get("client_id", client_id),
        mode=mode,
        confirm_live=confirm_live,
        timeout=timeout,
    )
    if not success:
        return None, None, error, debug

    acc_ok, acc_err, summary = adapter.get_account(timeout=timeout)
    pos_ok, pos_err, positions = adapter.get_positions(timeout=timeout)

    if not acc_ok:
        return None, None, acc_err, adapter.get_messages()
    if not pos_ok:
        return None, None, pos_err, adapter.get_messages()

    portfolio_data = []
    for pos in positions.values():
        portfolio_data.append({
            "symbol": pos.symbol,
            "secType": pos.sec_type,
            "exchange": pos.exchange,
            "currency": pos.currency,
            "position": pos.quantity,
            "marketPrice": pos.market_price,
            "marketValue": pos.market_value,
            "averageCost": pos.avg_cost,
            "unrealizedPNL": pos.unrealized_pnl,
            "realizedPNL": pos.realized_pnl,
            "account": pos.account,
        })

    account_data = _build_account_data(summary)

    return portfolio_data, account_data, None, adapter.get_messages()


def _format_duration(duration):
    duration_map = {
        "1D": "1 D",
        "5D": "5 D",
        "1M": "1 M",
        "3M": "3 M",
        "6M": "6 M",
        "1Y": "1 Y",
    }
    return duration_map.get(duration, "1 M")


def _format_bar_size(interval):
    bar_map = {
        "1min": "1 min",
        "5min": "5 mins",
        "15min": "15 mins",
        "1h": "1 hour",
        "1d": "1 day",
    }
    return bar_map.get(interval, "1 day")


def fetch_historical_data(host, port, client_id, symbol, duration, bar_size, mode="paper", confirm_live=False, timeout=30):
    """
    Obtiene datos históricos vía Trading Engine.

    Returns:
        tuple: (dataframe, error_message, debug_messages)
    """
    adapter = get_adapter()

    success, error, _, debug = connect_to_ib(
        host=host,
        port=port,
        client_id=client_id,
        mode=mode,
        confirm_live=confirm_live,
        timeout=timeout,
    )
    if not success:
        return None, error, debug

    ok, err, df = adapter.fetch_historical_data(
        symbol=symbol,
        duration=_format_duration(duration),
        bar_size=_format_bar_size(bar_size),
        timeout=timeout,
    )

    if not ok:
        return None, err, adapter.get_messages()

    return df, None, adapter.get_messages()


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

    # =========================================================================
    # Inicializar estado de sesión
    # =========================================================================
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
    # Nuevos estados para modo trading
    if 'trading_mode' not in st.session_state:
        st.session_state.trading_mode = "paper"  # Por defecto SIEMPRE paper
    if 'platform' not in st.session_state:
        st.session_state.platform = "TWS"
    if 'live_confirmed' not in st.session_state:
        st.session_state.live_confirmed = False
    if 'host' not in st.session_state:
        st.session_state.host = "127.0.0.1"
    if 'client_id' not in st.session_state:
        st.session_state.client_id = 1
    if 'timeout' not in st.session_state:
        st.session_state.timeout = 10
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'account_data' not in st.session_state:
        st.session_state.account_data = None
    if 'market_symbol' not in st.session_state:
        st.session_state.market_symbol = "AAPL"
    if 'market_duration' not in st.session_state:
        st.session_state.market_duration = "1M"
    if 'market_interval' not in st.session_state:
        st.session_state.market_interval = "1d"
    if 'auto_refresh_ticks' not in st.session_state:
        st.session_state.auto_refresh_ticks = bool(getattr(st, "autorefresh", None))
    if 'auto_refresh_interval' not in st.session_state:
        st.session_state.auto_refresh_interval = 5
    if 'tick_history' not in st.session_state:
        st.session_state.tick_history = {}
    if 'subs_cache' not in st.session_state:
        st.session_state.subs_cache = None
    if 'subs_cache_ts' not in st.session_state:
        st.session_state.subs_cache_ts = 0.0

    # Puerto actual según plataforma y modo
    port = get_port_for_mode(st.session_state.platform, st.session_state.trading_mode)

    # Engine status (para sidebar/health)
    adapter = get_adapter()
    engine_status = adapter.get_status()
    connected = engine_status.get("connected")
    ib_connected = engine_status.get("ib_connected")
    hb = engine_status.get("heartbeat", {})
    rc = engine_status.get("reconcile", {})
    re = engine_status.get("reconnect", {})
    hb_ok = bool(hb.get("last"))
    rc_ok = bool(rc.get("last"))
    re_attempts = re.get("attempts", 0)

    st.markdown("""
    <style>
        .health-bar { display:flex; gap:10px; margin: 6px 0 12px 0; }
        .health-pill { padding:6px 10px; border-radius:14px; font-size:12px; font-weight:600; }
        .health-ok { background: rgba(0,200,83,0.2); color:#00c853; border:1px solid #00c853; }
        .health-warn { background: rgba(255,193,7,0.2); color:#ffc107; border:1px solid #ffc107; }
        .health-bad { background: rgba(255,23,68,0.2); color:#ff1744; border:1px solid #ff1744; }
    </style>
    """, unsafe_allow_html=True)

    # Verificar si puede conectar (Live requiere confirmación)
    can_connect = st.session_state.trading_mode == "paper" or st.session_state.live_confirmed

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.title("📊 IB Dashboard")
        nav_options = {
            "market": "📈 Datos Históricos",
            "portfolio": "💼 Portfolio",
            "ml": "🤖 ML Trading",
            "config": "⚙️ Configuración",
            "debug": "🐛 Debug",
        }
        if st.session_state.get("main_nav") not in nav_options:
            st.session_state.main_nav = "market"
        selected_section = st.radio(
            "Sección",
            options=list(nav_options.keys()),
            format_func=lambda x: nav_options[x],
            key="main_nav",
        )
        st.markdown("---")
        # Estado de conexión bajo el título
        if st.session_state.connection_status:
            mode_class = "conn-paper" if st.session_state.trading_mode == "paper" else "conn-live"
            mode_icon = "🟢" if st.session_state.trading_mode == "paper" else "🔴"
            mode_text = "Paper" if st.session_state.trading_mode == "paper" else "Live"
            st.markdown(f"""
            <div class="connection-status {mode_class}">
                {mode_icon} Conectado a: {st.session_state.platform} {mode_text} (Puerto {port})
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="connection-status conn-disconnected">
                ⚪ Desconectado
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                '<div class="sidebar-hint">✗ Haz clic en "Test Conexión"</div>',
                unsafe_allow_html=True,
            )

        # Botón de Test Conexión (debajo del estado)
        test_btn = st.button(
            "🔌 Test Conexión",
            use_container_width=True,
            type="secondary",
            key="btn_test_connection_sidebar",
            disabled=not can_connect,
        )

        if test_btn:
            st.toast("Iniciando test de conexión...", icon="🔌")
            if st.session_state.trading_mode == "live" and not st.session_state.live_confirmed:
                st.error("⛔ Debes confirmar el modo Live antes de conectar")
            else:
                progress_container = st.empty()
                status_container = st.empty()

                # Paso 1: Iniciar engine
                progress_container.info("🔄 Paso 1/4: Iniciando Trading Engine...")
                adapter = get_adapter()
                adapter.ensure_running()

                # Verificar que engine está corriendo
                engine_status = adapter.get_status()
                if not engine_status.get("running"):
                    st.error("❌ No se pudo iniciar el Trading Engine")
                    st.session_state.connection_info = {"error": "Engine failed to start", "debug": []}
                else:
                    status_container.success("✅ Engine iniciado")

                    # Resolver configuración
                    try:
                        resolved_cfg = adapter.resolve_connection(
                            mode=st.session_state.trading_mode,
                            host=st.session_state.host,
                            port=port,
                            client_id=st.session_state.client_id,
                        )
                    except Exception:
                        resolved_cfg = None

                    # Paso 2: Conectar a IB
                    mode_text = "Paper (Simulación)" if st.session_state.trading_mode == "paper" else "Live (Dinero Real)"
                    progress_container.info(
                        f"🔄 Paso 2/4: Conectando a {st.session_state.platform} ({mode_text})..."
                    )
                    if resolved_cfg:
                        logger.info(
                            "Intentando conectar a "
                            f"{resolved_cfg.get('host')}:{resolved_cfg.get('port')} "
                            f"con client_id={resolved_cfg.get('client_id')}"
                        )
                    else:
                        logger.info(
                            f"Intentando conectar a {st.session_state.host}:{port} "
                            f"con client_id={st.session_state.client_id}"
                        )

                    success, error, info, debug = connect_to_ib(
                        (resolved_cfg or {}).get("host", st.session_state.host),
                        (resolved_cfg or {}).get("port", port),
                        (resolved_cfg or {}).get("client_id", st.session_state.client_id),
                        timeout=st.session_state.timeout + 5,
                        mode=st.session_state.trading_mode,
                        confirm_live=st.session_state.live_confirmed,
                    )

                    st.session_state.debug_log = debug

                    if error or not success:
                        progress_container.error(f"❌ Error de conexión: {error or 'Unknown error'}")
                        st.session_state.connection_status = False
                        st.session_state.connection_info = {"error": error or "Connection failed", "debug": debug}
                    else:
                        status_container.success("✅ Conectado a IB")

                        # Paso 3: Obtener info de cuenta
                        progress_container.info("🔄 Paso 3/4: Obteniendo información de cuenta...")
                        acc_ok, acc_err, summary = adapter.get_account(timeout=st.session_state.timeout)

                        if not acc_ok:
                            progress_container.warning(f"⚠️ No se pudo obtener cuenta: {acc_err}")
                            st.session_state.connection_status = True
                            st.session_state.connection_info = {
                                "connected": True,
                                "accounts": info.get("accounts", []) if isinstance(info, dict) else [],
                                "account_info": {},
                                "debug": debug,
                                "mode": st.session_state.trading_mode,
                                "platform": st.session_state.platform,
                                "port": port,
                                "warning": acc_err,
                            }
                        else:
                            account_info = _build_account_data(summary)
                            progress_container.success("✅ Paso 4/4: Conexión completada exitosamente!")
                            st.session_state.connection_status = True
                            accounts = info.get("accounts", []) if isinstance(info, dict) else []
                            st.session_state.connection_info = {
                                "connected": True,
                                "accounts": accounts,
                                "account_info": account_info,
                                "net_liquidation": account_info.get("NetLiquidation", {}).get("value"),
                                "debug": debug,
                                "mode": st.session_state.trading_mode,
                                "platform": st.session_state.platform,
                                "port": port,
                            }

                        st.rerun()

        # Selector de modo tipo pestañas
        st.markdown("**Modo de Trading**")
        selected_mode = st.radio(
            "Modo",
            options=["paper", "live"],
            format_func=lambda x: "🟢 Paper" if x == "paper" else "🔴 Live",
            index=0 if st.session_state.trading_mode == "paper" else 1,
            key="mode_selector_main",
            horizontal=True,
            label_visibility="collapsed",
        )

        if selected_mode != st.session_state.trading_mode:
            old_mode = st.session_state.trading_mode
            if selected_mode == "live":
                st.session_state.live_confirmed = False
            st.session_state.connection_status = False
            st.session_state.connection_info = None
            st.session_state.trading_mode = selected_mode
            port = get_port_for_mode(st.session_state.platform, selected_mode)
            log_mode_change(old_mode, selected_mode, st.session_state.platform, port)
            st.rerun()

        if st.session_state.trading_mode == "live":
            st.markdown("""
            <div class="live-warning">
                <h4>⚠️ ADVERTENCIA</h4>
                <p>Vas a conectar a tu cuenta <strong>REAL</strong>.</p>
                <p>Las órdenes ejecutadas afectarán tu dinero real.</p>
            </div>
            """, unsafe_allow_html=True)

            live_confirm = st.checkbox(
                "✅ Entiendo que estoy usando dinero real",
                value=st.session_state.live_confirmed,
                key="live_confirm_checkbox_main",
            )
            st.session_state.live_confirmed = live_confirm

            if not live_confirm:
                st.error("⛔ Debes confirmar para usar modo Live")

        # Health bar compacta
        conn_class = "health-ok" if connected else "health-bad"
        ib_class = "health-ok" if ib_connected else "health-bad"
        hb_class = "health-ok" if hb_ok else "health-warn"
        rc_class = "health-ok" if rc_ok else "health-warn"
        re_class = "health-warn" if re_attempts else "health-ok"

        st.markdown(f"""
        <div class="health-bar sidebar">
            <div class="health-pill {conn_class}">Connected: {connected}</div>
            <div class="health-pill {ib_class}">IB: {ib_connected}</div>
            <div class="health-pill {hb_class}">Heartbeat: {hb.get('last') or '—'}</div>
            <div class="health-pill {rc_class}">Reconcile: {rc.get('last') or '—'}</div>
            <div class="health-pill {re_class}">Reconnect: {re_attempts}</div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # ÁREA PRINCIPAL
    # =========================================================================
    symbol = st.session_state.market_symbol.strip().upper()
    duration = st.session_state.market_duration
    interval = st.session_state.market_interval

    # Título/estado global removidos para dejar el centro limpio

    # =========================================================================
    # SECCIÓN: Datos Históricos
    # =========================================================================
    if selected_section == "market":
        st.subheader("📈 Datos de Mercado")

        cols = st.columns([2, 1, 1, 1])
        with cols[0]:
            st.text_input(
                "Símbolo",
                value=st.session_state.market_symbol,
                help="Símbolo de la acción (ej: AAPL, MSFT, GOOGL)",
                key="market_symbol",
            )
        with cols[1]:
            duration_options = ["1D", "5D", "1M", "3M", "6M", "1Y"]
            duration_index = duration_options.index(st.session_state.market_duration) if st.session_state.market_duration in duration_options else 2
            st.selectbox(
                "Duración",
                options=duration_options,
                index=duration_index,
                help="Período de tiempo",
                key="market_duration",
            )
        with cols[2]:
            interval_options = ["1min", "5min", "15min", "1h", "1d"]
            interval_index = interval_options.index(st.session_state.market_interval) if st.session_state.market_interval in interval_options else 4
            st.selectbox(
                "Intervalo",
                options=interval_options,
                index=interval_index,
                help="Tamaño de cada barra",
                key="market_interval",
            )
        with cols[3]:
            fetch_btn = st.button(
                "📥 Obtener Datos",
                use_container_width=True,
                type="primary",
                disabled=not can_connect,
            )

        st.markdown("---")

        # Suscripciones activas (cache con TTL)
        adapter = get_adapter()
        subs_ttl = 5.0
        now = time.time()
        subs_ok = True
        subs_err = None
        subs = st.session_state.subs_cache or {}
        if now - st.session_state.subs_cache_ts >= subs_ttl or st.session_state.subs_cache is None:
            subs_ok, subs_err, subs = adapter.get_market_subscriptions(timeout=5)
            subs = subs or {}
            st.session_state.subs_cache = subs
            st.session_state.subs_cache_ts = now
        if subs_ok and subs:
            st.caption(f"📡 Suscripciones activas: {', '.join(sorted(subs.keys()))}")
        elif subs_ok:
            st.caption("📡 Suscripciones activas: ninguna")
        else:
            st.caption(f"📡 Suscripciones activas: error ({subs_err})")

        # Panel de ticks en vivo (cache)
        market_data = adapter.get_cached_market_data() or {}
        # Auto-refresh controls
        controls = st.columns([1, 1, 2])
        with controls[0]:
            st.toggle("Auto‑refresh", key="auto_refresh_ticks")
        with controls[1]:
            st.slider("Intervalo (s)", min_value=1, max_value=30, key="auto_refresh_interval")
        with controls[2]:
            st.caption("Actualiza el panel sin re-solicitar datos a IB")

        if st.session_state.auto_refresh_ticks:
            _auto_refresh(st.session_state.auto_refresh_interval, key="ticks_autorefresh")

        # Update tick history for sparklines
        tick_history = st.session_state.tick_history
        for sym, data in market_data.items():
            price = data.get("last")
            if not isinstance(price, (int, float)):
                price = data.get("mid")
            if not isinstance(price, (int, float)):
                price = data.get("close")
            if not isinstance(price, (int, float)):
                price = data.get("bid")
            if not isinstance(price, (int, float)):
                price = data.get("ask")
            if isinstance(price, (int, float)):
                series = tick_history.get(sym, [])
                series.append({
                    "ts": data.get("timestamp") or datetime.now().isoformat(),
                    "price": price,
                })
                if len(series) > 60:
                    series = series[-60:]
                tick_history[sym] = series
        st.session_state.tick_history = tick_history

        if market_data:
            st.subheader("📡 Live Ticks (cache)")
            for sym in sorted(market_data.keys()):
                data = market_data.get(sym, {})
                cols_tick = st.columns([1.5, 1.5, 2, 2, 2, 0.8])
                last = data.get("last")
                bid = data.get("bid")
                ask = data.get("ask")
                ts = data.get("timestamp", "")

                with cols_tick[0]:
                    st.markdown(f"**{sym}**")
                with cols_tick[1]:
                    st.caption("Last")
                    st.write(f"{last:.2f}" if isinstance(last, (int, float)) else "—")
                with cols_tick[2]:
                    st.caption("Bid / Ask")
                    bid_txt = f"{bid:.2f}" if isinstance(bid, (int, float)) else "—"
                    ask_txt = f"{ask:.2f}" if isinstance(ask, (int, float)) else "—"
                    st.write(f"{bid_txt} / {ask_txt}")
                with cols_tick[3]:
                    st.caption("Timestamp")
                    st.write(ts if ts else "—")
                with cols_tick[4]:
                    st.caption("Sparkline")
                    series = st.session_state.tick_history.get(sym, [])
                    if series:
                        df_spark = pd.DataFrame(series)
                        df_spark["ts"] = pd.to_datetime(df_spark["ts"], errors="coerce")
                        st.line_chart(
                            df_spark.set_index("ts")["price"],
                            height=70,
                            use_container_width=True,
                        )
                    else:
                        st.write("—")
                with cols_tick[5]:
                    if st.button("⛔", key=f"unsub_{sym}", help="Desuscribir"):
                        ok, err, _ = adapter.unsubscribe_market_data(sym, timeout=5)
                        if ok:
                            st.success(f"Desuscrito {sym}")
                            st.rerun()
                        else:
                            st.error(err or "Error al desuscribir")
        else:
            st.info("Sin ticks en cache (suscribe un símbolo para ver datos).")

        if fetch_btn:
            # Verificar permisos para Live
            if st.session_state.trading_mode == "live" and not st.session_state.live_confirmed:
                st.error("⛔ Debes confirmar el modo Live antes de conectar")
            else:
                with st.spinner(f"📡 Conectando y obteniendo datos de {symbol}..."):
                    df, error, debug = fetch_historical_data(
                        host=st.session_state.host,
                        port=port,
                        client_id=st.session_state.client_id + 1,
                        symbol=symbol,
                        duration=duration,
                        bar_size=interval,
                        mode=st.session_state.trading_mode,
                        confirm_live=st.session_state.live_confirmed,
                    )

                    st.session_state.debug_log = debug

                    if error:
                        st.error(f"❌ Error: {error}")
                        with st.expander("🐛 Ver debug log"):
                            for msg in debug:
                                st.text(msg)
                        st.info(f"""
                        **Verifica:**
                        1. {st.session_state.platform} está abierto y logueado
                        2. Puerto correcto ({port} para {st.session_state.platform} {'Paper' if st.session_state.trading_mode == 'paper' else 'Live'})
                        3. API habilitada en {st.session_state.platform}
                        4. El símbolo es válido
                        """)
                    elif df is not None and not df.empty:
                        st.session_state.data = df
                        st.session_state.last_symbol = symbol
                        st.session_state.connection_status = True
                        st.success(f"✓ {len(df)} barras obtenidas")
                        # Auto-suscribir market data para este símbolo
                        adapter.subscribe_market_data(symbol, timeout=5)
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
            # Ensure Date is datetime-like before formatting
            original_date = display_df['Date'].copy()
            display_df['Date'] = pd.to_datetime(display_df['Date'], errors='coerce')
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['Date'] = display_df['Date'].fillna(original_date.astype(str))
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
    # SECCIÓN: Portfolio
    # =========================================================================
    if selected_section == "portfolio":
        st.subheader("💼 Portfolio y Resumen de Cuenta")

        # Verificar permisos para Live
        can_load = st.session_state.trading_mode == "paper" or st.session_state.live_confirmed

        # Botón para cargar portfolio
        if st.button("🔄 Actualizar Portfolio", type="primary", key="refresh_portfolio", disabled=not can_load):
            if st.session_state.trading_mode == "live" and not st.session_state.live_confirmed:
                st.error("⛔ Debes confirmar el modo Live antes de conectar")
            else:
                with st.spinner("📡 Obteniendo portfolio y datos de cuenta..."):
                    portfolio, account_info, error, debug = fetch_portfolio(
                        host=st.session_state.host,
                        port=port,
                        client_id=st.session_state.client_id + 2,
                        mode=st.session_state.trading_mode,
                        confirm_live=st.session_state.live_confirmed,
                    )

                    st.session_state.debug_log = debug

                    if error:
                        st.error(f"❌ Error: {error}")
                        with st.expander("🐛 Ver debug log"):
                            for msg in debug:
                                st.text(msg)
                    else:
                        st.session_state.portfolio_data = portfolio
                        st.session_state.account_data = account_info
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

        # Información de configuración
        with st.expander("ℹ️ Configuración actual"):
            st.json({
                "Host": st.session_state.host,
                "Puerto": port,
                "Client ID": st.session_state.client_id,
                "Modo": "Paper Trading" if st.session_state.trading_mode == "paper" else "Live Trading",
                "Plataforma": st.session_state.platform,
                "Timeout": st.session_state.timeout
            })
            # Mostrar configuración efectiva (resuelta por engine/config/env)
            try:
                resolved = adapter.resolve_connection(
                    mode=st.session_state.trading_mode,
                    host=st.session_state.host,
                    port=port,
                    client_id=st.session_state.client_id,
                )
                st.caption("Configuración efectiva (engine/config/env)")
                st.json(resolved)
            except Exception:
                pass

    # =========================================================================
    # SECCIÓN: Configuración
    # =========================================================================
    if selected_section == "config":
        st.subheader("⚙️ Configuración del Dashboard")

        st.markdown("---")

        # Sección de Modo de Trading
        st.markdown("### 🎯 Modo de Trading")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Modo actual:**")
            if st.session_state.trading_mode == "paper":
                st.success("🟢 Paper Trading (Simulación)")
                st.caption("Las operaciones NO afectan dinero real")
            else:
                st.error("🔴 Live Trading (Dinero Real)")
                st.caption("⚠️ Las operaciones AFECTAN tu cuenta real")

        with col2:
            st.markdown("**Plataforma:**")
            st.info(f"🖥️ {st.session_state.platform}")
            st.caption(f"Puerto: {port}")

        st.markdown("---")

        # Configuración de conexión
        st.markdown("### 🔌 Configuración de Conexión")

        new_platform = st.selectbox(
            "Plataforma",
            options=["TWS", "Gateway"],
            index=0 if st.session_state.platform == "TWS" else 1,
            help="TWS = Trader Workstation, Gateway = IB Gateway",
            key="config_platform",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            new_host = st.text_input(
                "Host",
                value=st.session_state.host,
                key="config_host"
            )

        with col2:
            new_client_id = st.number_input(
                "Client ID",
                value=st.session_state.client_id,
                min_value=1,
                max_value=999,
                key="config_client_id"
            )

        with col3:
            new_timeout = st.number_input(
                "Timeout (segundos)",
                value=st.session_state.timeout,
                min_value=5,
                max_value=60,
                key="config_timeout"
            )

        st.markdown("---")

        # Referencia de puertos
        st.markdown("### 📡 Referencia de Puertos")

        port_df = pd.DataFrame({
            "Plataforma": ["TWS", "TWS", "Gateway", "Gateway"],
            "Modo": ["Paper", "Live", "Paper", "Live"],
            "Puerto": [7497, 7496, 4002, 4001],
            "Descripción": [
                "Simulación en TWS",
                "Dinero real en TWS",
                "Simulación en Gateway",
                "Dinero real en Gateway"
            ]
        })

        st.dataframe(port_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Botones de acción
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Guardar Configuración", type="primary", use_container_width=True):
                old_platform = st.session_state.platform
                st.session_state.host = new_host
                st.session_state.client_id = new_client_id
                st.session_state.timeout = new_timeout
                st.session_state.platform = new_platform
                if new_platform != old_platform:
                    st.session_state.connection_status = False
                st.success("✓ Configuración guardada")
                logger.info(
                    "[CONFIG] Guardada: "
                    f"platform={new_platform}, host={new_host}, "
                    f"client_id={new_client_id}, timeout={new_timeout}"
                )
                st.rerun()

        with col2:
            if st.button("🔄 Restablecer Valores por Defecto", use_container_width=True):
                st.session_state.host = DEFAULT_CONFIG["host"]
                st.session_state.client_id = DEFAULT_CONFIG["client_id"]
                st.session_state.timeout = DEFAULT_CONFIG["timeout"]
                st.session_state.trading_mode = DEFAULT_CONFIG["mode"]
                st.session_state.platform = DEFAULT_CONFIG["platform"]
                st.session_state.live_confirmed = False
                st.session_state.connection_status = False
                st.success("✓ Valores restablecidos a defecto (Paper Trading)")
                logger.info("[CONFIG] Restablecido a valores por defecto")
                st.rerun()

        st.markdown("---")

        # Resumen de configuración actual
        with st.expander("📋 Ver configuración completa"):
            st.json({
                "trading_mode": st.session_state.trading_mode,
                "platform": st.session_state.platform,
                "host": st.session_state.host,
                "port": port,
                "client_id": st.session_state.client_id,
                "timeout": st.session_state.timeout,
                "live_confirmed": st.session_state.live_confirmed,
                "connection_status": st.session_state.connection_status
            })

    # =========================================================================
    # SECCIÓN: Debug
    # =========================================================================
    if selected_section == "debug":
        st.subheader("🐛 Debug Log")

        st.subheader("🔌 Último Test de Conexión")
        if st.session_state.connection_info:
            info = st.session_state.connection_info
            if info.get("connected"):
                conn_mode = info.get("mode", "paper")
                conn_platform = info.get("platform", "TWS")
                conn_port = info.get("port", port)
                if conn_mode == "paper":
                    st.success(f"✓ CONEXIÓN EXITOSA - {conn_platform} Paper (Puerto {conn_port})")
                else:
                    st.warning(f"⚠️ CONEXIÓN EXITOSA - {conn_platform} LIVE (Puerto {conn_port}) - DINERO REAL")
            else:
                st.error(f"❌ Error de conexión: {info.get('error', 'Desconocido')}")
            with st.expander("📋 Detalle completo"):
                st.json(info)
        else:
            st.info("No hay test de conexión aún. Usa el botón en el sidebar.")

        st.markdown("---")

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
        st.subheader("🫀 Engine Health")
        adapter = get_adapter()
        engine_status = adapter.get_status()
        st.json({
            "running": engine_status.get("running"),
            "connected": engine_status.get("connected"),
            "ib_connected": engine_status.get("ib_connected"),
            "heartbeat": engine_status.get("heartbeat"),
            "reconcile": engine_status.get("reconcile"),
            "reconnect": engine_status.get("reconnect"),
        })

        st.markdown("---")
        st.subheader("📋 Estado de Session")
        st.json({
            "connection_status": st.session_state.connection_status,
            "trading_mode": st.session_state.trading_mode,
            "platform": st.session_state.platform,
            "port": port,
            "live_confirmed": st.session_state.live_confirmed,
            "last_symbol": st.session_state.last_symbol,
            "data_loaded": st.session_state.data is not None,
            "data_rows": len(st.session_state.data) if st.session_state.data is not None else 0
        })

    # =========================================================================
    # SECCIÓN: ML Trading
    # =========================================================================
    if selected_section == "ml":
        st.subheader("🤖 ML Trading Signals")
        
        col1, col2 = st.columns([2, 1])
        with col1:
             st.info("Estrategias disponibles: Enhanced SMA, Intraday Prediction")
        with col2:
             if st.button("🔄 Recargar Modelos"):
                 st.session_state.ml_engine.reload_models()
                 st.success("Modelos recargados")
             
        if 'ml_engine' not in st.session_state:
             st.session_state.ml_engine = MLEngine()
        
        ml_engine = st.session_state.ml_engine
        
        st.markdown("### 🔮 Predicción Bajo Demanda")
        
        st.markdown("### 🔮 Predicción Bajo Demanda")
        
        # Determine symbol
        symbol = st.session_state.last_symbol
        if not symbol:
            st.warning("⚠️ Selecciona un símbolo en 'Datos Históricos' primero")
        else:
            st.write(f"Símbolo seleccionado: **{symbol}**")
             
            if st.button("Ejecutar Análisis ML", type="primary"):
                 with st.spinner(f"Obteniendo datos extendidos para {symbol} y analizando..."):
                     # 1. Fetch sufficient history (e.g. 2 Years) to ensure ML features (SMA200) can be calculated
                     # regardless of what the user is inspecting visually.
                     adapter = get_adapter()
                     port = get_port_for_mode(st.session_state.platform, st.session_state.trading_mode)
                     
                     # Check connection first
                     if not st.session_state.connection_status:
                          st.error("⚠️ No hay conexión activa. Conecta primero en el sidebar.")
                     else:
                         # Fetch Daily for Enhanced SMA
                         df_daily, err, _ = fetch_historical_data(
                             st.session_state.host, port, st.session_state.client_id + 5,
                             symbol, "2Y", "1d", st.session_state.trading_mode, st.session_state.live_confirmed
                         )
                         
                         if err or df_daily is None or df_daily.empty:
                             st.error(f"Error obteniendo datos diarios: {err}")
                         else:
                             # 1. Enhanced SMA Analysis
                             st.write("#### 1. Enhanced SMA Strategy")
                             model_key = f"enhanced_sma_{symbol}"
                             res_sma = ml_engine.get_prediction(model_key, df_daily)
                             
                             if "error" in res_sma:
                                 st.error(f"Enhanced SMA Error: {res_sma['error']}")
                                 st.caption("Quizás el modelo no está entrenado para este símbolo.")
                             else:
                                 prob = res_sma.get("probability", 0)
                                 confidence = res_sma.get("confidence", 0)
                                 pred_class = res_sma.get("prediction", 0)
                                 
                                 c1, c2, c3 = st.columns(3)
                                 c1.metric("Probabilidad (Profitable)", f"{prob:.2f}")
                                 c2.metric("Confianza", f"{confidence:.2f}")
                                 c3.metric("Clase Predicha", f"{pred_class}")
                                 
                                 if prob > 0.65:
                                     st.success("✅ SEÑAL FUERTE BUY (High Confidence)")
                                 elif prob < 0.35:
                                     st.error("📉 SEÑAL FUERTE SELL/AVOID")
                                 else:
                                     st.warning("⚠️ NEUTRAL / NO TRADE")
                             
                             st.markdown("---")
                             
                             # 2. Intraday Predictor
                             # Fetch Intraday Data (e.g. 5 days of 15mins) to ensure sufficiency
                             # We use "15 mins" because that's what we trained on.
                             st.write("#### 2. Intraday Predictor")
                             df_intra, err_intra, _ = fetch_historical_data(
                                 st.session_state.host, port, st.session_state.client_id + 6,
                                 symbol, "5D", "15min", st.session_state.trading_mode, st.session_state.live_confirmed
                             )
                             
                             if err_intra or df_intra is None or df_intra.empty:
                                 st.warning(f"No se pudieron obtener datos intraday (15min): {err_intra}")
                             else:
                                 # Construct Key: we trained 'intraday_AAPL_15min' (no 's')
                                 # Dashboard usually sends '15min' or '15 mins'.
                                 # Let's standardize to what we trained: '15min'
                                 model_key_intra = f"intraday_{symbol}_15min" 
                                 
                                 res_intra = ml_engine.get_prediction(model_key_intra, df_intra)
                                 
                                 if "error" in res_intra:
                                     st.error(f"Intraday Error: {res_intra['error']}")
                                     st.caption(f"Intentado cargar: {model_key_intra}")
                                 else:
                                     pred_price = res_intra.get("prediction")
                                     curr_price = df_intra['Close'].iloc[-1]
                                     diff = pred_price - curr_price
                                     pct = (diff / curr_price) * 100
                                     
                                     c1, c2 = st.columns(2)
                                     c1.metric("Precio Actual", f"${curr_price:.2f}")
                                     c2.metric("Predicción (Next Bar)", f"${pred_price:.2f}", f"{pct:+.2f}%")



if __name__ == "__main__":
    main()
