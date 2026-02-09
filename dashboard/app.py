"""Dashboard de Trading con Interactive Brokers
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
from src.utils.market_data import normalize_ohlcv
from src.strategies.basic import (
  SmaCrossoverStrategy,
  RSIMeanReversionStrategy,
  RangeBreakoutStrategy,
)
from src.strategies import StrategyConfig
from src.ml.ml_engine import MLEngine


# =============================================================================
# Configuración de página (DEBE ser lo primero)
# =============================================================================
st.set_page_config(
  page_title="IB Trading Dashboard",
  page_icon="IB",
  layout="wide",
  initial_sidebar_state="expanded")

# Cargar tema GitHub centralizado
THEME_PATH = Path(__file__).with_name("theme.css")
if THEME_PATH.exists():
  st.markdown(f"<style>{THEME_PATH.read_text()}</style>", unsafe_allow_html=True)
else:
  logger.warning("theme.css no encontrado en %s", THEME_PATH)


# =============================================================================
# Iconografía minimalista
# =============================================================================
ICON_SVGS = {
  "chart": """<svg viewBox="0 0 16 16"aria-hidden="true">
          <path d="M2 13.5h12"></path>
          <path d="M3 11l3-3 2 2 4-5"></path>
      </svg>
  """,
  "briefcase": """<svg viewBox="0 0 16 16"aria-hidden="true">
          <rect x="2"y="5"width="12"height="9"rx="1.5"></rect>
          <path d="M6 5V3.5h4V5"></path>
          <path d="M2 8h12"></path>
      </svg>
  """,
  "settings": """<svg viewBox="0 0 16 16"aria-hidden="true">
          <circle cx="8"cy="8"r="2.5"></circle>
          <path d="M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2"></path>
          <path d="M3.2 3.2l1.4 1.4M11.4 11.4l1.4 1.4M12.8 3.2l-1.4 1.4M3.2 12.8l1.4-1.4"></path>
      </svg>
  """,
  "bug": """<svg viewBox="0 0 16 16"aria-hidden="true">
          <rect x="5"y="4"width="6"height="7"rx="3"></rect>
          <path d="M6 6.5h4M5 9h6"></path>
          <path d="M3.5 7H5M11 7h1.5M4 12l2-1M12 12l-2-1"></path>
      </svg>
  """,
  "pulse": """<svg viewBox="0 0 16 16"aria-hidden="true">
          <path d="M1.5 8h3l1.5-3 2 6 1.5-3h3.5"></path>
      </svg>
  """,
  "cpu": """<svg viewBox="0 0 16 16"aria-hidden="true">
          <rect x="4.5"y="4.5"width="7"height="7"rx="1"></rect>
          <path d="M3 6H2M3 10H2M14 6h-1M14 10h-1"></path>
          <path d="M6 3V2M10 3V2M6 14v-1M10 14v-1"></path>
      </svg>
  """,
  "plug": """<svg viewBox="0 0 16 16"aria-hidden="true">
          <path d="M6 2v4M10 2v4"></path>
          <path d="M5 6h6"></path>
          <path d="M8 6v3a3 3 0 0 1-3 3H4"></path>
      </svg>
  """,
  "candles": """<svg viewBox="0 0 16 16"aria-hidden="true">
          <path d="M4 3v10M4 5h2v4H4z"></path>
          <path d="M10 4v8M10 6h2v3h-2z"></path>
      </svg>
  """,
  "table": """<svg viewBox="0 0 16 16"aria-hidden="true">
          <rect x="2"y="3"width="12"height="10"rx="1"></rect>
          <path d="M2 7h12M2 10h12M6 3v10M10 3v10"></path>
      </svg>
  """,
}


def icon_html(name: str) -> str:
  svg = ICON_SVGS.get(name)
  if not svg:
    return ""
  return f'<span class="gh-icon">{svg}</span>'


def section_title(title: str, icon: str | None = None, small: bool = False) -> None:
  icon_markup = icon_html(icon) if icon else ""
  size_class = " small" if small else ""
  st.markdown(
    f'<div class="gh-section-title{size_class}">{icon_markup}{title}</div>',
    unsafe_allow_html=True,
  )


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
  logger.info(
    f"MODE CHANGE: {old_mode} → {new_mode} | Platform: {platform} | Port: {port}"
  )


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
  """Conecta al Trading Engine (single-writer). No conecta a IB directamente.

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
  """Obtiene el portfolio completo y resumen de cuenta vía Trading Engine.

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
    "10D": "10 D",
    "20D": "20 D",
    "1W": "1 W",
    "2W": "2 W",
    "1M": "1 M",
    "3M": "3 M",
    "6M": "6 M",
    "1Y": "1 Y",
    "2Y": "2 Y",
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
  """Obtiene datos históricos vía Trading Engine.

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

  colors = [
    '#ff1744' if row['Close'] < row['Open'] else '#00c853'
    for _, row in df.iterrows()
  ]

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
      decreasing_fillcolor='#ff1744'),
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
    plot_bgcolor='rgba(0,0,0,0)')

  fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

  return fig


def render_basic_strategies_panel(symbol: str, data_df: Optional[pd.DataFrame]) -> None:
  st.caption("Estrategias técnicas simples para señales rápidas (sin ejecutar órdenes).")

  strategy_type = st.selectbox(
    "Estrategia",
    ["SMA Crossover", "RSI Reversión", "Breakout de Rango"],
    key=f"basic_strategy_type_{symbol}",
  )

  params_col1, params_col2 = st.columns(2)
  with params_col1:
    sma_short = st.number_input("SMA corta", min_value=2, max_value=100, value=10, key=f"sma_short_{symbol}")
    rsi_len = st.number_input("RSI longitud", min_value=5, max_value=50, value=14, key=f"rsi_len_{symbol}")
    lookback = st.number_input("Breakout lookback", min_value=5, max_value=100, value=20, key=f"lookback_{symbol}")
  with params_col2:
    sma_long = st.number_input("SMA larga", min_value=5, max_value=300, value=30, key=f"sma_long_{symbol}")
    rsi_os = st.number_input("RSI oversold", min_value=5, max_value=50, value=30, key=f"rsi_os_{symbol}")
    rsi_ob = st.number_input("RSI overbought", min_value=50, max_value=95, value=70, key=f"rsi_ob_{symbol}")

  if st.button("Evaluar Señal", type="primary", key=f"basic_eval_{symbol}"):
    if data_df is None or data_df.empty:
      st.warning("Primero carga datos en 'Datos Históricos'.")
    else:
      df = normalize_ohlcv(data_df, schema="lower", set_index=False)
      df["symbol"] = symbol

      cfg = StrategyConfig(name=f"Basic {strategy_type}", symbols=[symbol])
      if strategy_type == "SMA Crossover":
        strategy = SmaCrossoverStrategy(cfg, sma_short=sma_short, sma_long=sma_long)
      elif strategy_type == "RSI Reversión":
        strategy = RSIMeanReversionStrategy(cfg, rsi_length=rsi_len, oversold=rsi_os, overbought=rsi_ob)
      else:
        strategy = RangeBreakoutStrategy(cfg, lookback=lookback)

      signals = strategy.calculate_signals(df)
      if not signals:
        st.info("No hay señal activa con los parámetros actuales.")
      else:
        last_signal = signals[-1]
        st.success(f"Señal: {last_signal.signal_type.value}")
        st.json(last_signal.to_dict())


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

  # Verificar si puede conectar (Live requiere confirmación)
  can_connect = st.session_state.trading_mode == "paper" or st.session_state.live_confirmed

  # =========================================================================
  # SIDEBAR
  # =========================================================================
  with st.sidebar:
    # Título estilo Degiro
    st.markdown('<div class="sidebar-title">IB Trading</div>', unsafe_allow_html=True)

    # Navegación profesional
    st.markdown('<div class="nav-section-title">NAVEGACIÓN</div>', unsafe_allow_html=True)
    nav_options = {
      "market": "Datos de Mercado",
      "portfolio": "Portfolio",
      "basic": "Estrategias",
      "ml": "ML Trading",
      "config": "Configuración",
      "debug": "Debug",
    }
    if st.session_state.get("main_nav") not in nav_options:
      st.session_state.main_nav = "market"
    selected_section = st.radio(
      "Sección",
      options=list(nav_options.keys()),
      format_func=lambda x: nav_options[x],
      key="main_nav",
      label_visibility="collapsed",
    )

    st.markdown("---")

    # Estado de conexión - estilo minimalista
    st.markdown('<div class="nav-section-title">CONEXIÓN</div>', unsafe_allow_html=True)
    if st.session_state.connection_status:
      mode_class = "conn-paper" if st.session_state.trading_mode == "paper" else "conn-live"
      mode_text = "Paper" if st.session_state.trading_mode == "paper" else "Live"
      dot_class = "connected"
      st.markdown(
        f"""<div class="connection-status {mode_class}">
        <span class="status-dot {dot_class}"></span>
        {st.session_state.platform} · {mode_text} · Puerto {port}
      </div>
      """,
        unsafe_allow_html=True,
      )
    else:
      st.markdown("""<div class="connection-status conn-disconnected">
        <span class="status-dot disconnected"></span>
        Sin conexión
      </div>
      """, unsafe_allow_html=True)
      st.markdown(
        '<div class="sidebar-hint">Haz clic en "Conectar" para iniciar</div>',
        unsafe_allow_html=True,
      )

    # Botón de conexión - estilo profesional
    test_btn = st.button(
      "Conectar",
      use_container_width=True,
      type="secondary",
      key="btn_test_connection_sidebar",
      disabled=not can_connect,
    )

    if test_btn:
      st.toast("Iniciando conexión...")
      if st.session_state.trading_mode == "live" and not st.session_state.live_confirmed:
        st.error("Debes confirmar el modo Live antes de conectar")
      else:
        progress_container = st.empty()
        status_container = st.empty()

        # Paso 1: Iniciar engine
        progress_container.info("Paso 1/4: Iniciando Trading Engine...")
        adapter = get_adapter()
        adapter.ensure_running()

        # Verificar que engine está corriendo
        engine_status = adapter.get_status()
        if not engine_status.get("running"):
          st.error("No se pudo iniciar el Trading Engine")
          st.session_state.connection_info = {"error": "Engine failed to start", "debug": []}
        else:
          status_container.success("Engine iniciado")

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
            f"Paso 2/4: Conectando a {st.session_state.platform} ({mode_text})..."
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
            progress_container.error(f"Error de conexión: {error or 'Unknown error'}")
            st.session_state.connection_status = False
            st.session_state.connection_info = {"error": error or "Connection failed", "debug": debug}
          else:
            status_container.success("Conectado a IB")

            # Paso 3: Obtener info de cuenta
            progress_container.info("Paso 3/4: Obteniendo información de cuenta...")
            acc_ok, acc_err, summary = adapter.get_account(timeout=st.session_state.timeout)

            if not acc_ok:
              progress_container.warning(f"No se pudo obtener cuenta: {acc_err}")
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
              progress_container.success("Paso 4/4: Conexión completada")
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

    # Selector de modo tipo pestañas - estilo Degiro
    st.markdown("---")
    st.markdown('<div class="nav-section-title">MODO</div>', unsafe_allow_html=True)
    selected_mode = st.radio(
      "Modo",
      options=["paper", "live"],
      format_func=lambda x: "Paper" if x == "paper" else "Live",
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
      st.markdown("""<div class="live-warning">
        <div class="live-warning-title">MODO LIVE ACTIVO</div>
        <div class="live-warning-text">
          Estás conectando a tu cuenta real.<br/>
          Las operaciones afectarán tu capital.
        </div>
      </div>
      """, unsafe_allow_html=True)

      live_confirm = st.checkbox(
        "Confirmo que entiendo los riesgos",
        value=st.session_state.live_confirmed,
        key="live_confirm_checkbox_main",
      )
      st.session_state.live_confirmed = live_confirm

      if not live_confirm:
        st.warning("Debes confirmar para usar modo Live")

    # Health bar compacta - estilo profesional
    st.markdown("---")
    st.markdown('<div class="nav-section-title">ESTADO</div>', unsafe_allow_html=True)
    conn_class = "active" if connected else "error"
    ib_class = "active" if ib_connected else "error"
    hb_class = "active" if hb_ok else "warning"
    rc_class = "active" if rc_ok else "warning"
    re_class = "warning" if re_attempts else "active"
    st.markdown(
      f"""<div class="health-bar">
      <div class="health-pill {conn_class}">Engine: {'On' if connected else 'Off'}</div>
      <div class="health-pill {ib_class}">IB: {'OK' if ib_connected else '—'}</div>
      <div class="health-pill {hb_class}">HB: {hb.get('last') or '—'}</div>
      <div class="health-pill {rc_class}">Rec: {rc.get('last') or '—'}</div>
      <div class="health-pill {re_class}">Retry: {re_attempts}</div>
    </div>
    """,
      unsafe_allow_html=True,
    )

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
    section_title("Datos de Mercado", "chart")

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
        "Obtener Datos",
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
      st.caption(f"Suscripciones activas: {', '.join(sorted(subs.keys()))}")
    elif subs_ok:
      st.caption("Suscripciones activas: ninguna")
    else:
      st.caption(f"Suscripciones activas: error ({subs_err})")

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
      section_title("Live Ticks (cache)", "pulse", small=True)
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
          if st.button("×", key=f"unsub_{sym}", help="Desuscribir"):
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
        st.error("Debes confirmar el modo Live antes de conectar")
      else:
        with st.spinner(f"Conectando y obteniendo datos de {symbol}..."):
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
            st.error(f"Error: {error}")
            with st.expander("Ver debug log"):
              for msg in debug:
                st.text(msg)
            st.info(f"""**Verifica:**
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
            st.warning("No se encontraron datos")

    # Mostrar datos si existen
    if st.session_state.data is not None and not st.session_state.data.empty:
      df = st.session_state.data

      # Métricas
      section_title("Métricas", "chart", small=True)

      current_price = df['Close'].iloc[-1]
      prev_close = df['Close'].iloc[-2] if len(df) > 1 else df['Open'].iloc[0]
      change = current_price - prev_close
      change_pct = (change / prev_close) * 100
      high = df['High'].max()
      low = df['Low'].min()

      col1, col2, col3, col4 = st.columns(4)

      with col1:
        st.metric(
          label="Precio Actual",
          value=f"${current_price:.2f}",
          delta=f"{change:.2f} ({change_pct:+.2f}%)")

      with col2:
        st.metric(label="Máximo", value=f"${high:.2f}")

      with col3:
        st.metric(label="Mínimo", value=f"${low:.2f}")

      with col4:
        avg_volume = df['Volume'].mean()
        st.metric(label="Vol. Promedio", value=f"{avg_volume:,.0f}")

      st.markdown("---")

      # Gráfico de velas
      section_title("Gráfico de Velas", "candles", small=True)
      chart = create_candlestick_chart(df, st.session_state.last_symbol or symbol)
      st.plotly_chart(chart, use_container_width=True)

      st.markdown("---")

      # Tabla de datos
      section_title("Últimos 20 Registros", "table", small=True)

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
        label="Descargar CSV",
        data=csv,
        file_name=f"{symbol}_historical_data.csv",
        mime="text/csv")
    else:
      st.info("Ingresa un símbolo y haz clic en 'Obtener Datos' para comenzar")

  # =========================================================================
  # SECCIÓN: Portfolio
  # =========================================================================
  if selected_section == "portfolio":
    section_title("Portfolio y Resumen de Cuenta", "briefcase")

    # Verificar permisos para Live
    can_load = st.session_state.trading_mode == "paper" or st.session_state.live_confirmed

    # Botón para cargar portfolio
    if st.button("Actualizar Portfolio", type="primary", key="refresh_portfolio", disabled=not can_load):
      if st.session_state.trading_mode == "live" and not st.session_state.live_confirmed:
        st.error("Debes confirmar el modo Live antes de conectar")
      else:
        with st.spinner("Obteniendo portfolio y datos de cuenta..."):
          portfolio, account_info, error, debug = fetch_portfolio(
            host=st.session_state.host,
            port=port,
            client_id=st.session_state.client_id + 2,
            mode=st.session_state.trading_mode,
            confirm_live=st.session_state.live_confirmed,
          )

          st.session_state.debug_log = debug

          if error:
            st.error(f"Error: {error}")
            with st.expander("Ver debug log"):
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
      section_title("Resumen General", "chart", small=True)

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
          label="Net Liquidation",
          value=f"${net_liq:,.2f}",
          help="Valor total de la cuenta")

      with col2:
        cash = get_value('TotalCashValue')
        st.metric(
          label="Total Cash",
          value=f"${cash:,.2f}",
          help="Efectivo disponible")

      with col3:
        stock_value = get_value('GrossPositionValue')
        if stock_value == 0:
          stock_value = get_value('StockMarketValue')
        st.metric(
          label="Stock Market Value",
          value=f"${stock_value:,.2f}",
          help="Valor en acciones")

      # Fila 2: Más métricas
      col1, col2, col3 = st.columns(3)

      with col1:
        buying_power = get_value('BuyingPower')
        st.metric(
          label="Buying Power",
          value=f"${buying_power:,.2f}",
          help="Poder de compra")

      with col2:
        unrealized = get_value('UnrealizedPnL')
        delta_color = "normal" if unrealized >= 0 else "inverse"
        st.metric(
          label="Unrealized P&L",
          value=f"${unrealized:,.2f}",
          delta=f"{'↑' if unrealized >= 0 else '↓'} {'Ganancia' if unrealized >= 0 else 'Pérdida'}",
          delta_color=delta_color,
          help="Ganancias/pérdidas no realizadas")

      with col3:
        realized = get_value('RealizedPnL')
        delta_color = "normal" if realized >= 0 else "inverse"
        st.metric(
          label="Realized P&L",
          value=f"${realized:,.2f}",
          delta=f"{'↑' if realized >= 0 else '↓'} {'Ganancia' if realized >= 0 else 'Pérdida'}",
          delta_color=delta_color,
          help="Ganancias/pérdidas realizadas")

      # Fila 3: Márgenes
      st.markdown("---")
      section_title("Márgenes y Liquidez", "chart", small=True)

      col1, col2, col3, col4 = st.columns(4)

      with col1:
        avail_funds = get_value('AvailableFunds')
        st.metric(
          label="Available Funds",
          value=f"${avail_funds:,.2f}")

      with col2:
        excess_liq = get_value('ExcessLiquidity')
        st.metric(
          label="Excess Liquidity",
          value=f"${excess_liq:,.2f}")

      with col3:
        init_margin = get_value('InitMarginReq')
        st.metric(
          label="Init Margin Req",
          value=f"${init_margin:,.2f}")

      with col4:
        maint_margin = get_value('MaintMarginReq')
        st.metric(
          label="Maint Margin Req",
          value=f"${maint_margin:,.2f}")

      # Cushion (indicador de salud de la cuenta)
      cushion = get_value('Cushion')
      if cushion > 0:
        st.markdown("---")
        cushion_pct = cushion * 100
        color = "green" if cushion_pct > 25 else "orange" if cushion_pct > 10 else "red"
        st.markdown(f"**Cushion (Margin Safety):** :{color}[{cushion_pct:.1f}%]")
        st.progress(min(cushion, 1.0))

      # =================================================================
      # POSICIONES DEL PORTFOLIO
      # =================================================================
      if st.session_state.portfolio_data:
        st.markdown("---")
        section_title("Posiciones Actuales", "briefcase", small=True)

        portfolio_df = pd.DataFrame(st.session_state.portfolio_data)

        if not portfolio_df.empty:
          # Calcular P&L total
          total_unrealized = portfolio_df['unrealizedPNL'].sum()
          total_realized = portfolio_df['realizedPNL'].sum()
          total_market_value = portfolio_df['marketValue'].sum()

          # Métricas de portfolio
          col1, col2, col3 = st.columns(3)
          with col1:
            st.metric("Posiciones", len(portfolio_df))
          with col2:
            st.metric("Valor Total", f"${total_market_value:,.2f}")
          with col3:
            color = "normal" if total_unrealized >= 0 else "inverse"
            st.metric(
              "P&L Total",
              f"${total_unrealized:,.2f}",
              delta_color=color
            )

          # Tabla de posiciones
          st.markdown("---")

          # Formatear DataFrame para mostrar
          display_portfolio = portfolio_df[[
            'symbol', 'position', 'marketPrice', 'marketValue',
            'averageCost', 'unrealizedPNL', 'realizedPNL']].copy()

          display_portfolio.columns = [
            'Símbolo', 'Posición', 'Precio', 'Valor Mercado',
            'Costo Promedio', 'P&L No Realizado', 'P&L Realizado']

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
            section_title("Composición del Portfolio", "chart", small=True)

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
              plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
          st.info("No hay posiciones abiertas en el portfolio")
      else:
        st.info("No hay posiciones en el portfolio")

      # =================================================================
      # TODOS LOS VALORES DE CUENTA (expandible)
      # =================================================================
      with st.expander("Ver todos los valores de cuenta"):
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
      st.info("Haz clic en 'Actualizar Portfolio' para cargar los datos de tu cuenta")

    # Información de configuración
    with st.expander("Configuración actual"):
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
    section_title("Configuración", "settings")

    st.markdown("---")

    # Sección de Modo de Trading
    st.markdown("### Modo de Trading")

    col1, col2 = st.columns(2)

    with col1:
      st.markdown("**Modo actual:**")
      if st.session_state.trading_mode == "paper":
        st.success("Paper Trading (Simulación)")
        st.caption("Las operaciones NO afectan dinero real")
      else:
        st.error("Live Trading (Dinero Real)")
        st.caption("Las operaciones afectan tu cuenta real")

    with col2:
      st.markdown("**Plataforma:**")
      st.info(f"{st.session_state.platform}")
      st.caption(f"Puerto: {port}")

    st.markdown("---")

    # Configuración de conexión
    st.markdown("### Configuración de Conexión")

    new_platform = st.selectbox(
      "Plataforma",
      options=["TWS", "Gateway"],
      index=0 if st.session_state.platform == "TWS"else 1,
      help="TWS = Trader Workstation, Gateway = IB Gateway",
      key="config_platform",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
      new_host = st.text_input(
        "Host",
        value=st.session_state.host,
        key="config_host")

    with col2:
      new_client_id = st.number_input(
        "Client ID",
        value=st.session_state.client_id,
        min_value=1,
        max_value=999,
        key="config_client_id")

    with col3:
      new_timeout = st.number_input(
        "Timeout (segundos)",
        value=st.session_state.timeout,
        min_value=5,
        max_value=60,
        key="config_timeout")

    st.markdown("---")

    # Referencia de puertos
    st.markdown("### Referencia de Puertos")

    port_df = pd.DataFrame({
      "Plataforma": ["TWS", "TWS", "Gateway", "Gateway"],
      "Modo": ["Paper", "Live", "Paper", "Live"],
      "Puerto": [7497, 7496, 4002, 4001],
      "Descripción": [
        "Simulación en TWS",
        "Dinero real en TWS",
        "Simulación en Gateway",
        "Dinero real en Gateway"]
    })

    st.dataframe(port_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Botones de acción
    col1, col2 = st.columns(2)

    with col1:
      if st.button("Guardar Configuración", type="primary", use_container_width=True):
        old_platform = st.session_state.platform
        st.session_state.host = new_host
        st.session_state.client_id = new_client_id
        st.session_state.timeout = new_timeout
        st.session_state.platform = new_platform
        if new_platform != old_platform:
          st.session_state.connection_status = False
        st.success("✓ Configuración guardada")
        logger.info(
          "[CONFIG] Guardada: "f"platform={new_platform}, host={new_host}, "f"client_id={new_client_id}, timeout={new_timeout}")
        st.rerun()

    with col2:
      if st.button("Restablecer Valores por Defecto", use_container_width=True):
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
    with st.expander("Ver configuración completa"):
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
    section_title("Debug Log", "bug")

    section_title("Último Test de Conexión", "plug", small=True)
    if st.session_state.connection_info:
      info = st.session_state.connection_info
      if info.get("connected"):
        conn_mode = info.get("mode", "paper")
        conn_platform = info.get("platform", "TWS")
        conn_port = info.get("port", port)
        if conn_mode == "paper":
          st.success(f"✓ CONEXIÓN EXITOSA - {conn_platform} Paper (Puerto {conn_port})")
        else:
          st.warning(f"CONEXIÓN EXITOSA - {conn_platform} LIVE (Puerto {conn_port}) - DINERO REAL")
      else:
        st.error(f"Error de conexión: {info.get('error', 'Desconocido')}")
      with st.expander("Detalle completo"):
        st.json(info)
    else:
      st.info("No hay test de conexión aún. Usa el botón en el sidebar.")

    st.markdown("---")

    if st.session_state.debug_log:
      st.markdown('<div class="debug-box">', unsafe_allow_html=True)
      for msg in st.session_state.debug_log:
        st.text(msg)
      st.markdown('</div>', unsafe_allow_html=True)

      if st.button("Limpiar log"):
        st.session_state.debug_log = []
        st.rerun()
    else:
      st.info("No hay mensajes de debug. Ejecuta una conexión o solicitud de datos.")

    st.markdown("---")
    section_title("Engine Health", "pulse", small=True)
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
    section_title("Estado de Session", "chart", small=True)
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
  # SECCIÓN: Estrategias Básicas
  # =========================================================================
  if selected_section == "basic":
    section_title("Estrategias Básicas", "pulse")
    symbol = st.session_state.last_symbol or "SPY"
    st.write(f"Símbolo activo: **{symbol}**")
    render_basic_strategies_panel(symbol, st.session_state.data)

  # =========================================================================
  # SECCIÓN: ML Trading
  # =========================================================================
  if selected_section == "ml":
    section_title("ML Trading", "cpu")
    section_title("Estrategias Básicas (extra)", "pulse", small=True)
    symbol = st.session_state.last_symbol or "SPY"
    st.write(f"Símbolo activo: **{symbol}**")
    render_basic_strategies_panel(symbol, st.session_state.data)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
      st.info("Estrategias disponibles: Enhanced SMA, Intraday Prediction")
    with col2:
      if st.button("Recargar Modelos"):
        st.session_state.ml_engine.reload_models()
        st.success("Modelos recargados")

    if 'ml_engine' not in st.session_state:
      st.session_state.ml_engine = MLEngine()

    ml_engine = st.session_state.ml_engine

    section_title("Predicción Bajo Demanda", "cpu", small=True)
    
    # Determine symbol
    symbol = st.session_state.last_symbol
    if not symbol:
      st.warning("Selecciona un símbolo en 'Datos Históricos' primero")
    else:
      st.write(f"Símbolo seleccionado: **{symbol}**")

      if st.button("Ejecutar Análisis ML", type="primary"):
        with st.spinner(f"Obteniendo datos extendidos para {symbol} y analizando..."):
          # 1. Fetch sufficient history (2Y) to ensure ML features (SMA200) can be calculated
          # regardless of what the user is inspecting visually.
          adapter = get_adapter()
          port = get_port_for_mode(st.session_state.platform, st.session_state.trading_mode)

          # Check connection first
          if not st.session_state.connection_status:
            st.error("No hay conexión activa. Conecta primero en el sidebar.")
          else:
            # Fetch Daily for Enhanced SMA
            df_daily, err, _ = fetch_historical_data(
              st.session_state.host, port, st.session_state.client_id + 5,
              symbol, "2Y", "1d", st.session_state.trading_mode, st.session_state.live_confirmed
            )

            if err or df_daily is None or df_daily.empty:
              st.error(f"Error obteniendo datos diarios: {err}")
            else:
              df_daily = normalize_ohlcv(df_daily, schema="upper", set_index=False)
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
                  st.success("SEÑAL FUERTE BUY (High Confidence)")
                elif prob < 0.35:
                  st.error("SEÑAL FUERTE SELL/AVOID")
                else:
                  st.warning("NEUTRAL / NO TRADE")

              st.markdown("---")

              # 2. Intraday Predictor
              # Fetch Intraday Data (1 month of 15 mins) to ensure sufficiency
              # We use "15 mins" because that's what we trained on.
              st.write("#### 2. Intraday Predictor")
              df_intra, err_intra, _ = fetch_historical_data(
                st.session_state.host, port, st.session_state.client_id + 6,
                symbol, "1M", "15min", st.session_state.trading_mode, st.session_state.live_confirmed
              )

              if err_intra or df_intra is None or df_intra.empty:
                st.warning(f"No se pudieron obtener datos intraday (15min): {err_intra}")
              else:
                df_intra = normalize_ohlcv(df_intra, schema="upper", set_index=False)
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
