"""
Aplicación web FastAPI para el dashboard de trading.

Proporciona una interfaz web para interactuar con el sistema de trading,
visualizar datos y monitorear estrategias.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.utils.logger import logger
from src.connection.ib_client import IBClient
from src.data.fetcher import HistoricalDataFetcher
from src.data.stream import MarketDataStream, TickData
from src.strategies import (
    MovingAverageCrossover,
    StrategyConfig,
    MAType,
)

# Estado global de la aplicación
app_state: Dict[str, Any] = {
    "ib_client": None,
    "fetcher": None,
    "stream": None,
    "strategies": {},
    "websocket_clients": [],
}


# ==================== Modelos Pydantic ====================

class ConnectionRequest(BaseModel):
    """Request para conectar a IB."""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1


class HistoricalDataRequest(BaseModel):
    """Request para datos históricos."""
    symbol: str
    duration: str = "1 M"
    bar_size: str = "1 day"


class StreamRequest(BaseModel):
    """Request para streaming."""
    symbols: List[str]


class StrategyRequest(BaseModel):
    """Request para crear estrategia."""
    name: str
    symbols: List[str]
    fast_period: int = 10
    slow_period: int = 30
    ma_type: str = "SMA"
    max_position_size: float = 10000.0


# ==================== Lifecycle ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación."""
    logger.info("Iniciando aplicación web...")
    yield
    # Cleanup
    logger.info("Cerrando aplicación web...")
    if app_state["ib_client"]:
        await app_state["ib_client"].disconnect()


def create_app() -> FastAPI:
    """Crea y configura la aplicación FastAPI."""
    app = FastAPI(
        title="IB Trading Dashboard",
        description="Dashboard para trading con Interactive Brokers",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Registrar rutas
    register_routes(app)

    return app


# ==================== Rutas ====================

def register_routes(app: FastAPI):
    """Registra todas las rutas de la API."""

    @app.get("/", response_class=HTMLResponse)
    async def home():
        """Página principal del dashboard."""
        return get_dashboard_html()

    # ==================== Conexión ====================

    @app.get("/api/status")
    async def get_status():
        """Obtiene el estado de la conexión."""
        client = app_state.get("ib_client")
        connected = client.is_connected() if client else False

        return {
            "connected": connected,
            "timestamp": datetime.now().isoformat(),
            "active_streams": len(app_state.get("stream", {})._subscriptions) if app_state.get("stream") else 0,
            "active_strategies": len(app_state.get("strategies", {})),
        }

    @app.post("/api/connect")
    async def connect(request: ConnectionRequest):
        """Conecta con Interactive Brokers."""
        try:
            client = IBClient(
                host=request.host,
                port=request.port,
                client_id=request.client_id,
            )
            await client.connect()

            app_state["ib_client"] = client
            app_state["fetcher"] = HistoricalDataFetcher(client.ib)
            app_state["stream"] = MarketDataStream(client.ib)

            logger.info("Conexión establecida via web")

            return {
                "success": True,
                "message": f"Conectado a IB en {request.host}:{request.port}",
            }

        except Exception as e:
            logger.error(f"Error conectando: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/disconnect")
    async def disconnect():
        """Desconecta de Interactive Brokers."""
        client = app_state.get("ib_client")

        if not client:
            return {"success": True, "message": "No hay conexión activa"}

        try:
            # Cancelar streams
            stream = app_state.get("stream")
            if stream:
                await stream.unsubscribe_all()

            await client.disconnect()

            app_state["ib_client"] = None
            app_state["fetcher"] = None
            app_state["stream"] = None

            return {"success": True, "message": "Desconectado de IB"}

        except Exception as e:
            logger.error(f"Error desconectando: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Datos Históricos ====================

    @app.post("/api/historical")
    async def get_historical_data(request: HistoricalDataRequest):
        """Obtiene datos históricos de un símbolo."""
        fetcher = app_state.get("fetcher")

        if not fetcher:
            raise HTTPException(status_code=400, detail="No hay conexión con IB")

        try:
            df = await fetcher.get_stock_bars(
                symbol=request.symbol,
                duration=request.duration,
                bar_size=request.bar_size,
            )

            if df.empty:
                return {"symbol": request.symbol, "data": [], "count": 0}

            # Convertir a formato JSON-friendly
            data = df.to_dict(orient="records")

            # Convertir fechas
            for row in data:
                if "date" in row and hasattr(row["date"], "isoformat"):
                    row["date"] = row["date"].isoformat()

            return {
                "symbol": request.symbol,
                "data": data,
                "count": len(data),
            }

        except Exception as e:
            logger.error(f"Error obteniendo datos históricos: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Streaming ====================

    @app.post("/api/stream/subscribe")
    async def subscribe_stream(request: StreamRequest):
        """Suscribe a streaming de símbolos."""
        stream = app_state.get("stream")

        if not stream:
            raise HTTPException(status_code=400, detail="No hay conexión con IB")

        try:
            results = await stream.subscribe_multiple(request.symbols)

            return {
                "success": True,
                "subscriptions": results,
            }

        except Exception as e:
            logger.error(f"Error suscribiendo: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/stream/unsubscribe")
    async def unsubscribe_stream(symbol: str):
        """Cancela suscripción de un símbolo."""
        stream = app_state.get("stream")

        if not stream:
            raise HTTPException(status_code=400, detail="No hay conexión con IB")

        try:
            success = await stream.unsubscribe(symbol)
            return {"success": success}

        except Exception as e:
            logger.error(f"Error desuscribiendo: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/stream/latest")
    async def get_latest_prices():
        """Obtiene últimos precios de símbolos suscritos."""
        stream = app_state.get("stream")

        if not stream:
            return {"prices": {}}

        latest = stream.get_all_latest()

        prices = {}
        for symbol, tick in latest.items():
            prices[symbol] = tick.to_dict()

        return {"prices": prices}

    # ==================== Estrategias ====================

    @app.post("/api/strategy/create")
    async def create_strategy(request: StrategyRequest):
        """Crea una nueva estrategia."""
        try:
            ma_type = MAType[request.ma_type.upper()]

            config = StrategyConfig(
                name=request.name,
                symbols=request.symbols,
                max_position_size=request.max_position_size,
            )

            strategy = MovingAverageCrossover(
                config=config,
                fast_period=request.fast_period,
                slow_period=request.slow_period,
                ma_type=ma_type,
            )

            app_state["strategies"][request.name] = strategy

            return {
                "success": True,
                "name": request.name,
                "params": strategy.get_strategy_params(),
            }

        except Exception as e:
            logger.error(f"Error creando estrategia: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/strategy/{name}/start")
    async def start_strategy(name: str):
        """Inicia una estrategia."""
        strategy = app_state["strategies"].get(name)

        if not strategy:
            raise HTTPException(status_code=404, detail="Estrategia no encontrada")

        strategy.start()
        return {"success": True, "state": strategy.state.name}

    @app.post("/api/strategy/{name}/stop")
    async def stop_strategy(name: str):
        """Detiene una estrategia."""
        strategy = app_state["strategies"].get(name)

        if not strategy:
            raise HTTPException(status_code=404, detail="Estrategia no encontrada")

        strategy.stop()
        return {"success": True, "state": strategy.state.name}

    @app.get("/api/strategy/{name}")
    async def get_strategy(name: str):
        """Obtiene información de una estrategia."""
        strategy = app_state["strategies"].get(name)

        if not strategy:
            raise HTTPException(status_code=404, detail="Estrategia no encontrada")

        return strategy.get_summary()

    @app.get("/api/strategies")
    async def list_strategies():
        """Lista todas las estrategias."""
        strategies = []

        for name, strategy in app_state["strategies"].items():
            strategies.append({
                "name": name,
                "state": strategy.state.name,
                "symbols": strategy.symbols,
                "positions": len(strategy.open_positions),
            })

        return {"strategies": strategies}

    # ==================== WebSocket para datos en tiempo real ====================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket para streaming de datos en tiempo real."""
        await websocket.accept()
        app_state["websocket_clients"].append(websocket)

        logger.info("Cliente WebSocket conectado")

        try:
            stream = app_state.get("stream")

            if stream:
                # Callback para enviar datos a WebSocket
                def send_tick(tick: TickData):
                    asyncio.create_task(
                        broadcast_tick(tick)
                    )

                stream.add_global_callback(send_tick)

            while True:
                # Mantener conexión y procesar mensajes
                data = await websocket.receive_text()

                # Procesar comandos del cliente
                if data == "ping":
                    await websocket.send_text("pong")

        except WebSocketDisconnect:
            logger.info("Cliente WebSocket desconectado")
        finally:
            if websocket in app_state["websocket_clients"]:
                app_state["websocket_clients"].remove(websocket)


async def broadcast_tick(tick: TickData):
    """Envía tick a todos los clientes WebSocket."""
    message = {
        "type": "tick",
        "data": tick.to_dict(),
    }

    for client in app_state["websocket_clients"]:
        try:
            await client.send_json(message)
        except Exception:
            pass


def get_dashboard_html() -> str:
    """Retorna el HTML del dashboard."""
    return """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IB Trading Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .status-connected { color: #22c55e; }
        .status-disconnected { color: #ef4444; }
        .card { @apply bg-white rounded-lg shadow-md p-6 mb-4; }
        .btn { @apply px-4 py-2 rounded font-medium transition-colors; }
        .btn-primary { @apply bg-blue-600 text-white hover:bg-blue-700; }
        .btn-danger { @apply bg-red-600 text-white hover:bg-red-700; }
        .btn-success { @apply bg-green-600 text-white hover:bg-green-700; }
        .input { @apply border rounded px-3 py-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-500; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <nav class="bg-blue-800 text-white p-4 shadow-lg">
        <div class="container mx-auto flex justify-between items-center">
            <h1 class="text-2xl font-bold">IB Trading Dashboard</h1>
            <div id="connection-status" class="flex items-center gap-2">
                <span class="w-3 h-3 rounded-full bg-red-500" id="status-indicator"></span>
                <span id="status-text">Desconectado</span>
            </div>
        </div>
    </nav>

    <main class="container mx-auto p-6">
        <!-- Conexion -->
        <div class="card">
            <h2 class="text-xl font-semibold mb-4">Conexion IB</h2>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                <div>
                    <label class="block text-sm font-medium mb-1">Host</label>
                    <input type="text" id="host" value="127.0.0.1" class="input">
                </div>
                <div>
                    <label class="block text-sm font-medium mb-1">Puerto</label>
                    <input type="number" id="port" value="7497" class="input">
                </div>
                <div>
                    <label class="block text-sm font-medium mb-1">Client ID</label>
                    <input type="number" id="client-id" value="1" class="input">
                </div>
                <div class="flex items-end gap-2">
                    <button onclick="connect()" class="btn btn-primary">Conectar</button>
                    <button onclick="disconnect()" class="btn btn-danger">Desconectar</button>
                </div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Datos Historicos -->
            <div class="card">
                <h2 class="text-xl font-semibold mb-4">Datos Historicos</h2>
                <div class="grid grid-cols-3 gap-4 mb-4">
                    <div>
                        <label class="block text-sm font-medium mb-1">Simbolo</label>
                        <input type="text" id="hist-symbol" value="AAPL" class="input">
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-1">Duracion</label>
                        <select id="hist-duration" class="input">
                            <option value="1 D">1 Dia</option>
                            <option value="1 W">1 Semana</option>
                            <option value="1 M" selected>1 Mes</option>
                            <option value="3 M">3 Meses</option>
                            <option value="1 Y">1 Año</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-1">Intervalo</label>
                        <select id="hist-barsize" class="input">
                            <option value="1 min">1 min</option>
                            <option value="5 mins">5 mins</option>
                            <option value="1 hour">1 hora</option>
                            <option value="1 day" selected>1 dia</option>
                        </select>
                    </div>
                </div>
                <button onclick="getHistoricalData()" class="btn btn-primary mb-4">Obtener Datos</button>
                <div id="chart-container" style="height: 300px;">
                    <canvas id="price-chart"></canvas>
                </div>
            </div>

            <!-- Streaming -->
            <div class="card">
                <h2 class="text-xl font-semibold mb-4">Streaming en Tiempo Real</h2>
                <div class="flex gap-4 mb-4">
                    <input type="text" id="stream-symbol" placeholder="AAPL,MSFT,GOOGL" class="input flex-1">
                    <button onclick="subscribeStream()" class="btn btn-success">Suscribir</button>
                </div>
                <div id="stream-data" class="overflow-auto max-h-64">
                    <table class="w-full text-sm">
                        <thead class="bg-gray-100">
                            <tr>
                                <th class="p-2 text-left">Simbolo</th>
                                <th class="p-2 text-right">Ultimo</th>
                                <th class="p-2 text-right">Bid</th>
                                <th class="p-2 text-right">Ask</th>
                                <th class="p-2 text-right">Volumen</th>
                            </tr>
                        </thead>
                        <tbody id="stream-table"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Estrategias -->
        <div class="card mt-6">
            <h2 class="text-xl font-semibold mb-4">Estrategias</h2>
            <div class="grid grid-cols-1 md:grid-cols-6 gap-4 mb-4">
                <div>
                    <label class="block text-sm font-medium mb-1">Nombre</label>
                    <input type="text" id="strat-name" value="MA Cross" class="input">
                </div>
                <div>
                    <label class="block text-sm font-medium mb-1">Simbolos</label>
                    <input type="text" id="strat-symbols" value="AAPL" class="input">
                </div>
                <div>
                    <label class="block text-sm font-medium mb-1">MA Rapida</label>
                    <input type="number" id="strat-fast" value="10" class="input">
                </div>
                <div>
                    <label class="block text-sm font-medium mb-1">MA Lenta</label>
                    <input type="number" id="strat-slow" value="30" class="input">
                </div>
                <div>
                    <label class="block text-sm font-medium mb-1">Tipo MA</label>
                    <select id="strat-matype" class="input">
                        <option value="SMA">SMA</option>
                        <option value="EMA">EMA</option>
                        <option value="WMA">WMA</option>
                    </select>
                </div>
                <div class="flex items-end">
                    <button onclick="createStrategy()" class="btn btn-primary w-full">Crear</button>
                </div>
            </div>
            <div id="strategies-list" class="overflow-auto">
                <table class="w-full text-sm">
                    <thead class="bg-gray-100">
                        <tr>
                            <th class="p-2 text-left">Nombre</th>
                            <th class="p-2 text-left">Estado</th>
                            <th class="p-2 text-left">Simbolos</th>
                            <th class="p-2 text-right">Posiciones</th>
                            <th class="p-2 text-center">Acciones</th>
                        </tr>
                    </thead>
                    <tbody id="strategies-table"></tbody>
                </table>
            </div>
        </div>

        <!-- Logs -->
        <div class="card mt-6">
            <h2 class="text-xl font-semibold mb-4">Logs</h2>
            <div id="logs" class="bg-gray-900 text-green-400 p-4 rounded font-mono text-sm h-48 overflow-auto"></div>
        </div>
    </main>

    <script>
        let priceChart = null;
        let ws = null;
        const streamData = {};

        // Utilidades
        function log(message) {
            const logs = document.getElementById('logs');
            const time = new Date().toLocaleTimeString();
            logs.innerHTML += `[${time}] ${message}\\n`;
            logs.scrollTop = logs.scrollHeight;
        }

        async function api(endpoint, method = 'GET', data = null) {
            const options = {
                method,
                headers: { 'Content-Type': 'application/json' },
            };
            if (data) options.body = JSON.stringify(data);

            const response = await fetch(`/api${endpoint}`, options);
            return response.json();
        }

        // Conexion
        async function connect() {
            const host = document.getElementById('host').value;
            const port = parseInt(document.getElementById('port').value);
            const clientId = parseInt(document.getElementById('client-id').value);

            log('Conectando a IB...');
            try {
                const result = await api('/connect', 'POST', { host, port, client_id: clientId });
                if (result.success) {
                    log('Conectado exitosamente');
                    updateStatus(true);
                    connectWebSocket();
                }
            } catch (e) {
                log('Error: ' + e.message);
            }
        }

        async function disconnect() {
            log('Desconectando...');
            try {
                await api('/disconnect', 'POST');
                log('Desconectado');
                updateStatus(false);
                if (ws) ws.close();
            } catch (e) {
                log('Error: ' + e.message);
            }
        }

        function updateStatus(connected) {
            const indicator = document.getElementById('status-indicator');
            const text = document.getElementById('status-text');
            indicator.className = `w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`;
            text.textContent = connected ? 'Conectado' : 'Desconectado';
        }

        // WebSocket
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'tick') {
                    updateStreamTable(msg.data);
                }
            };
            ws.onclose = () => log('WebSocket desconectado');
        }

        // Datos historicos
        async function getHistoricalData() {
            const symbol = document.getElementById('hist-symbol').value;
            const duration = document.getElementById('hist-duration').value;
            const barSize = document.getElementById('hist-barsize').value;

            log(`Obteniendo datos de ${symbol}...`);
            try {
                const result = await api('/historical', 'POST', {
                    symbol,
                    duration,
                    bar_size: barSize,
                });
                log(`Recibidas ${result.count} barras`);
                updateChart(result.data);
            } catch (e) {
                log('Error: ' + e.message);
            }
        }

        function updateChart(data) {
            const ctx = document.getElementById('price-chart').getContext('2d');

            if (priceChart) priceChart.destroy();

            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.map(d => d.date ? d.date.split('T')[0] : ''),
                    datasets: [{
                        label: 'Precio',
                        data: data.map(d => d.close),
                        borderColor: 'rgb(59, 130, 246)',
                        tension: 0.1,
                        fill: false,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                }
            });
        }

        // Streaming
        async function subscribeStream() {
            const input = document.getElementById('stream-symbol').value;
            const symbols = input.split(',').map(s => s.trim().toUpperCase());

            log(`Suscribiendo a ${symbols.join(', ')}...`);
            try {
                const result = await api('/stream/subscribe', 'POST', { symbols });
                log('Suscripcion activa');
            } catch (e) {
                log('Error: ' + e.message);
            }
        }

        function updateStreamTable(tick) {
            streamData[tick.symbol] = tick;

            const tbody = document.getElementById('stream-table');
            tbody.innerHTML = '';

            for (const [symbol, data] of Object.entries(streamData)) {
                tbody.innerHTML += `
                    <tr class="border-b">
                        <td class="p-2 font-medium">${symbol}</td>
                        <td class="p-2 text-right">${data.last?.toFixed(2) || '-'}</td>
                        <td class="p-2 text-right">${data.bid?.toFixed(2) || '-'}</td>
                        <td class="p-2 text-right">${data.ask?.toFixed(2) || '-'}</td>
                        <td class="p-2 text-right">${data.volume?.toLocaleString() || '-'}</td>
                    </tr>
                `;
            }
        }

        // Estrategias
        async function createStrategy() {
            const name = document.getElementById('strat-name').value;
            const symbols = document.getElementById('strat-symbols').value.split(',').map(s => s.trim());
            const fastPeriod = parseInt(document.getElementById('strat-fast').value);
            const slowPeriod = parseInt(document.getElementById('strat-slow').value);
            const maType = document.getElementById('strat-matype').value;

            log(`Creando estrategia ${name}...`);
            try {
                await api('/strategy/create', 'POST', {
                    name,
                    symbols,
                    fast_period: fastPeriod,
                    slow_period: slowPeriod,
                    ma_type: maType,
                });
                log('Estrategia creada');
                loadStrategies();
            } catch (e) {
                log('Error: ' + e.message);
            }
        }

        async function loadStrategies() {
            try {
                const result = await api('/strategies');
                const tbody = document.getElementById('strategies-table');
                tbody.innerHTML = '';

                for (const strat of result.strategies) {
                    tbody.innerHTML += `
                        <tr class="border-b">
                            <td class="p-2 font-medium">${strat.name}</td>
                            <td class="p-2">
                                <span class="px-2 py-1 rounded text-xs ${strat.state === 'RUNNING' ? 'bg-green-100 text-green-800' : 'bg-gray-100'}">
                                    ${strat.state}
                                </span>
                            </td>
                            <td class="p-2">${strat.symbols.join(', ')}</td>
                            <td class="p-2 text-right">${strat.positions}</td>
                            <td class="p-2 text-center">
                                <button onclick="startStrategy('${strat.name}')" class="text-green-600 hover:underline mr-2">Iniciar</button>
                                <button onclick="stopStrategy('${strat.name}')" class="text-red-600 hover:underline">Detener</button>
                            </td>
                        </tr>
                    `;
                }
            } catch (e) {
                console.error(e);
            }
        }

        async function startStrategy(name) {
            await api(`/strategy/${name}/start`, 'POST');
            log(`Estrategia ${name} iniciada`);
            loadStrategies();
        }

        async function stopStrategy(name) {
            await api(`/strategy/${name}/stop`, 'POST');
            log(`Estrategia ${name} detenida`);
            loadStrategies();
        }

        // Inicializar
        async function init() {
            const status = await api('/status');
            updateStatus(status.connected);
            if (status.connected) connectWebSocket();
            loadStrategies();
            log('Dashboard iniciado');
        }

        init();
    </script>
</body>
</html>
"""


# Crear instancia de la app
app = create_app()
