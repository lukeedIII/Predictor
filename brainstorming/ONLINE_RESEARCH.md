# 🔍 Online Research — Phase 1 & 2 Implementation

> **Date:** 2026-02-13  
> **Purpose:** Technical reference collected before implementation  

---

## 1. Binance WebSocket API

### Stream URLs (Public — No API Key Required)
```
# Individual streams
wss://stream.binance.com:9443/ws/btcusdt@trade       ← individual trades
wss://stream.binance.com:9443/ws/btcusdt@ticker       ← 24h rolling stats
wss://stream.binance.com:9443/ws/btcusdt@kline_1m     ← 1-min candle updates

# Combined stream (what we use — fewer connections)
wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@ticker
```

### Trade Stream Message Format
```json
{
  "e": "trade",           // Event type
  "E": 1707840000000,     // Event time (ms)
  "s": "BTCUSDT",         // Symbol
  "t": 123456789,         // Trade ID
  "p": "67827.58",        // Price (STRING, must parse to float!)
  "q": "0.00045",         // Quantity
  "b": 88776655,          // Buyer order ID
  "a": 88776656,          // Seller order ID
  "T": 1707840000000,     // Trade time (ms)
  "m": true               // Is buyer the maker? true = SELL pressure
}
```

### 24h Ticker Stream Message Format
```json
{
  "e": "24hrTicker",
  "s": "BTCUSDT",
  "p": "-110.42",         // Price change (STRING)
  "P": "-0.16",           // Price change % (STRING)
  "w": "67200.50",        // Weighted avg price
  "o": "67938.00",        // Open price (24h ago)
  "h": "68028.45",        // 24h High
  "l": "65118.00",        // 24h Low
  "c": "67827.58",        // Last price
  "v": "23474.75",        // Volume in BTC
  "q": "1559344860.53",   // Volume in USDT
  "b": "67827.57",        // Best bid price
  "B": "1.234",           // Best bid qty
  "a": "67827.58",        // Best ask price
  "A": "0.567"            // Best ask qty
}
```

### Combined Stream Wrapping
When using combined streams, messages are wrapped:
```json
{
  "stream": "btcusdt@trade",
  "data": { ... actual trade data ... }
}
```

### Connection Rules
- **24h limit**: connections auto-disconnect after 24 hours → must auto-reconnect
- **Ping/Pong**: server sends ping every 20 seconds, client must respond with pong
- **Rate limit**: max 5 incoming messages/second (outbound only, not a concern for us)
- **Max streams per connection**: 1024 (we only use 2)
- **All prices are STRINGS** — must parse with `float()`

---

## 2. Python WebSocket Libraries

### Decision: `websocket-client` (sync/threaded)
Our backend runs sync code in threads (not asyncio). `websocket-client` is the best fit:

```python
# Install
pip install websocket-client

# Usage — WebSocketApp with callbacks
import websocket

def on_message(ws, message):
    data = json.loads(message)
    process(data)

ws = websocket.WebSocketApp(
    "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@ticker",
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)
ws.run_forever(ping_interval=20, ping_timeout=10)
```

**Key features:**
- Built-in ping/pong handling via `ping_interval`
- `run_forever()` blocks — perfect for our daemon thread
- Automatic reconnect logic available
- Does NOT require asyncio

**Add to requirements.txt:**
```
websocket-client
```

### Alternative considered: `websockets` (async)
Not used because our api_server.py uses threading, not asyncio event loops for background tasks.

---

## 3. FastAPI WebSocket Endpoint (Inbound — Frontend ← Backend)

FastAPI has **built-in WebSocket** support. No extra packages needed:

```python
from fastapi import WebSocket, WebSocketDisconnect

# Connection manager for multiple frontend clients
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        for ws in list(self.active):
            try:
                await ws.send_json(data)
            except Exception:
                self.active.remove(ws)

manager = ConnectionManager()

@app.websocket("/ws/live")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # Keep connection alive — wait for client messages (or just sleep)
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
```

### Push Pattern (from background thread):
```python
import asyncio

def _push_to_frontend(data):
    """Called from sync BinanceWS thread → push to async WS clients."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.run_coroutine_threadsafe(manager.broadcast(data), loop)
```

**⚠️ GOTCHA:** FastAPI WebSocket `broadcast()` is async. If called from a sync/threaded context (like our BinanceWS daemon thread), we need `asyncio.run_coroutine_threadsafe()` to safely bridge between sync and async.

---

## 4. TradingView Lightweight Charts v5

### ⚠️ CRITICAL: v5 API Changes from v3
The v5 API is significantly different from v3 tutorials online:

```typescript
// ❌ OLD v3 API (most tutorials show this)
chart.addCandlestickSeries()

// ✅ NEW v5 API (what we must use)
import { CandlestickSeries, createChart } from 'lightweight-charts';
const series = chart.addSeries(CandlestickSeries);
```

### Chart Creation (React + TypeScript)
```typescript
import { createChart, CandlestickSeries, HistogramSeries, LineSeries } from 'lightweight-charts';

const chartContainerRef = useRef<HTMLDivElement>(null);
const chartRef = useRef<ReturnType<typeof createChart>>();

useEffect(() => {
  if (!chartContainerRef.current) return;
  
  const chart = createChart(chartContainerRef.current, {
    width: chartContainerRef.current.clientWidth,
    height: 400,
    layout: {
      background: { color: '#0A0E17' },
      textColor: '#9CA3AF',
    },
    grid: {
      vertLines: { color: '#1F2937' },
      horzLines: { color: '#1F2937' },
    },
    crosshair: {
      mode: 0, // CrosshairMode.Normal
    },
    timeScale: {
      borderColor: '#1F2937',
      timeVisible: true,
      secondsVisible: false,
    },
  });

  const candleSeries = chart.addSeries(CandlestickSeries, {
    upColor: '#10B981',
    downColor: '#EF4444',
    borderDownColor: '#EF4444',
    borderUpColor: '#10B981',
    wickDownColor: '#EF4444',
    wickUpColor: '#10B981',
  });

  // Volume histogram
  const volumeSeries = chart.addSeries(HistogramSeries, {
    priceFormat: { type: 'volume' },
    priceScaleId: 'volume',
  });
  chart.priceScale('volume').applyOptions({
    scaleMargins: { top: 0.8, bottom: 0 },
  });

  // MA overlays
  const ma7 = chart.addSeries(LineSeries, {
    color: '#F59E0B', lineWidth: 1, priceLineVisible: false,
    lastValueVisible: false,
  });
  
  chartRef.current = chart;
  return () => chart.remove();
}, []);
```

### Real-Time Update (THE KEY METHOD)
```typescript
// Update existing candle OR add new one
candleSeries.update({
  time: 1707840060 as UTCTimestamp,  // UNIX timestamp (seconds)
  open: 67800.5,
  high: 67850.2,
  low: 67790.1,
  close: 67827.58,
});

// ⚠️ IMPORTANT: time format is UNIX SECONDS (not milliseconds!)
// Binance gives ms → divide by 1000

// ⚠️ IMPORTANT: Do NOT call setData() for real-time updates
// Use update() — it's 100x more performant
```

### Time Format
```typescript
// lightweight-charts expects UNIX timestamp in SECONDS (not ms)
// Convert from Binance (ms) → lightweight-charts (s):
const time = Math.floor(binanceTimestamp / 1000) as UTCTimestamp;

// Or ISO date string: '2024-01-15'
// Or business day: { year: 2024, month: 1, day: 15 }
```

### Responsive Resize
```typescript
useEffect(() => {
  const observer = new ResizeObserver(entries => {
    const { width, height } = entries[0].contentRect;
    chartRef.current?.applyOptions({ width, height });
  });
  observer.observe(chartContainerRef.current!);
  return () => observer.disconnect();
}, []);
```

---

## 5. Dependencies to Add

### Python (requirements.txt)
```
websocket-client          # Binance WebSocket client (outbound)
# FastAPI already has built-in WebSocket (inbound) — no extra package needed
```

### Node (already installed)
```
lightweight-charts: ^5.1.0   ← already in package.json ✅
```

---

## 6. Architecture Summary

```
Binance Cloud                    Our Backend                      React Frontend
═══════════════                  ══════════════                   ═══════════════

wss://stream.binance.com         Python Thread                    Browser
  ┌─ btcusdt@trade ──────────→  BinanceWSClient  ──→ snapshot    
  └─ btcusdt@ticker ─────────→  (websocket-client)   (dict)     
                                     │                            
                                     ▼ (1/sec push)              
                                FastAPI /ws/live  ═══════════→  useWebSocket()
                                (built-in WebSocket)              React state
                                     │                               │
                                     │                               ▼
                                GET /api/live-price              StatusBar (UTC clock + price)
                                (REST fallback)                  TradingViewChart (candles)
                                                                 Dashboard (price header)
```

---

## 7. Gotchas & Warnings

1. **All Binance prices are STRINGS** — always `float(data["p"])`, never assume numeric
2. **Binance timestamps are MILLISECONDS** — divide by 1000 for lightweight-charts (seconds)
3. **lightweight-charts v5 uses `addSeries(CandlestickSeries)`** — NOT `addCandlestickSeries()`
4. **`series.update()` for real-time** — NEVER call `setData()` in a real-time loop
5. **24h WS disconnect** — Binance auto-disconnects after 24h, must auto-reconnect
6. **Thread-to-async bridge** — Use `asyncio.run_coroutine_threadsafe()` to push from sync BinanceWS thread to async FastAPI WebSocket broadcast
7. **FastAPI WebSocket needs ASGI** — Already running uvicorn (ASGI), so this works natively
