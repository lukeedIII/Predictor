"""
Nexus Shadow-Quant — FastAPI Backend
======================================
Local REST API serving all predictor/trader data to the Electron + React desktop UI.
Runs as a child process spawned by Electron's main process.
"""

import os
import sys
import json
import logging
import threading
import time
import traceback
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import asyncio
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import requests
import config
from predictor import NexusPredictor
from paper_trader import PaperTrader
from data_collector import DataCollector
from math_core import MathCore
import nexus_agent
from binance_ws import BinanceWSClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
#  GLOBAL STATE
# ============================================================

predictor: Optional[NexusPredictor] = None
trader: Optional[PaperTrader] = None
collector: Optional[DataCollector] = None
math_core: Optional[MathCore] = None
binance_ws: Optional[BinanceWSClient] = None

boot_status = {"stage": "starting", "progress": 0, "message": "Initializing..."}
_auto_trade_thread: Optional[threading.Thread] = None
_auto_trade_stop = threading.Event()
_event_loop: Optional[asyncio.AbstractEventLoop] = None  # for thread→async bridge

# Auto-retrain scheduler state
RETRAIN_INTERVAL_HOURS = 6
_retrain_stop = threading.Event()
_retrain_status = {
    "last_retrain": None,
    "next_retrain": None,
    "last_accuracy": None,
    "retrain_count": 0,
    "is_retraining": False,
    "last_error": None,
}


def _check_system_requirements():
    """Validate GPU and disk space before launching engines.
    Returns (ok: bool, error_message: str | None)"""
    import shutil
    
    # ── GPU Check: require NVIDIA GPU with compute capability >= 8.6 (RTX 3060+) ──
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "NVIDIA GPU not detected. Nexus requires an NVIDIA GPU (RTX 3060 or higher) with CUDA support."
        
        gpu_name = torch.cuda.get_device_name(0)
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        compute_cap = cc_major + cc_minor / 10
        
        # RTX 3060 = compute capability 8.6, RTX 4060 = 8.9, RTX 5080 = 12.0
        if compute_cap < 8.6:
            return False, f"GPU '{gpu_name}' (compute {cc_major}.{cc_minor}) is below minimum. Requires RTX 3060 or higher (compute 8.6+)."
        
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU check passed: {gpu_name} (compute {cc_major}.{cc_minor}, {vram_gb:.1f} GB VRAM)")
        
    except ImportError:
        return False, "PyTorch with CUDA not installed. Cannot detect GPU."
    except Exception as e:
        return False, f"GPU detection failed: {str(e)[:200]}"
    
    # ── Disk Space Check: require 20 GB free ──
    MIN_FREE_GB = 20
    try:
        disk = shutil.disk_usage(config.DATA_DIR)
        free_gb = disk.free / (1024**3)
        if free_gb < MIN_FREE_GB:
            return False, f"Insufficient disk space: {free_gb:.1f} GB free, need {MIN_FREE_GB} GB minimum."
        logging.info(f"Disk check passed: {free_gb:.1f} GB free")
    except Exception as e:
        logging.warning(f"Disk check skipped: {e}")
    
    return True, None


def _init_engines():
    """Initialize all engines in order, updating boot status."""
    global predictor, trader, collector, math_core, binance_ws, boot_status
    
    try:
        # ── System Requirements Check ──
        boot_status = {"stage": "syscheck", "progress": 5, "message": "Checking system requirements..."}
        ok, err = _check_system_requirements()
        if not ok:
            boot_status = {"stage": "error", "progress": -1, "message": f"⛔ {err}"}
            logging.error(f"System check FAILED: {err}")
            return
        
        boot_status = {"stage": "math", "progress": 10, "message": "Loading math engine..."}
        math_core = MathCore()
        
        boot_status = {"stage": "collector", "progress": 20, "message": "Connecting to Binance..."}
        collector = DataCollector()
        
        # Start real-time WebSocket feed (non-blocking)
        boot_status = {"stage": "websocket", "progress": 30, "message": "Starting live price feed..."}
        binance_ws = BinanceWSClient(on_price_update=_on_binance_price)
        binance_ws.start()
        
        boot_status = {"stage": "data", "progress": 40, "message": "Loading market data..."}
        collector.collect_and_save(limit=100)
        
        boot_status = {"stage": "predictor", "progress": 55, "message": "Initializing AI predictor..."}
        predictor = NexusPredictor()
        
        boot_status = {"stage": "training", "progress": 70, "message": "Checking model status..."}
        if not predictor.is_trained:
            boot_status["message"] = "Training AI model (first run)..."
            predictor.train()
        
        boot_status = {"stage": "trader", "progress": 85, "message": "Starting paper trader..."}
        trader = PaperTrader()
        
        boot_status = {"stage": "ready", "progress": 100, "message": "All systems online"}
        logging.info("All engines initialized successfully")
        
    except Exception as e:
        boot_status = {"stage": "error", "progress": -1, "message": f"Init error: {str(e)[:200]}"}
        logging.error(f"Engine init failed: {e}")
        traceback.print_exc()


def _auto_retrain_loop():
    """Background loop: retrains the model every RETRAIN_INTERVAL_HOURS."""
    import json as _json
    global _retrain_status
    
    # Wait for engines to be ready first
    while not _retrain_stop.is_set() and boot_status.get('stage') != 'ready':
        _retrain_stop.wait(5)
    
    retrain_history_path = os.path.join(config.LOG_DIR, 'retrain_history.json')
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    while not _retrain_stop.is_set():
        next_time = datetime.now() + timedelta(hours=RETRAIN_INTERVAL_HOURS)
        _retrain_status['next_retrain'] = next_time.isoformat()
        
        # Sleep until next retrain (check every 30s for shutdown)
        for _ in range(RETRAIN_INTERVAL_HOURS * 120):
            if _retrain_stop.is_set():
                return
            _retrain_stop.wait(30)
        
        if predictor is None or _retrain_stop.is_set():
            continue
        
        try:
            _retrain_status['is_retraining'] = True
            logging.info("[AUTO-RETRAIN] Starting scheduled retrain...")
            
            result = predictor.train()
            now = datetime.now().isoformat()
            acc = predictor.last_validation_accuracy if hasattr(predictor, 'last_validation_accuracy') else None
            
            _retrain_status.update({
                'last_retrain': now,
                'last_accuracy': acc,
                'retrain_count': _retrain_status['retrain_count'] + 1,
                'is_retraining': False,
                'last_error': None,
            })
            
            # Append to history log
            entry = {'timestamp': now, 'accuracy': acc, 'result': str(result)}
            history = []
            if os.path.exists(retrain_history_path):
                try:
                    with open(retrain_history_path, 'r') as f:
                        history = _json.load(f)
                except Exception:
                    history = []
            history.append(entry)
            # Keep last 100 retrains
            history = history[-100:]
            with open(retrain_history_path, 'w') as f:
                _json.dump(history, f, indent=2)
            
            logging.info(f"[AUTO-RETRAIN] Complete. Accuracy: {acc:.1f}%")
            
        except Exception as e:
            _retrain_status['is_retraining'] = False
            _retrain_status['last_error'] = str(e)[:200]
            logging.error(f"[AUTO-RETRAIN] Failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start engine init + auto-retrain scheduler + WS push on startup."""
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    
    init_thread = threading.Thread(target=_init_engines, daemon=True)
    init_thread.start()
    retrain_thread = threading.Thread(target=_auto_retrain_loop, daemon=True)
    retrain_thread.start()
    # Start periodic WS push to frontend clients
    push_task = asyncio.create_task(_ws_push_loop())
    yield
    _auto_trade_stop.set()
    _retrain_stop.set()
    push_task.cancel()
    if binance_ws:
        binance_ws.stop()


# ============================================================
#  WEBSOCKET — Real-time push to frontend
# ============================================================

class _WSConnectionManager:
    """Manages WebSocket connections to frontend clients."""
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logging.info(f"WS client connected ({len(self.active)} total)")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logging.info(f"WS client disconnected ({len(self.active)} total)")

    async def broadcast(self, data: dict):
        """Send data to all connected frontend clients."""
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

_ws_manager = _WSConnectionManager()


def _on_binance_price(price: float):
    """Callback from BinanceWSClient — bridge sync thread to async broadcast."""
    # We don't broadcast on every tick (too fast), the push loop handles it
    pass


async def _ws_push_loop():
    """Push live data to frontend WebSocket clients every ~1 second."""
    while True:
        try:
            if _ws_manager.active and binance_ws:
                snapshot = binance_ws.snapshot
                payload = {
                    "type": "state_update",
                    "price": snapshot.get("price"),
                    "change_24h": snapshot.get("change_24h"),
                    "change_pct": snapshot.get("change_pct"),
                    "high_24h": snapshot.get("high_24h"),
                    "low_24h": snapshot.get("low_24h"),
                    "volume_btc": snapshot.get("volume_btc"),
                    "volume_usdt": snapshot.get("volume_usdt"),
                    "ws_connected": snapshot.get("ws_connected", False),
                    "timestamp": time.time(),
                }
                # Enrich with prediction/bot data if available
                if predictor and predictor.is_trained:
                    try:
                        pred = predictor.get_prediction()
                        payload["prediction"] = _sanitize_for_json(pred)
                        payload["accuracy"] = predictor.last_validation_accuracy
                        payload["quant"] = _sanitize_for_json(predictor.last_quant_analysis or {})
                        payload["alt_signals"] = _sanitize_for_json(predictor.last_alt_signals or {})
                    except Exception:
                        pass
                if trader:
                    payload["bot_running"] = trader.is_running
                    payload["positions"] = _sanitize_for_json([
                        pos.to_dict() for pos in trader.positions
                    ]) if trader.positions else []
                    try:
                        payload["stats"] = _sanitize_for_json(trader.get_stats())
                    except Exception:
                        pass
                
                await _ws_manager.broadcast(payload)
        except Exception as e:
            logging.debug(f"WS push error: {e}")
        await asyncio.sleep(1)


# ============================================================
#  APP
# ============================================================

app = FastAPI(title="Nexus Shadow-Quant API", version=config.VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#  MODELS
# ============================================================

class TradeRequest(BaseModel):
    direction: str  # "LONG" or "SHORT"
    
class CloseRequest(BaseModel):
    position_index: int = 0
    reason: str = "MANUAL"


# ============================================================
#  ENDPOINTS
# ============================================================

@app.get("/api/boot-status")
def get_boot_status():
    """Real-time boot progress for the splash screen."""
    return boot_status


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """Real-time data push to frontend via WebSocket.
    
    The actual data is pushed by _ws_push_loop() every ~1 second.
    This endpoint just manages the connection lifecycle.
    """
    await _ws_manager.connect(ws)
    try:
        while True:
            # Keep connection alive — wait for client pings/messages
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_manager.disconnect(ws)
    except Exception:
        _ws_manager.disconnect(ws)


@app.get("/api/live-price")
def get_live_price():
    """REST fallback for live price + 24h stats.
    
    Prefer the WebSocket /ws/live for real-time updates.
    This endpoint is for one-shot queries or fallback.
    """
    if binance_ws and binance_ws.connected:
        return binance_ws.snapshot
    
    # Fallback to REST-based price
    price = _get_live_price()
    return {
        "price": price,
        "ws_connected": False,
        "source": "rest_fallback",
    }


@app.get("/api/status")
def get_status():
    """System health + model status."""
    if predictor is None:
        return {"ready": False, "boot": boot_status}
    
    return {
        "ready": True,
        "version": config.VERSION,
        "device": predictor.device,
        "model_trained": predictor.is_trained,
        "verified": predictor.is_statistically_verified,
        "validation_accuracy": predictor.last_validation_accuracy,
        "xgb_weight": predictor.xgb_weight,
        "lstm_weight": predictor.lstm_weight,
        "bot_running": trader.is_running if trader else False,
        "positions_count": len(trader.positions) if trader else 0,
    }

def _sanitize_for_json(obj):
    """Deep-convert numpy/torch/etc. types to JSON-safe Python types.
    Also replaces NaN/Infinity with None (null) — NaN breaks browser JSON.parse()."""
    import math
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, (np.integer, np.floating)) else k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(v) for v in obj.tolist()]
    if hasattr(obj, 'item'):  # other numeric scalars
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    return obj


@app.get("/api/prediction")
def get_prediction():
    """Current prediction + quant analysis."""
    if predictor is None:
        raise HTTPException(503, "Engine not ready")
    
    pred = predictor.get_prediction()
    quant = predictor.last_quant_analysis or {}
    alt = predictor.last_alt_signals or {}
    
    data = _sanitize_for_json({
        "prediction": pred,
        "quant": quant,
        "alt_signals": alt,
        "accuracy": predictor.last_validation_accuracy,
    })
    # _sanitize_for_json already converts NaN/Infinity → None recursively.
    return JSONResponse(content=data)


# ===== Phase 3: Feature Importance =====
@app.get("/api/feature-importance")
def get_feature_importance():
    """Sorted XGBoost feature importances with category labels."""
    if predictor is None or not predictor.is_trained:
        raise HTTPException(503, "Model not trained yet")
    
    # Category mapping for UI color-coding
    categories = {
        'close_ret_1': 'ohlcv', 'close_ret_5': 'ohlcv', 'close_ret_15': 'ohlcv',
        'high_low_range': 'ohlcv', 'close_open_range': 'ohlcv', 'volume_ratio': 'ohlcv',
        'sma_20_dist': 'technical', 'sma_50_dist': 'technical', 'rsi': 'technical',
        'volatility': 'technical', 'cycle_1': 'technical', 'cycle_2': 'technical',
        'hurst': 'microstructure', 'kalman_err_norm': 'microstructure', 'obi_sim': 'microstructure',
        'regime_id': 'quant', 'regime_confidence': 'quant', 'gjr_volatility': 'quant',
        'hawkes_intensity': 'quant', 'wass_drift': 'quant',
        'trend_5m': 'trend', 'trend_15m': 'trend', 'trend_1h': 'trend',
        'ret_60': 'trend', 'ret_240': 'trend', 'vol_regime': 'trend',
        'vwap_dist': 'volume', 'vol_momentum': 'volume',
        'eth_ret_5': 'cross_asset', 'eth_ret_15': 'cross_asset', 'eth_vol_ratio': 'cross_asset',
        'ethbtc_ret_5': 'cross_asset', 'ethbtc_trend': 'cross_asset',
        'gold_ret_15': 'cross_asset', 'gold_ret_60': 'cross_asset',
        'trade_intensity': 'microstructure', 'buy_sell_ratio': 'microstructure',
        'vwap_momentum': 'microstructure', 'tick_volatility': 'microstructure',
        'large_trade_ratio': 'microstructure',
    }
    
    importances = predictor.model.feature_importances_
    features = predictor.features
    
    result = sorted(
        [{"name": f, "importance": round(float(v), 4), "category": categories.get(f, "other")}
         for f, v in zip(features, importances)],
        key=lambda x: x["importance"], reverse=True
    )
    return {"features": result, "total": len(result)}


# ===== Phase 4: System Health =====
@app.get("/api/system-health")
def get_system_health():
    """System health: GPU, model age, data size, disk usage."""
    import torch
    
    health = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only",
        "gpu_vram_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if torch.cuda.is_available() else 0,
        "gpu_vram_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2) if torch.cuda.is_available() else 0,
        "model_trained": predictor.is_trained if predictor else False,
        "model_version": "v6.0",
        "feature_count": len(predictor.features) if predictor else 0,
        "validation_accuracy": predictor.last_validation_accuracy if predictor else 0,
        "ensemble_weights": {
            "xgb": predictor.xgb_weight if predictor else 1.0,
            "lstm": predictor.lstm_weight if predictor else 0.0,
        },
    }
    
    # Model file age
    if predictor and os.path.exists(predictor.model_path):
        model_mtime = os.path.getmtime(predictor.model_path)
        age_hours = (time.time() - model_mtime) / 3600
        health["model_age_hours"] = round(age_hours, 1)
        health["model_last_trained"] = datetime.fromtimestamp(model_mtime).isoformat()
    
    # Data file size
    if os.path.exists(config.MARKET_DATA_PARQUET_PATH):
        size_mb = os.path.getsize(config.MARKET_DATA_PARQUET_PATH) / (1024 * 1024)
        health["data_size_mb"] = round(size_mb, 1)
    elif os.path.exists(config.MARKET_DATA_PATH):
        size_mb = os.path.getsize(config.MARKET_DATA_PATH) / (1024 * 1024)
        health["data_size_mb"] = round(size_mb, 1)
    
    return health


# ===== Phase 4: CSV Export =====
@app.get("/api/export/trades")
def export_trades():
    """Export trade history as CSV download."""
    if trader is None:
        raise HTTPException(503, "Paper trader not available")
    
    import io
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "type", "direction", "entry_price", "exit_price", "pnl_pct", "size_btc"])
    
    history = trader.trade_history if hasattr(trader, 'trade_history') else []
    for t in history:
        writer.writerow([
            t.get("timestamp", ""),
            t.get("type", ""),
            t.get("direction", ""),
            t.get("entry_price", ""),
            t.get("exit_price", ""),
            t.get("pnl_pct", ""),
            t.get("size_btc", ""),
        ])
    
    from starlette.responses import StreamingResponse
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=nexus_trades_{datetime.now().strftime('%Y%m%d')}.csv"}
    )


@app.get("/api/market-data")
def get_market_data(limit: int = 200, interval: str = "1m"):
    """OHLCV data for price chart with candle aggregation.
    
    interval: 1m, 5m, 15m, 1h, 4h
    limit: number of aggregated candles to return
    """
    if predictor is None:
        raise HTTPException(503, "Engine not ready")
    
    # Map interval to minutes
    interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    interval_mins = interval_map.get(interval, 1)
    
    # Calculate how far back we need to go (in minutes) + 10% buffer
    lookback_minutes = int(limit * interval_mins * 1.1) + interval_mins
    
    # Load the FULL dataset and filter by date (avoids gap issues)
    try:
        if os.path.exists(config.MARKET_DATA_PARQUET_PATH):
            full_df = pd.read_parquet(config.MARKET_DATA_PARQUET_PATH)
        elif os.path.exists(config.MARKET_DATA_PATH):
            full_df = pd.read_csv(config.MARKET_DATA_PATH)
        else:
            return {"candles": [], "count": 0}
    except Exception:
        return {"candles": [], "count": 0}
    
    if full_df is None or full_df.empty:
        return {"candles": [], "count": 0}
    
    # Filter by date range (not row count!)
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    cutoff = full_df['timestamp'].max() - pd.Timedelta(minutes=lookback_minutes)
    df = full_df[full_df['timestamp'] >= cutoff].copy()
    
    if df.empty:
        return {"candles": [], "count": 0}
    
    if interval_mins > 1:
        # Aggregate 1-minute candles into larger intervals
        df = df.set_index('timestamp')
        agg = df.resample(f'{interval_mins}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(subset=['open'])
        agg = agg.reset_index()
        df = agg
    
    # Take last `limit` candles
    df = df.tail(limit).reset_index(drop=True)
    
    # Convert to JSON-friendly format — send UNIX ms timestamps for chart
    df['timestamp'] = (pd.to_datetime(df['timestamp']).astype('int64') // 10**6).astype('int64')
    candles = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
    
    return {"candles": candles, "count": len(candles)}


@app.get("/api/stats")
def get_stats():
    """Paper trader performance stats."""
    if trader is None:
        return {
            "balance": config.PAPER_STARTING_BALANCE,
            "starting_balance": config.PAPER_STARTING_BALANCE,
            "total_pnl": 0, "total_pnl_pct": 0,
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0,
            "profit_factor": 0, "kelly_fraction": 0.01,
            "unrealized_pnl": 0, "circuit_breaker": False,
            "position_open": False, "positions_count": 0, "leverage": 10
        }
    try:
        return _sanitize_for_json(trader.get_stats())
    except Exception as e:
        logging.error(f"Stats error: {e}")
        return {"balance": trader.balance, "starting_balance": trader.starting_balance,
                "total_pnl": 0, "total_pnl_pct": 0, "total_trades": 0, "winning_trades": 0,
                "losing_trades": 0, "win_rate": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0,
                "profit_factor": 0, "kelly_fraction": 0.01}


_price_cache = {"price": None, "ts": 0}

def _get_live_price() -> float:
    """Get live BTC price — WebSocket first (instant), REST fallback."""
    # 1. BEST: Real-time from Binance WebSocket (sub-second)
    if binance_ws and binance_ws.price > 0:
        return binance_ws.price
    
    now = time.time()
    if _price_cache["price"] and (now - _price_cache["ts"]) < 3:
        return _price_cache["price"]
    
    # 2. REST fallback: collector's exchange
    try:
        if collector and hasattr(collector, 'exchange'):
            ticker = collector.exchange.fetch_ticker("BTC/USDT")
            p = ticker.get("last")
            if p:
                _price_cache["price"] = p
                _price_cache["ts"] = now
                return p
    except Exception:
        pass
    
    # 3. Fallback: latest candle from disk
    try:
        df = pd.read_parquet(config.DATA_PATH)
        if len(df) > 0:
            p = float(df['close'].iloc[-1])
            _price_cache["price"] = p
            _price_cache["ts"] = now
            return p
    except Exception:
        pass
    
    return _price_cache.get("price")


@app.get("/api/positions")
def get_positions():
    """Open positions."""
    if trader is None:
        return {"positions": []}
    
    # Get LIVE price — cached with 3s TTL
    price = _get_live_price() or getattr(trader, '_last_price', None)
    
    positions = []
    for i, pos in enumerate(trader.positions):
        p = price or pos.entry_price
        positions.append({
            "index": i,
            **pos.to_dict(),
            "unrealized_pnl": round(pos.unrealized_pnl(p), 2),
            "unrealized_pnl_pct": round(pos.unrealized_pnl_pct(p), 2),
            "elapsed_seconds": (datetime.now() - pos.entry_time).total_seconds(),
        })
    
    return {"positions": positions, "current_price": price}


@app.get("/api/trade-history")
def get_trade_history(limit: int = 50):
    """Closed trades."""
    if trader is None:
        return {"trades": [], "total": 0}
    
    trades = trader.trade_history[-limit:]
    return {"trades": trades, "total": len(trader.trade_history)}


@app.get("/api/equity-history")
def get_equity_history():
    """Equity curve data."""
    if trader is None:
        return {"points": []}
    
    # Build from trade_history list (list of dicts)
    if not trader.trade_history:
        return {"points": []}
    
    points = []
    for rec in trader.trade_history:
        points.append({
            "timestamp": str(rec.get('timestamp_close', rec.get('timestamp', ''))),
            "balance": rec.get('balance_after', trader.starting_balance),
        })
    
    return {"points": points}


@app.get("/api/news")
def get_news():
    """News feed."""
    try:
        from twitter_scraper import CryptoNewsScraper
        scraper = CryptoNewsScraper()
        items = scraper.fetch_news("BTC", limit=10)
        return {"items": items or []}
    except Exception as e:
        return {"items": [], "error": str(e)}


@app.get("/api/cycles")
def get_cycles():
    """FFT market cycles for the HUD."""
    if math_core is None or predictor is None:
        return {"cycles": []}
    
    try:
        df = predictor._load_market_data_tail(n=200)
        if df is None or df.empty:
            return {"cycles": []}
        
        prices = df['close'].tail(100).values
        cycles = math_core.extract_cycles(prices, top_n=3)
        labels = ['Short', 'Mid', 'Long']
        
        return {"cycles": [{"label": l, "strength": round(c, 4)} for l, c in zip(labels, cycles)]}
    except Exception:
        return {"cycles": []}


# ─── TRADING ACTIONS ─────────────────────────────────

@app.post("/api/trade/open")
def open_trade(req: TradeRequest):
    """Open a manual position."""
    if trader is None or predictor is None:
        raise HTTPException(503, "Engine not ready")
    
    # Get current price
    try:
        price_data = collector.fetch_ohlcv(limit=2)
        current_price = float(price_data['close'].iloc[-1])
    except Exception:
        raise HTTPException(500, "Could not fetch current price")
    
    pred = predictor.get_prediction() if predictor.is_trained else {
        'confidence': 100, 'direction': req.direction.replace("LONG", "UP").replace("SHORT", "DOWN"),
        'hurst': 0.6, 'regime_label': 'MANUAL'
    }
    
    success = trader.open_position(req.direction, current_price, pred, volatility=0.005)
    trader._last_price = current_price
    
    if not success:
        raise HTTPException(400, "Could not open position (max positions or insufficient margin)")
    
    return {"success": True, "price": current_price}


@app.post("/api/trade/close")
def close_trade(req: CloseRequest):
    """Close a position."""
    if trader is None:
        raise HTTPException(503, "Engine not ready")
    
    if req.position_index >= len(trader.positions):
        raise HTTPException(404, "Position not found")
    
    try:
        price_data = collector.fetch_ohlcv(limit=2)
        current_price = float(price_data['close'].iloc[-1])
    except Exception:
        raise HTTPException(500, "Could not fetch current price")
    
    pos = trader.positions[req.position_index]
    result = trader.close_position(current_price, req.reason, pos)
    
    return {"success": True, "trade": result}


@app.post("/api/trade/close-all")
def close_all_trades():
    """Close all open positions."""
    if trader is None:
        raise HTTPException(503, "Engine not ready")
    
    try:
        price_data = collector.fetch_ohlcv(limit=2)
        current_price = float(price_data['close'].iloc[-1])
    except Exception:
        raise HTTPException(500, "Could not fetch current price")
    
    results = []
    for pos in list(trader.positions):
        r = trader.close_position(current_price, "MANUAL", pos)
        results.append(r)
    
    return {"success": True, "closed": len(results)}


@app.post("/api/bot/start")
def start_bot():
    """Start auto-trading loop."""
    global _auto_trade_thread
    if trader is None or predictor is None:
        raise HTTPException(503, "Engine not ready")
    
    if trader.is_running:
        return {"running": True, "message": "Already running"}
    
    trader.is_running = True
    _auto_trade_stop.clear()
    
    def _trade_loop():
        while not _auto_trade_stop.is_set() and trader.is_running:
            try:
                # Update data
                collector.collect_and_save(limit=5)
                
                # Get price
                price_data = collector.fetch_ohlcv(limit=2)
                if price_data is None or price_data.empty:
                    time.sleep(30)
                    continue
                current_price = float(price_data['close'].iloc[-1])
                trader._last_price = current_price
                
                # Get prediction for signal evaluation
                pred = predictor.get_prediction() if predictor.is_trained else None
                vol = float(pred.get('volatility', 0.005)) if pred else 0.005
                
                # update() handles: TP/SL/liquidation checks + new signal evaluation
                trader.update(current_price, pred, vol)
                
                time.sleep(config.UPDATE_INTERVAL_SEC)
                
            except Exception as e:
                logging.error(f"Auto-trade error: {e}")
                time.sleep(30)
    
    _auto_trade_thread = threading.Thread(target=_trade_loop, daemon=True)
    _auto_trade_thread.start()
    
    return {"running": True}


@app.post("/api/bot/stop")
def stop_bot():
    """Stop auto-trading."""
    if trader is None:
        raise HTTPException(503, "Engine not ready")
    
    trader.is_running = False
    _auto_trade_stop.set()
    
    return {"running": False}


@app.post("/api/train")
def trigger_training():
    """Trigger model retraining."""
    if predictor is None:
        raise HTTPException(503, "Engine not ready")
    
    def _train_bg():
        try:
            predictor.train()
            logging.info("Background training complete")
        except Exception as e:
            logging.error(f"Training failed: {e}")
    
    thread = threading.Thread(target=_train_bg, daemon=True)
    thread.start()
    
    return {"training": True, "message": "Training started in background"}


@app.get("/api/retrain-status")
def get_retrain_status():
    """Auto-retrain scheduler status."""
    return _retrain_status


# ============================================================
#  SETTINGS & API KEY MANAGEMENT
# ============================================================

def _mask_key(key: str) -> str:
    """Mask an API key for display: sk-****1234"""
    if not key or len(key) < 8:
        return ""
    return key[:4] + "****" + key[-4:]


def _load_env_keys() -> dict:
    """Load current .env keys."""
    env_path = os.path.join(config.DATA_ROOT, ".env")
    keys = {
        "GEMINI_API_KEY": "",
        "OPENAI_API_KEY": "",
        "BINANCE_API_KEY": "",
        "BINANCE_SECRET_KEY": "",
    }
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k in keys:
                        keys[k] = v
    return keys


def _save_env_keys(keys: dict):
    """Save keys to .env file."""
    env_path = os.path.join(config.DATA_ROOT, ".env")
    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    
    # Preserve any extra keys in existing .env
    existing = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    existing[k.strip()] = v.strip()
    
    existing.update(keys)
    
    with open(env_path, "w") as f:
        f.write("# Nexus Shadow-Quant — API Keys\n")
        f.write("# Auto-generated by Settings page\n\n")
        for k, v in existing.items():
            f.write(f'{k}="{v}"\n')
    
    # Reload into os.environ
    for k, v in keys.items():
        os.environ[k] = v


@app.get("/api/settings")
def get_settings():
    """Return current settings with masked API keys."""
    keys = _load_env_keys()
    return {
        "keys": {k: _mask_key(v) for k, v in keys.items()},
        "has_keys": {k: bool(v) for k, v in keys.items()},
        "data_root": config.DATA_ROOT,
        "version": config.VERSION,
        "is_installed": config.IS_INSTALLED,
    }


class SettingsUpdate(BaseModel):
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_secret_key: Optional[str] = None


@app.post("/api/settings")
def update_settings(body: SettingsUpdate):
    """Save API keys to .env file."""
    updates = {}
    if body.gemini_api_key is not None:
        updates["GEMINI_API_KEY"] = body.gemini_api_key
    if body.openai_api_key is not None:
        updates["OPENAI_API_KEY"] = body.openai_api_key
    if body.binance_api_key is not None:
        updates["BINANCE_API_KEY"] = body.binance_api_key
    if body.binance_secret_key is not None:
        updates["BINANCE_SECRET_KEY"] = body.binance_secret_key
    
    if not updates:
        raise HTTPException(400, "No keys provided")
    
    _save_env_keys(updates)
    
    # Update config module
    for k, v in updates.items():
        if hasattr(config, k):
            setattr(config, k, v)
    
    return {"saved": True, "keys_updated": list(updates.keys())}


@app.post("/api/settings/validate")
def validate_api_key(body: dict):
    """Test an API key against its provider."""
    provider = body.get("provider", "")
    key = body.get("key", "")
    
    if not key:
        raise HTTPException(400, "No key provided")
    
    try:
        if provider == "gemini":
            import requests as req
            r = req.get(
                f"https://generativelanguage.googleapis.com/v1/models?key={key}",
                timeout=10
            )
            if r.status_code == 200:
                return {"valid": True, "message": "Gemini API key is valid"}
            else:
                return {"valid": False, "message": f"Invalid key (HTTP {r.status_code})"}
        
        elif provider == "openai":
            import requests as req
            r = req.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10
            )
            if r.status_code == 200:
                return {"valid": True, "message": "OpenAI API key is valid"}
            else:
                return {"valid": False, "message": f"Invalid key (HTTP {r.status_code})"}
        
        elif provider == "binance":
            import ccxt
            ex = ccxt.binance({"apiKey": key, "enableRateLimit": True})
            ex.fetch_ticker("BTC/USDT")
            return {"valid": True, "message": "Binance connection successful"}
        
        else:
            raise HTTPException(400, f"Unknown provider: {provider}")
    
    except Exception as e:
        return {"valid": False, "message": str(e)[:100]}


@app.get("/api/system-check")
def system_check():
    """Run system compatibility check."""
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "system_check.py")],
            capture_output=True, text=True, timeout=30
        )
        return json.loads(result.stdout)
    except Exception as e:
        return {
            "gpu_name": "Unknown", "vram_gb": 0, "gpu_ok": False, "gpu_compute": 0,
            "disk_free_gb": 0, "disk_ok": False, "ram_gb": 0, "ram_ok": False,
            "cuda_version": "N/A", "errors": [str(e)], "warnings": []
        }


# ============================================================
#  FIRST-RUN MANAGEMENT
# ============================================================

_first_run_status = {"running": False, "complete": False, "progress": {}}


@app.get("/api/first-run-status")
def get_first_run_status():
    """Check if first-run setup is needed or in progress."""
    has_data = os.path.exists(config.MARKET_DATA_PARQUET_PATH)
    has_model = os.path.exists(os.path.join(config.MODEL_DIR, "predictor_v3.joblib"))
    needs_setup = not (has_data and has_model)
    return {
        "needs_setup": needs_setup,
        "running": _first_run_status["running"],
        "complete": _first_run_status["complete"],
        "progress": _first_run_status["progress"],
    }


@app.post("/api/first-run")
def trigger_first_run(body: dict = None):
    """Start the first-run setup process."""
    if _first_run_status["running"]:
        raise HTTPException(409, "First-run setup already in progress")
    
    days = (body or {}).get("days", 1095)
    
    def _run_setup():
        import subprocess
        _first_run_status["running"] = True
        _first_run_status["complete"] = False
        
        try:
            python_path = sys.executable
            script = os.path.join(os.path.dirname(__file__), "first_run.py")
            
            proc = subprocess.Popen(
                [python_path, script, "--json-progress", f"--days={days}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(__file__),
            )
            
            for line in proc.stdout:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        _first_run_status["progress"] = data
                    except json.JSONDecodeError:
                        pass
            
            proc.wait()
            _first_run_status["complete"] = proc.returncode == 0
        except Exception as e:
            _first_run_status["progress"] = {"stage": "error", "message": str(e)}
        finally:
            _first_run_status["running"] = False
    
    thread = threading.Thread(target=_run_setup, daemon=True)
    thread.start()
    
    return {"started": True, "days": days}


# ============================================================
#  NEXUS AGENT (AI CHAT)
# ============================================================

class AgentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.post("/api/agent/chat")
def agent_chat(req: AgentChatRequest):
    """Chat with the Nexus Agent. Returns AI analysis based on live state."""
    try:
        result = nexus_agent.chat(
            user_message=req.message,
            predictor=predictor,
            trader=trader,
            collector=collector,
            session_id=req.session_id,
        )
        return _sanitize_for_json(result)
    except Exception as e:
        logging.error(f"Agent chat error: {e}")
        return {"reply": f"⚠️ Error: {str(e)}", "provider": "error"}

@app.get("/api/agent/state")
def agent_state():
    """Get the raw state snapshot used by the agent."""
    state = nexus_agent.build_state_snapshot(predictor, trader, collector)
    return _sanitize_for_json(state)

@app.get("/api/agent/history")
def agent_history(session_id: str = None, limit: int = 50):
    """Get chat history from memory."""
    import nexus_memory
    messages = nexus_memory.get_recent_messages(session_id=session_id, limit=limit)
    return {"messages": messages, "session_id": session_id}

@app.get("/api/agent/knowledge")
def agent_knowledge(category: str = None):
    """Get learned knowledge items."""
    import nexus_memory
    items = nexus_memory.get_knowledge(category=category, limit=50)
    return {"knowledge": items, "total": len(items)}

@app.get("/api/agent/memory-stats")
def agent_memory_stats():
    """Get memory system statistics."""
    import nexus_memory
    return nexus_memory.get_stats()

@app.post("/api/agent/new-session")
def agent_new_session():
    """Start a new chat session."""
    sid = nexus_agent.new_session()
    return {"session_id": sid}


# ============================================================
#  NEWS INTELLIGENCE FEED
# ============================================================

_news_cache = {"items": [], "ts": 0}
_NEWS_CACHE_TTL = 300  # 5 minute cache

def _fetch_cryptopanic():
    """Fetch from CryptoPanic free API (no key = public posts)."""
    items = []
    try:
        resp = requests.get(
            "https://cryptopanic.com/api/free/v1/posts/?currencies=BTC&kind=news&public=true",
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            for post in (data.get("results") or [])[:8]:
                title = post.get("title", "")
                # CryptoPanic has votes for sentiment
                votes = post.get("votes", {})
                pos = votes.get("positive", 0) + votes.get("liked", 0)
                neg = votes.get("negative", 0) + votes.get("disliked", 0)
                total = pos + neg
                score = (pos - neg) / max(total, 1) if total > 0 else 0
                sentiment = "BULLISH" if score > 0.15 else ("BEARISH" if score < -0.15 else "NEUTRAL")
                source = (post.get("source", {}) or {}).get("title", "CryptoPanic")
                items.append({
                    "title": title,
                    "source": source,
                    "sentiment": sentiment,
                    "sentiment_score": round(score, 2),
                })
    except Exception as e:
        logging.debug(f"CryptoPanic fetch error: {e}")
    return items

def _fetch_coingecko_trending():
    """Fetch trending coins from CoinGecko (free, no key)."""
    items = []
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/search/trending",
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            coins = data.get("coins", [])[:3]
            for coin in coins:
                c = coin.get("item", {})
                name = c.get("name", "?")
                symbol = c.get("symbol", "?")
                rank = c.get("market_cap_rank", "?")
                price_change = c.get("data", {}).get("price_change_percentage_24h", {}).get("usd", 0) or 0
                sent = "BULLISH" if price_change > 2 else ("BEARISH" if price_change < -2 else "NEUTRAL")
                items.append({
                    "title": f"🔥 {name} ({symbol}) trending — #{rank} by market cap, 24h: {price_change:+.1f}%",
                    "source": "CoinGecko Trending",
                    "sentiment": sent,
                    "sentiment_score": round(min(1, max(-1, price_change / 10)), 2),
                })
    except Exception as e:
        logging.debug(f"CoinGecko trending error: {e}")
    return items

def _generate_market_signals():
    """Generate smart 'news' from live prediction/market data."""
    items = []
    try:
        if predictor and hasattr(predictor, 'get_prediction') and predictor.is_trained:
            pred = predictor.get_prediction()
            price = pred.get("current_price", 0)
            direction = pred.get("direction", "?")
            confidence = pred.get("confidence", 0)
            hurst = pred.get("hurst", 0.5)
            regime = pred.get("regime_label", "Unknown")
            
            # AI Prediction signal
            sent = "BULLISH" if direction == "LONG" else "BEARISH"
            score = (confidence / 100) if direction == "LONG" else -(confidence / 100)
            items.append({
                "title": f"🤖 AI predicts {'📈 LONG' if direction == 'LONG' else '📉 SHORT'} with {confidence:.0f}% confidence — Target: ${pred.get('target_price', 0):,.0f}",
                "source": "Nexus AI Engine",
                "sentiment": sent,
                "sentiment_score": round(score, 2),
            })
            
            # Regime signal
            regime_sent = {"TRENDING": "BULLISH", "MEAN_REVERTING": "NEUTRAL", "VOLATILE": "BEARISH", "CHAOTIC": "BEARISH"}
            items.append({
                "title": f"📊 Market regime: {regime} — Hurst exponent: {hurst:.3f} ({'trending' if hurst > 0.55 else 'mean-reverting' if hurst < 0.45 else 'random walk'})",
                "source": "Quant Engine",
                "sentiment": regime_sent.get(regime, "NEUTRAL"),
                "sentiment_score": round(hurst - 0.5, 2),
            })
            
            # Fear & Greed
            alt = predictor.last_alt_signals or {}
            fg = alt.get("fear_greed_index")
            if fg and fg != "?":
                try:
                    fg_val = int(fg) if isinstance(fg, (int, float, str)) else 50
                    fg_label = "Extreme Fear" if fg_val < 25 else "Fear" if fg_val < 40 else "Neutral" if fg_val < 60 else "Greed" if fg_val < 75 else "Extreme Greed"
                    fg_sent = "BEARISH" if fg_val < 35 else ("BULLISH" if fg_val > 55 else "NEUTRAL")
                    items.append({
                        "title": f"😱 Fear & Greed Index: {fg_val}/100 ({fg_label}) — {'contrarian buy signal' if fg_val < 25 else 'contrarian sell signal' if fg_val > 75 else 'no extreme'}",
                        "source": "Alternative.me",
                        "sentiment": fg_sent,
                        "sentiment_score": round((fg_val - 50) / 50, 2),
                    })
                except (ValueError, TypeError):
                    pass
            
            # Accuracy tracking
            acc = predictor.last_validation_accuracy
            if acc > 0:
                items.append({
                    "title": f"🎯 Model accuracy: {acc:.1f}% ({('outperforming' if acc > 52 else 'at') + ' baseline'}) — Statistically {'verified ✅' if predictor.is_statistically_verified else 'tracking...'}",
                    "source": "Validation Engine",
                    "sentiment": "BULLISH" if acc > 52 else "NEUTRAL",
                    "sentiment_score": round((acc - 50) / 20, 2),
                })
            
            # Quant engine signals
            q = predictor.last_quant_analysis or {}
            if q:
                flow = q.get("order_flow", {})
                bp = flow.get("buy_pressure", 0.5)
                if abs(bp - 0.5) > 0.05:
                    items.append({
                        "title": f"📈 Order flow: {'buyers dominating' if bp > 0.55 else 'sellers dominating'} — Buy pressure: {bp:.1%}",
                        "source": "Order Flow Analysis",
                        "sentiment": "BULLISH" if bp > 0.55 else "BEARISH",
                        "sentiment_score": round((bp - 0.5) * 4, 2),
                    })
                
                jump = q.get("jump_risk", {})
                jl = jump.get("level", "")
                if jl and jl != "LOW":
                    items.append({
                        "title": f"⚡ Jump risk: {jl} — Hawkes process detecting volatility clustering",
                        "source": "Hawkes Process",
                        "sentiment": "BEARISH",
                        "sentiment_score": -0.4 if jl == "HIGH" else -0.2,
                    })
        
        # Portfolio signals
        if trader:
            stats = trader.get_stats()
            pnl_pct = stats.get("total_pnl_pct", 0)
            n_pos = len(trader.positions)
            if n_pos > 0:
                items.append({
                    "title": f"💼 {n_pos} position{'s' if n_pos > 1 else ''} open — Portfolio PnL: {pnl_pct:+.2f}%, Balance: ${stats.get('balance', 10000):,.0f}",
                    "source": "Paper Trader",
                    "sentiment": "BULLISH" if pnl_pct > 0 else ("BEARISH" if pnl_pct < 0 else "NEUTRAL"),
                    "sentiment_score": round(min(1, max(-1, pnl_pct / 5)), 2),
                })
            
            dd = stats.get("max_drawdown_pct", 0)
            if dd > 5:
                items.append({
                    "title": f"⚠️ Drawdown alert: {dd:.1f}% max drawdown — {'circuit breaker zone' if dd > 15 else 'elevated risk'}",
                    "source": "Risk Manager",
                    "sentiment": "BEARISH",
                    "sentiment_score": round(min(-0.3, -dd / 20), 2),
                })
    except Exception as e:
        logging.debug(f"Market signal generation error: {e}")
    return items


@app.get("/api/news")
def get_news():
    """Aggregated crypto news + live market signals."""
    global _news_cache
    now = time.time()
    
    # Return cache if fresh
    if now - _news_cache["ts"] < _NEWS_CACHE_TTL and _news_cache["items"]:
        return {"items": _news_cache["items"]}
    
    all_items = []
    
    # 1. Market-derived signals (always available, always first)
    all_items.extend(_generate_market_signals())
    
    # 2. CryptoPanic headlines
    all_items.extend(_fetch_cryptopanic())
    
    # 3. CoinGecko trending
    all_items.extend(_fetch_coingecko_trending())
    
    # Cache the result
    _news_cache = {"items": all_items, "ts": now}
    
    return {"items": all_items}


# ============================================================
#  GRACEFUL SHUTDOWN
# ============================================================

@app.post("/api/shutdown")
async def shutdown():
    """Graceful shutdown endpoint — called by Electron on app close.
    Stops auto-trade/retrain threads, closes Binance WS, then exits."""
    logging.info("Shutdown requested via /api/shutdown")
    try:
        _auto_trade_stop.set()
        _retrain_stop.set()
        if binance_ws:
            binance_ws.stop()
    except Exception as e:
        logging.warning(f"Error during shutdown cleanup: {e}")
    # Schedule hard exit after giving the response time to flush
    asyncio.get_event_loop().call_later(0.5, lambda: os._exit(0))
    return {"status": "shutting_down"}


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("NEXUS_API_PORT", 8420))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
