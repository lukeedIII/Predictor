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
from gpu_game import GpuGame
from derivatives_feed import DerivativesFeed
import hf_sync

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
#  GLOBAL STATE
# ============================================================

predictor: Optional[NexusPredictor] = None
trader: Optional[PaperTrader] = None
collector: Optional[DataCollector] = None
math_core: Optional[MathCore] = None
binance_ws: Optional[BinanceWSClient] = None
derivs_feed: Optional[DerivativesFeed] = None

boot_status = {"stage": "starting", "progress": 0, "message": "Initializing..."}
_auto_trade_thread: Optional[threading.Thread] = None
_auto_trade_stop = threading.Event()
_event_loop: Optional[asyncio.AbstractEventLoop] = None  # for thread→async bridge

# GPU Game instance
gpu_game: Optional[GpuGame] = None
_gpu_game_stop = threading.Event()

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

# Continuous data collection (independent of trading bot)
_data_collect_stop = threading.Event()
DATA_COLLECT_INTERVAL_SEC = 60  # Collect every minute

# App boot timestamp — used by agent for lifetime awareness
_app_boot_time = time.time()

# Boot-gate: all HTTP requests (except /api/boot-status) are held until
# _init_engines() completes, preventing "predictor is None" 500 errors.
_boot_gate: asyncio.Event = asyncio.Event()


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
    global predictor, trader, collector, math_core, binance_ws, boot_status, gpu_game, derivs_feed
    
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
        
        # ── Derivatives Feed (funding, OI, basis) ──
        if getattr(config, 'DERIVATIVES_ENABLED', False):
            boot_status = {"stage": "derivatives", "progress": 48, "message": "Backfilling derivatives data..."}
            try:
                derivs_feed = DerivativesFeed()
                derivs_feed.backfill_history()
                derivs_feed.collect_snapshot()
                logging.info("[BOOT] Derivatives feed initialized")
            except Exception as e:
                logging.warning(f"[BOOT] Derivatives feed failed (non-fatal): {e}")
                derivs_feed = None
        
        boot_status = {"stage": "predictor", "progress": 55, "message": "Initializing AI predictor..."}
        predictor = NexusPredictor()
        
        boot_status = {"stage": "training", "progress": 70, "message": "Checking model status..."}
        if not predictor.is_trained:
            # ── HF AUTO-PULL CHECK ──
            # If no local models, try pulling from Hub first to save 6 hours of training
            if not hf_sync.has_models():
                boot_status["message"] = "Checking Hugging Face for remote models..."
                res = hf_sync.pull_from_hub()
                if res.get("success"):
                    logging.info(f"[BOOT] Successfully pulled models from HF: {res['path']}")
                    boot_status["message"] = "Models pulled from Hub!"
                    # Need to re-init predictor to pick up new files
                    predictor = NexusPredictor()
                else:
                    logging.info(f"[BOOT] HF pull skipped/failed: {res.get('error')}")

            if not predictor.is_trained:
                boot_status["message"] = "Training AI model (first run)..."
                predictor.train()  # 3-tuple return ignored during boot
        
        boot_status = {"stage": "trader", "progress": 85, "message": "Starting paper trader..."}
        trader = PaperTrader()
        
        # ── GPU Game ──
        boot_status = {"stage": "game", "progress": 92, "message": "Loading GPU game..."}
        gpu_game = GpuGame()
        # NOTE: tick thread is started by lifespan context manager (_gpu_game_tick_loop)
        
        boot_status = {"stage": "ready", "progress": 100, "message": "All systems online"}
        logging.info("All engines initialized successfully")
        
    except Exception as e:
        boot_status = {"stage": "error", "progress": -1, "message": f"Init error: {str(e)[:200]}"}
        logging.error(f"Engine init failed: {e}")
        traceback.print_exc()
    finally:
        # Always release the gate so requests don't hang forever on error.
        _boot_gate.set()
        logging.info("[BOOT-GATE] Released — API now accepting requests.")


def _continuous_data_collect():
    """Background loop: collects and saves market candles every 60s.
    Also snapshots WebSocket microstructure data (trades/sec, buy/sell ratio, spread).
    Runs INDEPENDENTLY of the trading bot so the retrain loop always has fresh data."""
    import pandas as _pd
    
    # Wait for engines to be ready first
    while not _data_collect_stop.is_set() and boot_status.get('stage') != 'ready':
        _data_collect_stop.wait(5)
    
    logging.info("[DATA-COLLECTOR] Continuous background collection started")
    candles_saved = 0
    micro_path = config.MICROSTRUCTURE_DATA_PATH
    
    while not _data_collect_stop.is_set():
        try:
            # ── 1. Collect OHLCV candles (always) ──
            if collector is not None:
                collector.collect_and_save(limit=5)
                candles_saved += 1
            
            # ── 2. Snapshot WebSocket microstructure data ──
            if binance_ws is not None and binance_ws.connected:
                snap = binance_ws.snapshot
                row = {
                    'timestamp': _pd.Timestamp.now(tz='UTC').floor('min'),
                    'ws_trades_per_sec': snap.get('trades_per_sec', 0.0),
                    'ws_buy_sell_ratio': snap.get('buy_sell_ratio', 1.0),
                    'ws_buy_volume_60s': snap.get('buy_volume_60s', 0.0),
                    'ws_sell_volume_60s': snap.get('sell_volume_60s', 0.0),
                    'ws_spread_bps': round(
                        (snap['ask'] - snap['bid']) / (snap['price'] + 1e-9) * 10000, 2
                    ) if snap.get('price', 0) > 0 else 0.0,
                    'ws_price': snap.get('price', 0.0),
                }
                new_row = _pd.DataFrame([row])
                
                try:
                    if os.path.exists(micro_path):
                        existing = _pd.read_parquet(micro_path)
                        combined = _pd.concat([existing, new_row]).drop_duplicates(
                            subset=['timestamp'], keep='last'
                        )
                        # Cap at 50K rows (~35 days of 1-min data)
                        combined = combined.tail(50_000)
                        combined.to_parquet(micro_path, index=False)
                    else:
                        new_row.to_parquet(micro_path, index=False)
                except Exception as e:
                    logging.debug(f"[DATA-COLLECTOR] Microstructure save error: {e}")
                
                # Push live snapshot to predictor for real-time predictions
                if predictor is not None:
                    predictor._live_ws_snapshot = snap
            
            if candles_saved % 60 == 0 and candles_saved > 0:  # Log every hour
                logging.info(f"[DATA-COLLECTOR] {candles_saved} cycles complete, micro file: {os.path.exists(micro_path)}")
            
            # ── 3. Derivatives snapshot (funding, OI, basis) ──
            if derivs_feed is not None:
                try:
                    derivs_feed.collect_snapshot()
                    # Push features to predictor for live use
                    if predictor is not None:
                        predictor._live_derivs_features = derivs_feed.get_features()
                    
                    # Periodic history update (every 5 minutes)
                    hist_interval = getattr(config, 'DERIVATIVES_HISTORY_INTERVAL', 300)
                    if candles_saved % (hist_interval // 60) == 0 and candles_saved > 0:
                        derivs_feed.collect_periodic_history()
                except Exception as e:
                    logging.debug(f"[DATA-COLLECTOR] Derivatives error: {e}")
                
        except Exception as e:
            logging.warning(f"[DATA-COLLECTOR] Collection error: {e}")
        
        # Sleep 60s in 5s chunks so shutdown is responsive
        for _ in range(12):
            if _data_collect_stop.is_set():
                return
            _data_collect_stop.wait(5)
    
    logging.info("[DATA-COLLECTOR] Stopped")


def _auto_retrain_loop():
    """Background loop: retraining schedule.
    
    Schedule:
      1h after boot  → First retrain (let data accumulate)
      Every 6h after → Full retrain with all accumulated 1m data
    """
    import json as _json
    global _retrain_status
    
    # Wait for engines to be ready first
    while not _retrain_stop.is_set() and boot_status.get('stage') != 'ready':
        _retrain_stop.wait(5)
    
    retrain_history_path = os.path.join(config.LOG_DIR, 'retrain_history.json')
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    def _do_retrain(data_source: str, label: str):
        """Execute a single retrain using the specified data source."""
        if predictor is None:
            logging.warning(f"[AUTO-RETRAIN] {label} — predictor not available, skipping")
            return
        
        try:
            _retrain_status['is_retraining'] = True
            logging.info(f"[AUTO-RETRAIN] {label} — starting...")
            
            if data_source == '1s' and binance_ws is not None:
                # Train on 1-second WebSocket candles
                candle_df = binance_ws.get_1s_candles()
                if len(candle_df) < 80:
                    logging.warning(f"[AUTO-RETRAIN] {label} — only {len(candle_df)} 1s candles, skipping")
                    _retrain_status['is_retraining'] = False
                    return
                logging.info(f"[AUTO-RETRAIN] Training on {len(candle_df):,} 1s candles")
                result = predictor.train_on_candles(candle_df, timeframe_label='1s')
            else:
                # Train on 1m candles (standard path)
                if collector is not None:
                    try:
                        collector.collect_and_save(limit=100)
                        logging.info("[AUTO-RETRAIN] Refreshed 1m market data before training")
                    except Exception as e:
                        logging.warning(f"[AUTO-RETRAIN] Pre-train data refresh failed: {e}")
                result = predictor.train()
                # Unpack promotion info from champion-challenger gate
                promotion = result[2] if isinstance(result, tuple) and len(result) >= 3 else None
            
            now = datetime.now().isoformat()
            acc = predictor.last_validation_accuracy if hasattr(predictor, 'last_validation_accuracy') else None
            
            # Add promotion metadata to retrain status
            promotion_info = {}
            if promotion is not None:
                promotion_info = {
                    'last_promotion': promotion.get('promoted', None),
                    'last_promotion_reason': promotion.get('reason', None),
                }
            
            _retrain_status.update({
                'last_retrain': now,
                'last_accuracy': acc,
                'retrain_count': _retrain_status['retrain_count'] + 1,
                'is_retraining': False,
                'last_error': None,
                **promotion_info,
            })
            
            # Append to history log (with delta tracking)
            history = []
            if os.path.exists(retrain_history_path):
                try:
                    with open(retrain_history_path, 'r') as f:
                        history = _json.load(f)
                except Exception:
                    history = []
            
            # Compute delta vs previous accuracy
            prev_acc = None
            if history:
                prev_acc = history[-1].get('accuracy')
            
            delta = None
            trend = '→'  # neutral
            if acc is not None and prev_acc is not None:
                delta = round(acc - prev_acc, 2)
                if delta > 0.1:
                    trend = '↑'
                elif delta < -0.1:
                    trend = '↓'
                else:
                    trend = '→'
            
            # Count consecutive improvements/regressions
            streak = 0
            if delta is not None and delta > 0:
                streak = 1
                for h in reversed(history):
                    if h.get('delta') is not None and h['delta'] > 0:
                        streak += 1
                    else:
                        break
            elif delta is not None and delta < 0:
                streak = -1
                for h in reversed(history):
                    if h.get('delta') is not None and h['delta'] < 0:
                        streak -= 1
                    else:
                        break
            
            entry = {
                'timestamp': now,
                'accuracy': acc,
                'prev_accuracy': prev_acc,
                'delta': delta,
                'trend': trend,
                'streak': streak,
                'label': label,
                'data_source': data_source,
                'feature_count': len(getattr(predictor, 'features', [])),
            }
            # Append champion-challenger metrics if available
            if promotion is not None:
                entry['promoted'] = promotion.get('promoted')
                entry['promotion_reason'] = promotion.get('reason')
                entry['champion_logloss'] = promotion.get('champion_logloss')
                entry['challenger_logloss'] = promotion.get('challenger_logloss')
                entry['champion_accuracy'] = promotion.get('champion_accuracy')
                entry['challenger_accuracy'] = promotion.get('challenger_accuracy')
            
            # Append drift monitor report if available
            drift_report = getattr(predictor, 'last_drift_report', None)
            if drift_report is not None:
                entry['drift_severity'] = drift_report.get('overall_severity')
                feat_drift = drift_report.get('feature_drift', {})
                entry['drift_feature_psi'] = feat_drift.get('mean_psi')
                entry['drift_features_drifted'] = feat_drift.get('features_drifted')
                cal_drift = drift_report.get('calibration_drift', {})
                entry['drift_brier_score'] = cal_drift.get('brier_score')
                _retrain_status['last_drift_severity'] = drift_report.get('overall_severity')
                _retrain_status['last_drift_report'] = drift_report
            
            # Append walk-forward evaluation results if available
            if promotion is not None and 'walk_forward' in promotion:
                wf = promotion['walk_forward']
                entry['wf_mean_accuracy'] = wf.get('mean_accuracy')
                entry['wf_std_accuracy'] = wf.get('std_accuracy')
                entry['wf_min_accuracy'] = wf.get('min_accuracy')
                entry['wf_max_accuracy'] = wf.get('max_accuracy')
                entry['wf_mean_logloss'] = wf.get('mean_logloss')
                entry['wf_folds'] = wf.get('n_folds')
            
            history.append(entry)
            history = history[-100:]
            with open(retrain_history_path, 'w') as f:
                _json.dump(history, f, indent=2)
            
            promoted_str = ""
            if promotion is not None:
                promoted_str = f" | {'PROMOTED' if promotion.get('promoted') else 'REJECTED'}: {promotion.get('reason', '')}"
            drift_str = f" | Drift: {entry.get('drift_severity', 'N/A')}" if drift_report else ""
            wf_str = ""
            if entry.get('wf_mean_accuracy') is not None:
                wf_str = f" | WF: {entry['wf_mean_accuracy']:.1f}%±{entry['wf_std_accuracy']:.1f}%"
            delta_str = f" (Δ {delta:+.2f}%)" if delta is not None else ""
            acc_str = f"{acc:.1f}%{delta_str}" if acc is not None else "N/A"
            logging.info(f"[AUTO-RETRAIN] {label} — complete. Accuracy: {acc_str} {trend}{promoted_str}{drift_str}{wf_str}")
            
        except Exception as e:
            _retrain_status['is_retraining'] = False
            _retrain_status['last_error'] = str(e)[:200]
            logging.error(f"[AUTO-RETRAIN] {label} — failed: {e}")
    
    # ── First retrain: 1 hour after boot ──
    FIRST_RETRAIN_MINUTES = 60
    next_time = datetime.now() + timedelta(minutes=FIRST_RETRAIN_MINUTES)
    _retrain_status['next_retrain'] = next_time.isoformat()
    logging.info(f"[AUTO-RETRAIN] First retrain scheduled in {FIRST_RETRAIN_MINUTES}min")
    
    for _ in range(FIRST_RETRAIN_MINUTES * 6):  # check every 10s
        if _retrain_stop.is_set():
            return
        _retrain_stop.wait(10)
    
    if _retrain_stop.is_set():
        return
    
    _do_retrain('1m', 'First retrain (1h after boot)')
    
    # ── Ongoing: every 6 hours ──
    while not _retrain_stop.is_set():
        next_time = datetime.now() + timedelta(hours=RETRAIN_INTERVAL_HOURS)
        _retrain_status['next_retrain'] = next_time.isoformat()
        
        # Sleep until next retrain (check every 30s for shutdown)
        for _ in range(RETRAIN_INTERVAL_HOURS * 120):
            if _retrain_stop.is_set():
                return
            _retrain_stop.wait(30)
        
        if _retrain_stop.is_set():
            return
        
        _do_retrain('1m', f'Scheduled 6h retrain #{_retrain_status["retrain_count"] + 1}')


def _gpu_game_tick_loop():
    """Background loop: update ASS price + mine coins every 30s."""
    import time as _time
    _time.sleep(5)  # wait for startup
    while not _gpu_game_stop.is_set():
        try:
            if gpu_game:
                gpu_game.tick()
        except Exception as e:
            logging.error(f"GPU game tick error: {e}")
        _gpu_game_stop.wait(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start engine init + auto-retrain scheduler + WS push on startup."""
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    
    init_thread = threading.Thread(target=_init_engines, daemon=True)
    init_thread.start()
    retrain_thread = threading.Thread(target=_auto_retrain_loop, daemon=True)
    retrain_thread.start()
    data_thread = threading.Thread(target=_continuous_data_collect, daemon=True)
    data_thread.start()
    gpu_tick_thread = threading.Thread(target=_gpu_game_tick_loop, daemon=True)
    gpu_tick_thread.start()
    # Slow prediction refresh thread (2s cycle — feeds cached data to fast WS push)
    pred_thread = threading.Thread(target=_prediction_refresh_loop, daemon=True)
    pred_thread.start()
    # Start periodic WS push to frontend clients (200ms fast ticks)
    push_task = asyncio.create_task(_ws_push_loop())
    yield
    _auto_trade_stop.set()
    _retrain_stop.set()
    _data_collect_stop.set()
    _gpu_game_stop.set()
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


# ─── Cached prediction state (updated by slow thread) ────────────
_cached_prediction: dict = {}        # latest prediction + quant + accuracy
_cached_prediction_lock = threading.Lock()
_FAST_TICK_INTERVAL = 0.2            # 200ms → 5 ticks/sec for price data
_SLOW_TICK_INTERVAL = 2.0            # 2s    → prediction/quant refresh


def _prediction_refresh_loop():
    """Background thread: recompute prediction + quant every SLOW_TICK_INTERVAL.

    This keeps the expensive model inference off the fast WS push path.
    Results are stored in `_cached_prediction` and merged by the fast loop.
    """
    global _cached_prediction
    while True:
        try:
            if predictor and predictor.is_trained:
                pred = predictor.get_prediction()
                live_acc = getattr(predictor, '_live_accuracy', 0.0)
                live_samples = getattr(predictor, '_live_accuracy_samples', 0)
                training_acc = (
                    getattr(predictor, '_training_accuracy', 0.0)
                    or predictor.last_validation_accuracy
                )
                quant = (
                    predictor.quant_engine.get_ui_summary()
                    if predictor.quant_engine and predictor.quant_initialized
                    else predictor.last_quant_analysis or {}
                )
                alt = predictor.last_alt_signals or {}

                enrichment = {
                    "prediction": _sanitize_for_json(pred),
                    "accuracy": live_acc if live_samples >= 3 else training_acc,
                    "accuracy_source": "live" if live_samples >= 3 else "training",
                    "live_accuracy_samples": live_samples,
                    "training_accuracy": training_acc,
                    "quant": _sanitize_for_json(quant),
                    "alt_signals": _sanitize_for_json(alt),
                    "derivatives": _sanitize_for_json(
                        derivs_feed.get_snapshot_dict() if derivs_feed else {}
                    ),
                }

                with _cached_prediction_lock:
                    _cached_prediction = enrichment
        except Exception as e:
            logging.debug(f"Prediction refresh error: {e}")
        time.sleep(_SLOW_TICK_INTERVAL)


async def _ws_push_loop():
    """Push live data to frontend WebSocket clients at high frequency.

    ┌──────────────┬─────────────────────────────────────────────────┐
    │ FAST (200ms) │ Price, 24h stats, volume, positions, bot stats │
    │ SLOW (2s)    │ Prediction, accuracy, quant, alt_signals       │
    └──────────────┴─────────────────────────────────────────────────┘

    The slow parts are computed by `_prediction_refresh_loop` in a background
    thread and merged via `_cached_prediction` on each fast tick.
    """
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

                # ── Merge cached prediction data (computed on slow thread) ──
                with _cached_prediction_lock:
                    if _cached_prediction:
                        payload.update(_cached_prediction)

                # ── Trader / positions (cheap — in-memory read) ──
                if trader:
                    payload["bot_running"] = trader.is_running
                    if trader.positions:
                        live_price = snapshot.get("price") or 0
                        pos_list = []
                        for pos in trader.positions:
                            d = pos.to_dict()
                            if live_price > 0:
                                d["unrealized_pnl"] = round(pos.unrealized_pnl(live_price), 2)
                                d["unrealized_pnl_pct"] = round(pos.unrealized_pnl_pct(live_price), 2)
                            pos_list.append(d)
                        payload["positions"] = _sanitize_for_json(pos_list)
                    else:
                        payload["positions"] = []
                    try:
                        payload["stats"] = _sanitize_for_json(trader.get_stats())
                    except Exception:
                        pass

                await _ws_manager.broadcast(payload)
        except Exception as e:
            logging.debug(f"WS push error: {e}")
        await asyncio.sleep(_FAST_TICK_INTERVAL)


# ============================================================
#  APP
# ============================================================

app = FastAPI(title="Nexus Shadow-Quant API", version=config.VERSION, lifespan=lifespan)

# Strict CORS: allow only Electron renderer and local dev server origins.
# 'allow_origins=["*"]' would expose the local API to any webpage the user
# opens in a browser tab — a significant CSRF/SSRF risk.
_ALLOWED_ORIGINS = [
    "file://",                  # Electron renderer (file:// scheme)
    "null",                     # Electron on some platforms sends 'null' origin
    "http://localhost",
    "http://localhost:5173",    # Vite dev server
    "http://localhost:5174",
    "http://127.0.0.1",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def boot_gate_middleware(request, call_next):
    """Return 503 (not a hang) for all non-status requests until boot is done.

    Returning 503 is deterministic: the frontend splash screen already polls
    /api/boot-status and retries; clients get a clear signal with Retry-After
    rather than an opaque connection timeout.
    WebSocket upgrade requests are also passed through so the WS module can
    gate them itself with a 1013 close frame.
    /health/live is always passed through (liveness vs. readiness split).
    """
    passthrough = (
        request.url.path.startswith("/api/boot-status")
        or request.url.path == "/health/live"   # liveness always passes
        or request.url.path.startswith("/ws")   # WS handshake — WS module gates
        or request.url.path.startswith("/api/ws")
    )
    if not passthrough and not _boot_gate.is_set():
        return JSONResponse(
            {
                "status": "booting",
                "stage": boot_status.get("stage"),
                "progress": boot_status.get("progress"),
                "message": "Service is initialising. Retry shortly.",
            },
            status_code=503,
            headers={"Retry-After": "2"},
        )
    return await call_next(request)


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


@app.get("/health/live", tags=["health"])
def health_live():
    """Liveness probe — always 200 while the process is alive.

    Kubernetes / Docker can use this to detect crash-loops.
    Deliberately never returns 503: if the process responds at all, it's live.
    """
    return {"status": "alive", "uptime_seconds": round(time.time() - _app_boot_time, 1)}


@app.get("/health/ready", tags=["health"])
def health_ready():
    """Readiness probe — 200 when boot is complete, 503 while initialising.

    Clients / load-balancers poll this endpoint to know when to start
    sending real traffic. It complements /health/live which is always 200.
    """
    if _boot_gate.is_set():
        return {"status": "ready", "stage": boot_status.get("stage")}
    raise HTTPException(
        status_code=503,
        detail={
            "status": "initialising",
            "stage": boot_status.get("stage"),
            "progress": boot_status.get("progress"),
        },
        headers={"Retry-After": "2"},
    )


# ── Settings persistence (JSON file) ─────────────────

def _load_settings() -> dict:
    """Load settings from JSON file."""
    if os.path.exists(config.SETTINGS_PATH):
        try:
            with open(config.SETTINGS_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_settings(data: dict):
    """Save settings to JSON file."""
    os.makedirs(os.path.dirname(config.SETTINGS_PATH), exist_ok=True)
    with open(config.SETTINGS_PATH, 'w') as f:
        json.dump(data, f, indent=2)


@app.get("/api/settings")
def get_settings_json():
    """Return merged settings: JSON persistence (first_run_done etc.) + masked env keys."""
    json_settings = _load_settings()
    # Also try to add masked API key info
    try:
        keys = _load_env_keys()
        json_settings["keys"] = {k: _mask_key(v) for k, v in keys.items()}
        json_settings["has_keys"] = {k: bool(v) for k, v in keys.items()}
    except Exception:
        pass
    json_settings["data_root"] = config.DATA_ROOT
    json_settings["version"] = config.VERSION
    
    # Beta features status
    user_beta = json_settings.get('beta_features', {})
    json_settings["beta_features_available"] = {
        k: {
            "label": v["label"],
            "description": v["description"],
            "enabled": user_beta.get(k, v["default"]),
        }
        for k, v in config.BETA_FEATURES.items()
    }
    
    # Active model architecture
    json_settings["model_arch"] = json_settings.get("model_arch", config.DEFAULT_MODEL_ARCH)
    try:
        if predictor:
            json_settings["running_model_arch"] = getattr(predictor, 'active_arch', config.DEFAULT_MODEL_ARCH)
    except Exception:
        pass
    
    return json_settings


@app.post("/api/settings")
def post_settings(body: dict):
    """Merge incoming settings into the stored JSON file."""
    current = _load_settings()
    current.update(body)
    _save_settings(current)
    return current


@app.post("/api/settings/keys")
def post_settings_keys(body: dict):
    """Store an API key securely in the .env file."""
    key = body.get("key")
    value = body.get("value")
    if not key or not value:
        raise HTTPException(status_code=400, detail="key and value required")

    env_path = os.path.join(config.DATA_ROOT, ".env")
    # Read existing lines, replace or append
    lines = []
    replaced = False
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    replaced = True
                else:
                    lines.append(line)
    if not replaced:
        lines.append(f"{key}={value}\n")
    with open(env_path, 'w') as f:
        f.writelines(lines)

    # Reload into current environment
    os.environ[key] = value
    logging.info(f"API key '{key}' saved to .env")
    return {"status": "ok", "key": key}


@app.get("/api/settings/key-status/{env_key}")
def get_key_status(env_key: str):
    """Check if an environment variable / API key is configured."""
    value = os.environ.get(env_key, "")
    return {"key": env_key, "is_set": bool(value), "masked": f"{'*' * 8}…{value[-4:]}" if len(value) > 4 else ""}


# ========================= BETA FEATURES =========================

@app.get("/api/beta-features")
def get_beta_features():
    """Return available beta features and their current status."""
    settings = _load_settings()
    user_beta = settings.get('beta_features', {})
    result = {}
    for key, feat in config.BETA_FEATURES.items():
        result[key] = {
            "label": feat["label"],
            "description": feat["description"],
            "enabled": user_beta.get(key, feat["default"]),
        }
    return result


@app.post("/api/beta-features")
def post_beta_features(body: dict):
    """Toggle beta features. Body: {"feature_key": true/false}."""
    settings = _load_settings()
    beta = settings.get('beta_features', {})
    for key, value in body.items():
        if key in config.BETA_FEATURES:
            beta[key] = bool(value)
    settings['beta_features'] = beta
    _save_settings(settings)
    return {"status": "ok", "beta_features": beta}


# ========================= MODEL SELECTOR =========================

@app.get("/api/models")
def get_models():
    """List available model architectures with VRAM requirements, availability, and real checkpoint accuracies."""
    import torch
    import re
    settings = _load_settings()
    active_arch = settings.get('model_arch', config.DEFAULT_MODEL_ARCH)
    beta_enabled = settings.get('beta_features', {}).get('model_selector', False)
    
    # VRAM info
    vram_total = 0
    vram_free = 0
    gpu_name = "N/A"
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
            vram_free = round((torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)) / 1e9, 1)
        except Exception:
            pass
    
    # ── Scan checkpoint dir for real accuracy data ──
    checkpoint_dir = os.path.join(config.MODEL_DIR, "checkpoints")
    checkpoint_data = {}  # arch_name -> {best_acc, epochs, count, checkpoints[]}
    if os.path.isdir(checkpoint_dir):
        # Pattern: {arch}_epoch_{N}_acc{F}.pth
        ckpt_re = re.compile(r'^(.+?)_epoch_(\d+)_acc([\d.]+)\.pth$')
        for fname in os.listdir(checkpoint_dir):
            m = ckpt_re.match(fname)
            if m:
                arch_key = m.group(1)
                epoch_num = int(m.group(2))
                acc_val = float(m.group(3))
                if arch_key not in checkpoint_data:
                    checkpoint_data[arch_key] = {
                        "best_acc": 0.0,
                        "max_epoch": 0,
                        "count": 0,
                        "checkpoints": [],
                    }
                cd = checkpoint_data[arch_key]
                cd["best_acc"] = max(cd["best_acc"], acc_val)
                cd["max_epoch"] = max(cd["max_epoch"], epoch_num)
                cd["count"] += 1
                cd["checkpoints"].append({
                    "epoch": epoch_num,
                    "accuracy": round(acc_val * 100, 1),
                    "filename": fname,
                })
        # Sort checkpoints by epoch
        for cd in checkpoint_data.values():
            cd["checkpoints"].sort(key=lambda x: x["epoch"])
    
    models = []
    for key, info in config.MODEL_ARCHITECTURES.items():
        # Check if pretrained weights exist on disk
        model_file = os.path.join(config.MODEL_DIR, info.get("model_file", ""))
        pretrained_file = os.path.join(config.MODEL_DIR, info.get("pretrained_file", ""))
        has_weights = os.path.exists(model_file) or os.path.exists(pretrained_file)
        
        # File size on disk
        file_size_mb = 0
        for f in [model_file, pretrained_file]:
            if os.path.exists(f):
                file_size_mb = round(os.path.getsize(f) / 1e6)
                break
        
        # Real checkpoint accuracy data
        ckpt = checkpoint_data.get(key, {})
        best_acc = round(ckpt.get("best_acc", 0) * 100, 1) if ckpt else None
        epochs_trained = ckpt.get("max_epoch", 0) if ckpt else 0
        checkpoint_count = ckpt.get("count", 0) if ckpt else 0
        checkpoints = ckpt.get("checkpoints", []) if ckpt else []
        
        models.append({
            "key": key,
            "label": info["label"],
            "params": info["params"],
            "vram_gb": info["vram_gb"],
            "description": info["description"],
            "has_weights": has_weights,
            "file_size_mb": file_size_mb,
            "vram_ok": vram_free >= info["vram_gb"],
            "is_active": key == active_arch,
            # Real checkpoint data
            "best_checkpoint_acc": best_acc,
            "epochs_trained": epochs_trained,
            "checkpoint_count": checkpoint_count,
            "checkpoints": checkpoints,
        })
    
    # Also report the currently running architecture from the predictor
    running_arch = None
    try:
        if predictor:
            running_arch = getattr(predictor, 'active_arch', config.DEFAULT_MODEL_ARCH)
    except Exception:
        pass
    
    return {
        "models": models,
        "active_arch": active_arch,
        "running_arch": running_arch,
        "beta_enabled": beta_enabled,
        "gpu": {
            "name": gpu_name,
            "vram_total_gb": vram_total,
            "vram_free_gb": vram_free,
        },
    }


@app.post("/api/models/select")
def select_model(body: dict):
    """Select a model architecture. Requires restart to take effect.
    Body: {"arch": "small_jamba"}
    """
    arch = body.get("arch", "")
    if arch not in config.MODEL_ARCHITECTURES:
        raise HTTPException(status_code=400, detail=f"Unknown architecture: {arch}. Available: {list(config.MODEL_ARCHITECTURES.keys())}")
    
    # Check VRAM
    import torch
    required_vram = config.MODEL_ARCHITECTURES[arch]["vram_gb"]
    if torch.cuda.is_available():
        try:
            free = (torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)) / 1e9
            if free < required_vram:
                return {
                    "status": "warning",
                    "message": f"⚠️ {arch} needs {required_vram} GB VRAM but only {free:.1f} GB free. Model will auto-fallback to {config.DEFAULT_MODEL_ARCH} at startup.",
                    "arch": arch,
                }
        except Exception:
            pass
    
    # Check if weights exist
    info = config.MODEL_ARCHITECTURES[arch]
    model_file = os.path.join(config.MODEL_DIR, info.get("model_file", ""))
    pretrained_file = os.path.join(config.MODEL_DIR, info.get("pretrained_file", ""))
    has_weights = os.path.exists(model_file) or os.path.exists(pretrained_file)
    
    # Save to settings
    settings = _load_settings()
    settings['model_arch'] = arch
    # Also auto-enable the beta feature
    beta = settings.get('beta_features', {})
    beta['model_selector'] = True
    settings['beta_features'] = beta
    _save_settings(settings)
    
    logging.info(f"🔄 Model architecture changed to: {arch} (restart required)")
    
    return {
        "status": "ok",
        "arch": arch,
        "label": info["label"],
        "has_weights": has_weights,
        "requires_restart": True,
        "message": f"Model set to {info['label']}. Restart the server to apply.",
    }


@app.post("/api/models/hf-search")
def hf_search_models(body: dict):
    """Search HuggingFace Hub for compatible financial/time-series models.
    Body: {"query": "bitcoin prediction transformer"}
    """
    import requests as req
    query = body.get("query", "bitcoin prediction transformer")
    
    try:
        # Search HuggingFace API for relevant models
        params = {
            "search": query,
            "filter": "pytorch",
            "sort": "downloads",
            "direction": "-1",
            "limit": 12,
        }
        r = req.get("https://huggingface.co/api/models", params=params, timeout=15)
        r.raise_for_status()
        raw_models = r.json()
        
        models = []
        for m in raw_models:
            model_id = m.get("id", "")
            # Filter for finance/crypto/time-series related models
            tags = m.get("tags", [])
            pipeline_tag = m.get("pipeline_tag", "")
            
            # Include models that are relevant to financial prediction
            relevance_keywords = [
                "finance", "crypto", "bitcoin", "stock", "trading", "forecast",
                "time-series", "timeseries", "prediction", "price", "quant",
                "transformer", "lstm", "regression", "tabular",
            ]
            model_text = f"{model_id} {' '.join(tags)} {pipeline_tag} {m.get('cardData', {}).get('license', '')}".lower()
            
            # Either matches relevance keywords or the user explicitly searched for it
            is_relevant = any(kw in model_text for kw in relevance_keywords) or any(kw in query.lower() for kw in model_id.lower().split("/"))
            
            if not is_relevant and len(models) >= 5:
                continue  # Skip irrelevant results once we have enough relevant ones
            
            author = m.get("author", model_id.split("/")[0] if "/" in model_id else "unknown")
            
            # Get description from card data
            description = ""
            card_data = m.get("cardData", {})
            if isinstance(card_data, dict):
                description = card_data.get("description", "") or card_data.get("summary", "")
            if not description:
                description = pipeline_tag or "PyTorch model"
            
            models.append({
                "id": model_id,
                "author": author,
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "description": str(description)[:200],
                "tags": tags[:8],
            })
        
        return {"models": models[:10]}
    
    except Exception as e:
        logging.error(f"HuggingFace search error: {e}")
        raise HTTPException(status_code=502, detail=f"HuggingFace search failed: {str(e)}")


@app.post("/api/models/hf-download")
def hf_download_model(body: dict):
    """Download a model from HuggingFace Hub.
    Body: {"model_id": "author/model-name"}
    """
    import requests as req
    model_id = body.get("model_id", "")
    if not model_id or "/" not in model_id:
        raise HTTPException(status_code=400, detail="Invalid model_id. Use format: author/model-name")
    
    safe_name = model_id.replace("/", "_").replace(".", "_")
    download_dir = os.path.join(config.MODEL_DIR, "hf_models", safe_name)
    os.makedirs(download_dir, exist_ok=True)
    
    try:
        # List files in the model repo
        r = req.get(f"https://huggingface.co/api/models/{model_id}", timeout=15)
        r.raise_for_status()
        model_info = r.json()
        
        siblings = model_info.get("siblings", [])
        
        # Find downloadable model files (weights, config, etc.)
        target_files = []
        for f in siblings:
            fname = f.get("rfilename", "")
            # Download model weights, config, and tokenizer files
            if any(fname.endswith(ext) for ext in [".bin", ".pt", ".pth", ".safetensors", ".json", ".txt"]):
                if any(fname.endswith(ext) for ext in [".bin", ".pt", ".pth", ".safetensors"]):
                    target_files.insert(0, fname)  # Weights first
                else:
                    target_files.append(fname)
        
        if not target_files:
            raise HTTPException(status_code=404, detail=f"No downloadable model files found for {model_id}")
        
        # Only download the main weight file + config (not everything)
        weight_files = [f for f in target_files if any(f.endswith(e) for e in [".bin", ".pt", ".pth", ".safetensors"])]
        config_files = [f for f in target_files if f.endswith(".json")]
        
        files_to_download = (weight_files[:2] + config_files[:2])  # Max 4 files
        
        downloaded = []
        total_size = 0
        for fname in files_to_download:
            url = f"https://huggingface.co/{model_id}/resolve/main/{fname}"
            local_path = os.path.join(download_dir, fname.replace("/", "_"))
            
            logging.info(f"📥 Downloading: {url}")
            dr = req.get(url, stream=True, timeout=120)
            dr.raise_for_status()
            
            with open(local_path, "wb") as out:
                for chunk in dr.iter_content(chunk_size=8192):
                    out.write(chunk)
                    total_size += len(chunk)
            
            downloaded.append(fname)
            logging.info(f"  ✅ Saved: {local_path} ({os.path.getsize(local_path) / 1e6:.1f} MB)")
        
        # Register as a custom architecture in settings
        settings = _load_settings()
        hf_models = settings.get("hf_models", {})
        hf_models[safe_name] = {
            "model_id": model_id,
            "download_dir": download_dir,
            "files": downloaded,
            "total_size_mb": round(total_size / 1e6, 1),
            "downloaded_at": __import__("datetime").datetime.now().isoformat(),
        }
        settings["hf_models"] = hf_models
        _save_settings(settings)
        
        logging.info(f"✅ HuggingFace model downloaded: {model_id} ({total_size / 1e6:.1f} MB)")
        
        return {
            "status": "ok",
            "model_id": model_id,
            "files": downloaded,
            "total_size_mb": round(total_size / 1e6, 1),
            "download_dir": download_dir,
            "message": f"✅ Downloaded {model_id} ({total_size / 1e6:.1f} MB). Model saved to {download_dir}. Integration with predictor coming soon.",
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"HuggingFace download error for {model_id}: {e}")
        raise HTTPException(status_code=502, detail=f"Download failed: {str(e)}")


# ========================= HARDWARE MONITOR =========================

@app.get("/api/hardware")
def get_hardware_stats():
    """Real-time hardware stats: CPU, RAM, GPU temp/util/VRAM, disk."""
    import psutil

    stats: dict = {}

    # ── CPU ──────────────────────────────────────────────
    stats["cpu"] = {
        "percent": psutil.cpu_percent(interval=0.3),
        "per_core": psutil.cpu_percent(interval=0, percpu=True),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "freq_mhz": round(psutil.cpu_freq().current) if psutil.cpu_freq() else 0,
    }

    # CPU temp (Windows: may need Open Hardware Monitor / LibreHardwareMonitor)
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            # Try common sensor names
            for name in ['coretemp', 'k10temp', 'cpu_thermal', 'acpitz']:
                if name in temps and temps[name]:
                    stats["cpu"]["temp_c"] = round(temps[name][0].current, 1)
                    break
            # Fallback: first available sensor
            if "temp_c" not in stats["cpu"] and temps:
                first_key = list(temps.keys())[0]
                if temps[first_key]:
                    stats["cpu"]["temp_c"] = round(temps[first_key][0].current, 1)
    except Exception:
        pass

    # ── RAM (split: this app vs others) ───────────────────
    mem = psutil.virtual_memory()
    try:
        proc = psutil.Process(os.getpid())
        app_rss = proc.memory_info().rss
        # Include child processes (training workers, etc.)
        for child in proc.children(recursive=True):
            try:
                app_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        app_rss = 0
    others_used = max(0, mem.used - app_rss)
    stats["ram"] = {
        "total_gb": round(mem.total / 1e9, 1),
        "used_gb": round(mem.used / 1e9, 1),
        "app_used_gb": round(app_rss / 1e9, 2),
        "others_used_gb": round(others_used / 1e9, 1),
        "available_gb": round(mem.available / 1e9, 1),
        "percent": mem.percent,
    }

    # ── GPU (NVIDIA via pynvml for REAL VRAM) ────────────
    gpu = {"available": False}
    try:
        from pynvml import (
            nvmlInit, nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetName, nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetTemperature, nvmlDeviceGetUtilizationRates,
            nvmlDeviceGetFanSpeed, nvmlDeviceGetPowerUsage,
            nvmlDeviceGetEnforcedPowerLimit,
            NVML_TEMPERATURE_GPU,
        )
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        gpu["available"] = True

        # Name
        raw_name = nvmlDeviceGetName(handle)
        gpu["name"] = raw_name.decode('utf-8') if isinstance(raw_name, bytes) else raw_name

        # VRAM — real dedicated GPU memory (not shared / not torch tensor-only)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        gpu["vram_total_gb"] = round(mem_info.total / 1e9, 1)
        gpu["vram_used_gb"] = round(mem_info.used / 1e9, 2)
        gpu["vram_free_gb"] = round(mem_info.free / 1e9, 2)
        gpu["vram_percent"] = round(mem_info.used / max(mem_info.total, 1) * 100, 1)

        # Temp, utilization
        gpu["temp_c"] = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        rates = nvmlDeviceGetUtilizationRates(handle)
        gpu["utilization_percent"] = rates.gpu
        gpu["memory_utilization_percent"] = rates.memory

        # Fan
        try:
            gpu["fan_speed_percent"] = nvmlDeviceGetFanSpeed(handle)
        except Exception:
            pass

        # Power
        try:
            gpu["power_draw_w"] = round(nvmlDeviceGetPowerUsage(handle) / 1000, 1)
            gpu["power_limit_w"] = round(nvmlDeviceGetEnforcedPowerLimit(handle) / 1000, 1)
        except Exception:
            pass

    except ImportError:
        # Fallback to torch if pynvml not available
        try:
            import torch
            if torch.cuda.is_available():
                gpu["available"] = True
                gpu["name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                total = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
                gpu["vram_total_gb"] = round(total / 1e9, 1)
                gpu["vram_used_gb"] = round(torch.cuda.memory_reserved(0) / 1e9, 2)
                gpu["vram_percent"] = round(torch.cuda.memory_reserved(0) / max(total, 1) * 100, 1)
        except Exception:
            pass
    except Exception:
        pass
    stats["gpu"] = gpu

    # ── Disk ─────────────────────────────────────────────
    try:
        disk = psutil.disk_usage('/')
        stats["disk"] = {
            "total_gb": round(disk.total / 1e9, 1),
            "used_gb": round(disk.used / 1e9, 1),
            "free_gb": round(disk.free / 1e9, 1),
            "percent": disk.percent,
        }
    except Exception:
        # Windows: try the drive where the app is located
        try:
            drive = os.path.splitdrive(config.DATA_ROOT)[0] + '\\'
            disk = psutil.disk_usage(drive)
            stats["disk"] = {
                "total_gb": round(disk.total / 1e9, 1),
                "used_gb": round(disk.used / 1e9, 1),
                "free_gb": round(disk.free / 1e9, 1),
                "percent": disk.percent,
            }
        except Exception:
            stats["disk"] = {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent": 0}

    # ── System uptime ────────────────────────────────────
    try:
        boot_time = psutil.boot_time()
        uptime_sec = time.time() - boot_time
        days = int(uptime_sec // 86400)
        hours = int((uptime_sec % 86400) // 3600)
        mins = int((uptime_sec % 3600) // 60)
        stats["uptime"] = f"{days}d {hours}h {mins}m" if days > 0 else f"{hours}h {mins}m"
    except Exception:
        stats["uptime"] = "N/A"

    return stats


# ========================= TELEGRAM =========================

@app.post("/api/telegram/test")
def test_telegram():
    """Send a test message via Telegram to verify bot configuration."""
    try:
        from telegram_notifier import telegram
        result = telegram.test_connection()
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/telegram/status")
def telegram_status():
    """Check if Telegram notifications are enabled."""
    try:
        from telegram_notifier import telegram
        return {"enabled": telegram.is_enabled}
    except Exception:
        return {"enabled": False}


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
    
    # Accuracy: prefer live accuracy (from real predictions), fallback to training accuracy
    live_acc = getattr(predictor, '_live_accuracy', 0.0)
    live_samples = getattr(predictor, '_live_accuracy_samples', 0)
    training_acc = getattr(predictor, '_training_accuracy', 0.0) or predictor.last_validation_accuracy
    
    accuracy = live_acc if live_samples >= 3 else training_acc
    
    data = _sanitize_for_json({
        "prediction": pred,
        "quant": quant,
        "alt_signals": alt,
        "accuracy": accuracy,
        "accuracy_source": "live" if live_samples >= 3 else "training",
        "live_accuracy": live_acc,
        "live_accuracy_samples": live_samples,
        "training_accuracy": training_acc,
    })
    # _sanitize_for_json already converts NaN/Infinity → None recursively.
    return JSONResponse(content=data)


@app.get("/api/prediction/trajectory")
def get_prediction_trajectory(steps: int = 5, interval: str = "1m"):
    """Project future price trajectory based on SmallJamba prediction.

    Returns ~5 future price points for the chart ghost line overlay.
    The projection uses the model's 3-class prediction (UP/FLAT/DOWN)
    and confidence to extrapolate a directional trajectory from current price.

    Query params:
        steps: number of future candles to project (1-10, default 5)
        interval: timeframe for spacing (1m, 5m, 15m, 1h, 4h, 1d)
    """
    import math as _math

    # Clamp steps
    steps = max(1, min(steps, 10))

    # Get current price
    price = _get_live_price()
    if not price or price <= 0:
        raise HTTPException(503, "No live price available")

    # Get prediction
    pred = None
    if predictor is not None:
        try:
            pred = predictor.get_prediction()
        except Exception:
            pass

    # Default to FLAT if no prediction
    direction = "FLAT"
    confidence = 0.33
    probabilities = {"UP": 0.33, "FLAT": 0.34, "DOWN": 0.33}

    if pred and isinstance(pred, dict):
        direction = pred.get("direction", pred.get("signal", "FLAT"))
        confidence = float(pred.get("confidence", 0.33))
        probabilities = {
            "UP": float(pred.get("prob_up", pred.get("up_prob", 0.33))),
            "FLAT": float(pred.get("prob_flat", pred.get("flat_prob", 0.34))),
            "DOWN": float(pred.get("prob_down", pred.get("down_prob", 0.33))),
        }

    # Timeframe → seconds per candle
    tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
    candle_sec = tf_seconds.get(interval, 60)

    # Base move per candle: scale by confidence and recent volatility
    # Use a conservative 0.05% per minute as baseline, scaled by timeframe
    base_pct = 0.0005 * _math.sqrt(candle_sec / 60)  # sqrt scaling for timeframe

    # Direction multiplier
    if direction == "UP":
        dir_mult = 1.0
    elif direction == "DOWN":
        dir_mult = -1.0
    else:
        dir_mult = 0.0  # FLAT = no directional move

    # Scale by confidence (how sure the model is)
    move_pct = base_pct * dir_mult * confidence

    # Build trajectory points
    now = int(time.time())
    candle_start = (now // candle_sec) * candle_sec
    last_candle_time = candle_start  # Current candle's open time

    trajectory = []
    current_price = price
    for i in range(1, steps + 1):
        future_time = last_candle_time + (i * candle_sec)
        # Apply move with slight diminishing returns (uncertainty grows)
        decay = 1.0 / (1.0 + 0.1 * i)  # Confidence decays slightly per step
        current_price = current_price * (1.0 + move_pct * decay)
        trajectory.append({
            "time": future_time,
            "value": round(current_price, 2),
        })

    data = _sanitize_for_json({
        "trajectory": trajectory,
        "anchor": {"time": last_candle_time, "value": round(price, 2)},
        "direction": direction,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "model": "SmallJamba",
        "steps": steps,
        "interval": interval,
    })
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
@app.get("/api/health")
@app.get("/api/system-health")
def get_system_health():
    """System health: GPU, model age, data size, disk usage."""
    try:
        import torch
        
        health = {
            "model_trained": getattr(predictor, 'is_trained', False) if predictor else False,
            "model_version": "v6.0",
            "feature_count": len(getattr(predictor, 'features', [])) if predictor else 0,
            "validation_accuracy": getattr(predictor, 'last_validation_accuracy', 0) if predictor else 0,
            "ensemble_weights": {
                "xgb": getattr(predictor, 'xgb_weight', 1.0) if predictor else 1.0,
                "lstm": getattr(predictor, 'lstm_weight', 0.0) if predictor else 0.0,
            },
        }
        
        # GPU info — handle attribute differences across PyTorch versions
        try:
            if torch.cuda.is_available():
                health["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                total_mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
                health["gpu_vram_total_gb"] = round(total_mem / 1e9, 1)
                health["gpu_vram_used_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
            else:
                health["gpu_name"] = "CPU-only"
                health["gpu_vram_total_gb"] = 0
                health["gpu_vram_used_gb"] = 0
        except Exception:
            health["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
            health["gpu_vram_total_gb"] = 0
            health["gpu_vram_used_gb"] = 0
        
        # Model file age
        if predictor and hasattr(predictor, 'model_path') and os.path.exists(predictor.model_path):
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
        
        # ── Uptime & Retrain Schedule ──────────────────
        uptime_sec = time.time() - _app_boot_time
        uptime_h = int(uptime_sec // 3600)
        uptime_m = int((uptime_sec % 3600) // 60)
        health["uptime"] = f"{uptime_h}h {uptime_m}m" if uptime_h > 0 else f"{uptime_m}m"
        health["uptime_seconds"] = round(uptime_sec)
        
        health["is_retraining"] = _retrain_status.get("is_retraining", False)
        health["retrain_count"] = _retrain_status.get("retrain_count", 0)
        health["last_retrain"] = _retrain_status.get("last_retrain")
        
        # Next retrain countdown
        nxt = _retrain_status.get("next_retrain")
        if nxt:
            try:
                from datetime import datetime as _dt
                next_dt = _dt.fromisoformat(nxt)
                now_dt = _dt.now()
                delta_sec = max(0, (next_dt - now_dt).total_seconds())
                rem_h = int(delta_sec // 3600)
                rem_m = int((delta_sec % 3600) // 60)
                if delta_sec <= 0:
                    health["next_retrain_countdown"] = "imminent"
                elif rem_h > 0:
                    health["next_retrain_countdown"] = f"{rem_h}h {rem_m}m"
                else:
                    health["next_retrain_countdown"] = f"{rem_m}m"
                health["next_retrain_ts"] = nxt
            except Exception:
                health["next_retrain_countdown"] = "unknown"
        else:
            health["next_retrain_countdown"] = "pending"
        # ── AI Services Status ──────────────────────────
        ai_services = {}
        
        # Ollama — ping local server
        try:
            ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            ol_resp = requests.get(f"{ollama_base}/api/tags", timeout=2)
            if ol_resp.status_code == 200:
                ol_data = ol_resp.json()
                models = [m.get("name", "?") for m in (ol_data.get("models") or [])]
                # Check which model is actively loaded
                active_model = None
                try:
                    ps_resp = requests.get(f"{ollama_base}/api/ps", timeout=2)
                    if ps_resp.status_code == 200:
                        running = ps_resp.json().get("models", [])
                        if running:
                            active_model = running[0].get("name", "?")
                except Exception:
                    pass
                ai_services["ollama"] = {
                    "connected": True,
                    "models": models[:5],
                    "active_model": active_model,
                    "model_count": len(models),
                }
            else:
                ai_services["ollama"] = {"connected": False, "reason": f"HTTP {ol_resp.status_code}"}
        except requests.exceptions.ConnectionError:
            ai_services["ollama"] = {"connected": False, "reason": "Not running"}
        except Exception:
            ai_services["ollama"] = {"connected": False, "reason": "Unreachable"}
        
        # OpenAI — check if API key is configured
        openai_key = getattr(config, 'OPENAI_API_KEY', '') or os.environ.get('OPENAI_API_KEY', '')
        ai_services["openai"] = {
            "connected": bool(openai_key and len(openai_key) > 10),
            "key_preview": f"{openai_key[:8]}...{openai_key[-4:]}" if openai_key and len(openai_key) > 12 else None,
        }
        
        # Gemini — check if API key is configured
        gemini_key = getattr(config, 'GEMINI_API_KEY', '') or os.environ.get('GEMINI_API_KEY', '')
        ai_services["gemini"] = {
            "connected": bool(gemini_key and len(gemini_key) > 10),
            "key_preview": f"{gemini_key[:8]}...{gemini_key[-4:]}" if gemini_key and len(gemini_key) > 12 else None,
        }
        
        health["ai_services"] = ai_services
        
        return health
    except Exception as e:
        logging.error(f"Health endpoint error: {e}")
        return {
            "gpu_name": "Unknown",
            "gpu_vram_total_gb": 0,
            "gpu_vram_used_gb": 0,
            "model_trained": False,
            "model_version": "v6.0",
            "feature_count": 0,
            "validation_accuracy": 0,
            "ensemble_weights": {"xgb": 1.0, "lstm": 0.0},
            "error": str(e),
        }


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
def get_market_data(limit: int = 500, interval: str = "1m"):
    """OHLCV data for price chart — fetched from Binance klines API (free, no key).
    
    Supports: 1m, 5m, 15m, 1h, 4h, 1d
    Returns up to 1000 candles per request.
    Falls back to local data if Binance is unreachable.
    """
    global _kline_cache
    
    # Validate interval
    valid_intervals = {"1m", "5m", "15m", "1h", "4h", "1d"}
    if interval not in valid_intervals:
        interval = "1m"
    limit = min(max(limit, 10), 1000)
    
    # Cache key: interval + limit
    cache_key = f"{interval}_{limit}"
    now = time.time()
    
    # Return cache if fresh (5 min for larger TFs, 30s for 1m)
    cache_ttl = 30 if interval == "1m" else 300
    if cache_key in _kline_cache:
        cached = _kline_cache[cache_key]
        if now - cached["ts"] < cache_ttl:
            return {"candles": cached["candles"], "count": len(cached["candles"])}
    
    # ─── Primary: Binance REST API (free, no key) ──────
    candles = _fetch_binance_klines(interval, limit)
    
    # ─── Fallback: local parquet/csv ───────────────────
    if not candles:
        candles = _fetch_local_candles(interval, limit)
    
    # Cache result
    _kline_cache[cache_key] = {"candles": candles, "ts": now}
    
    return {"candles": candles, "count": len(candles)}


# In-memory kline cache
_kline_cache: dict = {}


def _fetch_binance_klines(interval: str, limit: int) -> list:
    """Fetch OHLCV klines from Binance public API (free, no key needed)."""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            logging.warning(f"Binance klines HTTP {resp.status_code}")
            return []
        
        raw = resp.json()
        candles = []
        for k in raw:
            # Binance kline format: [openTime, open, high, low, close, volume, closeTime, ...]
            candles.append({
                "timestamp": int(k[0]),  # open time in ms
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        
        logging.info(f"Binance klines: {len(candles)} {interval} candles fetched")
        return candles
        
    except requests.exceptions.Timeout:
        logging.warning("Binance klines: timeout")
        return []
    except Exception as e:
        logging.warning(f"Binance klines error: {e}")
        return []


def _fetch_local_candles(interval: str, limit: int) -> list:
    """Fallback: load from local parquet/csv and aggregate."""
    interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    interval_mins = interval_map.get(interval, 1)
    lookback_minutes = int(limit * interval_mins * 1.1) + interval_mins
    
    try:
        if os.path.exists(config.MARKET_DATA_PARQUET_PATH):
            full_df = pd.read_parquet(config.MARKET_DATA_PARQUET_PATH)
        elif os.path.exists(config.MARKET_DATA_PATH):
            full_df = pd.read_csv(config.MARKET_DATA_PATH)
        else:
            return []
    except Exception:
        return []
    
    if full_df is None or full_df.empty:
        return []
    
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
    cutoff = full_df['timestamp'].max() - pd.Timedelta(minutes=lookback_minutes)
    df = full_df[full_df['timestamp'] >= cutoff].copy()
    
    if df.empty:
        return []
    
    if interval_mins > 1:
        df = df.set_index('timestamp')
        agg = df.resample(f'{interval_mins}min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna(subset=['open'])
        df = agg.reset_index()
    
    df = df.tail(limit).reset_index(drop=True)
    df['timestamp'] = (pd.to_datetime(df['timestamp']).astype('int64') // 10**6).astype('int64')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')


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


# NOTE: /api/news is defined below in the NEWS INTELLIGENCE FEED section (aggregated feed)


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


# ─── Paper Trading Aliases (frontend uses /api/paper/*) ──────

@app.post("/api/paper/start")
def paper_start():
    """Alias for /api/bot/start — used by PaperTrading page."""
    return start_bot()


@app.post("/api/paper/stop")
def paper_stop():
    """Alias for /api/bot/stop — used by PaperTrading page."""
    return stop_bot()


@app.post("/api/paper/reset")
def paper_reset():
    """Reset paper trader: close all positions, clear history, reset balance."""
    if trader is None:
        raise HTTPException(503, "Engine not ready")
    trader.is_running = False
    _auto_trade_stop.set()
    # Close all positions
    for pos in list(trader.positions):
        try:
            trader.close_position(pos, pos.entry_price, "RESET")
        except Exception:
            pass
    trader.positions.clear()
    trader.trade_history.clear()
    trader.balance = trader.starting_balance
    return {"status": "reset", "balance": trader.balance}


@app.get("/api/paper/trades")
def paper_trades(limit: int = 50):
    """Alias for /api/trade-history — used by PaperTrading page."""
    return get_trade_history(limit)


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


@app.get("/api/retrain-history")
def get_retrain_history(limit: int = 20):
    """Training history with accuracy deltas and trend analysis."""
    import json as _json
    retrain_history_path = os.path.join(config.LOG_DIR, 'retrain_history.json')
    
    if not os.path.exists(retrain_history_path):
        return {"entries": [], "summary": None}
    
    try:
        with open(retrain_history_path, 'r') as f:
            history = _json.load(f)
    except Exception:
        return {"entries": [], "summary": None}
    
    # Return most recent entries (newest first)
    entries = list(reversed(history[-limit:]))
    
    # Build summary statistics
    accuracies = [e['accuracy'] for e in history if e.get('accuracy') is not None]
    deltas = [e['delta'] for e in history if e.get('delta') is not None]
    
    summary = None
    if accuracies:
        summary = {
            "total_retrains": len(history),
            "best_accuracy": round(max(accuracies), 2),
            "worst_accuracy": round(min(accuracies), 2),
            "latest_accuracy": round(accuracies[-1], 2) if accuracies else None,
            "avg_delta": round(sum(deltas) / len(deltas), 2) if deltas else None,
            "positive_retrains": sum(1 for d in deltas if d > 0),
            "negative_retrains": sum(1 for d in deltas if d < 0),
            "neutral_retrains": sum(1 for d in deltas if d == 0),
            "current_streak": history[-1].get('streak', 0) if history else 0,
        }
    
    return {"entries": entries, "summary": summary}

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


# NOTE: /api/settings GET is defined above (merged JSON settings + env keys)


class SettingsUpdate(BaseModel):
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_secret_key: Optional[str] = None


@app.post("/api/settings/keys-legacy")
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
    
    # Update config module + os.environ so all modules see the new keys immediately
    for k, v in updates.items():
        if hasattr(config, k):
            setattr(config, k, v)
        os.environ[k] = v
    
    # Also update nexus_agent's cached key if OpenAI key changed
    if "OPENAI_API_KEY" in updates:
        try:
            import nexus_agent
            nexus_agent._OPENAI_API_KEY = updates["OPENAI_API_KEY"]
        except Exception:
            pass
    
    return {"saved": True, "keys_updated": list(updates.keys())}


# ── Hugging Face Sync Endpoints ──────────────────────
@app.post("/api/settings/hf-sync/push")
async def hf_sync_push():
    """Manual push of local models to HF Hub."""
    result = hf_sync.push_to_hub()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result

@app.post("/api/settings/hf-sync/pull")
async def hf_sync_pull():
    """Manual pull of latest models from HF Hub."""
    result = hf_sync.pull_from_hub()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    # Re-init predictor states after pull
    global predictor
    if predictor is not None:
        predictor = NexusPredictor()
    return result


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
#  GPU FARM (AssGPU Mini-Game)
# ============================================================

# gpu_game is initialized by _init_engines() and available as module global

class TransferRequest(BaseModel):
    amount: float

class MergeRequest(BaseModel):
    card_id_1: int
    card_id_2: int

class SellAssRequest(BaseModel):
    amount: float

@app.get("/api/game/state")
def game_state():
    """Get full GPU Farm game state."""
    if gpu_game is None:
        raise HTTPException(503, "GPU game not initialized")
    return gpu_game.get_state()

@app.post("/api/game/transfer")
def game_transfer(req: TransferRequest):
    """Transfer USD from trading wallet to game wallet (one-way)."""
    if gpu_game is None:
        raise HTTPException(503, "GPU game not initialized")
    if req.amount <= 0:
        raise HTTPException(400, "Amount must be positive")
    # Deduct from paper trader balance
    if trader and hasattr(trader, 'balance'):
        if trader.balance < req.amount:
            raise HTTPException(400, f"Insufficient trading balance (${trader.balance:.2f})")
        trader.balance -= req.amount
        trader._save_equity_point()
    result = gpu_game.transfer_in(req.amount)
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "Transfer failed"))
    result["trading_balance"] = round(trader.balance, 2) if trader else 0
    # Log to NexusLogger
    try:
        from nexus_logger import get_logger
        get_logger().log_game_transfer(req.amount, result["game_balance"])
    except Exception:
        pass
    return result

@app.post("/api/game/buy-card")
def game_buy_card():
    """Buy a new 10 Series GPU card."""
    if gpu_game is None:
        raise HTTPException(503, "GPU game not initialized")
    result = gpu_game.buy_card()
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "Buy failed"))
    try:
        from nexus_logger import get_logger
        card = result.get("card", {})
        get_logger().log_game_buy_card(card.get("tier", 1), "10s", result.get("cost", 0), result.get("game_balance", 0))
    except Exception:
        pass
    return result

@app.post("/api/game/merge")
def game_merge(req: MergeRequest):
    """Merge two same-tier GPU cards into next tier."""
    if gpu_game is None:
        raise HTTPException(503, "GPU game not initialized")
    result = gpu_game.merge_cards(req.card_id_1, req.card_id_2)
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "Merge failed"))
    try:
        from nexus_logger import get_logger
        new_card = result.get("new_card", {})
        get_logger().log_game_merge(new_card.get("tier", 2) - 1, new_card.get("tier", 2), new_card.get("label", "?"))
    except Exception:
        pass
    return result

@app.post("/api/game/sell-ass")
def game_sell_ass(req: SellAssRequest):
    """Sell ASS coin at current market price."""
    if gpu_game is None:
        raise HTTPException(503, "GPU game not initialized")
    if req.amount <= 0:
        raise HTTPException(400, "Amount must be positive")
    result = gpu_game.sell_ass(req.amount)
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "Sell failed"))
    try:
        from nexus_logger import get_logger
        get_logger().log_game_sell_ass(result.get("amount_sold", 0), result.get("price", 0), result.get("usd_received", 0))
    except Exception:
        pass
    return result

@app.get("/api/game/price-history")
def game_price_history():
    """Get ASS coin price history for charting."""
    if gpu_game is None:
        raise HTTPException(503, "GPU game not initialized")
    state = gpu_game.get_state()
    return {"history": state.get("ass_price_history", [])}

# Background game tick thread is started by lifespan (_gpu_game_tick_loop)


# ============================================================
#  NEXUS AGENT (AI CHAT)
# ============================================================

class AgentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.post("/api/agent/chat")
def agent_chat(req: AgentChatRequest):
    """Chat with the Nexus Agent — streams response as SSE (multi-provider)."""
    from starlette.responses import StreamingResponse
    import nexus_memory

    def sse_generator():
        try:
            sid = req.session_id or nexus_agent.get_or_create_session()

            # Save user message
            user_msg_id = nexus_memory.save_message(sid, "user", req.message)

            # Build state + multi-layer message stack
            state = nexus_agent.build_state_snapshot(predictor, trader, collector)
            state_json = json.dumps(state, indent=2, default=str)
            knowledge_context = nexus_memory.get_knowledge_summary()

            # Conversation history
            conv_history = nexus_memory.get_recent_messages(session_id=sid, limit=10)
            conv_history = [m for m in conv_history if m['id'] != user_msg_id]

            # Build multi-layer message stack (for OpenAI)
            messages = nexus_agent.build_messages(
                user_message=req.message,
                state_json=state_json,
                conversation_history=conv_history,
                knowledge_context=knowledge_context,
            )

            # Flat prompt for Ollama / Gemini
            full_prompt = (
                nexus_agent.MASTER_SYSTEM_PROMPT + "\n\n"
                + nexus_agent.DEVELOPER_PROMPT + "\n\n"
                + nexus_agent.REF_MODEL_STACK
                + f"\n\n[LIVE STATE]\n```json\n{state_json}\n```"
            )
            if knowledge_context:
                full_prompt += f"\n\n{knowledge_context}"

            # ── Gather available providers (same logic as nexus_agent.chat) ──
            openai_key = getattr(config, 'OPENAI_API_KEY', None) or os.environ.get('OPENAI_API_KEY', '')
            if not openai_key:
                try:
                    env_path = os.path.join(config.DATA_ROOT, ".env")
                    if os.path.exists(env_path):
                        with open(env_path, 'r', encoding='utf-8', errors='replace') as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith("OPENAI_API_KEY="):
                                    openai_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                except Exception:
                    pass
            if not openai_key:
                openai_key = nexus_agent._OPENAI_API_KEY

            gemini_key = os.environ.get('GEMINI_API_KEY', '')
            if not gemini_key:
                try:
                    env_path = os.path.join(config.DATA_ROOT, ".env")
                    if os.path.exists(env_path):
                        with open(env_path, 'r', encoding='utf-8', errors='replace') as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith("GEMINI_API_KEY="):
                                    gemini_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                except Exception:
                    pass

            ollama_model = nexus_agent._get_ollama_model()

            # Build provider priority list
            preferred = nexus_agent._get_provider_priority()
            providers = []
            if preferred == 'ollama':
                if ollama_model: providers.append(('ollama', ollama_model))
                if openai_key: providers.append(('openai', openai_key))
                if gemini_key: providers.append(('gemini', gemini_key))
            elif preferred == 'openai':
                if openai_key: providers.append(('openai', openai_key))
                if ollama_model: providers.append(('ollama', ollama_model))
                if gemini_key: providers.append(('gemini', gemini_key))
            elif preferred == 'gemini':
                if gemini_key: providers.append(('gemini', gemini_key))
                if ollama_model: providers.append(('ollama', ollama_model))
                if openai_key: providers.append(('openai', openai_key))
            elif preferred == 'embedded':
                try:
                    import embedded_llm
                    if embedded_llm.is_available():
                        providers.append(('embedded', None))
                except ImportError:
                    pass
                if ollama_model: providers.append(('ollama', ollama_model))
                if openai_key: providers.append(('openai', openai_key))
                if gemini_key: providers.append(('gemini', gemini_key))
            else:
                if ollama_model: providers.append(('ollama', ollama_model))
                if openai_key: providers.append(('openai', openai_key))
                if gemini_key: providers.append(('gemini', gemini_key))

            # Add embedded LLM as last-resort fallback (if not already preferred)
            if preferred != 'embedded':
                try:
                    import embedded_llm
                    if embedded_llm.is_available():
                        providers.append(('embedded', None))
                except ImportError:
                    pass

            if not providers:
                yield f"data: {json.dumps({'content': '⚠️ No LLM available. Install Ollama, or add an OpenAI/Gemini API key in Settings.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # ── Try providers in priority order ──
            last_error = ""
            for provider_name, credential in providers:
                try:
                    full_reply = ""

                    if provider_name == 'openai':
                        # Stream from OpenAI
                        for chunk in nexus_agent._call_openai_stream(messages, credential):
                            full_reply += chunk
                            yield f"data: {json.dumps({'content': chunk})}\n\n"
                        provider_label = "gpt-4.1-mini"

                    elif provider_name == 'ollama':
                        # Ollama (non-streaming, send as single chunk)
                        full_reply = nexus_agent._call_ollama(
                            full_prompt, req.message, credential,
                            conversation_history=conv_history
                        )
                        # Stream word-by-word for smooth UX
                        words = full_reply.split(' ')
                        for i, word in enumerate(words):
                            chunk = word + (' ' if i < len(words) - 1 else '')
                            yield f"data: {json.dumps({'content': chunk})}\n\n"
                        provider_label = f"ollama:{credential}"

                    elif provider_name == 'gemini':
                        # Gemini (non-streaming, send as single chunk)
                        full_reply = nexus_agent._call_gemini(
                            full_prompt, req.message, credential,
                            conversation_history=conv_history
                        )
                        words = full_reply.split(' ')
                        for i, word in enumerate(words):
                            chunk = word + (' ' if i < len(words) - 1 else '')
                            yield f"data: {json.dumps({'content': chunk})}\n\n"
                        provider_label = f"gemini-2.0-flash"

                    elif provider_name == 'embedded':
                        import embedded_llm
                        embedded_prompt = (
                            "You are Dr. Nexus, a quantitative trading AI analyst. "
                            "Analyze the live state data below and answer the user's question. "
                            "Use markdown: headers (##), bold (**), tables, bullet points. "
                            "Start analytical responses with: # 🔮 Dr. Nexus | [Title]\n\n"
                            f"[LIVE STATE]\n```json\n{state_json}\n```"
                        )
                        full_reply = embedded_llm.generate(
                            system_prompt=embedded_prompt,
                            user_message=req.message,
                            conversation_history=conv_history,
                            max_new_tokens=800,
                            temperature=0.7,
                        )
                        words = full_reply.split(' ')
                        for i, word in enumerate(words):
                            chunk = word + (' ' if i < len(words) - 1 else '')
                            yield f"data: {json.dumps({'content': chunk})}\n\n"
                        provider_label = f"embedded:{embedded_llm.MODEL_LABEL}"

                    else:
                        continue

                    yield f"data: {json.dumps({'meta': {'provider': provider_label}})}\n\n"
                    yield "data: [DONE]\n\n"

                    # Save full reply to memory
                    if full_reply:
                        nexus_memory.save_message(sid, "agent", full_reply, provider=provider_label)

                    return  # Success — exit generator

                except Exception as e:
                    last_error = str(e)[:200]
                    logging.warning(f"[Dr. Nexus SSE] {provider_name} failed: {last_error}, trying next...")
                    continue

            # All providers failed
            yield f"data: {json.dumps({'content': f'⚠️ All LLM providers failed. Last error: {last_error}'})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logging.error(f"Agent stream error: {e}")
            yield f"data: {json.dumps({'content': f'⚠️ Error: {str(e)}'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

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

@app.get("/api/agent/sessions")
def agent_sessions():
    """List all chat sessions with titles."""
    import nexus_memory
    sessions = nexus_memory.get_all_sessions()
    current = nexus_agent.get_or_create_session()
    return {"sessions": sessions, "current_session_id": current}

@app.delete("/api/agent/sessions/{session_id}")
def agent_delete_session(session_id: str):
    """Delete a chat session and all its messages."""
    import nexus_memory
    deleted = nexus_memory.delete_session(session_id)
    return {"deleted": deleted, "session_id": session_id}


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
                    "headline": title,
                    "source": source,
                    "sentiment": sentiment.lower(),
                    "sentiment_score": round(score, 2),
                    "url": post.get("url", ""),
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
                    "headline": f"🔥 {name} ({symbol}) trending — #{rank} by market cap, 24h: {price_change:+.1f}%",
                    "source": "CoinGecko Trending",
                    "sentiment": sent.lower(),
                    "sentiment_score": round(min(1, max(-1, price_change / 10)), 2),
                })
    except Exception as e:
        logging.debug(f"CoinGecko trending error: {e}")
    return items


def _analyze_headline_sentiment(title: str) -> tuple:
    """Keyword-based sentiment scoring for news headlines."""
    t = title.lower()
    bullish = ['surge', 'soar', 'rally', 'breakout', 'bullish', 'record high',
                'all-time high', 'ath', 'moon', 'adoption', 'approval', 'buy',
                'accumulate', 'inflow', 'upgrade', 'etf', 'partnership',
                'institutional', 'support', 'recover', 'bounce', 'gain', 'rising',
                'outperform', 'profit', 'milestone', 'breakthrough', 'launch',
                'integrate', 'grow', 'expand']
    bearish = ['crash', 'plunge', 'dump', 'bearish', 'sell-off', 'selloff',
                'hack', 'exploit', 'fraud', 'ban', 'crackdown', 'regulation',
                'lawsuit', 'sec', 'fine', 'collapse', 'bankruptcy', 'liquidat',
                'outflow', 'decline', 'drop', 'fall', 'warn', 'risk', 'fear',
                'scam', 'rug pull', 'arrest', 'sanction', 'attack', 'vulnerability']
    bull_hits = sum(1 for kw in bullish if kw in t)
    bear_hits = sum(1 for kw in bearish if kw in t)
    if bull_hits > bear_hits:
        return "BULLISH", round(min(0.3 + bull_hits * 0.15, 0.9), 2)
    elif bear_hits > bull_hits:
        return "BEARISH", round(max(-0.3 - bear_hits * 0.15, -0.9), 2)
    return "NEUTRAL", 0.0


def _fetch_rss_news():
    """Fetch real crypto news from multiple RSS feeds (free, no API key)."""
    items = []
    feeds = [
        ("https://cointelegraph.com/rss", "CoinTelegraph"),
        ("https://www.coindesk.com/arc/outboundfeeds/rss/", "CoinDesk"),
        ("https://bitcoinmagazine.com/feed", "Bitcoin Magazine"),
        ("https://www.newsbtc.com/feed/", "NewsBTC"),
    ]
    try:
        import feedparser
    except ImportError:
        logging.warning("feedparser not installed — run: pip install feedparser")
        return items

    for url, source_name in feeds:
        try:
            # Use requests with timeout then parse the content
            # (feedparser.parse(url) can hang without timeout)
            resp = requests.get(url, timeout=6, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NexusBot/1.0"
            })
            if resp.status_code != 200:
                logging.warning(f"RSS {source_name}: HTTP {resp.status_code}")
                continue
            feed = feedparser.parse(resp.content)
            if not feed.entries:
                logging.warning(f"RSS {source_name}: 0 entries (bozo={feed.bozo})")
                continue
            logging.info(f"RSS {source_name}: {len(feed.entries)} entries fetched")
            for entry in feed.entries[:4]:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                link = entry.get("link", "")
                # Parse time
                pub = entry.get("published", entry.get("updated", ""))
                time_str = ""
                if pub:
                    try:
                        from email.utils import parsedate_to_datetime
                        dt = parsedate_to_datetime(pub)
                        from datetime import datetime, timezone
                        delta = datetime.now(timezone.utc) - dt
                        hrs = int(delta.total_seconds() // 3600)
                        if hrs < 1:
                            time_str = f"{int(delta.total_seconds() // 60)}m ago"
                        elif hrs < 24:
                            time_str = f"{hrs}h ago"
                        else:
                            time_str = f"{hrs // 24}d ago"
                    except Exception:
                        pass

                sentiment, score = _analyze_headline_sentiment(title)
                items.append({
                    "headline": title,
                    "source": source_name,
                    "sentiment": sentiment.lower(),
                    "sentiment_score": score,
                    "time": time_str,
                    "url": link,
                })
        except requests.exceptions.Timeout:
            logging.warning(f"RSS {source_name}: timeout after 6s")
        except Exception as e:
            logging.warning(f"RSS feed error ({source_name}): {e}")
    
    logging.info(f"RSS total: {len(items)} items from {len(feeds)} feeds")
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
                "headline": f"🤖 AI predicts {'📈 LONG' if direction == 'LONG' else '📉 SHORT'} with {confidence:.0f}% confidence — Target: ${pred.get('target_price', 0):,.0f}",
                "source": "Nexus AI Engine",
                "sentiment": sent.lower(),
                "sentiment_score": round(score, 2),
            })
            
            # Regime signal
            regime_sent = {"TRENDING": "BULLISH", "MEAN_REVERTING": "NEUTRAL", "VOLATILE": "BEARISH", "CHAOTIC": "BEARISH"}
            items.append({
                "headline": f"📊 Market regime: {regime} — Hurst exponent: {hurst:.3f} ({'trending' if hurst > 0.55 else 'mean-reverting' if hurst < 0.45 else 'random walk'})",
                "source": "Quant Engine",
                "sentiment": regime_sent.get(regime, "NEUTRAL").lower(),
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
                        "headline": f"😱 Fear & Greed Index: {fg_val}/100 ({fg_label}) — {'contrarian buy signal' if fg_val < 25 else 'contrarian sell signal' if fg_val > 75 else 'no extreme'}",
                        "source": "Alternative.me",
                        "sentiment": fg_sent.lower(),
                        "sentiment_score": round((fg_val - 50) / 50, 2),
                    })
                except (ValueError, TypeError):
                    pass
            
            # Accuracy tracking
            acc = predictor.last_validation_accuracy
            if acc > 0:
                items.append({
                    "headline": f"🎯 Model accuracy: {acc:.1f}% ({('outperforming' if acc > 52 else 'at') + ' baseline'}) — Statistically {'verified ✅' if predictor.is_statistically_verified else 'tracking...'}",
                    "source": "Validation Engine",
                    "sentiment": "bullish" if acc > 52 else "neutral",
                    "sentiment_score": round((acc - 50) / 20, 2),
                })
            
            # Quant engine signals
            q = predictor.last_quant_analysis or {}
            if q:
                flow = q.get("order_flow", {})
                bp = flow.get("buy_pressure", 0.5)
                if abs(bp - 0.5) > 0.05:
                    items.append({
                        "headline": f"📈 Order flow: {'buyers dominating' if bp > 0.55 else 'sellers dominating'} — Buy pressure: {bp:.1%}",
                        "source": "Order Flow Analysis",
                        "sentiment": "bullish" if bp > 0.55 else "bearish",
                        "sentiment_score": round((bp - 0.5) * 4, 2),
                    })
                
                jump = q.get("jump_risk", {})
                jl = jump.get("level", "")
                if jl and jl != "LOW":
                    items.append({
                        "headline": f"⚡ Jump risk: {jl} — Hawkes process detecting volatility clustering",
                        "source": "Hawkes Process",
                        "sentiment": "bearish",
                        "sentiment_score": -0.4 if jl == "HIGH" else -0.2,
                    })
        
        # Portfolio signals
        if trader:
            stats = trader.get_stats()
            pnl_pct = stats.get("total_pnl_pct", 0)
            n_pos = len(trader.positions)
            if n_pos > 0:
                items.append({
                    "headline": f"💼 {n_pos} position{'s' if n_pos > 1 else ''} open — Portfolio PnL: {pnl_pct:+.2f}%, Balance: ${stats.get('balance', 10000):,.0f}",
                    "source": "Paper Trader",
                    "sentiment": "bullish" if pnl_pct > 0 else ("bearish" if pnl_pct < 0 else "neutral"),
                    "sentiment_score": round(min(1, max(-1, pnl_pct / 5)), 2),
                })
            
            dd = stats.get("max_drawdown_pct", 0)
            if dd > 5:
                items.append({
                    "headline": f"⚠️ Drawdown alert: {dd:.1f}% max drawdown — {'circuit breaker zone' if dd > 15 else 'elevated risk'}",
                    "source": "Risk Manager",
                    "sentiment": "bearish",
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
    
    # 3. RSS feeds (CoinTelegraph, CoinDesk, Bitcoin Magazine, NewsBTC)
    all_items.extend(_fetch_rss_news())
    
    # 4. CoinGecko trending
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
#  DERIVATIVES DATA
# ============================================================

@app.get("/api/derivatives")
async def get_derivatives():
    """Real-time derivatives data: funding rate, OI, basis, mark-index spread."""
    if derivs_feed is None:
        return {
            "enabled": False,
            "message": "Derivatives feed not initialized",
            "snapshot": {},
            "features": {},
        }
    return {
        "enabled": True,
        "snapshot": derivs_feed.get_snapshot_dict(),
        "features": derivs_feed.get_features(),
        "feature_names": derivs_feed.get_feature_names(),
    }


# ============================================================
#  PROBABILITY CALIBRATION (Phase 2)
# ============================================================

@app.get("/api/calibration")
async def get_calibration():
    """Probability calibration diagnostics: ECE, Brier score, fit status."""
    if predictor is None or predictor.prob_calibrator is None:
        return {
            "enabled": False,
            "message": "Calibrator not available",
            "diagnostics": {},
        }
    return {
        "enabled": True,
        "diagnostics": predictor.prob_calibrator.get_diagnostics(),
    }


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("NEXUS_API_PORT", 8420))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
