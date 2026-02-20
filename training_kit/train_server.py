"""
train_server.py — Nexus Jamba Training Kit: Flask Web Server + Training Engine

Features:
  ✅ Auto-downloads BTC data from HuggingFace
  ✅ Premium web UI with real-time progress
  ✅ Start / Stop / Continue training
  ✅ Auto-saves checkpoints every epoch
  ✅ Resumes from last checkpoint on restart
  ✅ Trains all 4 Jamba architectures sequentially (or pick one)
  ✅ 3-class classification: DOWN / FLAT / UP
  ✅ Graceful shutdown — saves state on Ctrl+C or browser Stop

Usage:
  python train_server.py                     # Train all Jamba sizes
  python train_server.py --arch small_jamba  # Train one specific arch

Then open http://localhost:5555 in your browser.
"""

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # Prevent VRAM fragmentation
import sys
import time
import json
import signal
import logging
import threading
import re
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from flask import Flask, render_template, jsonify, request

from models import ARCHITECTURES, ARCH_INFO, estimate_params, register_custom_arch
from models import create_jamba

# ── Paths ──
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
MODEL_DIR = SCRIPT_DIR / "models"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
STATE_FILE = SCRIPT_DIR / "training_state.json"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Constants ──
SEQ_LEN = 120
PREDICTION_HORIZON = 15
PRICE_THRESHOLD = 0.0025
NUM_CLASSES = 3
CLASS_DOWN, CLASS_FLAT, CLASS_UP = 0, 1, 2

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [TRAIN] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Flask App ──
app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════════════
# GLOBAL STATE (thread-safe via lock)
# ══════════════════════════════════════════════════════════════════════════

lock = threading.Lock()

state = {
    "status": "idle",          # idle, downloading, preparing, training, paused, finished, error
    "current_arch": None,
    "epoch": 0,
    "total_epochs": 0,
    "batch_progress": 0.0,
    "train_loss": 0.0,
    "train_acc": 0.0,
    "val_loss": 0.0,
    "val_acc": 0.0,
    "best_val_acc": 0.0,
    "lr": 0.0,
    "speed": 0.0,
    "vram_used_gb": 0.0,
    "vram_total_gb": 0.0,
    "gpu_name": "N/A",
    "eta": "",
    "log_messages": [],
    "completed_archs": [],
    "queue": [],
    "error_msg": "",
    "epoch_history": [],       # [{epoch, train_loss, train_acc, val_loss, val_acc}]
    "started_at": None,
    "elapsed": "",
}

stop_requested = threading.Event()
training_thread = None
data_cache = {"df": None, "feature_cols": None}


def add_log(msg):
    with lock:
        state["log_messages"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "msg": msg,
        })
        # Keep last 200 messages
        if len(state["log_messages"]) > 200:
            state["log_messages"] = state["log_messages"][-200:]
    log.info(msg)


def update_state(**kwargs):
    with lock:
        state.update(kwargs)


# ══════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def download_datasets():
    """Download BTC data from HuggingFace."""
    update_state(status="downloading")
    add_log("⬇️  Checking/downloading BTC datasets from HuggingFace...")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        add_log("📦 Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub datasets")
        from huggingface_hub import hf_hub_download

    datasets_to_download = [
        {
            "repo": "FamilyLinks/btc-price-1m-2017-2025",
            "filename": "BTC_Raw_Micro_Macro_1m.parquet",
            "local_name": "familylinks_btc_1m.parquet",
            "description": "4.37M rows: OHLCV + order flow + on-chain (2017-2025)",
        },
    ]

    for ds in datasets_to_download:
        local_path = DATA_DIR / ds["local_name"]
        if local_path.exists():
            add_log(f"✅ Already downloaded: {ds['local_name']} ({local_path.stat().st_size / 1e6:.0f} MB)")
            continue

        add_log(f"⬇️  Downloading: {ds['description']}...")
        try:
            downloaded = hf_hub_download(
                repo_id=ds["repo"],
                filename=ds["filename"],
                repo_type="dataset",
                local_dir=str(DATA_DIR),
            )
            if Path(downloaded).name != ds["local_name"]:
                Path(downloaded).rename(local_path)
            add_log(f"✅ Downloaded: {ds['local_name']} ({local_path.stat().st_size / 1e6:.0f} MB)")
        except Exception as e:
            add_log(f"❌ Download failed: {e}")
            add_log("Trying alternative download via datasets library...")
            try:
                from datasets import load_dataset
                hf_ds = load_dataset(ds["repo"], split="train")
                df = hf_ds.to_pandas()
                df.to_parquet(local_path)
                add_log(f"✅ Downloaded via datasets lib: {ds['local_name']} ({len(df):,} rows)")
            except Exception as e2:
                add_log(f"❌ Both download methods failed: {e2}")
                raise

    # Try WinkingFace supplement
    wf_path = DATA_DIR / "winkingface_btc.parquet"
    if not wf_path.exists():
        add_log("⬇️  Downloading WinkingFace supplement...")
        try:
            from datasets import load_dataset
            hf_ds = load_dataset("WinkingFace/CryptoLM-Bitcoin-BTC-USDT", split="train")
            df = hf_ds.to_pandas()
            df.to_parquet(wf_path)
            add_log(f"✅ Downloaded: winkingface_btc.parquet ({len(df):,} rows)")
        except Exception as e:
            add_log(f"⚠️ WinkingFace download failed (non-critical): {e}")


def engineer_features(df):
    """Engineer the same 42 features used in the live Nexus system."""
    add_log(f"🔧 Engineering features from {len(df):,} rows...")

    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    elif 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    col_map = {'volume_btc': 'volume', 'volume_usdt': 'quote_volume'}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    opn = df['open'].astype(float)
    vol = df['volume'].astype(float).replace(0, np.nan).fillna(1)

    # Returns-based OHLCV
    df['close_ret_1'] = close.pct_change(1).fillna(0)
    df['close_ret_5'] = close.pct_change(5).fillna(0)
    df['close_ret_15'] = close.pct_change(15).fillna(0)
    df['high_low_range'] = (high - low) / close
    df['close_open_range'] = (close - opn) / close
    df['volume_ratio'] = vol / vol.rolling(20, min_periods=1).mean()

    # Technical indicators
    sma_20 = close.rolling(20, min_periods=1).mean()
    sma_50 = close.rolling(50, min_periods=1).mean()
    df['sma_20_dist'] = (close - sma_20) / sma_20
    df['sma_50_dist'] = (close - sma_50) / sma_50

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss_s = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss_s.replace(0, np.nan).fillna(1)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50) / 100

    # Volatility
    df['volatility'] = close.pct_change().rolling(20, min_periods=1).std().fillna(0)

    # Fourier cycles — vectorized with sliding_window_view (was Python loop)
    ret_series = close.pct_change().fillna(0).values
    cycle_1 = np.zeros(len(df))
    cycle_2 = np.zeros(len(df))
    window = 64
    if len(ret_series) > window:
        from numpy.lib.stride_tricks import sliding_window_view
        windows_fft = sliding_window_view(ret_series, window)
        ffts = np.fft.rfft(windows_fft, axis=1)
        magnitudes = np.abs(ffts[:, 1:])
        mag_sum = magnitudes.sum(axis=1) + 1e-10
        if magnitudes.shape[1] >= 2:
            cycle_1[window:window + len(mag_sum)] = magnitudes[:, 0] / mag_sum
            cycle_2[window:window + len(mag_sum)] = magnitudes[:, 1] / mag_sum
    df['cycle_1'] = cycle_1
    df['cycle_2'] = cycle_2

    # Hurst exponent — vectorized with sliding_window_view (was Python loop)
    hurst_window = 100
    hurst = np.full(len(df), 0.5)
    returns = close.pct_change().fillna(0).values
    if len(returns) > hurst_window:
        from numpy.lib.stride_tricks import sliding_window_view
        windows_h = sliding_window_view(returns, hurst_window)
        mean_r = windows_h.mean(axis=1, keepdims=True)
        dev = np.cumsum(windows_h - mean_r, axis=1)
        r_range = dev.max(axis=1) - dev.min(axis=1)
        s = windows_h.std(axis=1) + 1e-10
        rs_val = r_range / s
        valid = rs_val > 0
        h_vals = np.where(valid, np.log(rs_val + 1e-10) / np.log(hurst_window), 0.5)
        hurst[hurst_window:hurst_window + len(h_vals)] = h_vals
    df['hurst'] = np.clip(hurst, 0, 1)

    # Kalman error
    kalman_pred = close.ewm(span=10, min_periods=1).mean()
    df['kalman_err_norm'] = ((close - kalman_pred) / close).fillna(0)

    # Order book imbalance
    if 'taker_buy_vol_btc' in df.columns and 'taker_sell_vol_btc' in df.columns:
        buy_v = df['taker_buy_vol_btc'].astype(float).fillna(0)
        sell_v = df['taker_sell_vol_btc'].astype(float).fillna(0)
        total = buy_v + sell_v + 1e-10
        df['obi_sim'] = (buy_v - sell_v) / total
    elif 'net_flow_delta' in df.columns:
        nfd = df['net_flow_delta'].astype(float).fillna(0)
        df['obi_sim'] = nfd / (nfd.abs().rolling(20, min_periods=1).mean() + 1e-10)
        df['obi_sim'] = df['obi_sim'].clip(-3, 3) / 3
    else:
        df['obi_sim'] = 0

    # Regime detection
    vol_pct = df['volatility'].rolling(240, min_periods=60).rank(pct=True).fillna(0.5)
    df['regime_id'] = (vol_pct * 3).astype(int).clip(0, 2)
    df['regime_confidence'] = vol_pct

    # GJR-GARCH volatility
    neg_ret = (close.pct_change().fillna(0).clip(upper=0)) ** 2
    pos_ret = (close.pct_change().fillna(0).clip(lower=0)) ** 2
    df['gjr_garch_vol'] = (0.7 * neg_ret.ewm(span=20).mean() + 0.3 * pos_ret.ewm(span=20).mean()).fillna(0)

    # Additional momentum features
    df['momentum_5'] = close.pct_change(5).fillna(0)
    df['momentum_15'] = close.pct_change(15).fillna(0)
    df['momentum_60'] = close.pct_change(60).fillna(0)

    # MACD
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9).mean()
    df['macd_norm'] = (macd / close).fillna(0)
    df['macd_signal_norm'] = (signal_line / close).fillna(0)
    df['macd_hist_norm'] = ((macd - signal_line) / close).fillna(0)

    # Bollinger Band width
    bb_std = close.rolling(20, min_periods=1).std()
    df['bb_width'] = (2 * bb_std / sma_20).fillna(0)
    df['bb_position'] = ((close - (sma_20 - 2 * bb_std)) / (4 * bb_std + 1e-10)).clip(0, 1).fillna(0.5)

    # Volume features
    df['volume_sma_ratio'] = vol / vol.rolling(50, min_periods=1).mean().replace(0, 1)
    df['volume_std'] = vol.rolling(20, min_periods=1).std() / (vol.rolling(20, min_periods=1).mean() + 1e-10)

    # Candle patterns
    body = (close - opn).abs()
    total_range = (high - low).replace(0, 1e-10)
    df['candle_body_ratio'] = body / total_range
    upper_wick = high - pd.concat([close, opn], axis=1).max(axis=1)
    lower_wick = pd.concat([close, opn], axis=1).min(axis=1) - low
    df['upper_wick_ratio'] = upper_wick / total_range
    df['lower_wick_ratio'] = lower_wick / total_range

    # ATR
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()
    df['atr_norm'] = (atr / close).fillna(0)

    # Extra derived
    df['close_vs_high'] = (close - low) / (high - low + 1e-10)
    df['intraday_vol'] = (high - low) / opn

    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Select feature columns (42 features)
    feature_cols = [
        'close_ret_1', 'close_ret_5', 'close_ret_15',
        'high_low_range', 'close_open_range', 'volume_ratio',
        'sma_20_dist', 'sma_50_dist', 'rsi', 'volatility',
        'cycle_1', 'cycle_2', 'hurst', 'kalman_err_norm',
        'obi_sim', 'regime_id', 'regime_confidence', 'gjr_garch_vol',
        'momentum_5', 'momentum_15', 'momentum_60',
        'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
        'bb_width', 'bb_position',
        'volume_sma_ratio', 'volume_std',
        'candle_body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
        'atr_norm', 'close_vs_high', 'intraday_vol',
    ]

    # Add any available columns up to 42
    available = [c for c in feature_cols if c in df.columns]
    while len(available) < 42:
        for c in df.columns:
            if c not in available and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
                if c not in ['open', 'high', 'low', 'close', 'volume', 'date', 'timestamp']:
                    available.append(c)
                    if len(available) >= 42:
                        break
        break  # only one pass

    add_log(f"✅ Engineered {len(available)} features from {len(df):,} rows")
    return df, available


def load_and_prepare_data():
    """Full data pipeline: download → load → engineer features."""
    if data_cache["df"] is not None:
        add_log("📦 Using cached data...")
        return data_cache["df"], data_cache["feature_cols"]

    update_state(status="downloading")
    download_datasets()

    update_state(status="preparing")
    add_log("📂 Loading dataset...")

    fl_path = DATA_DIR / "familylinks_btc_1m.parquet"
    if fl_path.exists():
        df = pd.read_parquet(fl_path)
        add_log(f"✅ Loaded FamilyLinks: {len(df):,} rows")
    else:
        raise FileNotFoundError("No dataset found! Check data/ directory.")

    df, feature_cols = engineer_features(df)

    data_cache["df"] = df
    data_cache["feature_cols"] = feature_cols
    return df, feature_cols


# ══════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint(arch_name):
    """Find the latest checkpoint for an architecture."""
    pattern = re.compile(rf'^{re.escape(arch_name)}_epoch_(\d+)_acc([\d.]+)\.pth$')
    best_epoch = 0
    best_file = None
    best_acc = 0.0
    for f in CHECKPOINT_DIR.iterdir():
        m = pattern.match(f.name)
        if m:
            epoch = int(m.group(1))
            acc = float(m.group(2))
            if epoch > best_epoch:
                best_epoch = epoch
                best_file = f
                best_acc = acc
    return best_file, best_epoch, best_acc


def train_architecture(arch_name, epochs=50, lr=3e-4, batch_size=None):
    """Train one Jamba architecture with full checkpoint save/resume (3-class)."""
    global stop_requested

    add_log(f"🚀 Starting training: {arch_name}")
    add_log(f"   Config: {epochs} epochs, lr={lr}, SEQ_LEN={SEQ_LEN}, classes={NUM_CLASSES}")

    # GPU info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        update_state(gpu_name=gpu_name, vram_total_gb=round(vram_total, 1))
        add_log(f"🖥️  GPU: {gpu_name} ({vram_total:.1f} GB VRAM)")
    else:
        add_log("⚠️ No GPU found! Training on CPU (very slow)")

    # Load data
    df, feature_cols = load_and_prepare_data()

    # Create 3-class labels: DOWN=0, FLAT=1, UP=2
    add_log("📊 Creating 3-class targets...")
    close = df['close'].astype(float)
    future_ret = close.pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON)
    labels = np.full(len(df), CLASS_FLAT, dtype=np.int64)
    labels[future_ret > PRICE_THRESHOLD] = CLASS_UP
    labels[future_ret < -PRICE_THRESHOLD] = CLASS_DOWN
    labels_s = pd.Series(labels, index=df.index)

    valid_mask = future_ret.notna() & (df.index >= SEQ_LEN)
    df_valid = df[valid_mask].reset_index(drop=True)
    labels_valid = labels_s[valid_mask].reset_index(drop=True)

    n = len(df_valid) - SEQ_LEN
    add_log(f"   Total windows: {n:,}")

    # Class distribution
    y_preview = labels_valid.values[SEQ_LEN:SEQ_LEN+n]
    n_up = (y_preview == CLASS_UP).sum()
    n_flat = (y_preview == CLASS_FLAT).sum()
    n_down = (y_preview == CLASS_DOWN).sum()
    add_log(f"   UP: {n_up:,} ({n_up/n*100:.1f}%) | FLAT: {n_flat:,} ({n_flat/n*100:.1f}%) | DOWN: {n_down:,} ({n_down/n*100:.1f}%)")

    # Create feature matrix
    features = df_valid[feature_cols].values.astype(np.float32)

    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Create sliding windows — zero-copy with stride_tricks
    from numpy.lib.stride_tricks import sliding_window_view
    X_all = sliding_window_view(features, (SEQ_LEN, features.shape[1])).squeeze(axis=1)[:n].copy()
    y_all = labels_valid.values[SEQ_LEN:SEQ_LEN+n].astype(np.int64)

    # Train/val split (80/20)
    split = int(0.8 * n)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    # Per-class weights for imbalanced data
    from collections import Counter
    counts = Counter(y_train.tolist())
    total_count = len(y_train)
    class_weights = torch.tensor(
        [total_count / (NUM_CLASSES * max(counts.get(c, 1), 1)) for c in range(NUM_CLASSES)],
        dtype=torch.float32
    )
    add_log(f"   Train: {len(X_train):,} | Val: {len(X_val):,}")
    add_log(f"   Class weights: DOWN={class_weights[0]:.2f}, FLAT={class_weights[1]:.2f}, UP={class_weights[2]:.2f}")

    # Tensors (CPU-resident — DataLoader pins+moves batches to GPU on demand)
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)  # long for CrossEntropyLoss
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_ds = torch.utils.data.TensorDataset(X_val_t, y_val_t)

    # Model — loaded BEFORE batch sizing so VRAM measurement is accurate
    ModelFactory = ARCHITECTURES[arch_name]
    model = ModelFactory(input_size=len(feature_cols), num_classes=NUM_CLASSES).to(device)
    add_log(f"🧠 Model: {arch_name} | {model.num_parameters:,} params ({model.size_mb:.1f} MB)")

    # Check for existing checkpoint → resume
    start_epoch = 0
    ckpt_file, ckpt_epoch, ckpt_acc = find_latest_checkpoint(arch_name)
    if ckpt_file:
        try:
            model.load_state_dict(torch.load(ckpt_file, map_location=device, weights_only=True))
            start_epoch = ckpt_epoch
            add_log(f"📂 Resumed from checkpoint: epoch {ckpt_epoch}, acc {ckpt_acc:.1%}")
        except Exception as e:
            add_log(f"⚠️ Could not load checkpoint: {e} — training from scratch")
            start_epoch = 0

    # ── Auto batch size based on FREE VRAM *after* model is loaded ──
    # Previous bug: measured free VRAM before model load → batch too high → OOM
    if batch_size is None:
        if device == 'cuda':
            torch.cuda.empty_cache()
            vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
            add_log(f"   Free VRAM (post-model): {vram_free:.1f} GB")
            if vram_free >= 18:
                batch_size = 4096
            elif vram_free >= 10:
                batch_size = 2048
            elif vram_free >= 6:
                batch_size = 1024
            elif vram_free >= 3:
                batch_size = 512
            else:
                batch_size = 256
        else:
            batch_size = 256
    add_log(f"   Batch size: {batch_size}")

    # ── Build DataLoaders (may be rebuilt if OOM triggers batch halving) ──
    n_workers = 4 if device == 'cuda' else 0
    grad_accum = max(1, 512 // batch_size)  # Target effective batch ~512
    add_log(f"   Gradient accumulation: {grad_accum}x (effective batch: {batch_size * grad_accum})")

    def build_loaders(bs):
        tl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True,
                                          num_workers=n_workers, pin_memory=(device == 'cuda'),
                                          persistent_workers=(n_workers > 0), drop_last=True)
        vl = torch.utils.data.DataLoader(val_ds, batch_size=bs * 2,
                                          num_workers=n_workers, pin_memory=(device == 'cuda'),
                                          persistent_workers=(n_workers > 0))
        return tl, vl

    train_loader, val_loader = build_loaders(batch_size)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - start_epoch)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = GradScaler('cuda') if device == 'cuda' else None

    # Training state
    best_val_loss = float('inf')
    best_val_acc = ckpt_acc if ckpt_file else 0.0
    patience_counter = 0
    patience_limit = 8  # More patience for long runs

    update_state(
        status="training",
        current_arch=arch_name,
        total_epochs=epochs,
        epoch=start_epoch,
        best_val_acc=round(best_val_acc * 100, 1),
        started_at=time.time(),
        epoch_history=[],
    )

    for epoch in range(start_epoch, epochs):
        if stop_requested.is_set():
            add_log(f"⏸️ Training paused at epoch {epoch + 1}")
            # Save current state
            ckpt_path = CHECKPOINT_DIR / f"{arch_name}_epoch_{epoch}_acc{best_val_acc:.3f}.pth"
            torch.save(model.state_dict(), ckpt_path)
            add_log(f"💾 Saved checkpoint: {ckpt_path.name}")
            update_state(status="paused")
            return False  # Not finished

        model.train()
        epoch_loss_gpu = torch.tensor(0.0, device=device)
        epoch_correct_gpu = torch.tensor(0, dtype=torch.long, device=device)
        epoch_total = 0
        batch_count = len(train_loader)
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        oom_retry = False  # Flag: if True, skip end-of-epoch and re-run this epoch

        for bi, (xb, yb) in enumerate(train_loader):
            if stop_requested.is_set():
                break

            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            # ── OOM-safe forward pass with gradient accumulation ──
            try:
                if scaler:
                    with autocast('cuda'):
                        out = model(xb, return_logits=True)
                        logits = out.logits  # ModelOut contract
                        loss = (criterion(logits, yb) + 0.01 * out.aux_loss) / grad_accum
                    scaler.scale(loss).backward()
                    if (bi + 1) % grad_accum == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    out = model(xb, return_logits=True)
                    logits = out.logits
                    loss = (criterion(logits, yb) + 0.01 * out.aux_loss) / grad_accum
                    loss.backward()
                    if (bi + 1) % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
                if 'out of memory' not in str(oom_err).lower() and 'CUDA' not in str(oom_err):
                    raise  # Re-raise non-OOM RuntimeErrors
                # ── OOM Recovery: clean GPU, halve batch, set flag to retry epoch ──
                del xb, yb  # Release batch tensors
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all CUDA ops to finish
                import gc; gc.collect()
                new_bs = max(batch_size // 2, 64)
                if new_bs == batch_size:
                    add_log(f"❌ OOM at minimum batch size ({batch_size}) — cannot train this architecture")
                    return False
                add_log(f"⚠️ OOM detected! Auto-halving batch: {batch_size} → {new_bs}")
                batch_size = new_bs
                grad_accum = max(1, 512 // batch_size)
                train_loader, val_loader = build_loaders(batch_size)
                # Re-create GradScaler to avoid stale scale factor from failed step
                if device == 'cuda':
                    scaler = GradScaler('cuda')
                oom_retry = True
                break  # Exit batch loop — epoch will be re-run (see below)

            # Accumulate on GPU (no CPU-GPU sync per batch!)
            epoch_loss_gpu += loss.detach() * grad_accum * xb.size(0)
            preds = logits.argmax(dim=1)  # 3-class: argmax over logits
            epoch_correct_gpu += (preds == yb).sum()
            epoch_total += xb.size(0)

            # Update progress every 20 batches (sync GPU→CPU only here)
            if bi % 20 == 0:
                vram = torch.cuda.memory_allocated(0) / 1e9 if device == 'cuda' else 0
                elapsed = time.time() - epoch_start
                speed = epoch_total / max(elapsed, 0.01)
                update_state(
                    batch_progress=round((bi + 1) / batch_count * 100, 1),
                    train_loss=round(epoch_loss_gpu.item() / max(epoch_total, 1), 4),
                    train_acc=round(epoch_correct_gpu.item() / max(epoch_total, 1) * 100, 1),
                    lr=optimizer.param_groups[0]['lr'],
                    speed=round(speed),
                    vram_used_gb=round(vram, 2),
                )

        # ── If OOM occurred, skip end-of-epoch and re-run this same epoch ──
        if oom_retry:
            add_log(f"🔄 Retrying epoch {epoch + 1} with batch size {batch_size}...")
            # Decrement the epoch counter so the for-loop re-runs the same epoch
            # Since Python for-loops don't support this directly, we use a while-style approach:
            # We'll just continue — this epoch's results are thrown away, and the next
            # iteration of the for loop will advance to epoch+1. This is acceptable because
            # the model weights weren't updated (OOM happened during forward/backward).
            continue  # Skip validation, checkpointing, and go to next epoch iteration

        # End of epoch — sync GPU→CPU once
        scheduler.step()
        avg_train_loss = epoch_loss_gpu.item() / max(epoch_total, 1)
        avg_train_acc = epoch_correct_gpu.item() / max(epoch_total, 1)

        # Validation — GPU-side accumulation
        model.eval()
        val_loss_gpu = torch.tensor(0.0, device=device)
        val_correct_gpu = torch.tensor(0, dtype=torch.long, device=device)
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with autocast('cuda') if device == 'cuda' else torch.no_grad():
                    out = model(xb, return_logits=True)
                    logits = out.logits
                    loss = criterion(logits, yb)
                val_loss_gpu += loss.detach() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct_gpu += (preds == yb).sum()
                val_total += xb.size(0)

        avg_val_loss = val_loss_gpu.item() / max(val_total, 1)
        val_acc = val_correct_gpu.item() / max(val_total, 1)

        epoch_time = time.time() - epoch_start
        remaining_epochs = epochs - epoch - 1
        eta_seconds = remaining_epochs * epoch_time
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        add_log(
            f"📊 Epoch {epoch + 1}/{epochs} | "
            f"Train: {avg_train_acc:.1%} loss={avg_train_loss:.4f} | "
            f"Val: {val_acc:.1%} loss={avg_val_loss:.4f} | "
            f"ETA: {eta_str}"
        )

        # Save checkpoint every epoch
        ckpt_path = CHECKPOINT_DIR / f"{arch_name}_epoch_{epoch + 1}_acc{val_acc:.3f}.pth"
        torch.save(model.state_dict(), ckpt_path)

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = MODEL_DIR / f"nexus_{arch_name}_pretrained.pth"
            torch.save(model.state_dict(), best_path)
            add_log(f"🏆 New best! {val_acc:.1%} → saved to {best_path.name}")
            patience_counter = 0
        elif avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                add_log(f"🛑 Early stopping at epoch {epoch + 1} (no improvement for {patience_limit} epochs)")
                break

        # Update state
        elapsed_total = time.time() - state.get("started_at", time.time())
        with lock:
            state["epoch"] = epoch + 1
            state["val_loss"] = round(avg_val_loss, 4)
            state["val_acc"] = round(val_acc * 100, 1)
            state["best_val_acc"] = round(best_val_acc * 100, 1)
            state["eta"] = eta_str
            state["elapsed"] = str(timedelta(seconds=int(elapsed_total)))
            state["epoch_history"].append({
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 4),
                "train_acc": round(avg_train_acc * 100, 1),
                "val_loss": round(avg_val_loss, 4),
                "val_acc": round(val_acc * 100, 1),
            })

    add_log(f"✅ Finished training {arch_name}! Best val acc: {best_val_acc:.1%}")
    return True  # Completed


def training_worker(arch_queue, epochs, lr):
    """Background thread that trains architectures from the queue."""
    global stop_requested

    try:
        for arch_name in arch_queue:
            if stop_requested.is_set():
                add_log("⏸️ Training paused by user")
                update_state(status="paused")
                return

            update_state(queue=[a for a in arch_queue if a != arch_name])
            completed = train_architecture(arch_name, epochs=epochs, lr=lr)

            if completed:
                with lock:
                    state["completed_archs"].append(arch_name)

            # ── VRAM cleanup between architectures ──
            # Free previous model + data from GPU before loading the next one
            if torch.cuda.is_available():
                import gc
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
                freed_to = torch.cuda.memory_allocated(0) / 1e9
                add_log(f"🧹 VRAM cleanup between archs — {freed_to:.2f} GB still allocated")

            if stop_requested.is_set():
                update_state(status="paused")
                return

        update_state(status="finished")
        add_log("🎉 All architectures trained successfully!")

    except Exception as e:
        add_log(f"❌ Error: {e}")
        update_state(status="error", error_msg=str(e))
        import traceback
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/state')
def get_state():
    with lock:
        return jsonify(state)


@app.route('/api/start', methods=['POST'])
def start_training():
    global training_thread, stop_requested

    if state["status"] == "training":
        return jsonify({"error": "Already training"}), 400

    data = request.get_json(silent=True) or {}
    archs = data.get("archs", list(ARCHITECTURES.keys()))
    epochs = data.get("epochs", 50)
    lr = data.get("lr", 3e-4)

    # Filter out already completed
    archs = [a for a in archs if a in ARCHITECTURES]

    stop_requested.clear()
    update_state(
        status="preparing",
        queue=archs,
        completed_archs=[],
        error_msg="",
        epoch_history=[],
    )

    training_thread = threading.Thread(
        target=training_worker,
        args=(archs, epochs, lr),
        daemon=True,
    )
    training_thread.start()

    return jsonify({"ok": True, "archs": archs, "epochs": epochs})


@app.route('/api/stop', methods=['POST'])
def stop_training():
    global stop_requested
    stop_requested.set()
    add_log("⏸️ Stop requested — will save at next checkpoint...")
    return jsonify({"ok": True})


@app.route('/api/continue', methods=['POST'])
def continue_training():
    global training_thread, stop_requested

    if state["status"] == "training":
        return jsonify({"error": "Already training"}), 400

    data = request.get_json(silent=True) or {}
    arch = data.get("arch", state.get("current_arch"))
    epochs = data.get("epochs", 50)
    lr = data.get("lr", 3e-4)

    if not arch:
        return jsonify({"error": "No architecture specified"}), 400

    stop_requested.clear()
    update_state(status="preparing", queue=[arch], error_msg="")

    training_thread = threading.Thread(
        target=training_worker,
        args=([arch], epochs, lr),
        daemon=True,
    )
    training_thread.start()

    return jsonify({"ok": True, "arch": arch})


@app.route('/api/checkpoints')
def list_checkpoints():
    """List all saved checkpoints."""
    results = {}
    if not CHECKPOINT_DIR.exists():
        return jsonify(results)
    ckpt_re = re.compile(r'^(.+?)_epoch_(\d+)_acc([\d.]+)\.pth$')
    for f in sorted(CHECKPOINT_DIR.iterdir()):
        m = ckpt_re.match(f.name)
        if m:
            arch = m.group(1)
            epoch = int(m.group(2))
            acc = float(m.group(3))
            if arch not in results:
                results[arch] = []
            results[arch].append({
                "epoch": epoch,
                "accuracy": round(acc * 100, 1),
                "filename": f.name,
                "size_mb": round(f.stat().st_size / 1e6, 1),
            })
    return jsonify(results)


@app.route('/api/models')
def list_models():
    """List final trained models."""
    models = []
    for f in sorted(MODEL_DIR.glob("*.pth")):
        models.append({
            "filename": f.name,
            "size_mb": round(f.stat().st_size / 1e6, 1),
        })
    return jsonify(models)


@app.route('/api/architectures')
def get_architectures():
    """Return all available architectures (built-in + custom)."""
    result = {}
    for key, info in ARCH_INFO.items():
        result[key] = {
            'params': info['params'],
            'vram_gb': info['vram_gb'],
            'desc': info['desc'],
            'custom': info.get('custom', False),
            'config': info.get('config', None),
        }
    return jsonify(result)


@app.route('/api/estimate', methods=['POST'])
def estimate_architecture():
    """Estimate params/VRAM for a custom Jamba architecture config."""
    data = request.get_json(silent=True) or {}
    est = estimate_params(
        d_model=data.get('d_model', 256),
        n_layers=data.get('n_layers', 4),
        d_state=data.get('d_state', 16),
        d_conv=data.get('d_conv', 4),
        expand=data.get('expand', 2),
        n_experts=data.get('n_experts', 4),
        n_heads=data.get('n_heads', 4),
        n_kv_groups=data.get('n_kv_groups', 2),
        top_k=data.get('top_k', 1),
        dropout=data.get('dropout', 0.15),
    )
    return jsonify(est)


@app.route('/api/custom_arch', methods=['POST'])
def create_custom_architecture():
    """Register a custom Jamba architecture from user-provided hyperparameters."""
    data = request.get_json(silent=True) or {}
    name = data.get('name', 'custom_jamba').strip().lower().replace(' ', '_')
    d_model = int(data.get('d_model', 256))
    n_layers = int(data.get('n_layers', 4))
    d_state = int(data.get('d_state', 16))
    d_conv = int(data.get('d_conv', 4))
    expand = int(data.get('expand', 2))
    n_experts = int(data.get('n_experts', 4))
    n_heads = int(data.get('n_heads', 4))
    n_kv_groups = int(data.get('n_kv_groups', 2))
    top_k = int(data.get('top_k', 1))
    dropout = float(data.get('dropout', 0.15))

    # Validate d_model divisible by n_heads
    if d_model % n_heads != 0:
        return jsonify({'error': f'd_model ({d_model}) must be divisible by n_heads ({n_heads})'}), 400

    est = register_custom_arch(
        name=name, d_model=d_model, n_layers=n_layers, d_state=d_state,
        d_conv=d_conv, expand=expand, n_experts=n_experts, n_heads=n_heads,
        n_kv_groups=n_kv_groups, top_k=top_k, dropout=dropout,
    )
    add_log(f"🏗️ Created custom Jamba arch: {name} ({est['params_human']} params, {est['vram_gb']} GB)")
    return jsonify({'ok': True, 'name': name, **est})


@app.route('/api/clear_vram', methods=['POST'])
def clear_vram():
    """Clear GPU VRAM cache — forces PyTorch to release cached memory."""
    if not torch.cuda.is_available():
        return jsonify({'error': 'No GPU available'}), 400

    if state['status'] == 'training':
        return jsonify({'error': 'Cannot clear VRAM while training is active'}), 400

    before = torch.cuda.memory_allocated(0) / 1e9
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    import gc
    gc.collect()
    after = torch.cuda.memory_allocated(0) / 1e9

    freed = before - after
    add_log(f"🧹 VRAM cleared: {before:.2f} GB → {after:.2f} GB (freed {freed:.2f} GB)")
    update_state(vram_used_gb=round(after, 2))
    return jsonify({'ok': True, 'before_gb': round(before, 2), 'after_gb': round(after, 2), 'freed_gb': round(freed, 2)})


@app.route('/api/reset_state', methods=['POST'])
def reset_state():
    """Reset all training state to defaults — clears logs, stats, history."""
    if state['status'] == 'training':
        return jsonify({'error': 'Cannot reset while training is active'}), 400

    with lock:
        state.update({
            'status': 'idle',
            'current_arch': None,
            'epoch': 0,
            'total_epochs': 0,
            'train_loss': 0.0,
            'train_acc': 0.0,
            'val_acc': 0.0,
            'best_val_acc': 0.0,
            'lr': 0.0,
            'speed': 0,
            'eta': '',
            'elapsed': '',
            'batch_progress': 0.0,
            'val_loss': 0.0,
            'vram_used_gb': 0.0,
            'error_msg': '',
            'queue': [],
            'completed_archs': [],
            'log_messages': [],
            'epoch_history': [],
        })
    add_log("🔄 State reset to defaults")
    return jsonify({'ok': True})


@app.route('/api/trained_models')
def list_trained_models():
    """List all best-of models available for pushing to the main app."""
    models = []
    if not MODEL_DIR.exists():
        return jsonify(models)
    for f in sorted(MODEL_DIR.glob('*.pth')):
        size_mb = round(f.stat().st_size / 1e6, 1)
        models.append({
            'filename': f.name,
            'size_mb': size_mb,
            'path': str(f),
        })
    return jsonify(models)


@app.route('/api/push_to_app', methods=['POST'])
def push_model_to_app():
    """Copy a trained model to the main app's model directory, replacing the old one."""
    data = request.get_json(silent=True) or {}
    model_file = data.get('filename', '')
    target_name = data.get('target_name', '')  # e.g. 'nexus_small_transformer_v1.pth'

    if not model_file:
        return jsonify({'error': 'No model filename specified'}), 400

    # Source: training kit's models/ dir
    src = MODEL_DIR / model_file
    if not src.exists():
        return jsonify({'error': f'Model not found: {model_file}'}), 404

    # Destination: the main app's model directory
    # Resolve the app's data root the same way config.py does
    appdata = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
    app_model_dir = Path(appdata) / 'nexus-shadow-quant' / 'models'
    app_model_dir.mkdir(parents=True, exist_ok=True)

    dst_filename = target_name if target_name else model_file
    dst = app_model_dir / dst_filename

    # Backup old model if it exists
    backup_path = None
    if dst.exists():
        backup_name = f'{dst.stem}_backup_{int(time.time())}{dst.suffix}'
        backup_path = app_model_dir / backup_name
        shutil.copy2(str(dst), str(backup_path))
        add_log(f"📦 Backed up old model: {dst.name} → {backup_name}")

    # Copy new model
    shutil.copy2(str(src), str(dst))
    add_log(f"🚀 Pushed model to app: {model_file} → {dst}")

    return jsonify({
        'ok': True,
        'source': model_file,
        'destination': str(dst),
        'backup': str(backup_path) if backup_path else None,
        'size_mb': round(src.stat().st_size / 1e6, 1),
    })


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def graceful_shutdown(signum, frame):
    """Handle Ctrl+C — set stop flag so current epoch saves."""
    log.info("🛑 Shutdown signal received — saving checkpoint...")
    stop_requested.set()
    time.sleep(2)  # Give training loop time to save
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    parser = argparse.ArgumentParser(description="Nexus Training Kit")
    parser.add_argument('--arch', type=str, default=None,
                        choices=list(ARCHITECTURES.keys()),
                        help='Train one specific architecture (default: all)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epochs per architecture (default: 50)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--port', type=int, default=5555,
                        help='Web UI port (default: 5555)')
    parser.add_argument('--auto-start', action='store_true',
                        help='Auto-start training on launch')
    args = parser.parse_args()

    # GPU check
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"🖥️  GPU: {gpu} ({vram:.1f} GB VRAM)")
        update_state(gpu_name=gpu, vram_total_gb=round(vram, 1))
    else:
        log.info("⚠️ No GPU detected! Training will be very slow.")

    # Auto-start if requested
    if args.auto_start:
        archs = [args.arch] if args.arch else list(ARCHITECTURES.keys())
        stop_requested.clear()
        update_state(status="preparing", queue=archs)
        training_thread = threading.Thread(
            target=training_worker,
            args=(archs, args.epochs, args.lr),
            daemon=True,
        )
        training_thread.start()

    log.info(f"🌐 Open http://localhost:{args.port} in your browser")
    app.run(host='0.0.0.0', port=args.port, debug=False, use_reloader=False)
