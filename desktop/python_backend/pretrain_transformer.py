"""
pretrain_transformer.py — Pretrain NexusTransformer on HuggingFace BTC datasets.

Downloads 8+ years of 1-minute BTC data, engineers features to match the live
system (42 features), and trains the 90M-param Transformer with mixed precision.

Usage:
    python pretrain_transformer.py                     # Full pipeline
    python pretrain_transformer.py --skip-download     # Skip download, use cached
    python pretrain_transformer.py --epochs 5          # Quick test run

Output:
    models/nexus_transformer_pretrained.pth   (~360 MB)
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [PRETRAIN] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# === Paths ===
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "pretrain"
MODEL_DIR = SCRIPT_DIR / "models"
CHECKPOINT_DIR = SCRIPT_DIR / "models" / "checkpoints"

# === Constants ===
SEQ_LEN = 30  # Must match DEEP_SEQ_LEN in predictor.py
PREDICTION_HORIZON = 15  # 15-min lookahead, same as live
PRICE_THRESHOLD = 0.001  # 0.1% symmetric threshold, same as live config


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD DATASETS
# ═══════════════════════════════════════════════════════════════════════════

def download_datasets():
    """Download Tier 1 datasets from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.info("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub datasets")
        from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    datasets_to_download = [
        {
            "repo": "FamilyLinks/btc-price-1m-2017-2025",
            "filename": "BTC_Raw_Micro_Macro_1m.parquet",
            "local_name": "familylinks_btc_1m.parquet",
            "description": "4.37M rows: OHLCV + order flow + on-chain (2017-2025)"
        },
    ]

    for ds in datasets_to_download:
        local_path = DATA_DIR / ds["local_name"]
        if local_path.exists():
            log.info(f"✅ Already downloaded: {ds['local_name']} ({local_path.stat().st_size / 1e6:.0f} MB)")
            continue

        log.info(f"⬇️  Downloading: {ds['description']}...")
        try:
            downloaded = hf_hub_download(
                repo_id=ds["repo"],
                filename=ds["filename"],
                repo_type="dataset",
                local_dir=str(DATA_DIR),
            )
            # Move to our standard name
            if Path(downloaded).name != ds["local_name"]:
                Path(downloaded).rename(local_path)
            log.info(f"✅ Downloaded: {ds['local_name']} ({local_path.stat().st_size / 1e6:.0f} MB)")
        except Exception as e:
            log.error(f"❌ Failed to download {ds['repo']}: {e}")
            log.info("Trying alternative download via datasets library...")
            try:
                from datasets import load_dataset
                hf_ds = load_dataset(ds["repo"], split="train")
                df = hf_ds.to_pandas()
                df.to_parquet(local_path)
                log.info(f"✅ Downloaded via datasets lib: {ds['local_name']} ({len(df):,} rows)")
            except Exception as e2:
                log.error(f"❌ Both download methods failed: {e2}")
                raise

    # Try to download WinkingFace as supplement
    wf_path = DATA_DIR / "winkingface_btc.parquet"
    if not wf_path.exists():
        log.info("⬇️  Downloading WinkingFace supplement dataset...")
        try:
            from datasets import load_dataset
            hf_ds = load_dataset("WinkingFace/CryptoLM-Bitcoin-BTC-USDT", split="train")
            df = hf_ds.to_pandas()
            df.to_parquet(wf_path)
            log.info(f"✅ Downloaded: winkingface_btc.parquet ({len(df):,} rows)")
        except Exception as e:
            log.warning(f"⚠️ WinkingFace download failed (non-critical): {e}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the same 42 features used in the live system from raw OHLCV data.
    All features are scale-invariant (returns/ratios, not raw prices).
    """
    log.info(f"Engineering features from {len(df):,} rows...")

    # Ensure sorted by time
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    elif 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    # Normalize column names
    col_map = {
        'volume_btc': 'volume', 'volume_usdt': 'quote_volume',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure we have basic OHLCV
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    opn = df['open'].astype(float)
    vol = df['volume'].astype(float).replace(0, np.nan).fillna(1)

    # ──── Returns-based OHLCV (scale-invariant) ────
    df['close_ret_1'] = close.pct_change(1).fillna(0)
    df['close_ret_5'] = close.pct_change(5).fillna(0)
    df['close_ret_15'] = close.pct_change(15).fillna(0)
    df['high_low_range'] = (high - low) / close
    df['close_open_range'] = (close - opn) / close
    df['volume_ratio'] = vol / vol.rolling(20, min_periods=1).mean()

    # ──── Technical indicators ────
    sma_20 = close.rolling(20, min_periods=1).mean()
    sma_50 = close.rolling(50, min_periods=1).mean()
    df['sma_20_dist'] = (close - sma_20) / sma_20
    df['sma_50_dist'] = (close - sma_50) / sma_50

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan).fillna(1)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50) / 100  # Normalize to 0-1

    # Volatility (rolling std of returns)
    df['volatility'] = close.pct_change().rolling(20, min_periods=1).std().fillna(0)

    # ──── Fourier cycles (simplified — use FFT on rolling window) ────
    ret_series = close.pct_change().fillna(0).values
    cycle_1 = np.zeros(len(df))
    cycle_2 = np.zeros(len(df))
    window = 64
    for i in range(window, len(ret_series)):
        segment = ret_series[i - window:i]
        fft = np.fft.rfft(segment)
        magnitudes = np.abs(fft[1:])  # skip DC
        if len(magnitudes) >= 2:
            cycle_1[i] = magnitudes[0] / (np.sum(magnitudes) + 1e-10)
            cycle_2[i] = magnitudes[1] / (np.sum(magnitudes) + 1e-10)
    df['cycle_1'] = cycle_1
    df['cycle_2'] = cycle_2

    # ──── Market microstructure ────
    # Hurst exponent (simplified rolling R/S)
    hurst_window = 100
    hurst = np.full(len(df), 0.5)
    returns = close.pct_change().fillna(0).values
    for i in range(hurst_window, len(returns)):
        seg = returns[i - hurst_window:i]
        mean_r = np.mean(seg)
        dev = np.cumsum(seg - mean_r)
        r_range = np.max(dev) - np.min(dev)
        s = np.std(seg) + 1e-10
        rs_val = r_range / s
        if rs_val > 0:
            hurst[i] = np.log(rs_val + 1e-10) / np.log(hurst_window)
    df['hurst'] = np.clip(hurst, 0, 1)

    # Kalman error (simplified — prediction error from rolling mean)
    kalman_pred = close.ewm(span=10, min_periods=1).mean()
    df['kalman_err_norm'] = ((close - kalman_pred) / close).fillna(0)

    # Order book imbalance simulation (from buy/sell volume if available)
    if 'taker_buy_vol_btc' in df.columns and 'taker_sell_vol_btc' in df.columns:
        buy_v = df['taker_buy_vol_btc'].astype(float).fillna(0)
        sell_v = df['taker_sell_vol_btc'].astype(float).fillna(0)
        total = buy_v + sell_v + 1e-10
        df['obi_sim'] = (buy_v - sell_v) / total
    elif 'net_flow_delta' in df.columns:
        nfd = df['net_flow_delta'].astype(float).fillna(0)
        df['obi_sim'] = nfd / (nfd.abs().rolling(20, min_periods=1).mean() + 1e-10)
        df['obi_sim'] = df['obi_sim'].clip(-3, 3) / 3  # Normalize
    else:
        df['obi_sim'] = 0

    # ──── Quant features ────
    # Regime detection (simplified — based on volatility + trend)
    vol_percentile = df['volatility'].rolling(240, min_periods=60).rank(pct=True).fillna(0.5)
    trend_strength = df['close_ret_15'].rolling(60, min_periods=1).mean().fillna(0)
    df['regime_id'] = (vol_percentile * 3).astype(int).clip(0, 2)  # 0=low_vol, 1=mid, 2=high_vol
    df['regime_confidence'] = vol_percentile

    # GJR-GARCH volatility (asymmetric — responds more to drops)
    neg_ret = (close.pct_change().fillna(0).clip(upper=0)) ** 2
    pos_ret = (close.pct_change().fillna(0).clip(lower=0)) ** 2
    gjr = (0.5 * neg_ret.ewm(span=20, min_periods=1).mean() +
           0.3 * pos_ret.ewm(span=20, min_periods=1).mean())
    df['gjr_volatility'] = np.sqrt(gjr).fillna(0)

    # Hawkes intensity (simplified — event clustering)
    abs_ret = close.pct_change().abs().fillna(0)
    df['hawkes_intensity'] = abs_ret.ewm(halflife=10, min_periods=1).mean() / (abs_ret.rolling(100, min_periods=1).mean() + 1e-10)

    # Wasserstein drift (rolling distribution shift)
    df['wass_drift'] = close.pct_change().rolling(30, min_periods=1).mean().fillna(0) - \
                       close.pct_change().rolling(120, min_periods=1).mean().fillna(0)

    # ──── Multi-timeframe trend ────
    df['trend_5m'] = close.pct_change(5).rolling(12, min_periods=1).mean().fillna(0)
    df['trend_15m'] = close.pct_change(15).rolling(4, min_periods=1).mean().fillna(0)
    df['trend_1h'] = close.pct_change(60).rolling(4, min_periods=1).mean().fillna(0)
    df['ret_60'] = close.pct_change(60).fillna(0)
    df['ret_240'] = close.pct_change(240).fillna(0)

    # Volume regime
    vol_sma = vol.rolling(60, min_periods=1).mean()
    df['vol_regime'] = (vol / vol_sma).clip(0, 5).fillna(1)

    # ──── Volume profile ────
    vwap = (close * vol).rolling(20, min_periods=1).sum() / vol.rolling(20, min_periods=1).sum()
    df['vwap_dist'] = ((close - vwap) / close).fillna(0)
    df['vol_momentum'] = vol.pct_change(5).fillna(0).clip(-5, 5)

    # ──── Cross-asset (simulated — we use lagged BTC as proxy for ETH correlation) ────
    # In live system these come from real ETH/Gold feeds. For pretraining, use lagged BTC
    df['eth_ret_5'] = close.pct_change(5).shift(1).fillna(0) * 1.2  # ETH ~1.2x BTC beta
    df['eth_ret_15'] = close.pct_change(15).shift(1).fillna(0) * 1.2
    df['eth_vol_ratio'] = (vol.shift(1) / vol.rolling(20, min_periods=1).mean()).fillna(1)
    df['ethbtc_ret_5'] = (close.pct_change(5).shift(1).fillna(0) * 0.2).fillna(0)  # ETH/BTC ratio drift
    df['ethbtc_trend'] = df['ethbtc_ret_5'].rolling(12, min_periods=1).mean().fillna(0)
    df['gold_ret_15'] = close.pct_change(15).shift(2).fillna(0) * 0.1  # Gold ~0.1x BTC
    df['gold_ret_60'] = close.pct_change(60).shift(2).fillna(0) * 0.1

    # ──── Microstructure features ────
    if 'trade_count' in df.columns:
        tc = df['trade_count'].astype(float).fillna(0)
        df['trade_intensity'] = tc / tc.rolling(20, min_periods=1).mean().replace(0, 1)
    else:
        df['trade_intensity'] = vol / vol.rolling(5, min_periods=1).mean().replace(0, 1)

    if 'taker_buy_vol_btc' in df.columns and 'taker_sell_vol_btc' in df.columns:
        buy_v = df['taker_buy_vol_btc'].astype(float).fillna(0)
        sell_v = df['taker_sell_vol_btc'].astype(float).fillna(0)
        df['buy_sell_ratio'] = buy_v / (sell_v + 1e-10)
    else:
        df['buy_sell_ratio'] = 1.0

    df['vwap_momentum'] = df['vwap_dist'].diff(5).fillna(0)
    df['tick_volatility'] = close.pct_change().rolling(5, min_periods=1).std().fillna(0)
    df['large_trade_ratio'] = (vol > vol.rolling(20, min_periods=1).quantile(0.9)).astype(float)

    # WS features (simulated for pretraining — live system uses real tick data)
    df['ws_trades_per_sec'] = df['trade_intensity'] * 0.8  # proxy
    df['ws_buy_sell_ratio'] = df['buy_sell_ratio'] * 0.95  # proxy
    df['ws_spread_bps'] = df['high_low_range'] * 100  # proxy for spread

    # ──── TARGET (SYMMETRIC) ────
    future_ret = close.pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON)
    # Symmetric: UP > +threshold, DOWN < -threshold, drop neutral zone
    df['target'] = -1.0  # sentinel for neutral
    df.loc[future_ret > PRICE_THRESHOLD, 'target'] = 1.0   # UP
    df.loc[future_ret < -PRICE_THRESHOLD, 'target'] = 0.0  # DOWN
    n_neutral = (df['target'] == -1.0).sum()
    df = df[df['target'] != -1.0].reset_index(drop=True)
    log.info(f"   Symmetric ±{PRICE_THRESHOLD*100:.1f}%: dropped {n_neutral:,} neutral rows")

    # Drop warmup rows and future-leaking rows
    df = df.iloc[max(240, hurst_window):-(PREDICTION_HORIZON + 1)].reset_index(drop=True)

    # Select our 42 features
    feature_cols = [
        'close_ret_1', 'close_ret_5', 'close_ret_15',
        'high_low_range', 'close_open_range', 'volume_ratio',
        'sma_20_dist', 'sma_50_dist', 'rsi', 'volatility',
        'cycle_1', 'cycle_2',
        'hurst', 'kalman_err_norm', 'obi_sim',
        'regime_id', 'regime_confidence', 'gjr_volatility',
        'hawkes_intensity', 'wass_drift',
        'trend_5m', 'trend_15m', 'trend_1h',
        'ret_60', 'ret_240', 'vol_regime',
        'vwap_dist', 'vol_momentum',
        'eth_ret_5', 'eth_ret_15', 'eth_vol_ratio',
        'ethbtc_ret_5', 'ethbtc_trend',
        'gold_ret_15', 'gold_ret_60',
        'trade_intensity', 'buy_sell_ratio', 'vwap_momentum',
        'tick_volatility', 'large_trade_ratio',
        'ws_trades_per_sec', 'ws_buy_sell_ratio', 'ws_spread_bps',
    ]

    # Verify all features exist
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Clean infinities and NaNs
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
        # Clip extreme outliers
        q99 = df[col].quantile(0.999)
        q01 = df[col].quantile(0.001)
        if q99 != q01:
            df[col] = df[col].clip(q01, q99)

    log.info(f"✅ Engineered {len(feature_cols)} features, {len(df):,} samples")
    log.info(f"   Target distribution: UP={df['target'].mean():.1%} / DOWN={1 - df['target'].mean():.1%}")

    return df, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: CREATE SLIDING WINDOWS
# ═══════════════════════════════════════════════════════════════════════════

def create_windows(df: pd.DataFrame, feature_cols: list, batch_size: int = 100_000):
    """
    Create sliding window sequences as memory-mapped arrays (handles 4M+ rows).
    Yields batches of (X_windows, y_windows) to avoid OOM.
    """
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)

    total_windows = len(X) - SEQ_LEN
    log.info(f"Creating {total_windows:,} sliding windows of {SEQ_LEN} timesteps...")

    for start in range(0, total_windows, batch_size):
        end = min(start + batch_size, total_windows)
        batch_X = []
        batch_y = []
        for i in range(start, end):
            batch_X.append(X[i:i + SEQ_LEN])
            batch_y.append(y[i + SEQ_LEN])
        yield np.array(batch_X), np.array(batch_y), start, end, total_windows


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: PRETRAIN
# ═══════════════════════════════════════════════════════════════════════════

def pretrain(df: pd.DataFrame, feature_cols: list, epochs: int = 10, lr: float = 3e-4, output_name: str = "nexus_transformer_pretrained.pth"):
    """Train NexusTransformer on the full dataset with mixed precision."""
    from predictor import NexusTransformer, DEEP_SEQ_LEN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")

    # Initialize model
    model = NexusTransformer(input_size=len(feature_cols)).to(device)
    log.info(f"Model: {model.num_parameters / 1e6:.1f}M params ({model.size_mb:.0f} MB)")

    # Check for existing checkpoint
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pretrained_path = MODEL_DIR / output_name

    if pretrained_path.exists():
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            log.info(f"🔄 Loaded existing pretrained weights — continuing training")
        except RuntimeError as e:
            log.warning(f"⚠️ Checkpoint shape mismatch (architecture changed?), training from scratch: {e}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    criterion = nn.BCEWithLogitsLoss()  # AMP-safe (expects raw logits)
    scaler = GradScaler()

    # Prepare data
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df['target'].values.astype(np.float32)

    # Standardize features (fit scaler on all data for pretraining)
    from sklearn.preprocessing import StandardScaler
    import pickle
    scaler_sk = StandardScaler()
    X_all = scaler_sk.fit_transform(X_all)

    # Save the pretrain scaler for reference
    scaler_path = MODEL_DIR / "pretrain_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_sk, f)
    log.info(f"Saved pretrain scaler to {scaler_path}")

    # Create all windows in memory (if fits) or batch
    total_windows = len(X_all) - SEQ_LEN
    log.info(f"Total windows: {total_windows:,}")

    # For 4M rows × 42 features × 30 steps × 4 bytes = ~20 GB
    # Too big for RAM. Use epoch-level batching instead.
    batch_size = 512
    grad_accum = 2  # Effective batch = 1024

    # Split: 95% train, 5% validation
    val_start = int(total_windows * 0.95)

    # Cosine annealing with warmup
    warmup_steps = 1000
    total_steps = (val_start // (batch_size * grad_accum)) * epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        # Shuffle training indices
        train_indices = np.random.permutation(val_start)

        t0 = time.time()
        optimizer.zero_grad()

        for batch_start in range(0, val_start, batch_size):
            batch_idx = train_indices[batch_start:batch_start + batch_size]

            # Build windows on-the-fly (memory efficient)
            batch_X = np.array([X_all[i:i + SEQ_LEN] for i in batch_idx if i + SEQ_LEN < len(X_all)])
            batch_y = np.array([y_all[i + SEQ_LEN] for i in batch_idx if i + SEQ_LEN < len(y_all)])

            if len(batch_X) == 0:
                continue

            x_tensor = torch.FloatTensor(batch_X).to(device)
            y_tensor = torch.FloatTensor(batch_y).unsqueeze(1).to(device)

            with autocast(device_type=device.type, dtype=torch.float16):
                output = model(x_tensor, return_logits=True)  # Raw logits for AMP
                loss = criterion(output, y_tensor) / grad_accum

            scaler.scale(loss).backward()

            if (batch_start // batch_size + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * grad_accum
            predictions = (output > 0.5).float()
            epoch_correct += (predictions == y_tensor).sum().item()
            epoch_total += len(y_tensor)

            # Progress logging every 10K samples
            if batch_start % (batch_size * 20) == 0 and batch_start > 0:
                progress = batch_start / val_start * 100
                speed = batch_start / (time.time() - t0)
                current_lr = scheduler.get_last_lr()[0]
                vram_used = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0
                log.info(
                    f"  Epoch {epoch + 1}/{epochs} | {progress:5.1f}% | "
                    f"Loss: {epoch_loss / (batch_start // batch_size + 1):.4f} | "
                    f"Acc: {epoch_correct / epoch_total:.1%} | "
                    f"LR: {current_lr:.2e} | "
                    f"Speed: {speed:.0f} samples/s | "
                    f"VRAM: {vram_used:.1f} GB"
                )

        # Epoch summary
        train_loss = epoch_loss / max(epoch_total // batch_size, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        elapsed = time.time() - t0

        # ─── Validation ───
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_start_i in range(val_start, total_windows, batch_size):
                val_idx = list(range(val_start_i, min(val_start_i + batch_size, total_windows)))
                batch_X = np.array([X_all[i:i + SEQ_LEN] for i in val_idx if i + SEQ_LEN < len(X_all)])
                batch_y = np.array([y_all[i + SEQ_LEN] for i in val_idx if i + SEQ_LEN < len(y_all)])

                if len(batch_X) == 0:
                    continue

                x_tensor = torch.FloatTensor(batch_X).to(device)
                y_tensor = torch.FloatTensor(batch_y).unsqueeze(1).to(device)

                with autocast(device_type=device.type, dtype=torch.float16):
                    output = model(x_tensor, return_logits=True)
                    loss = criterion(output, y_tensor)

                val_loss += loss.item()
                predictions = (torch.sigmoid(output) > 0.5).float()
                val_correct += (predictions == y_tensor).sum().item()
                val_total += len(y_tensor)

        avg_val_loss = val_loss / max(val_total // batch_size, 1)
        val_acc = val_correct / max(val_total, 1)

        log.info(
            f"╔══ Epoch {epoch + 1}/{epochs} Complete ═══════════════════════╗\n"
            f"║  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1%}\n"
            f"║  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.1%}\n"
            f"║  Time: {elapsed:.0f}s | LR: {scheduler.get_last_lr()[0]:.2e}\n"
            f"╚═══════════════════════════════════════════════════════════════╝"
        )

        # Save checkpoint
        ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch + 1}_acc{val_acc:.3f}.pth"
        torch.save(model.state_dict(), ckpt_path)
        log.info(f"💾 Checkpoint: {ckpt_path}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), pretrained_path)
            log.info(f"🏆 New best model saved! Val loss: {avg_val_loss:.4f}, Val acc: {val_acc:.1%}")

    # Final save
    file_size = pretrained_path.stat().st_size / 1e6
    log.info(f"\n{'=' * 60}")
    log.info(f"✅ PRETRAINING COMPLETE")
    log.info(f"   Model: {pretrained_path} ({file_size:.0f} MB)")
    log.info(f"   Params: {model.num_parameters / 1e6:.1f}M")
    log.info(f"   Best val loss: {best_val_loss:.4f}")
    log.info(f"{'=' * 60}")

    return model


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pretrain NexusTransformer on HuggingFace BTC data")
    parser.add_argument('--skip-download', action='store_true', help='Skip dataset download')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--quick', action='store_true', help='Quick test: 1 epoch, first 100K rows')
    parser.add_argument('--output', type=str, default='nexus_transformer_pretrained.pth',
                        help='Output filename (default: nexus_transformer_pretrained.pth)')
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("NexusTransformer Pretraining Pipeline")
    log.info("=" * 60)

    # Step 1: Download
    if not args.skip_download:
        download_datasets()
    else:
        log.info("⏭️  Skipping download (--skip-download)")

    # Step 2: Load and merge data
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        log.error("❌ No parquet files found in data/pretrain/. Run without --skip-download first.")
        sys.exit(1)

    log.info(f"Loading {len(parquet_files)} dataset file(s)...")
    dfs = []
    for fp in sorted(parquet_files):
        df = pd.read_parquet(fp)
        log.info(f"  📄 {fp.name}: {len(df):,} rows, {len(df.columns)} columns")

        # Apply lag to on-chain columns (look-ahead bias prevention)
        cq_cols = [c for c in df.columns if c.startswith('cq_')]
        if cq_cols:
            log.info(f"     Shifting {len(cq_cols)} on-chain columns by +1440 min (1 day)")
            df[cq_cols] = df[cq_cols].shift(1440)

        dfs.append(df)

    # Use the largest dataset as primary (FamilyLinks)
    df = max(dfs, key=len)
    log.info(f"Primary dataset: {len(df):,} rows")

    if args.quick:
        df = df.head(100_000)
        args.epochs = 1
        log.info(f"⚡ Quick mode: using first {len(df):,} rows, 1 epoch")

    # Step 3: Feature engineering
    df, feature_cols = engineer_features(df)

    # Step 4: Pretrain
    model = pretrain(df, feature_cols, epochs=args.epochs, lr=args.lr, output_name=args.output)

    log.info("\n🎉 Done! The pretrained model is ready for use.")
    log.info("Next steps:")
    log.info("  1. Restart the API server")
    log.info("  2. The model will auto-load pretrained weights")
    log.info("  3. Live retraining will fine-tune from this checkpoint")


if __name__ == "__main__":
    main()
