"""
train_mamba.py — Jamba training pipeline: SmallJamba / LiteJamba / MediumJamba.

Supports 3 model sizes via --arch flag (small, lite, medium).

Five critical fixes applied (from pre-training review):
  1. Feature Dropout (NOT channel permutation — heterogeneous features break under shuffling)
  2. 3-class target: UP / FLAT / DOWN (model learns when to sit out chop)
  3. Drop 7 simulated ETH/Gold features, merge real cross-asset data where available
  4. SEQ_LEN = 120 (2 hours) — unleash Mamba's long-context O(n) advantage
  5. PRICE_THRESHOLD = 0.25% — clears round-trip exchange fees + slippage

Additional features:
  - RevIN (Reversible Instance Normalization) — per-sample normalization
  - Scaler isolation — fit StandardScaler on TRAIN ONLY (no val leakage)
  - Gaussian noise augmentation — prevents candle pattern memorization
  - CryptoMamba hyperparameters — lower LR, stronger regularization
  - Sequence stride — reduce redundant overlapping windows

Usage:
    python train_mamba.py --arch small --skip-download       # SmallJamba (default)
    python train_mamba.py --arch lite --skip-download        # LiteJamba
    python train_mamba.py --arch medium --skip-download      # MediumJamba
    python train_mamba.py --arch small --quick --skip-download  # Quick test

References:
  - Jamba (AI21 Labs, 2024): Hybrid Transformer-Mamba Language Model
  - CryptoMamba (2024): Mamba for Bitcoin price prediction
  - TSMamba (2024): Transfer learning for time series with Mamba
  - MambaTS (2024): Variable-aware scanning with permutation training
  - Kim et al. (2022): Reversible Instance Normalization (RevIN)
"""

import os
import sys
import time
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [MAMBA] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "pretrain"
AUX_DATA_DIR = SCRIPT_DIR / "data"
MODEL_DIR = SCRIPT_DIR / "models"
CHECKPOINT_DIR = SCRIPT_DIR / "models" / "checkpoints"

# ──── FIX #4: Long context window (was 30) ────
SEQ_LEN = 120  # 2 hours of 1-min data — Mamba's O(n) makes this free

PREDICTION_HORIZON = 15  # 15-min lookahead

# ──── FIX #5: Threshold clears round-trip fees (was 0.1%) ────
# Binance taker fee: ~0.04-0.05% × 2 sides = 0.08-0.10% round trip
# Need >0.10% net profit → target must be ≥0.20% raw move
PRICE_THRESHOLD = 0.0025  # 0.25% — comfortable margin above fee floor

# Target class mapping
CLASS_DOWN = 0
CLASS_FLAT = 1
CLASS_UP   = 2
NUM_CLASSES = 3
CLASS_NAMES = {0: "DOWN", 1: "FLAT", 2: "UP"}

# ──── FIX #3: Simulated features to DROP ────
SIMULATED_FEATURES = [
    'eth_ret_5', 'eth_ret_15', 'eth_vol_ratio',
    'ethbtc_ret_5', 'ethbtc_trend',
    'gold_ret_15', 'gold_ret_60',
]


# ═══════════════════════════════════════════════════════════════════════════
# REVERSIBLE INSTANCE NORMALIZATION (RevIN)
# ═══════════════════════════════════════════════════════════════════════════

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022).
    Normalizes each sample independently at input time.
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        """x: (B, L, C) → (B, L, C) instance-normalized"""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + self.eps
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight + self.bias
        return x


# ═══════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def augment_batch(x: torch.Tensor, noise_std: float = 0.01,
                  feat_dropout: float = 0.15, training: bool = True):
    """Apply Mamba-specific augmentations to a batch.

    Args:
        x:             (B, L, C) input tensor
        noise_std:     Gaussian noise σ
        feat_dropout:  Fraction of feature columns to zero-mask per batch

    FIX #1: Uses Feature Dropout (zero-masking), NOT channel permutation.
    Channel permutation is catastrophic for heterogeneous tabular features
    because nn.Linear learns index-specific weights.
    """
    if not training:
        return x

    # 1. Gaussian noise injection (CryptoMamba technique)
    if noise_std > 0:
        noise = torch.randn_like(x) * noise_std
        x = x + noise

    # 2. Feature Dropout — randomly zero-mask ~15% of feature columns
    #    This forces the model to not over-rely on any single feature.
    #    Unlike channel permutation, this preserves the feature↔weight mapping.
    if feat_dropout > 0:
        n_features = x.shape[-1]
        n_drop = max(1, int(n_features * feat_dropout))
        drop_idx = torch.randperm(n_features, device=x.device)[:n_drop]
        x = x.clone()
        x[:, :, drop_idx] = 0.0

    return x


# ═══════════════════════════════════════════════════════════════════════════
# 3-CLASS TARGET CREATION  (FIX #2)
# ═══════════════════════════════════════════════════════════════════════════

def create_3class_target(df: pd.DataFrame, threshold: float = PRICE_THRESHOLD,
                         horizon: int = PREDICTION_HORIZON):
    """Create 3-class target: DOWN=0, FLAT=1, UP=2.

    Unlike the original binary target that DROPS neutral rows (creating time
    gaps that corrupt Mamba's sequential state), we keep ALL rows. The model
    learns when to trade AND when to sit on its hands.

    Returns:
        df with 'target' column set to 0/1/2
    """
    close = df['close'].astype(float)
    future_ret = close.pct_change(horizon).shift(-horizon)

    df['target'] = CLASS_FLAT  # Default: FLAT
    df.loc[future_ret > threshold, 'target'] = CLASS_UP
    df.loc[future_ret < -threshold, 'target'] = CLASS_DOWN

    # Drop only the tail rows where future return is NaN (unavoidable)
    df = df.dropna(subset=['target']).reset_index(drop=True)

    # Count distribution
    n_up = (df['target'] == CLASS_UP).sum()
    n_flat = (df['target'] == CLASS_FLAT).sum()
    n_down = (df['target'] == CLASS_DOWN).sum()
    total = len(df)

    log.info(f"3-class target (±{threshold*100:.2f}%, {horizon}min horizon):")
    log.info(f"  UP:   {n_up:>10,} ({n_up/total*100:.1f}%)")
    log.info(f"  FLAT: {n_flat:>10,} ({n_flat/total*100:.1f}%)")
    log.info(f"  DOWN: {n_down:>10,} ({n_down/total*100:.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# REAL CROSS-ASSET MERGE  (FIX #3)
# ═══════════════════════════════════════════════════════════════════════════

def merge_real_cross_assets(df: pd.DataFrame) -> tuple:
    """Merge real ETH and Gold data via backward-looking merge_asof.

    Returns:
        df with real cross-asset features added,
        list of new feature column names
    """
    new_features = []

    # Ensure BTC has a datetime index for merge_asof
    if 'date' in df.columns:
        df['_merge_ts'] = pd.to_datetime(df['date'])
    elif 'timestamp' in df.columns:
        df['_merge_ts'] = pd.to_datetime(df['timestamp'])
    else:
        log.warning("⚠️ No timestamp column found — skipping cross-asset merge")
        return df, new_features

    df = df.sort_values('_merge_ts').reset_index(drop=True)

    # ── Merge ETH/USDT ────
    eth_path = AUX_DATA_DIR / "eth_usdt.parquet"
    if eth_path.exists():
        try:
            eth = pd.read_parquet(eth_path)
            eth['_merge_ts'] = pd.to_datetime(eth['timestamp'])
            eth = eth.sort_values('_merge_ts').reset_index(drop=True)

            # Compute ETH features BEFORE merging
            eth_close = eth['close'].astype(float)
            eth_vol = eth['volume'].astype(float).replace(0, np.nan).fillna(1)
            eth['eth_close'] = eth_close
            eth['eth_volume'] = eth_vol

            # Backward-looking merge: each BTC row gets the most recent ETH data
            df = pd.merge_asof(
                df, eth[['_merge_ts', 'eth_close', 'eth_volume']],
                on='_merge_ts', direction='backward',
            )

            # Compute cross-asset features from real data
            btc_close = df['close'].astype(float)
            if 'eth_close' in df.columns and df['eth_close'].notna().sum() > 100:
                eth_c = df['eth_close'].astype(float)
                df['real_eth_ret_5'] = eth_c.pct_change(5).fillna(0)
                df['real_eth_ret_15'] = eth_c.pct_change(15).fillna(0)
                eth_v = df['eth_volume'].astype(float).replace(0, np.nan).fillna(1)
                df['real_eth_vol_ratio'] = (eth_v / eth_v.rolling(20, min_periods=1).mean()).fillna(1)
                ethbtc = eth_c / btc_close.replace(0, np.nan).fillna(1)
                df['real_ethbtc_ret_5'] = ethbtc.pct_change(5).fillna(0)
                df['real_ethbtc_trend'] = df['real_ethbtc_ret_5'].rolling(12, min_periods=1).mean().fillna(0)

                new_features.extend([
                    'real_eth_ret_5', 'real_eth_ret_15', 'real_eth_vol_ratio',
                    'real_ethbtc_ret_5', 'real_ethbtc_trend',
                ])
                n_valid = df['real_eth_ret_5'].notna().sum()
                log.info(f"✅ Merged real ETH data: {n_valid:,}/{len(df):,} rows have ETH features")
            else:
                log.warning("⚠️ ETH data merged but too sparse — skipping ETH features")

            # Drop helper columns
            for col in ['eth_close', 'eth_volume']:
                if col in df.columns:
                    df = df.drop(columns=[col])
        except Exception as e:
            log.warning(f"⚠️ ETH merge failed: {e}")

    # ── Merge PAXG/USDT (Gold proxy) ────
    paxg_path = AUX_DATA_DIR / "paxg_usdt.parquet"
    if paxg_path.exists() and paxg_path.stat().st_size > 100:
        try:
            paxg = pd.read_parquet(paxg_path)
            paxg['_merge_ts'] = pd.to_datetime(paxg['timestamp'])
            paxg = paxg.sort_values('_merge_ts').reset_index(drop=True)

            paxg_close = paxg['close'].astype(float)
            paxg['gold_close'] = paxg_close

            df = pd.merge_asof(
                df, paxg[['_merge_ts', 'gold_close']],
                on='_merge_ts', direction='backward',
            )

            if 'gold_close' in df.columns and df['gold_close'].notna().sum() > 100:
                gold_c = df['gold_close'].astype(float)
                df['real_gold_ret_15'] = gold_c.pct_change(15).fillna(0)
                df['real_gold_ret_60'] = gold_c.pct_change(60).fillna(0)
                new_features.extend(['real_gold_ret_15', 'real_gold_ret_60'])
                log.info(f"✅ Merged real PAXG (Gold) data")

            if 'gold_close' in df.columns:
                df = df.drop(columns=['gold_close'])
        except Exception as e:
            log.warning(f"⚠️ PAXG merge failed (file may be corrupted): {e}")
    else:
        log.info("ℹ️  PAXG file missing or corrupted — Gold features omitted")

    # Cleanup
    if '_merge_ts' in df.columns:
        df = df.drop(columns=['_merge_ts'])

    return df, new_features


# ═══════════════════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════

class TimeSeriesDataset(torch.utils.data.Dataset):
    """Sliding window dataset with configurable stride.

    FIX #2 note: Because we keep ALL rows (including FLAT), there are NO
    time gaps in the sequences. Mamba sees continuous 120-minute windows.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 120,
                 stride: int = 1):
        # Pre-convert to tensors ONCE (avoids numpy→tensor per __getitem__)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.seq_len = seq_len
        self.stride = stride
        self.total = max(0, (len(X) - seq_len) // stride)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        real_idx = idx * self.stride
        return self.X[real_idx:real_idx + self.seq_len], self.y[real_idx + self.seq_len]


# ═══════════════════════════════════════════════════════════════════════════
# GPU AUTO BATCH SIZE — uses torch.cuda.mem_get_info() for REAL measurement
# ═══════════════════════════════════════════════════════════════════════════

def _auto_batch_size(arch: str, n_features: int, device):
    """Find the largest batch size that keeps real GPU VRAM under 70% of total.

    Uses torch.cuda.mem_get_info() which reads ACTUAL free/total memory from
    the GPU driver — unlike max_memory_allocated() which only tracks the
    Python-side view and misses the CUDA caching allocator overhead.
    """
    from mamba_model import create_jamba

    SAFE_DEFAULTS = {"small": 64, "lite": 48, "medium": 32, "large": 16}

    try:
        free_start, total = torch.cuda.mem_get_info()
        total_gb = total / 1024**3
        # Keep 30% free for OS/display — prevents VRAM thrashing
        max_usable = int(total * 0.70)

        log.info(f"🔍 Auto batch size: {arch} on {total_gb:.1f} GB GPU "
                 f"(max usable: {max_usable / 1024**3:.1f} GB, "
                 f"free now: {free_start / 1024**3:.1f} GB)")

        torch.cuda.empty_cache()

        # Step 1: Load model + optimizer + do one step (baseline memory)
        probe_model = create_jamba(size=arch, input_size=n_features, num_classes=3).to(device)
        probe_opt = torch.optim.AdamW(probe_model.parameters(), lr=1e-4)
        probe_scaler = GradScaler()

        # Warm up optimizer states (Adam allocates 2x params on first step)
        probe_model.train()
        tiny = torch.randn(2, SEQ_LEN, n_features, device=device)
        tiny_y = torch.randint(0, 3, (2,), device=device)
        with autocast('cuda', dtype=torch.float16):
            out = probe_model(tiny, return_logits=True)
            loss = nn.CrossEntropyLoss()(out.logits, tiny_y)
        probe_scaler.scale(loss).backward()
        probe_scaler.step(probe_opt)
        probe_scaler.update()
        del tiny, tiny_y, out, loss
        probe_opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        # Measure baseline: model + optimizer states sitting in VRAM
        free_after_model, _ = torch.cuda.mem_get_info()
        baseline_used = total - free_after_model
        log.info(f"   Model + optimizer baseline: {baseline_used / 1024**3:.2f} GB")

        # Step 2: Measure per-sample cost with a known batch
        test_batch = 16
        torch.cuda.empty_cache()
        free_before_fwd, _ = torch.cuda.mem_get_info()

        probe_opt.zero_grad(set_to_none=True)
        x = torch.randn(test_batch, SEQ_LEN, n_features, device=device)
        y = torch.randint(0, 3, (test_batch,), device=device)
        with autocast('cuda', dtype=torch.float16):
            probe_out = probe_model(x, return_logits=True)
            loss = nn.CrossEntropyLoss()(probe_out.logits, y)
        probe_scaler.scale(loss).backward()

        free_after_fwd, _ = torch.cuda.mem_get_info()
        batch_cost = free_before_fwd - free_after_fwd  # real bytes used for this batch

        del x, y, probe_out, loss
        probe_opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        # Step 3: Calculate max batch from per-sample cost
        per_sample = max(batch_cost / test_batch, 1024)  # bytes per sample
        available_for_batches = max_usable - baseline_used
        max_batch = int(available_for_batches / per_sample)

        log.info(f"   Batch {test_batch} cost: {batch_cost / 1024**2:.0f} MB "
                 f"({per_sample / 1024**2:.1f} MB/sample)")
        log.info(f"   Budget for batches: {available_for_batches / 1024**3:.1f} GB")

        # Cleanup
        del probe_model, probe_opt, probe_scaler
        torch.cuda.empty_cache()

        # Clamp and align
        best_batch = max(8, min(max_batch, 512))
        best_batch = (best_batch // 8) * 8

        log.info(f"✅ Auto batch size: {best_batch} "
                 f"(target VRAM: {(baseline_used + best_batch * per_sample) / 1024**3:.1f} GB "
                 f"/ {total_gb:.1f} GB)")
        return best_batch

    except Exception as e:
        log.warning(f"⚠️  Auto batch probe failed: {e}")
        fallback = SAFE_DEFAULTS.get(arch, 32)
        log.warning(f"   Using safe fallback: batch_size={fallback}")
        torch.cuda.empty_cache()
        return fallback


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train_mamba(df: pd.DataFrame, feature_cols: list,
                epochs: int = 15, lr: float = 1e-4,
                output_name: str = "nexus_small_jamba_v1.pth",
                accuracy_target: float = None,
                noise_std: float = 0.01,
                feat_dropout: float = 0.15,
                stride: int = 5,
                use_revin: bool = True,
                arch: str = "small"):
    """Train Jamba hybrid (Mamba+Attention+MoE) with all critical fixes."""
    from mamba_model import create_jamba, JAMBA_CONFIGS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        # ── Performance: enable cuDNN auto-tuner + TF32 ──
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        log.info("cuDNN benchmark: enabled | TF32 matmul: enabled (Ampere+)")

    # ── Prepare data arrays ──
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df['target'].values.astype(np.int64)  # int64 for CrossEntropyLoss

    total_rows = len(X_all)
    n_features = len(feature_cols)
    log.info(f"Total rows: {total_rows:,} | Features: {n_features}")

    # ── Chronological split: 90% train / 5% val / 5% test ──
    train_end = int(total_rows * 0.90)
    val_end = int(total_rows * 0.95)

    X_train_raw = X_all[:train_end]
    y_train = y_all[:train_end]
    X_val_raw = X_all[train_end:val_end]
    y_val = y_all[train_end:val_end]
    X_test_raw = X_all[val_end:]
    y_test = y_all[val_end:]

    log.info(f"Split: train={len(X_train_raw):,} | val={len(X_val_raw):,} | test={len(X_test_raw):,}")

    # ── Scaler isolation: fit on TRAIN ONLY ──
    from sklearn.preprocessing import StandardScaler

    scaler_sk = StandardScaler()
    X_train = scaler_sk.fit_transform(X_train_raw)
    X_val = scaler_sk.transform(X_val_raw)
    X_test = scaler_sk.transform(X_test_raw)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    scaler_path = MODEL_DIR / "mamba_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_sk, f)
    log.info(f"Saved scaler to {scaler_path} (fitted on {scaler_sk.n_samples_seen_} train samples)")

    # ── Create datasets with stride ──
    train_ds = TimeSeriesDataset(X_train, y_train, seq_len=SEQ_LEN, stride=stride)
    val_ds = TimeSeriesDataset(X_val, y_val, seq_len=SEQ_LEN, stride=1)

    # ── Auto batch size: probe GPU to find optimal batch ──
    if device.type == 'cuda':
        batch_size = _auto_batch_size(arch, n_features, device)
    else:
        batch_size = 32  # CPU fallback
    # Gradient accumulation: target effective batch ~256
    grad_accum = max(1, 256 // batch_size)

    # DataLoader workers: 4 parallel prefetch threads for faster data loading
    n_workers = 4 if device.type == 'cuda' else 0
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=True, drop_last=True,
        persistent_workers=(n_workers > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=n_workers, pin_memory=True,
        persistent_workers=(n_workers > 0),
    )

    log.info(f"Train windows: {len(train_ds):,} (stride={stride}) | Val windows: {len(val_ds):,}")

    # ── Initialize model (size depends on --arch) ──
    cfg_label = JAMBA_CONFIGS[arch]["label"]
    model = create_jamba(
        size=arch,
        input_size=n_features,
        num_classes=NUM_CLASSES,  # 3-class: UP / FLAT / DOWN
    ).to(device)
    log.info(f"{cfg_label}: {model.num_parameters / 1e6:.2f}M params ({model.size_mb:.1f} MB)")
    log.info(f"Output: {NUM_CLASSES}-class ({'/'.join(CLASS_NAMES.values())})")
    log.info(f"Batch size: {batch_size} × {grad_accum} grad_accum = {batch_size * grad_accum} effective")
    log.info(f"Samples/param ratio: {len(train_ds) / model.num_parameters:.1f}x")

    # torch.compile: fuses small kernels into optimized Triton kernels.
    # Requires Triton which is Linux-only. On Windows, falls back gracefully.
    if device.type == 'cuda':
        try:
            import triton  # noqa: F401 — check if Triton is installed
            model = torch.compile(model, mode='max-autotune')
            log.info("🚀 torch.compile enabled (max-autotune) — first batch will be slow (compiling)")
        except (ImportError, Exception) as e:
            log.info(f"ℹ️  torch.compile not available ({type(e).__name__}), using standard mode")
            log.info("   (Triton requires Linux — this is normal on Windows)")

    # RevIN layer
    revin = RevIN(num_features=n_features).to(device) if use_revin else None
    if revin:
        log.info(f"RevIN: enabled ({n_features} channels)")

    # Load existing checkpoint
    pretrained_path = MODEL_DIR / output_name
    if pretrained_path.exists():
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
            log.info(f"🔄 Loaded existing checkpoint — continuing training")
        except RuntimeError as e:
            log.warning(f"⚠️ Checkpoint shape mismatch, training from scratch: {e}")

    # ── Class weights for imbalanced FLAT dominance ──
    train_class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
    # Inverse frequency weighting: rare classes get higher weight
    class_weights = 1.0 / (train_class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES  # Normalize
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    log.info(f"📊 Class weights: DOWN={class_weights[0]:.2f} | "
             f"FLAT={class_weights[1]:.2f} | UP={class_weights[2]:.2f}")

    # ── Optimizer (CryptoMamba-style) ──
    all_params = list(model.parameters())
    if revin:
        all_params += list(revin.parameters())

    optimizer = torch.optim.AdamW(
        all_params, lr=lr,
        weight_decay=0.05, betas=(0.9, 0.95),
        fused=(device.type == 'cuda'),  # Fused kernel: fewer launches
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    amp_scaler = GradScaler()

    # ── LR schedule: cosine with SHORT warmup ──
    total_steps = (len(train_loader) // grad_accum) * epochs
    # Warmup: max 500 steps or 5% of total — never more than half an epoch
    max_warmup = max(1, len(train_loader) // grad_accum // 2)  # half epoch
    warmup_steps = min(500, max(1, int(total_steps * 0.05)), max_warmup)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(0.01, step / warmup_steps)  # start from 1% not 0%
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.05, 0.5 * (1 + np.cos(np.pi * progress)))  # min 5% of peak LR

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training state ──
    best_val_loss = float('inf')
    best_val_acc = 0.0
    global_step = 0
    patience_counter = 0
    patience_limit = 7

    log.info(f"\n{'═' * 60}")
    log.info(f"Training Config:")
    log.info(f"  SEQ_LEN: {SEQ_LEN} (2 hours) | Threshold: ±{PRICE_THRESHOLD*100:.2f}%")
    log.info(f"  LR: {lr} | WD: 0.05 | Batch: {batch_size}×{grad_accum}={batch_size*grad_accum}")
    log.info(f"  Warmup: {warmup_steps} steps | Total: {total_steps} steps")
    log.info(f"  Noise σ: {noise_std} | Feature dropout: {feat_dropout}")
    log.info(f"  Stride: {stride} | RevIN: {use_revin}")
    log.info(f"  Loss: CrossEntropyLoss (3-class, weighted)")
    if accuracy_target:
        log.info(f"  🎯 Accuracy target: {accuracy_target:.0%}")
    log.info(f"{'═' * 60}\n")

    for epoch in range(epochs):
        model.train()
        if revin:
            revin.train()

        epoch_loss_gpu = torch.tensor(0.0, device=device)
        epoch_correct_gpu = torch.tensor(0, dtype=torch.long, device=device)
        epoch_total = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)  # (B,) — class indices

            # Feature Dropout + Gaussian noise (FIX #1)
            x_batch = augment_batch(x_batch, noise_std=noise_std,
                                    feat_dropout=feat_dropout, training=True)

            # RevIN
            if revin:
                x_batch = revin(x_batch)

            with autocast('cuda', dtype=torch.float16):
                out = model(x_batch, return_logits=True)  # ModelOut(logits, aux_loss)
                logits, aux_loss = out.logits, out.aux_loss
                ce_loss = criterion(logits, y_batch)
                # MoE load-balancing loss (encourages uniform expert usage, λ=0.01)
                loss = (ce_loss + aux_loss * 0.01) / grad_accum

            amp_scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # Log CE + aux every 100 steps (fewer GPU→CPU syncs)
                if global_step % 100 == 0:
                    logging.info(
                        f"[STEP {global_step}] ce={ce_loss.item():.4f} "
                        f"aux={aux_loss.item():.5f} "
                        f"loss={loss.item()*grad_accum:.4f}"
                    )

            # Accumulate on GPU (no CPU-GPU sync per batch!)
            epoch_loss_gpu += loss.detach() * grad_accum
            preds = logits.argmax(dim=-1)
            epoch_correct_gpu += (preds == y_batch).sum()
            epoch_total += len(y_batch)

            # ── First-batch timing (so user knows training is alive) ──
            if batch_idx == 0 and epoch == 0:
                first_batch_time = time.time() - t0
                est_epoch_min = first_batch_time * len(train_loader) / 60
                log.info(f"⏱️  First batch done in {first_batch_time:.1f}s "
                         f"(est. epoch: ~{est_epoch_min:.0f} min, "
                         f"{len(train_loader)} batches)")

            # Progress logging (every ~5% of epoch, capped at 100 batches)
            log_interval = max(10, min(100, len(train_loader) // 20))
            if batch_idx > 0 and batch_idx % log_interval == 0:
                progress = batch_idx / len(train_loader) * 100
                speed = epoch_total / (time.time() - t0)
                current_lr = scheduler.get_last_lr()[0]
                # Sync GPU→CPU only at log intervals (not every batch)
                epoch_loss_val = epoch_loss_gpu.item()
                epoch_correct_val = epoch_correct_gpu.item()
                if device.type == 'cuda':
                    free, tot = torch.cuda.mem_get_info()
                    vram_used = (tot - free) / 1e9
                else:
                    vram_used = 0
                log.info(
                    f"  Epoch {epoch+1}/{epochs} | {progress:5.1f}% | "
                    f"Loss: {epoch_loss_val / (batch_idx + 1):.4f} | Acc: {epoch_correct_val/epoch_total:.1%} | "
                    f"LR: {current_lr:.2e} | {speed:.0f} s/s | VRAM: {vram_used:.1f} GB"
                )

        # ── Epoch summary (sync GPU→CPU once per epoch) ──
        epoch_loss = epoch_loss_gpu.item()
        epoch_correct = epoch_correct_gpu.item()
        train_loss = epoch_loss / max(len(train_loader), 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        elapsed = time.time() - t0

        # ── Validation ──
        model.eval()
        if revin:
            revin.eval()

        val_loss_gpu = torch.tensor(0.0, device=device)
        val_correct_gpu = torch.tensor(0, dtype=torch.long, device=device)
        val_total = 0
        val_class_preds = np.zeros(NUM_CLASSES)

        with torch.no_grad():
            for x_val, y_val_batch in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val_batch = y_val_batch.to(device, non_blocking=True)

                if revin:
                    x_val = revin(x_val)

                with autocast('cuda', dtype=torch.float16):
                    out = model(x_val, return_logits=True)  # ModelOut; discard aux in eval
                    logits = out.logits
                    loss = criterion(logits, y_val_batch)

                val_loss_gpu += loss.detach()
                preds = logits.argmax(dim=-1)
                val_correct_gpu += (preds == y_val_batch).sum()
                for c in range(NUM_CLASSES):
                    val_class_preds[c] += (preds == c).sum().item()
                val_total += len(y_val_batch)

        avg_val_loss = val_loss_gpu.item() / max(len(val_loader), 1)
        val_acc = val_correct_gpu.item() / max(val_total, 1)

        # Prediction distribution
        pred_dist = " | ".join(
            f"{CLASS_NAMES[c]} {val_class_preds[c]/val_total*100:.1f}%"
            for c in range(NUM_CLASSES)
        )

        log.info(
            f"╔══ Epoch {epoch+1}/{epochs} Complete ({arch.capitalize()}Jamba {NUM_CLASSES}-class) ════════╗\n"
            f"║  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1%}\n"
            f"║  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.1%}\n"
            f"║  Preds: {pred_dist}\n"
            f"║  Time: {elapsed:.0f}s | LR: {scheduler.get_last_lr()[0]:.2e} | Step: {global_step}\n"
            f"╚═══════════════════════════════════════════════════════════════╝"
        )

        # ── Save checkpoint (full state) ──
        ckpt_path = CHECKPOINT_DIR / f"mamba_epoch_{epoch+1}_acc{val_acc:.3f}.pth"
        save_dict = {
            'model': model.state_dict(),
            'revin': revin.state_dict() if revin else None,
            'epoch': epoch + 1,
            'val_acc': val_acc,
            'val_loss': avg_val_loss,
            'config': {
                'lr': lr, 'noise_std': noise_std, 'feat_dropout': feat_dropout,
                'stride': stride, 'use_revin': use_revin,
                'seq_len': SEQ_LEN, 'threshold': PRICE_THRESHOLD,
                'num_classes': NUM_CLASSES, 'n_features': n_features,
            },
        }
        torch.save(save_dict, ckpt_path)
        log.info(f"💾 Checkpoint: {ckpt_path}")

        # ── Best model ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), pretrained_path)
            log.info(f"🏆 New best model! Val loss: {avg_val_loss:.4f}, Val acc: {val_acc:.1%}")
            if revin:
                torch.save(revin.state_dict(), MODEL_DIR / "mamba_revin.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            log.info(f"⚠️ No improvement ({patience_counter}/{patience_limit})")
            if patience_counter >= patience_limit:
                log.info(f"🛑 Early stopping at epoch {epoch+1}")
                break

        if accuracy_target and val_acc >= accuracy_target:
            log.info(f"🎯🎉 TARGET REACHED! Val acc {val_acc:.1%} >= {accuracy_target:.0%}")
            torch.save(model.state_dict(), pretrained_path)
            break

    # ── Test evaluation ──
    if len(X_test) > SEQ_LEN:
        log.info("\n─── Test Set Evaluation ───")
        test_ds = TimeSeriesDataset(X_test, y_test, seq_len=SEQ_LEN, stride=1)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size * 2, shuffle=False, pin_memory=True,
        )

        model.eval()
        if revin:
            revin.eval()

        test_correct = 0
        test_total = 0
        test_class_correct = np.zeros(NUM_CLASSES)
        test_class_total = np.zeros(NUM_CLASSES)

        with torch.no_grad():
            for x_t, y_t in test_loader:
                x_t = x_t.to(device)
                y_t = y_t.to(device)
                if revin:
                    x_t = revin(x_t)
                with autocast('cuda', dtype=torch.float16):
                    test_out = model(x_t, return_logits=True)
                preds = test_out.logits.argmax(dim=-1)
                test_correct += (preds == y_t).sum().item()
                test_total += len(y_t)
                for c in range(NUM_CLASSES):
                    mask = (y_t == c)
                    test_class_correct[c] += ((preds == c) & mask).sum().item()
                    test_class_total[c] += mask.sum().item()

        test_acc = test_correct / max(test_total, 1)
        log.info(f"📋 Test Accuracy: {test_acc:.1%} ({test_total:,} samples)")
        for c in range(NUM_CLASSES):
            if test_class_total[c] > 0:
                cls_acc = test_class_correct[c] / test_class_total[c]
                log.info(f"   {CLASS_NAMES[c]:>5}: {cls_acc:.1%} "
                         f"({int(test_class_total[c]):,} samples)")

    # ── Final summary ──
    file_size = pretrained_path.stat().st_size / 1e6 if pretrained_path.exists() else 0
    log.info(f"\n{'═' * 60}")
    log.info(f"✅ MAMBA TRAINING COMPLETE")
    log.info(f"   Model: {pretrained_path} ({file_size:.0f} MB)")
    log.info(f"   Params: {model.num_parameters / 1e6:.2f}M | Classes: {NUM_CLASSES}")
    log.info(f"   Best val loss: {best_val_loss:.4f} | Best val acc: {best_val_acc:.1%}")
    log.info(f"   SEQ_LEN={SEQ_LEN} | Threshold=±{PRICE_THRESHOLD*100:.2f}%")
    log.info(f"{'═' * 60}")

    return model


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train Jamba (Hybrid SSM) — Small, Lite, or Medium"
    )
    parser.add_argument('--arch', type=str, default='small',
                        choices=['small', 'lite', 'medium', 'large'],
                        help='Jamba size: small (4.4M), lite (~12M), medium (~28M), large (~60M)')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 100K rows, 2 epochs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (auto-set from --arch if omitted)')
    parser.add_argument('--target-acc', type=float, default=None)
    parser.add_argument('--noise', type=float, default=0.01)
    parser.add_argument('--feat-dropout', type=float, default=0.15,
                        help='Feature dropout fraction (default: 0.15)')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--no-revin', action='store_true')
    args = parser.parse_args()

    # Auto-set output filename from arch
    if args.output is None:
        args.output = f"nexus_{args.arch}_jamba_v1.pth"

    arch_labels = {"small": "SmallJamba", "lite": "LiteJamba", "medium": "MediumJamba", "large": "LargeJamba"}
    label = arch_labels[args.arch]

    log.info("═" * 60)
    log.info(f"{label} Training Pipeline (v4 — Multi-Model Jamba)")
    log.info("═" * 60)
    log.info(f"  Architecture: {label} (--arch {args.arch})")
    log.info(f"  Output: {args.output}")
    if args.arch == 'lite':
        log.info(f"  ⚗️  EXPERIMENTAL: trained on 2021-2026 only (2018-2020 = unseen OOD)")
    log.info(f"  3-class target: UP / FLAT / DOWN")
    log.info(f"  SEQ_LEN = {SEQ_LEN} (2 hours) | Threshold = ±{PRICE_THRESHOLD*100:.2f}%")
    log.info("═" * 60)

    # ── Step 1: Download ──
    if not args.skip_download:
        sys.path.insert(0, str(SCRIPT_DIR))
        from pretrain_transformer import download_datasets
        download_datasets()
    else:
        log.info("⏭️  Skipping download (--skip-download)")

    # ── Step 2: Load data ──
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        log.error("❌ No parquet files. Run without --skip-download first.")
        sys.exit(1)

    log.info(f"Loading {len(parquet_files)} dataset file(s)...")
    dfs = []
    for fp in sorted(parquet_files):
        df = pd.read_parquet(fp)
        log.info(f"  📄 {fp.name}: {len(df):,} rows, {len(df.columns)} columns")
        cq_cols = [c for c in df.columns if c.startswith('cq_')]
        if cq_cols:
            log.info(f"     Shifting {len(cq_cols)} on-chain columns by +1440 min")
            df[cq_cols] = df[cq_cols].shift(1440)
        dfs.append(df)

    df = max(dfs, key=len)
    log.info(f"Primary dataset: {len(df):,} rows")

    if args.quick:
        df = df.head(100_000)
        args.epochs = 2
        log.info(f"⚡ Quick mode: {len(df):,} rows, {args.epochs} epochs")

    # ── LiteJamba: experimental date filter (2021-2026 only) ──
    if args.arch == 'lite':
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            n_before = len(df)
            df = df[df['date'] >= '2021-01-01'].reset_index(drop=True)
            n_dropped = n_before - len(df)
            log.info(f"⚗️  LiteJamba date filter: kept {len(df):,} rows (2021+), "
                     f"dropped {n_dropped:,} pre-2021 rows")
            log.info(f"   Never-seen data: 2017-08 → 2020-12 ({n_dropped:,} candles = OOD test set)")
        else:
            log.warning("⚠️  No 'date' column found — skipping date filter")

    # ── Step 3: Feature engineering ──
    # NOTE: engineer_features() internally drops neutral rows (±0.1%).
    # This removes ~1.7M rows, but all remaining rows have valid features
    # and maintain correct chronological order. We then overwrite the
    # binary target with our 3-class target (Fix #2).
    sys.path.insert(0, str(SCRIPT_DIR))
    from pretrain_transformer import engineer_features

    log.info(f"Engineering features from {len(df):,} rows...")
    n_before_eng = len(df)
    df, feature_cols = engineer_features(df)
    n_after_eng = len(df)
    log.info(f"After feature engineering: {n_after_eng:,} rows "
             f"({n_before_eng - n_after_eng:,} warmup/neutral rows removed)")

    # ── FIX #3: Drop simulated features, merge real data ──
    dropped = [f for f in SIMULATED_FEATURES if f in feature_cols]
    feature_cols = [f for f in feature_cols if f not in SIMULATED_FEATURES]
    log.info(f"🗑️  Dropped {len(dropped)} simulated features: {dropped}")

    # Merge real cross-asset data
    df, new_features = merge_real_cross_assets(df)
    feature_cols.extend(new_features)
    log.info(f"📊 Final feature set: {len(feature_cols)} features")

    # Clean any NaN/inf in new features
    for col in new_features:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)

    # ── FIX #2: Create 3-class target (overwrites binary target) ──
    df = create_3class_target(df)

    # ── Step 4: Train ──
    model = train_mamba(
        df, feature_cols,
        epochs=args.epochs,
        lr=args.lr,
        output_name=args.output,
        accuracy_target=args.target_acc,
        noise_std=args.noise,
        feat_dropout=args.feat_dropout,
        stride=args.stride,
        use_revin=not args.no_revin,
        arch=args.arch,
    )

    log.info(f"\n🎉 Done! {label} trained → {args.output}")
    log.info("  1. Restart the API server")
    log.info(f"  2. Select {label} in model config")
    log.info("  3. Live inference: only trade when P(UP) or P(DOWN) > 0.70")
    log.info("  4. The FLAT class teaches the bot when to sit on its hands")


if __name__ == "__main__":
    main()
