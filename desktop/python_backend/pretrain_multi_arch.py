"""
pretrain_multi_arch.py — Pretrain right-sized models on HuggingFace BTC datasets.

Three architectures, all taking (batch, 30, 42) → binary UP/DOWN:
  A) SmallTransformer  — 4-layer, d_model=256, ~4M params
  B) TCN               — Dilated causal convolutions, ~2M params
  C) HybridCNNTrans    — Conv1D local → Transformer global, ~5M params

Usage:
    python pretrain_multi_arch.py --arch small_transformer --epochs 10 --skip-download
    python pretrain_multi_arch.py --arch tcn --epochs 10 --skip-download
    python pretrain_multi_arch.py --arch hybrid --epochs 10 --skip-download

All architectures reuse the same feature engineering as pretrain_transformer.py.
"""

import os
import sys
import gc
import time
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
SEQ_LEN = 30
PREDICTION_HORIZON = 15
PRICE_THRESHOLD = 0.001


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE A: Small Transformer (~4M params)
# ═══════════════════════════════════════════════════════════════════════════
class SmallTransformer(nn.Module):
    """Right-sized Transformer: 4 layers, d_model=256, 8 heads.
    ~4M params — proper ratio for 2.5M training windows.
    """
    def __init__(self, input_size=42, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.15):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # [CLS] token + positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, SEQ_LEN + 1, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer encoder (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_logits=False):
        B = x.shape[0]
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :x.shape[1], :]
        x = self.pos_dropout(x)
        x = self.encoder(x)
        cls_out = x[:, 0, :]
        logits = self.head(cls_out)
        if return_logits:
            return logits
        return self.sigmoid(logits)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE A2: Medium Transformer (~15M params)
# ═══════════════════════════════════════════════════════════════════════════
class MediumTransformer(nn.Module):
    """Medium Transformer: 6 layers, d_model=512, 8 heads.
    ~15M params — between small (3.2M) and the original (152M).
    """
    def __init__(self, input_size=42, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.15):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, SEQ_LEN + 1, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_logits=False):
        B = x.shape[0]
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :x.shape[1], :]
        x = self.pos_dropout(x)
        x = self.encoder(x)
        cls_out = x[:, 0, :]
        logits = self.head(cls_out)
        if return_logits:
            return logits
        return self.sigmoid(logits)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE A3: Mid-Large Transformer (~32M params)
# ═══════════════════════════════════════════════════════════════════════════
class MidLargeTransformer(nn.Module):
    """Mid-Large Transformer: 8 layers, d_model=768, 12 heads.
    ~32M params — testing where overfitting starts to creep in.
    """
    def __init__(self, input_size=42, d_model=768, nhead=12, num_layers=8,
                 dim_feedforward=3072, dropout=0.15):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, SEQ_LEN + 1, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model), enable_nested_tensor=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_logits=False):
        B = x.shape[0]
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :x.shape[1], :]
        x = self.pos_dropout(x)
        x = self.encoder(x)
        cls_out = x[:, 0, :]
        logits = self.head(cls_out)
        if return_logits:
            return logits
        return self.sigmoid(logits)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE B: Temporal Convolutional Network (~2M params)
# ═══════════════════════════════════════════════════════════════════════════
class CausalConvBlock(nn.Module):
    """Single causal conv block with residual connection."""
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.15):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # Causal padding (left only)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.trim = padding  # How many timesteps to trim (causal)

    def forward(self, x):
        # x: (B, C, T)
        residual = x
        out = self.conv1(x)
        if self.trim > 0:
            out = out[:, :, :-self.trim]  # Trim future (causal)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        if self.trim > 0:
            out = out[:, :, :-self.trim]
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)

        return out + residual  # Residual connection


class TCNModel(nn.Module):
    """Temporal Convolutional Network with dilated causal convolutions.
    ~2M params — lightweight, fast, respects temporal ordering natively.

    Receptive field with dilations [1,2,4,8,16] and kernel=3:
      (kernel-1) * sum(dilations) + 1 = 2*31 + 1 = 63 timesteps
      → covers all 30 timesteps easily.
    """
    def __init__(self, input_size=42, channels=256, num_blocks=5, kernel_size=3, dropout=0.15):
        super().__init__()

        # Input projection: features → channels
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, channels),
            nn.GELU(),
        )

        # Dilated causal conv stack
        self.blocks = nn.ModuleList([
            CausalConvBlock(channels, kernel_size=kernel_size,
                           dilation=2**i, dropout=dropout)
            for i in range(num_blocks)
        ])

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')

    def forward(self, x, return_logits=False):
        # x: (B, T, F) → project to (B, T, C)
        x = self.input_proj(x)

        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        # Global average pooling over time dimension
        x = x.mean(dim=2)  # (B, C)

        logits = self.head(x)
        if return_logits:
            return logits
        return self.sigmoid(logits)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE C: Hybrid CNN + Transformer (~5M params)
# ═══════════════════════════════════════════════════════════════════════════
class HybridCNNTransformer(nn.Module):
    """CNN extracts local patterns → Transformer reasons globally.
    ~5M params — best of both worlds.

    Stage 1: 3 Conv1D layers (kernel=3) detect local candlestick patterns
    Stage 2: 2 Transformer layers with [CLS] token for global context
    """
    def __init__(self, input_size=42, cnn_channels=256, d_model=256,
                 nhead=8, num_transformer_layers=2, dropout=0.15):
        super().__init__()

        # Stage 1: Local pattern extraction (CNN)
        self.cnn = nn.Sequential(
            # Layer 1: raw features → patterns
            nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            # Layer 2: patterns → higher patterns
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            # Layer 3: refine
            nn.Conv1d(cnn_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # Stage 2: Global context (Transformer)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, SEQ_LEN + 1, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')

    def forward(self, x, return_logits=False):
        B = x.shape[0]

        # Stage 1: CNN — (B, T, F) → (B, F, T) → Conv → (B, C, T) → (B, T, C)
        x = x.transpose(1, 2)  # Conv1d expects (B, C, T)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # Back to (B, T, d_model)

        # Stage 2: Transformer with [CLS]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :x.shape[1], :]
        x = self.pos_dropout(x)
        x = self.transformer(x)

        cls_out = x[:, 0, :]
        logits = self.head(cls_out)
        if return_logits:
            return logits
        return self.sigmoid(logits)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════
try:
    from predictor import NexusTransformer
except ImportError:
    NexusTransformer = None

try:
    from mamba_model import SmallMamba
except ImportError:
    SmallMamba = None

ARCHITECTURES = {
    'small_transformer': SmallTransformer,
    'medium_transformer': MediumTransformer,
    'midlarge_transformer': MidLargeTransformer,
    'tcn': TCNModel,
    'hybrid': HybridCNNTransformer,
}
if NexusTransformer is not None:
    ARCHITECTURES['nexus_transformer'] = NexusTransformer
if SmallMamba is not None:
    ARCHITECTURES['small_mamba'] = SmallMamba


def get_model(arch_name: str, input_size: int = 42) -> nn.Module:
    """Create model by architecture name."""
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch_name}. Choose from: {list(ARCHITECTURES.keys())}")
    return ARCHITECTURES[arch_name](input_size=input_size)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP (shared by all architectures)
# ═══════════════════════════════════════════════════════════════════════════
def pretrain(df: pd.DataFrame, feature_cols: list, arch_name: str,
             epochs: int = 10, lr: float = 3e-4, output_name: str = None,
             accuracy_target: float = None):
    """Train any architecture on the full dataset with mixed precision.
    
    Args:
        accuracy_target: Stop early if val accuracy reaches this threshold (e.g. 0.80 for 80%).
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")

    # Initialize model
    model = get_model(arch_name, input_size=len(feature_cols)).to(device)
    log.info(f"Architecture: {arch_name}")
    log.info(f"Model: {model.num_parameters / 1e6:.1f}M params ({model.size_mb:.0f} MB)")
    log.info(f"Samples/param ratio: {len(df) / model.num_parameters:.1f}x")

    # Output name
    if output_name is None:
        output_name = f"nexus_{arch_name}_pretrained.pth"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pretrained_path = MODEL_DIR / output_name

    # Load existing checkpoint if available
    if pretrained_path.exists():
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            log.info(f"🔄 Loaded existing checkpoint — continuing training")
        except RuntimeError as e:
            log.warning(f"⚠️ Checkpoint shape mismatch, training from scratch: {e}")

    # Prepare data
    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df['target'].values.astype(np.float32)

    # ── Class balance fix (prevents 99% SHORT bias) ──────────
    n_pos = float(y_all.sum())                      # LONG labels (1)
    n_neg = float(len(y_all) - n_pos)               # SHORT labels (0)
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        log.info(f"📊 Label distribution: LONG={n_pos:,.0f} ({n_pos/len(y_all)*100:.1f}%) | "
                 f"SHORT={n_neg:,.0f} ({n_neg/len(y_all)*100:.1f}%) | pos_weight={pos_weight.item():.3f}")
    else:
        pos_weight = torch.tensor([1.0], device=device)
        log.warning(f"⚠️ Degenerate labels: all {'LONG' if n_pos == len(y_all) else 'SHORT'}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler()

    # Standardize
    from sklearn.preprocessing import StandardScaler
    import pickle
    scaler_sk = StandardScaler()
    X_all = scaler_sk.fit_transform(X_all)

    scaler_path = MODEL_DIR / f"pretrain_scaler_{arch_name}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_sk, f)
    log.info(f"Saved scaler to {scaler_path}")

    # Windows
    total_windows = len(X_all) - SEQ_LEN
    log.info(f"Total windows: {total_windows:,}")

    batch_size = 256  # lowered from 512 — OOM-retry will halve if needed
    grad_accum = 4   # Effective batch = 1024

    # Split: 95% train, 5% val
    val_start = int(total_windows * 0.95)

    # Cosine annealing with warmup
    warmup_steps = 500
    total_steps = (val_start // (batch_size * grad_accum)) * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    global_step = 0
    patience_counter = 0
    patience_limit = 5  # Early stop if val loss doesn't improve for 5 epochs

    if accuracy_target:
        log.info(f"🎯 Accuracy target: {accuracy_target:.0%} — will stop early if reached")

    for epoch in range(epochs):
        # ── OOM-RETRY LOOP: halve batch & restart epoch on OOM ──
        while True:
            try:
                model.train()
                epoch_loss = 0
                epoch_correct = 0
                epoch_total = 0

                train_indices = np.random.permutation(val_start)
                t0 = time.time()
                optimizer.zero_grad()

                for batch_start in range(0, val_start, batch_size):
                    batch_idx = train_indices[batch_start:batch_start + batch_size]

                    batch_X = np.array([X_all[i:i + SEQ_LEN] for i in batch_idx if i + SEQ_LEN < len(X_all)])
                    batch_y = np.array([y_all[i + SEQ_LEN] for i in batch_idx if i + SEQ_LEN < len(y_all)])

                    if len(batch_X) == 0:
                        continue

                    x_tensor = torch.FloatTensor(batch_X).to(device)
                    y_tensor = torch.FloatTensor(batch_y).unsqueeze(1).to(device)

                    with autocast('cuda', dtype=torch.float16):
                        output = model(x_tensor, return_logits=True)
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

                    # Progress logging
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

                break  # epoch completed successfully

            except torch.cuda.OutOfMemoryError:
                old_bs = batch_size
                batch_size = max(16, batch_size // 2)
                log.warning(f"💥 OOM detected! Auto-halving batch size: {old_bs} → {batch_size}")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                if batch_size <= 16:
                    log.error(f"❌ OOM even at batch_size=16 — aborting")
                    raise
                log.info(f"🔄 Retrying epoch {epoch + 1} with batch_size={batch_size}...")

        # Epoch summary
        train_loss = epoch_loss / max(epoch_total // batch_size, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        elapsed = time.time() - t0

        # ─── Validation ───
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_pred_long = 0

        with torch.no_grad():
            for val_start_i in range(val_start, total_windows, batch_size):
                val_idx = list(range(val_start_i, min(val_start_i + batch_size, total_windows)))
                batch_X = np.array([X_all[i:i + SEQ_LEN] for i in val_idx if i + SEQ_LEN < len(X_all)])
                batch_y = np.array([y_all[i + SEQ_LEN] for i in val_idx if i + SEQ_LEN < len(y_all)])

                if len(batch_X) == 0:
                    continue

                x_tensor = torch.FloatTensor(batch_X).to(device)
                y_tensor = torch.FloatTensor(batch_y).unsqueeze(1).to(device)

                with autocast('cuda', dtype=torch.float16):
                    output = model(x_tensor, return_logits=True)
                    loss = criterion(output, y_tensor)

                val_loss += loss.item()
                predictions = (torch.sigmoid(output) > 0.5).float()
                val_correct += (predictions == y_tensor).sum().item()
                val_pred_long += predictions.sum().item()
                val_total += len(y_tensor)

        avg_val_loss = val_loss / max(val_total // batch_size, 1)
        val_acc = val_correct / max(val_total, 1)

        # Prediction distribution check (bias detector)
        pred_long = val_pred_long / max(val_total, 1) * 100
        pred_short = 100 - pred_long
        bias_warning = ""
        if pred_long < 20 or pred_long > 80:
            bias_warning = " ⚠️ DIRECTIONAL BIAS DETECTED"

        log.info(
            f"╔══ Epoch {epoch + 1}/{epochs} Complete ({arch_name}) ══════════════╗\n"
            f"║  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1%}\n"
            f"║  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.1%}\n"
            f"║  Predictions: LONG {pred_long:.1f}% | SHORT {pred_short:.1f}%{bias_warning}\n"
            f"║  Time: {elapsed:.0f}s | LR: {scheduler.get_last_lr()[0]:.2e}\n"
            f"╚═══════════════════════════════════════════════════════════════╝"
        )

        # Save checkpoint
        ckpt_path = CHECKPOINT_DIR / f"{arch_name}_epoch_{epoch + 1}_acc{val_acc:.3f}.pth"
        torch.save(model.state_dict(), ckpt_path)
        log.info(f"💾 Checkpoint: {ckpt_path}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), pretrained_path)
            log.info(f"🏆 New best model saved! Val loss: {avg_val_loss:.4f}, Val acc: {val_acc:.1%}")
            patience_counter = 0
        else:
            patience_counter += 1
            log.info(f"⚠️ No improvement ({patience_counter}/{patience_limit})")
            if patience_counter >= patience_limit:
                log.info(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                break

        # Check accuracy target
        if accuracy_target and val_acc >= accuracy_target:
            log.info(f"🎯🎉 ACCURACY TARGET REACHED! Val acc {val_acc:.1%} >= {accuracy_target:.0%}")
            log.info(f"   Stopping training at epoch {epoch + 1}")
            torch.save(model.state_dict(), pretrained_path)
            log.info(f"🏆 Target model saved to {pretrained_path}")
            break

    # Final summary
    file_size = pretrained_path.stat().st_size / 1e6 if pretrained_path.exists() else 0
    log.info(f"\n{'=' * 60}")
    log.info(f"✅ PRETRAINING COMPLETE — {arch_name}")
    log.info(f"   Model: {pretrained_path} ({file_size:.0f} MB)")
    log.info(f"   Params: {model.num_parameters / 1e6:.1f}M")
    log.info(f"   Samples/param: {len(df) / model.num_parameters:.1f}x")
    log.info(f"   Best val loss: {best_val_loss:.4f} | Best val acc: {best_val_acc:.1%}")
    log.info(f"{'=' * 60}")

    return model


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Pretrain right-sized models on BTC data")
    parser.add_argument('--arch', type=str, default='small_transformer',
                        choices=list(ARCHITECTURES.keys()),
                        help='Architecture to train (default: small_transformer)')
    parser.add_argument('--skip-download', action='store_true', help='Skip dataset download')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--quick', action='store_true', help='Quick test: 1 epoch, first 100K rows')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: nexus_{arch}_pretrained.pth)')
    parser.add_argument('--target-acc', type=float, default=None,
                        help='Stop early when val accuracy reaches this (e.g. 0.80 for 80%%)')
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"Multi-Architecture Pretraining Pipeline")
    log.info(f"Architecture: {args.arch}")
    log.info("=" * 60)

    # Step 1: Download (reuse from pretrain_transformer)
    if not args.skip_download:
        sys.path.insert(0, str(SCRIPT_DIR))
        from pretrain_transformer import download_datasets
        download_datasets()
    else:
        log.info("⏭️  Skipping download (--skip-download)")

    # Step 2: Load data
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        log.error("❌ No parquet files found. Run pretrain_transformer.py first to download data.")
        sys.exit(1)

    log.info(f"Loading {len(parquet_files)} dataset file(s)...")
    dfs = []
    for fp in sorted(parquet_files):
        df = pd.read_parquet(fp)
        log.info(f"  📄 {fp.name}: {len(df):,} rows, {len(df.columns)} columns")

        cq_cols = [c for c in df.columns if c.startswith('cq_')]
        if cq_cols:
            log.info(f"     Shifting {len(cq_cols)} on-chain columns by +1440 min (1 day)")
            df[cq_cols] = df[cq_cols].shift(1440)

        dfs.append(df)

    df = max(dfs, key=len)
    log.info(f"Primary dataset: {len(df):,} rows")

    if args.quick:
        df = df.head(100_000)
        args.epochs = 1
        log.info(f"⚡ Quick mode: using first {len(df):,} rows, 1 epoch")

    # Step 3: Feature engineering (reuse from pretrain_transformer)
    sys.path.insert(0, str(SCRIPT_DIR))
    from pretrain_transformer import engineer_features
    log.info(f"Engineering features from {len(df):,} rows...")
    df, feature_cols = engineer_features(df)

    # Step 4: Pretrain
    model = pretrain(df, feature_cols, arch_name=args.arch,
                     epochs=args.epochs, lr=args.lr, output_name=args.output,
                     accuracy_target=args.target_acc)

    log.info(f"\n🎉 Done! {args.arch} model ready.")
    log.info(f"Next: try other architectures or meta-ensemble all models.")


if __name__ == "__main__":
    main()
