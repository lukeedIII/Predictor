"""
models.py — All Nexus Transformer architectures (self-contained).

Four architectures, all taking (batch, 30, 42) → binary UP/DOWN:
  A) SmallTransformer   — 4-layer, d_model=256,  ~3.2M params  (0.2 GB VRAM)
  B) MediumTransformer  — 6-layer, d_model=512,  ~19M params   (0.6 GB VRAM)
  C) MidLargeTransformer— 8-layer, d_model=768,  ~32M params   (1.5 GB VRAM)
  D) NexusTransformer   — 12-layer, d_model=1024, ~152M params (2.5 GB VRAM)

With 24GB VRAM (RTX 3090), you can train ALL of these with large batch sizes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE A: Small Transformer (~3.2M params)
# ═══════════════════════════════════════════════════════════════════════════

class SmallTransformer(nn.Module):
    """Right-sized Transformer: 4 layers, d_model=256, 8 heads.
    ~3.2M params — very fast training, good baseline.
    """

    def __init__(self, input_size=42, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 31, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
        x = self.input_proj(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
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
# ARCHITECTURE B: Medium Transformer (~19M params)
# ═══════════════════════════════════════════════════════════════════════════

class MediumTransformer(nn.Module):
    """Medium Transformer: 6 layers, d_model=512, 8 heads.
    ~19M params — balanced speed/accuracy.
    """

    def __init__(self, input_size=42, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 31, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
        x = self.input_proj(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
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
# ARCHITECTURE C: Mid-Large Transformer (~32M params)
# ═══════════════════════════════════════════════════════════════════════════

class MidLargeTransformer(nn.Module):
    """Mid-Large Transformer: 8 layers, d_model=768, 12 heads.
    ~32M params — higher capacity, great for 24GB VRAM GPUs.
    """

    def __init__(self, input_size=42, d_model=768, nhead=12, num_layers=8,
                 dim_feedforward=3072, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 31, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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
        x = self.input_proj(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
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
# ARCHITECTURE D: NexusTransformer (Original, ~152M params)
# ═══════════════════════════════════════════════════════════════════════════

class NexusTransformer(nn.Module):
    """Original NexusTransformer: 12 layers, d_model=1024, 16 heads.
    ~152M params — highest capacity, needs ~2.5GB VRAM.
    """

    def __init__(self, input_size=42, d_model=1024, nhead=16, num_layers=12,
                 dim_feedforward=4096, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 31, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
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
        x = self.input_proj(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
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

ARCHITECTURES = {
    'small_transformer': SmallTransformer,
    'medium_transformer': MediumTransformer,
    'midlarge_transformer': MidLargeTransformer,
    'nexus_transformer': NexusTransformer,
}

ARCH_INFO = {
    'small_transformer':    {'params': '3.2M',  'vram_gb': 0.2, 'desc': '4L, d256 — fast baseline'},
    'medium_transformer':   {'params': '19M',   'vram_gb': 0.6, 'desc': '6L, d512 — balanced'},
    'midlarge_transformer': {'params': '32M',   'vram_gb': 1.5, 'desc': '8L, d768 — high capacity'},
    'nexus_transformer':    {'params': '152M',  'vram_gb': 2.5, 'desc': '12L, d1024 — maximum power'},
}
