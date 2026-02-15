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
# CUSTOM TRANSFORMER (User-Configurable)
# ═══════════════════════════════════════════════════════════════════════════

class CustomTransformer(nn.Module):
    """Fully configurable Transformer — design your own architecture.

    All hyperparameters are exposed:
      d_model, nhead, num_layers, dim_feedforward, dropout, seq_len
    """

    def __init__(self, input_size=42, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.15, seq_len=30):
        super().__init__()
        self.config = {
            'input_size': input_size, 'd_model': d_model, 'nhead': nhead,
            'num_layers': num_layers, 'dim_feedforward': dim_feedforward,
            'dropout': dropout, 'seq_len': seq_len,
        }
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        head_hidden = max(64, d_model // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
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


def estimate_params(d_model=256, nhead=8, num_layers=4, dim_feedforward=1024,
                    input_size=42, seq_len=30):
    """Estimate parameter count WITHOUT creating a model (for live UI preview)."""
    # Input projection: Linear(input_size, d_model) + LayerNorm(d_model)
    proj = input_size * d_model + d_model + 2 * d_model  # weight + bias + LN

    # Positional embedding + CLS token
    pos = (seq_len + 1) * d_model + d_model

    # Transformer encoder layers (each):
    # Self-attention: Q, K, V projections + output projection
    attn = 4 * (d_model * d_model + d_model)  # Wq, Wk, Wv, Wo + biases
    # FFN: two linear layers
    ffn = d_model * dim_feedforward + dim_feedforward + dim_feedforward * d_model + d_model
    # Two LayerNorms
    ln = 4 * d_model
    per_layer = attn + ffn + ln
    encoder = num_layers * per_layer

    # Head
    head_hidden = max(64, d_model // 2)
    head = 2 * d_model + d_model * head_hidden + head_hidden + head_hidden * 1 + 1

    total = proj + pos + encoder + head
    vram_mb = total * 4 / (1024 * 1024)  # float32
    return {
        'params': total,
        'params_human': f"{total / 1e6:.1f}M" if total >= 1e6 else f"{total / 1e3:.0f}K",
        'vram_mb': round(vram_mb, 1),
        'vram_gb': round(vram_mb / 1024, 2),
    }


def register_custom_arch(name, d_model=256, nhead=8, num_layers=4,
                         dim_feedforward=1024, dropout=0.15, seq_len=30):
    """Register a custom architecture in the global ARCHITECTURES dict."""
    # Create a factory function that captures the config
    def factory(input_size=42):
        return CustomTransformer(
            input_size=input_size, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, seq_len=seq_len,
        )

    ARCHITECTURES[name] = factory
    est = estimate_params(d_model, nhead, num_layers, dim_feedforward, 42, seq_len)
    ARCH_INFO[name] = {
        'params': est['params_human'],
        'vram_gb': est['vram_gb'],
        'desc': f'{num_layers}L, d{d_model}, {nhead}H — custom',
        'custom': True,
        'config': {
            'd_model': d_model, 'nhead': nhead, 'num_layers': num_layers,
            'dim_feedforward': dim_feedforward, 'dropout': dropout, 'seq_len': seq_len,
        },
    }
    return est


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
    'small_transformer':    {'params': '3.2M',  'vram_gb': 0.2, 'desc': '4L, d256, 8H — fast baseline'},
    'medium_transformer':   {'params': '19M',   'vram_gb': 0.6, 'desc': '6L, d512, 8H — balanced'},
    'midlarge_transformer': {'params': '32M',   'vram_gb': 1.5, 'desc': '8L, d768, 12H — high capacity'},
    'nexus_transformer':    {'params': '152M',  'vram_gb': 2.5, 'desc': '12L, d1024, 16H — maximum power'},
}

