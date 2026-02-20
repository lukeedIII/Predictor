"""
models.py — Jamba Model Registry for Nexus Training Kit.

All architectures are Jamba (Hybrid Transformer-Mamba SSM) models,
imported from the main python_backend's mamba_model.py.

Four presets, all taking (batch, SEQ_LEN, n_features) → 3-class (DOWN/FLAT/UP):
  A) SmallJamba    — 4L, d256,  4 experts  — fast baseline
  B) LiteJamba     — 6L, d384,  4 experts  — balanced
  C) MediumJamba   — 8L, d512,  6 experts  — high capacity
  D) LargeJamba    — 12L, d768, 8 experts  — maximum power
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════════
# IMPORT JAMBA FROM MAIN BACKEND
# ═══════════════════════════════════════════════════════════════════════════

_backend_dir = Path(__file__).parent.parent / "desktop" / "python_backend"
if _backend_dir.exists() and str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from mamba_model import SmallJamba, create_jamba, JAMBA_CONFIGS  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE REGISTRY — Jamba-only
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_jamba_params(cfg, input_size=36, num_classes=3):
    """Estimate parameter count for a Jamba config WITHOUT creating a model."""
    d = cfg["d_model"]
    n_layers = cfg["n_layers"]
    expand = cfg["expand"]
    d_state = cfg["d_state"]
    d_conv = cfg["d_conv"]
    n_experts = cfg["n_experts"]
    n_heads = cfg["n_heads"]
    n_kv = cfg["n_kv_groups"]
    d_inner = d * expand

    # Input projection: Linear(input_size, d_model)
    proj = input_size * d + d

    # Per-layer costs (approximate):
    # Mamba block: in_proj(d, 2*d_inner) + conv1d(d_inner, d_conv) + x_proj(d_inner, d_state*2+1) +
    #              dt_proj(d_inner, d_inner) + out_proj(d_inner, d) + A_log + D + RMSNorm
    mamba_block = (d * 2 * d_inner + 2 * d_inner +  # in_proj + bias
                   d_inner * d_conv +  # conv1d
                   d_inner * (d_state * 2 + 1) +  # x_proj
                   d_inner * d_inner + d_inner +  # dt_proj
                   d_inner * d + d +  # out_proj
                   d_inner * d_state +  # A_log
                   d_inner +  # D
                   d)  # RMSNorm

    # Attention block: Q/K/V projections + out + RMSNorm
    head_dim = d // n_heads
    attn_block = (d * d +  # Q proj
                  d * (n_kv * head_dim) * 2 +  # K, V proj
                  d * d +  # out proj
                  d)  # RMSNorm

    # MoE FFN (on alternating layers): n_experts * (gate + up + down) + router
    ffn_expert = 2 * d * d_inner + d_inner * d  # gate_proj + up_proj + down_proj
    moe_ffn = n_experts * ffn_expert + d * n_experts  # + router weight
    plain_ffn = 2 * d * d_inner + d_inner * d  # non-MoE layers

    # Count Mamba vs Attention blocks (Jamba ratio: every 4th is attention)
    n_attn = max(1, n_layers // 4)
    n_mamba = n_layers - n_attn

    total_layers = 0
    for i in range(n_layers):
        is_attn = (i % 4 == 3) or (i == n_layers - 1 and n_layers > 1)
        use_moe = (i % 2 == 1) and n_experts > 1
        if is_attn:
            total_layers += attn_block
        else:
            total_layers += mamba_block
        # FFN
        if use_moe:
            total_layers += moe_ffn
        else:
            total_layers += plain_ffn

    # Classification head
    head = d * (d // 2) + (d // 2) + (d // 2) * num_classes + num_classes + d  # + final RMSNorm

    total = proj + total_layers + head
    vram_mb = total * 4 / (1024 * 1024)  # float32
    return {
        'params': total,
        'params_human': f"{total / 1e6:.1f}M" if total >= 1e6 else f"{total / 1e3:.0f}K",
        'vram_mb': round(vram_mb, 1),
        'vram_gb': round(vram_mb / 1024, 2),
    }


# Build registries from JAMBA_CONFIGS
ARCHITECTURES = {}
ARCH_INFO = {}

_JAMBA_DESC = {
    "small":  "4L, d256, 4 exp — fast baseline",
    "lite":   "6L, d384, 4 exp — balanced",
    "medium": "8L, d512, 6 exp — high capacity",
    "large":  "12L, d768, 8 exp — maximum power",
}

for size_key, cfg in JAMBA_CONFIGS.items():
    arch_key = f"{size_key}_jamba"
    # Factory that captures `size_key` for late evaluation
    ARCHITECTURES[arch_key] = (lambda sk: lambda input_size=36, num_classes=3:
                                create_jamba(sk, input_size=input_size, num_classes=num_classes))(size_key)

    # Estimate params for display (without creating model)
    est = _estimate_jamba_params(cfg)
    ARCH_INFO[arch_key] = {
        'params': est['params_human'],
        'vram_gb': est['vram_gb'],
        'desc': _JAMBA_DESC.get(size_key, f"{cfg['label']}"),
    }


def estimate_params(d_model=256, n_layers=4, d_state=16, d_conv=4,
                    expand=2, n_experts=4, n_heads=4, n_kv_groups=2,
                    top_k=1, dropout=0.15, input_size=36, num_classes=3):
    """Estimate params/VRAM for a custom Jamba configuration (for live UI preview)."""
    cfg = {
        "d_model": d_model, "n_layers": n_layers, "d_state": d_state,
        "d_conv": d_conv, "expand": expand, "n_experts": n_experts,
        "n_heads": n_heads, "n_kv_groups": n_kv_groups, "top_k": top_k,
        "dropout": dropout,
    }
    return _estimate_jamba_params(cfg, input_size=input_size, num_classes=num_classes)


def register_custom_arch(name, d_model=256, n_layers=4, d_state=16,
                         d_conv=4, expand=2, n_experts=4, n_heads=4,
                         n_kv_groups=2, top_k=1, dropout=0.15):
    """Register a custom Jamba architecture."""
    def factory(input_size=36, num_classes=3):
        return SmallJamba(
            input_size=input_size, d_model=d_model, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout,
            num_classes=num_classes, n_heads=n_heads, n_kv_groups=n_kv_groups,
            n_experts=n_experts, top_k=top_k,
        )

    ARCHITECTURES[name] = factory
    est = estimate_params(
        d_model=d_model, n_layers=n_layers, d_state=d_state, d_conv=d_conv,
        expand=expand, n_experts=n_experts, n_heads=n_heads,
        n_kv_groups=n_kv_groups, top_k=top_k, dropout=dropout,
    )
    ARCH_INFO[name] = {
        'params': est['params_human'],
        'vram_gb': est['vram_gb'],
        'desc': f'{n_layers}L, d{d_model}, {n_experts} exp — custom',
        'custom': True,
        'config': {
            'd_model': d_model, 'n_layers': n_layers, 'd_state': d_state,
            'd_conv': d_conv, 'expand': expand, 'n_experts': n_experts,
            'n_heads': n_heads, 'n_kv_groups': n_kv_groups, 'top_k': top_k,
            'dropout': dropout,
        },
    }
    return est
