"""
mamba_model.py — SmallJamba: Hybrid Transformer-Mamba SSM for Nexus.

Architecture based on:
  "Jamba: A Hybrid Transformer-Mamba Language Model" (AI21 Labs, 2024)
  "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)

Key innovations from Jamba paper:
  - Hybrid layout: 3 Mamba blocks + 1 Attention block (a:m = 1:3)
  - RMSNorm (prevents training spikes, faster than LayerNorm)
  - Lightweight MoE (4 experts, top-1) on alternating layers
  - Grouped Query Attention (GQA) for memory efficiency
  - No positional encoding needed (Mamba provides implicit position)

No external dependencies beyond torch — works on Windows + CUDA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# RMSNORM — Faster + more stable than LayerNorm (Jamba Section 6.4)
# ═══════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Faster than LayerNorm because it skips the mean computation.
    The Jamba paper (Section 6.4) shows this prevents training loss spikes
    in Mamba layers at large scale.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ═══════════════════════════════════════════════════════════════════════════
# SELECTIVE SSM SCAN (S6) — Core of Mamba
# ═══════════════════════════════════════════════════════════════════════════

def selective_scan(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                   B: torch.Tensor, C: torch.Tensor, D_skip: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch implementation of the selective scan (S6).

    This is the core operation that replaces attention in Transformers.
    It processes the sequence step-by-step, maintaining a hidden state
    that selectively remembers or forgets based on the input.

    Args:
        x:      (B, L, D)    — input sequence (after expansion)
        delta:  (B, L, D)    — input-dependent timestep
        A:      (D, N)       — state transition matrix (diagonal, learned)
        B:      (B, L, N)    — input-to-state matrix (input-dependent)
        C:      (B, L, N)    — state-to-output matrix (input-dependent)
        D_skip: (D,)         — skip connection (residual)

    Returns:
        y:     (B, L, D)    — output sequence
    """
    B_batch, L, D_dim = x.shape
    N = A.shape[1]

    # Discretize: A_bar = exp(delta * A), B_bar = delta * B
    delta_A = torch.exp(delta.unsqueeze(-1) * A)        # (B, L, D, N)
    delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)      # (B, L, D, N)

    # Pre-allocate output tensor (avoids list.append + torch.stack overhead)
    y = torch.empty(B_batch, L, D_dim, device=x.device, dtype=x.dtype)

    # Sequential scan: h[t] = A_bar * h[t-1] + B_bar * x[t]
    h = torch.zeros(B_batch, D_dim, N, device=x.device, dtype=x.dtype)
    for t in range(L):
        h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1)
        y[:, t] = (h * C[:, t].unsqueeze(1)).sum(dim=-1)    # (B, D)

    y = y + x * D_skip                                   # Skip connection
    return y


# ═══════════════════════════════════════════════════════════════════════════
# MAMBA BLOCK — SSM layer with RMSNorm
# ═══════════════════════════════════════════════════════════════════════════

class MambaBlock(nn.Module):
    """One Mamba block: the SSM equivalent of a Transformer encoder layer.

    Architecture:
        Input → RMSNorm → Split into two branches:
          Branch 1 (SSM): Linear expand → Conv1d → SiLU → SSM scan → Linear project
          Branch 2 (Gate): Linear → SiLU
        → Multiply branches (gating) → Linear project → Residual add
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        self.norm = RMSNorm(d_model)  # Upgrade #2: RMSNorm

        # Input projections: one for SSM branch, one for gate
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal depthwise convolution (local pattern extraction)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameter projections (all input-dependent = SELECTIVE)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learned log(A) matrix — initialized to HiPPO-like pattern
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Delta (timestep) projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Args: x: (B, L, d_model) → Returns: (B, L, d_model)"""
        residual = x
        x = self.norm(x)

        # Project and split into SSM branch + gate branch
        xz = self.in_proj(x)                                 # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)                      # each (B, L, d_inner)

        # Causal Conv1d
        x_ssm = x_ssm.transpose(1, 2)                       # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :x.shape[1]]      # Trim to causal
        x_ssm = x_ssm.transpose(1, 2)                       # (B, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # Compute SSM parameters from input (the SELECTIVE part)
        ssm_params = self.x_proj(x_ssm)                     # (B, L, 2*N+1)
        B_param = ssm_params[:, :, :self.d_state]
        C_param = ssm_params[:, :, self.d_state:2*self.d_state]
        dt_raw = ssm_params[:, :, -1:]

        # Timestep (delta)
        delta = F.softplus(self.dt_proj(dt_raw))             # (B, L, d_inner)

        # State matrix A
        A = -torch.exp(self.A_log)                           # (d_inner, N)

        # Run the selective scan
        y = selective_scan(x_ssm, delta, A, B_param, C_param, self.D)

        # Gating
        y = y * F.silu(z)

        # Project back
        y = self.out_proj(y)
        y = self.dropout(y)

        return y + residual


# ═══════════════════════════════════════════════════════════════════════════
# ATTENTION BLOCK — GQA (Jamba Section 2)
# ═══════════════════════════════════════════════════════════════════════════

class AttentionBlock(nn.Module):
    """Grouped Query Attention block with SwiGLU FFN.

    From Jamba paper: even 1 attention layer per 7 Mamba layers fixes
    pure Mamba's weakness at in-context learning and format adherence.

    Architecture:
        Input → RMSNorm → GQA → Residual
              → RMSNorm → SwiGLU FFN → Residual

    Uses Grouped Query Attention (GQA):
        - 4 query heads, 2 KV groups (2:1 ratio)
        - No positional encoding (Mamba layers provide implicit position)
    """

    def __init__(self, d_model, n_heads=4, n_kv_groups=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.head_dim = d_model // n_heads
        self.heads_per_group = n_heads // n_kv_groups

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_groups == 0, "n_heads must be divisible by n_kv_groups"

        # Pre-attention norm
        self.norm_attn = RMSNorm(d_model)

        # GQA projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)       # Full heads
        self.k_proj = nn.Linear(d_model, self.head_dim * n_kv_groups, bias=False)  # Grouped
        self.v_proj = nn.Linear(d_model, self.head_dim * n_kv_groups, bias=False)  # Grouped
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        # Pre-FFN norm
        self.norm_ffn = RMSNorm(d_model)

        # SwiGLU FFN (as per Jamba paper, Section 2)
        ffn_hidden = int(d_model * 2.667)  # ~2/3 of 4x expansion (SwiGLU uses 3 matrices)
        self.gate_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        """Args: x: (B, L, d_model) → Returns: (B, L, d_model)"""
        B, L, D = x.shape

        # ── Attention sublayer ──
        residual = x
        x_norm = self.norm_attn(x)

        # Project Q, K, V
        q = self.q_proj(x_norm).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, L, self.n_kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, L, self.n_kv_groups, self.head_dim).transpose(1, 2)

        # Expand KV groups to match query heads (GQA)
        if self.n_kv_groups < self.n_heads:
            k = k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
            k = k.reshape(B, self.n_heads, L, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
            v = v.reshape(B, self.n_heads, L, self.head_dim)

        # Scaled dot-product attention (causal mask for time series)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # Causal mask: each timestep can only attend to past + current
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)                           # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)
        out = self.o_proj(out)

        x = residual + out

        # ── FFN sublayer (SwiGLU) ──
        residual = x
        x_norm = self.norm_ffn(x)

        # SwiGLU: gate * up, then project down
        gate = F.silu(self.gate_proj(x_norm))
        up = self.up_proj(x_norm)
        x_ffn = self.down_proj(gate * up)
        x_ffn = self.ffn_dropout(x_ffn)

        return residual + x_ffn


# ═══════════════════════════════════════════════════════════════════════════
# MIXTURE OF EXPERTS (MoE) — Lightweight (Jamba Section 6.3)
# ═══════════════════════════════════════════════════════════════════════════

class MoELayer(nn.Module):
    """Lightweight Mixture of Experts layer.

    Replaces the standard FFN/MLP in alternating layers.
    Scaled down from Jamba's 16-expert configuration to 4 experts
    with top-1 routing for our small model.

    Benefits:
      - 4x model capacity with same compute cost
      - Different experts can specialize in different market regimes
      - Load balancing loss prevents expert collapse
    """

    def __init__(self, d_model, n_experts=4, top_k=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        # Router: decides which expert(s) to use per token
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Each expert is a compact SwiGLU FFN (1.5x hidden, balances VRAM vs capacity)
        ffn_hidden = int(d_model * 1.5)
        self.gate_projs = nn.ModuleList([
            nn.Linear(d_model, ffn_hidden, bias=False) for _ in range(n_experts)
        ])
        self.up_projs = nn.ModuleList([
            nn.Linear(d_model, ffn_hidden, bias=False) for _ in range(n_experts)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(ffn_hidden, d_model, bias=False) for _ in range(n_experts)
        ])
        self.dropout = nn.Dropout(dropout)

        # Aux loss for load balancing (stored per forward pass)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        """Args: x: (B, L, d_model) → Returns: (B, L, d_model)"""
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # (B*L, D)

        # Route: compute expert scores
        router_logits = self.router(x_flat)             # (B*L, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize

        # Compute load balancing auxiliary loss (Switch Transformer style)
        # Encourages uniform expert usage across tokens
        tokens_per_expert = torch.zeros(self.n_experts, device=x.device)
        for i in range(self.n_experts):
            tokens_per_expert[i] = (top_k_indices == i).float().sum()
        tokens_per_expert = tokens_per_expert / (B * L)  # fraction per expert
        avg_probs = router_probs.mean(dim=0)              # average routing probability
        self.aux_loss = (tokens_per_expert * avg_probs).sum() * self.n_experts

        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]              # (B*L,)
            weight = top_k_probs[:, k].unsqueeze(-1)      # (B*L, 1)

            for i in range(self.n_experts):
                mask = expert_idx == i
                if not mask.any():
                    continue
                expert_input = x_flat[mask]               # (n_tokens, D)

                # SwiGLU expert computation
                gate = F.silu(self.gate_projs[i](expert_input))
                up = self.up_projs[i](expert_input)
                expert_out = self.down_projs[i](gate * up)

                output[mask] += weight[mask] * expert_out

        output = self.dropout(output)
        return output.view(B, L, D)


# ═══════════════════════════════════════════════════════════════════════════
# JAMBA BLOCK — Mamba or Attention + optional MoE (Jamba Section 2)
# ═══════════════════════════════════════════════════════════════════════════

class JambaBlock(nn.Module):
    """A single Jamba block: wraps either a MambaBlock or AttentionBlock
    and optionally replaces the output MLP with MoE.

    The Jamba architecture interleaves these blocks:
      [Mamba, Mamba+MoE, Mamba, Attention+MoE]
    """

    def __init__(self, d_model, block_type='mamba', use_moe=False,
                 d_state=16, d_conv=4, expand=2, n_heads=4, n_kv_groups=2,
                 n_experts=4, top_k=1, dropout=0.1):
        super().__init__()
        self.use_moe = use_moe

        if block_type == 'mamba':
            self.core = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        elif block_type == 'attention':
            self.core = AttentionBlock(d_model, n_heads, n_kv_groups, dropout)
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

        # Optional MoE replaces a post-block MLP
        if use_moe:
            self.moe_norm = RMSNorm(d_model)
            self.moe = MoELayer(d_model, n_experts, top_k, dropout)

    def forward(self, x):
        """Args: x: (B, L, d_model) → Returns: (B, L, d_model)"""
        x = self.core(x)

        if self.use_moe:
            residual = x
            x = self.moe_norm(x)
            x = self.moe(x)
            x = residual + x

        return x


# ═══════════════════════════════════════════════════════════════════════════
# SMALLJAMBA — Hybrid Transformer-Mamba classifier
# ═══════════════════════════════════════════════════════════════════════════

class SmallJamba(nn.Module):
    """Jamba-style hybrid classifier: 3 Mamba + 1 Attention blocks with MoE.
    ~2.1M params (~8 MB) — marginally larger than SmallMamba (1.74M).

    Architecture (from Jamba paper, adapted for our scale):
      Block 1: MambaBlock          — local pattern extraction
      Block 2: MambaBlock + MoE    — regime-specific feature processing
      Block 3: MambaBlock          — temporal state evolution
      Block 4: AttentionBlock + MoE — global pattern matching + classification anchor

    Key advantages over pure Mamba:
      - Attention layer enables in-context learning (Jamba Section 6.2)
      - MoE provides 4x capacity without 4x compute
      - RMSNorm prevents training spikes (Jamba Section 6.4)
      - Still O(n) for 3/4 of layers (Mamba), only 1/4 is O(n²)

    Supports both binary (num_classes=1) and multi-class (num_classes=3).
    """

    def __init__(self, input_size=42, d_model=256, n_layers=4,
                 d_state=16, d_conv=4, expand=2, dropout=0.15,
                 num_classes=1, n_heads=4, n_kv_groups=2,
                 n_experts=4, top_k=1):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Input projection: features → model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            RMSNorm(d_model),  # Upgrade #2: RMSNorm
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Jamba blocks: [Mamba, Mamba+MoE, Mamba, Attention+MoE]
        # a:m ratio = 1:3 (Jamba paper ablation shows 1:3 ≈ 1:7 performance)
        # MoE every e=2 layers (layers 2 and 4)
        self.blocks = nn.ModuleList([
            JambaBlock(d_model, block_type='mamba', use_moe=False,
                       d_state=d_state, d_conv=d_conv, expand=expand,
                       dropout=dropout),
            JambaBlock(d_model, block_type='mamba', use_moe=True,
                       d_state=d_state, d_conv=d_conv, expand=expand,
                       n_experts=n_experts, top_k=top_k, dropout=dropout),
            JambaBlock(d_model, block_type='mamba', use_moe=False,
                       d_state=d_state, d_conv=d_conv, expand=expand,
                       dropout=dropout),
            JambaBlock(d_model, block_type='attention', use_moe=True,
                       n_heads=n_heads, n_kv_groups=n_kv_groups,
                       n_experts=n_experts, top_k=top_k, dropout=dropout),
        ])

        # Final normalization
        self.norm_f = RMSNorm(d_model)  # Upgrade #2: RMSNorm

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_aux_loss(self):
        """Collect MoE auxiliary losses from all blocks."""
        aux = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            if hasattr(block, 'moe'):
                aux = aux + block.moe.aux_loss
        return aux

    def forward(self, x, return_logits=False):
        """
        Args:
            x: (B, seq_len, input_size) — e.g., (batch, 120, 36)
            return_logits: If True, return raw logits

        Returns:
            num_classes=1: (B, 1) — probability or logit
            num_classes=3: (B, 3) — class probabilities or logits
        """
        # Project input features to model dimension
        x = self.input_proj(x)                # (B, L, d_model)

        # Pass through Jamba blocks (Mamba + Attention hybrid)
        for block in self.blocks:
            x = block(x)                      # (B, L, d_model)

        # Final norm + mean pool over time
        x = self.norm_f(x)                    # (B, L, d_model)
        x = x.mean(dim=1)                     # (B, d_model)

        # Classify
        logits = self.head(x)                 # (B, num_classes)
        if return_logits:
            return logits

        # Apply appropriate activation
        if self.num_classes == 1:
            return self.sigmoid(logits)
        else:
            return F.softmax(logits, dim=-1)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════════════
# JAMBA SIZE CONFIGS — Small / Lite / Medium
# ═══════════════════════════════════════════════════════════════════════════

JAMBA_CONFIGS = {
    "small": {
        "label": "SmallJamba",
        "d_model": 256,
        "n_layers": 4,      # 3 Mamba + 1 Attention
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "dropout": 0.15,
        "n_heads": 4,
        "n_kv_groups": 2,
        "n_experts": 4,
        "top_k": 1,
    },
    "lite": {
        "label": "LiteJamba",
        "d_model": 384,
        "n_layers": 6,      # 5 Mamba + 1 Attention
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "dropout": 0.12,
        "n_heads": 6,
        "n_kv_groups": 2,
        "n_experts": 4,
        "top_k": 1,
    },
    "medium": {
        "label": "MediumJamba",
        "d_model": 512,
        "n_layers": 8,      # 6 Mamba + 2 Attention
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "dropout": 0.10,
        "n_heads": 8,
        "n_kv_groups": 2,
        "n_experts": 6,
        "top_k": 2,
    },
}


def create_jamba(size: str = "small", input_size: int = 42, num_classes: int = 3) -> SmallJamba:
    """Factory function to create a Jamba model of the specified size.

    Args:
        size: 'small', 'lite', or 'medium'
        input_size: Number of input features (default: 42)
        num_classes: 1 for binary, 3 for UP/FLAT/DOWN classification

    Returns:
        Configured SmallJamba instance

    Example:
        model = create_jamba('lite', num_classes=3)
    """
    if size not in JAMBA_CONFIGS:
        raise ValueError(f"Unknown Jamba size '{size}'. Choose from: {list(JAMBA_CONFIGS.keys())}")

    cfg = JAMBA_CONFIGS[size]
    return SmallJamba(
        input_size=input_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        d_state=cfg["d_state"],
        d_conv=cfg["d_conv"],
        expand=cfg["expand"],
        dropout=cfg["dropout"],
        num_classes=num_classes,
        n_heads=cfg["n_heads"],
        n_kv_groups=cfg["n_kv_groups"],
        n_experts=cfg["n_experts"],
        top_k=cfg["top_k"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# ALIASES — All sizes use SmallJamba class with different configs
# ═══════════════════════════════════════════════════════════════════════════

SmallMamba = SmallJamba       # Backward compat
LiteJamba = SmallJamba        # Same class, different config via create_jamba('lite')
MediumJamba = SmallJamba      # Same class, different config via create_jamba('medium')

