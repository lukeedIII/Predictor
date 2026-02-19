# 🚀 BaseJamba Training Kit

**Model**: SmallJamba (4.4M params — hybrid Mamba SSM + Attention + MoE)  
**Data**: BTC 1-minute candles (4.37M rows) + ETH cross-asset features  
**Target GPU**: RTX 3080 24GB (or any CUDA GPU with 8GB+ VRAM)

---

## Quick Start (3 steps)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note**: Make sure you have PyTorch with CUDA support. If not:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```

### 2. Run training
```bash
train.bat
```
Or manually:
```bash
python train_mamba.py --arch small --epochs 25 --lr 1e-4 --stride 5 --skip-download
```

### 3. Upload to HuggingFace (after training completes)
```bash
set HUGGINGFACE_TOKEN=hf_your_token_here
python push_model.py
```

---

## Folder Structure
```
training_kit/
├── train.bat              ← One-click launcher
├── train_mamba.py         ← Training script (all fixes applied)
├── mamba_model.py         ← Jamba model architecture
├── push_model.py          ← HuggingFace upload script
├── requirements.txt       ← Python dependencies
├── TRAINING_README.md     ← This file
├── data/
│   ├── pretrain/
│   │   └── familylinks_btc_1m.parquet  (428 MB)
│   └── eth_usdt.parquet                (8 MB)
└── models/                ← Created during training
    ├── nexus_small_jamba_v1.pth   (best model)
    ├── mamba_scaler.pkl           (feature scaler)
    ├── mamba_revin.pth            (RevIN normalization)
    └── checkpoints/               (per-epoch snapshots)
```

---

## What to Expect

| Metric | RTX 5080 (16GB) | RTX 3080 (24GB) est. |
|---|---|---|
| Batch Size | 136 (auto) | ~200+ (auto) |
| Speed | ~185 s/s | ~120-160 s/s |
| Epoch Time | ~42 min | ~40-55 min |
| VRAM Used | ~14.8 GB | ~18-20 GB |

- **Early stopping** at patience=7 (stops when val loss stops improving)
- **Checkpoints** saved every epoch — training can resume from last checkpoint
- Total training: ~5-10 epochs before early stopping ≈ **3-8 hours**

## Optimizations Applied
- ✅ Chunked selective scan (120→15 Python iterations per Mamba layer)
- ✅ Vectorized MoE routing (no nested Python loops)
- ✅ Fused AdamW optimizer (single-kernel optimizer step)
- ✅ TF32 matmul + cuDNN benchmark enabled
- ✅ AMP mixed precision (FP16 forward + FP32 gradients)
- ✅ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

## Troubleshooting

**"No parquet files"**: Make sure `data/pretrain/familylinks_btc_1m.parquet` exists.

**CUDA OOM**: The auto-batch probe targets 70% VRAM. If it still OOMs, add `--batch 64` to force a smaller batch.

**Slow first batch**: The first batch takes 10-15s (CUDA warmup + JIT compilation). This is normal. Real speed shows from batch 2 onwards.
