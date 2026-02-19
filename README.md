<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-banner.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/logo-banner.svg">
  <img alt="Nexus Shadow-Quant" src="assets/logo-banner.svg" width="100%">
</picture>

<br/>
<br/>

[![Version](https://img.shields.io/badge/version-7.0.0-6C63FF?style=for-the-badge&labelColor=0D0D0D)](https://github.com/lukeedIII/Predictor)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=0D0D0D)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=white&labelColor=0D0D0D)](https://react.dev)
[![Electron](https://img.shields.io/badge/Electron-40-47848F?style=for-the-badge&logo=electron&logoColor=white&labelColor=0D0D0D)](https://electronjs.org)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white&labelColor=0D0D0D)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/Tests-24%20Passing-00D084?style=for-the-badge&labelColor=0D0D0D)](desktop/python_backend/tests/test_core.py)
[![License](https://img.shields.io/badge/License-MIT-F7B731?style=for-the-badge&labelColor=0D0D0D)](LICENSE)

<br/>

> **Nexus Shadow-Quant** is an autonomous, institutional-grade Bitcoin intelligence suite.  
> It ingests live market data, runs a 16-model quant engine, and deploys a self-supervised  
> **Jamba Hybrid SSM** to forecast price direction — all running locally on your machine.

<br/>

</div>

---

## ⚡ What It Does — At a Glance

| Capability | Detail |
|:-----------|:-------|
| 🧠 **Core Model** | **Jamba Hybrid SSM** — Mamba blocks + Attention + Mixture of Experts (MoE) |
| 📐 **Prediction Task** | P(BTC up **≥ +0.30%** within 15 minutes) — 3-class: UP / FLAT / DOWN |
| 🏗️ **Training Set** | Last **500,000** 1-minute candles (~1 year of live data, auto-refreshed) |
| 📡 **Data Pipeline** | Binance REST + WebSocket · 42 scale-invariant features · zero raw-price leakage |
| 🔬 **Quant Engine** | 16 institutional models: HMM, GJR-GARCH, Heston, Rough Vol, PPO RL, TDA, RQA... |
| 💹 **Paper Trader** | Long/short simulation · multi-position · Kelly sizing · fee-adjusted PnL |
| 🤖 **Dr. Nexus AI** | Branded analyst — OpenAI → Gemini → Ollama → embedded Qwen 0.5B fallback |
| 🖥️ **Dashboard** | Electron + React · drag-and-drop grid · WebSocket push · light/dark theme |

---

## 📸 Platform Overview

<div align="center">

<img src="assets/demo.gif" alt="Nexus Shadow-Quant Dashboard" width="100%">

<p><em>Real-time BTC forecasting · 16-model Quant Intelligence · Dr. Nexus AI Analyst</em></p>

</div>

---

## 🧬 The Prediction Target (Exactly)

This is **not** a naive next-candle-up/down classifier.

```
Label Definition  (predictor.py — zero ambiguity)
─────────────────────────────────────────────────
Horizon  : 15 minutes (close-to-close)
UP    = 1 : close[t+15] > close[t] × 1.003    →  +0.30% move
DOWN  = 0 : anything else (including small moves < +0.30%)

The +0.30% hurdle ≈ total fees + slippage.
The model learns when a trade is worth taking — not just price direction.
```

> The UI surfaces this as **UP / DOWN + Confidence %** via softmax probabilities.

---

## 🏛️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Electron Shell                          │
│                                                                │
│  ┌─────────────────── React Frontend ───────────────────────┐  │
│  │  PriceCard · SignalBadge · TradingView Chart              │  │
│  │  Quant Intelligence (16 models) · Dr. Nexus AI           │  │
│  │  World Clock · Swiss Weather · Hardware Monitor          │  │
│  │  Paper Trading (equity curve, PnL) · Layout Presets      │  │
│  └────────────────────┬──────────────────────────────────────┘  │
│                       │  WebSocket (push ~1s) + REST            │
│                       │  localhost:8420                         │
│  ┌────────────────────┴──────────────────────────────────────┐  │
│  │                Python Backend (FastAPI + Uvicorn)          │  │
│  │                                                            │  │
│  │  ┌──────────────┐  ┌───────────────┐  ┌───────────────┐  │  │
│  │  │ DataCollector│→ │FeatureEngine  │→ │  Predictor    │  │  │
│  │  │ (Binance WS) │  │ (42 features) │  │ (Jamba SSM)   │  │  │
│  │  └──────────────┘  └───────────────┘  └───────┬───────┘  │  │
│  │                                               │           │  │
│  │  ┌────────────────────────────────────────────┴───────┐  │  │
│  │  │         QuantEngine  (16 institutional models)      │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │         PaperTrader  (thread-safe simulation)         │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

> **100% local.** External calls: Binance (market data) · OpenAI/Gemini (optional) · HuggingFace (model sync, optional)

---

## 🧠 Jamba Hybrid SSM — The Core Model

Adapted from the **AI21 Labs Jamba architecture (2024)** for financial time series:

| Component | Role |
|:----------|:-----|
| **Mamba blocks** | O(n) selective state space — sequential pattern recognition |
| **Attention blocks** | Global context via Grouped Query Attention (GQA) — memory efficient |
| **Mixture of Experts (MoE)** | 4–8 experts, top-k routing — capacity without compute cost |
| **RMSNorm** | Faster + more numerically stable than LayerNorm |
| **3-class head** | Softmax → P(UP), P(FLAT), P(DOWN) |

### Available Sizes

| Model | Params | VRAM | Architecture | Best For |
|:------|:------:|:----:|:-------------|:---------|
| **SmallJamba** | 4.4M | ~0.2 GB | 3 Mamba + 1 Attn · 4 experts (top-1) | Low VRAM · fastest inference |
| **LiteJamba** ⚗️ | ~12M | ~0.5 GB | 5 Mamba + 1 Attn · 4 experts (top-1) | OOD test — trained 2021-2026 only |
| **MediumJamba** | ~28M | ~1.2 GB | 6 Mamba + 2 Attn · 6 experts (top-2) | Balanced capacity |
| **LargeJamba** 🔥 | ~60M | ~3.5 GB | 9 Mamba + 3 Attn · 8 experts (top-2) | Maximum capacity |

> ⚗️ **LiteJamba is deliberately OOD-tested:** trained exclusively on 2021–2026 data. The model has **never seen 2018–2020**, making that period a true out-of-distribution test of regime generalization.

### Multi-Model Ensemble

Run multiple Jamba variants simultaneously for stronger consensus signals:

```
SmallJamba  →  UP (68%)  ─┐
                           ├─  Ensemble: UP (71%)  ✅  High confidence → trade
LiteJamba   →  UP (74%)  ─┘

SmallJamba  →  UP (55%)  ─┐
                           ├─  Ensemble: FLAT  ⏸️  Disagreement → skip
LiteJamba   → DOWN (60%) ─┘
```

**GPU memory for common combos:**

| Combination | VRAM |
|:------------|:----:|
| SmallJamba alone | ~0.2 GB |
| Small + Lite | ~0.7 GB |
| Small + Medium | ~1.4 GB |
| Small + Large | ~3.7 GB |
| All four | ~5.4 GB |

### Training Commands

```powershell
cd desktop\python_backend
.\venv\Scripts\Activate.ps1

python train_mamba.py --arch small  --skip-download   # SmallJamba  (4.4M)
python train_mamba.py --arch lite   --skip-download   # LiteJamba   (~12M)
python train_mamba.py --arch medium --skip-download   # MediumJamba (~28M)
python train_mamba.py --arch large  --skip-download   # LargeJamba  (~60M)
python train_mamba.py --arch small  --quick --skip-download  # 60s smoke test
```

Each variant saves as `nexus_{size}_jamba_v1.pth` automatically.

---

## 📊 Feature Set — 42 Scale-Invariant Features

All features are **price-level agnostic**: returns, ratios, and z-scores — never raw prices. The exact same `_engineer_features()` function runs at training time and live inference time.

| Family | Features |
|:-------|:---------|
| **Returns & Momentum** | 1m / 5m / 15m / 1h / 4h returns · RSI (Kalman-smoothed) · volume momentum |
| **Candle Geometry** | High-low range ratio · close-open body ratio |
| **Trend Context** | SMA distance ratios · multi-timeframe trend flags (5m / 15m / 1h) |
| **Volatility & Risk** | Rolling realized vol · GJR-GARCH asymmetry proxy · vol regime ratio |
| **Cycles & Fractals** | FFT dominant periods · rolling Hurst exponent |
| **Microstructure** | Tick volatility · whale ratio · buy/sell pressure |
| **Drift & Jumps** | Wasserstein drift · Hawkes self-exciting intensity |
| **Cross-Asset** | ETH returns / vol · ETH/BTC trend · PAXG returns |
| **Live WS Signals** | Trades/sec · WS buy-sell ratio · spread (bps) |

---

## 🔬 Quant Intelligence Engine — 16 Models

### Used in the Predictor Feature Vector (No Lookahead)

| Feature | Source |
|:--------|:-------|
| Regime ID + confidence | Hurst-based HMM (vectorized) |
| GJR-GARCH volatility | Vectorized rolling fit |
| Hawkes intensity | Self-exciting proxy (vectorized) |
| Wasserstein drift | Distribution shift metric (vectorized) |

### UI Diagnostics — Real-Time Institutional Panel

| # | Model | Output |
|:--|:------|:-------|
| 1 | **HMM Regime** | BULL / SIDEWAYS / BEAR + state probabilities |
| 2 | **GJR-GARCH** | Forecast vol · asymmetry γ · conditional vol |
| 3 | **Heston SV** | Current / mean vol · leverage ρ |
| 4 | **Rough Vol** | Hurst H · roughness score |
| 5 | **OFI** | Buy/sell pressure · normalized order flow |
| 6 | **EMD** | Top-3 empirical mode cycle strengths |
| 7 | **HHT** | Dominant frequency · period in minutes |
| 8 | **Wavelets** | Trend strength · signal vs noise ratio |
| 9 | **Merton Jump** | Detected · probability · direction · risk level |
| 10 | **Bates SVJ** | Jump intensity · risk score |
| 11 | **MF-DFA** | ΔH · spectral width · multifractal interpretation |
| 12 | **TDA** | Persistence · complexity · topology score |
| 13 | **RQA** | Determinism · recurrence rate |
| 14 | **Almgren-Chriss** | Optimal execution trajectory · market impact |
| 15 | **PPO RL Agent** | Action distribution (HOLD/BUY/SELL) · value |
| 16 | **Basic Metrics** | RSI · Momentum · Sharpe · VWAP distance |

---

## 💹 Paper Trading Engine

Fully simulated, zero real orders — every parameter is config-driven:

| Parameter | Default | Description |
|:----------|:-------:|:------------|
| **Max concurrent positions** | 3 | Multi-position support |
| **Min confidence gate** | 30% | Adaptive — self-adjusts based on recent performance |
| **Position sizing** | Half-Kelly | On available balance |
| **Leverage** | 10× | Configurable |
| **Fees** | 0.04% + 0.01% | Binance taker + slippage (both open and close) |
| **Max hold time** | 2 hours | `PAPER_MAX_HOLD_SEC` |
| **Circuit breaker** | 20% drawdown | Halts trading |
| **Cooldown** | 60 seconds | Min time between trades |

**Exit triggers:** TP/SL (ATR-scaled), trailing stop, prediction flip, max hold, liquidation.

**Accounting:** every trade record includes `gross_pnl_usd`, `pnl_usd` (net of fees), `entry_fee`, `exit_fee`, `total_fee`. Stats expose `net_sharpe_ratio` and cumulative fee drag.

---

## ✅ Engineering Standards (Code-Verified)

<details>
<summary><b>Click to expand full checklist</b></summary>

- ✅ Scale-invariant features — returns/ratios only, no raw price leakage
- ✅ Identical feature engineering path for training and live inference
- ✅ Label includes +0.30% hurdle (fee/slippage-aware target)
- ✅ Strict temporal split — no shuffle, no future data
- ✅ Exponential recency weighting for market adaptation
- ✅ Probability calibration (Platt scaling)
- ✅ Live prediction validation after 15-minute horizon
- ✅ **Champion-Challenger gate** — challenger must match or beat production on logloss + accuracy
- ✅ **Drift monitoring** — 3-channel: feature PSI + prediction distribution shift + Brier/ECE calibration
- ✅ **Fee-adjusted net-PnL** — gross/net/fee breakdown per trade, total fees in stats
- ✅ **Rolling walk-forward evaluation** — K=5 expanding-window folds, logged per retrain
- ✅ **XGBoost early stopping** — eval-set logloss, `early_stopping_rounds=30`
- ✅ **Regime-based trade gating** — Hurst chaos filter + vol-regime bounds + win-rate gate
- ✅ **Dynamic class-imbalance correction** — `scale_pos_weight` = neg/pos ratio per training call
- ✅ **Gap detection + quarantine** — gaps >5 min detected, quarantined rows excluded from training
- ✅ **Semantic HMM state ordering** — states sorted by mean return for stable BULL/BEAR labels
- ✅ **RQA/TDA computational guardrails** — max 200-point windows, vectorized `cdist`
- ✅ **Thread-safe model access** — `threading.RLock` for model swap and read
- ✅ **Thread-safe PaperTrader** — `threading.RLock` on all balance/position mutations
- ✅ **Bounded feedback log** — `deque(maxlen=2000)` prevents unbounded memory growth
- ✅ **Boot-gate middleware** — FastAPI holds requests until init completes (120s failsafe)
- ✅ **MoE aux loss propagation** — auxiliary loss backpropagated through all Jamba variants
- ✅ **Gradient clipping** — `clip_grad_norm_(max_norm=1.0)` in all training loops
- ✅ **Pinned dependency versions** — `requirements.txt` uses compatible-range constraints
- ✅ **Backtest horizon alignment** — all scripts use `config.PREDICTION_HORIZON_MINUTES`
- ✅ **Hugging Face Model Sync** — cloud backup/restore to skip initial training
- ✅ **Drag-and-drop dashboard** — react-grid-layout with JSON layout persistence
- ✅ **3 saveable layout presets** + reset to default
- ✅ **Light / Dark theme** — CSS variable scoping + localStorage persistence
- ✅ **World Clock** — 6 financial hubs (NYSE/LSE/SIX/MOEX/TSE/SSE) with live market status
- ✅ **Swiss Weather widget** — live conditions for Zürich
- ✅ **Dr. Nexus AI** — dual-mode prompting: Analysis Card format + conversational mode
- ✅ **Provider badges** — every AI response shows which LLM generated it
- ✅ **Embedded fallback LLM** — Qwen2.5-0.5B locally, no API key required
- ✅ **Test suite** — 24 pytest tests across 8 areas
- ✅ **Clean MIT license**

</details>

---

## 📈 Performance Benchmark

| Metric | Value | Notes |
|:-------|:-----:|:------|
| Offline backtest accuracy | **50.71%** | Walk-forward, 3.15M candles |
| Sharpe ratio (backtest) | **0.88** | Treat as "promising but modest edge" |
| Prediction latency | **~5 ms** | GPU (RTX 3060+) |
| Training time | **~6 hrs** | 500K candles, RTX 3060 |
| VRAM (SmallJamba inference) | **~0.2 GB** | Minimum footprint |

---

## 🚀 Installation

### Prerequisites

| Software | Version | Required |
|:---------|:-------:|:--------:|
| [Python](https://python.org) | 3.12 (3.10+) | ✅ |
| [Node.js](https://nodejs.org) | 20 LTS+ | ✅ |
| NVIDIA GPU + CUDA | RTX 3060+ | ✅ Recommended |
| [Git](https://git-scm.com) | any | Optional |

> SmallJamba can run inference on CPU (~50 ms), but GPU is strongly recommended for training and real-time use (~5 ms).

### Setup

```powershell
# 1. Clone
git clone https://github.com/lukeedIII/Predictor.git
cd Predictor

# 2. Python backend
cd desktop\python_backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 3. API keys  (create .env — only Binance is required)
New-Item .env -ItemType File
# Add to .env:
#   BINANCE_API_KEY=your_key
#   BINANCE_SECRET_KEY=your_secret
#   OPENAI_API_KEY=your_key   (optional — Dr. Nexus AI)

# 4. Frontend
cd ..
npm install

# 5. Launch
npm run dev
```

### Train a Model (before first launch)

```powershell
cd desktop\python_backend
.\venv\Scripts\Activate.ps1
python train_mamba.py --arch small --skip-download    # ~6 hrs on RTX 3060
```

### Run Tests

```powershell
cd desktop\python_backend
.\venv\Scripts\Activate.ps1
python -m pytest tests/ -v
# 24 tests — label creation, causal integrity, gap detection, HMM ordering,
# PSI drift, Sharpe, champion-challenger config, RQA/TDA guardrails
```

### Reproduce Backtest

```powershell
cd desktop\python_backend
.\venv\Scripts\Activate.ps1
python run_backtest_parallel.py    # all CPU cores
# or
python run_backtest.py             # single-threaded
```

---

## 🏗️ Project Structure

```
Predictor/
├── desktop/
│   ├── python_backend/        # FastAPI + all ML/quant logic
│   │   ├── api_server.py      # Main FastAPI app + boot-gate middleware
│   │   ├── predictor.py       # NexusPredictor — XGBoost + Jamba SSM ensemble
│   │   ├── mamba_model.py     # SmallJamba / LiteJamba / MediumJamba / LargeJamba
│   │   ├── train_mamba.py     # Standalone training script (all model sizes)
│   │   ├── paper_trader.py    # Thread-safe paper trading engine (RLock)
│   │   ├── quant_engine.py    # 16-model institutional quant engine
│   │   ├── config.py          # All tuneable parameters
│   │   ├── requirements.txt   # Pinned dependencies
│   │   └── tests/             # 24 pytest tests
│   └── src/                   # React + TypeScript frontend (Vite)
├── training_kit/              # Standalone training utilities
├── assets/                    # Logo, banner, demo GIF
└── README.md
```

---

## 🗺️ Changelog

<details>
<summary><b>v7.0.0 — Jamba Edition (current)</b></summary>

- 🧠 Full Jamba Hybrid SSM implementation (4 sizes: Small / Lite / Medium / Large)
- 🔥 MoE auxiliary loss propagation fixed (prevents expert collapse)
- 🛡️ Gradient clipping in all training paths (`max_norm=1.0`)
- 🔒 PaperTrader thread safety — `threading.RLock` + bounded deque
- ⚡ FastAPI boot-gate middleware — zero startup race conditions
- 🎯 AI Trajectory Overlay on TradingView chart
- 🔌 WebSocket real-time push (~1s latency, replaces polling)
- 📊 Multi-Model ensemble scoring

</details>

<details>
<summary><b>v6.x — Institutional Alpha Series</b></summary>

- Champion-Challenger deployment gate
- 3-channel drift monitoring (PSI + calibration + prediction distribution)
- Fee-adjusted net-PnL with Binance taker + slippage accounting
- Rolling walk-forward evaluation (K=5 folds)
- XGBoost early stopping
- Regime-based 3-layer trade gating
- Hugging Face Model Sync

</details>

---

## 🛡️ Security

- **No secrets in code** — credentials loaded from `.env` (git-ignored)
- **Paper trading only** — no real exchange orders, ever
- **Local-first** — model weights, candle data, and trade history stay on your machine
- **Minimal external calls** — Binance market data · optional AI API · optional HF sync

---

## ⚠️ Disclaimer

Nexus Shadow-Quant is an **educational and research tool**. It is not financial advice. Cryptocurrency markets are highly volatile and unpredictable. Past model performance does not guarantee future results. You are solely responsible for any decisions you make.

---

<div align="center">

**v7.0.0 Jamba Edition**

Built locally with ⚡ by **G-luc**

</div>
