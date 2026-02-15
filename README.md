<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-banner.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/logo-banner.svg">
  <img alt="Nexus Shadow-Quant" src="assets/logo-banner.svg" width="100%">
</picture>

<br/>

[![Version](https://img.shields.io/badge/version-6.4.0-blue?style=flat-square)](https://github.com/lukeedIII/Predictor)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Electron](https://img.shields.io/badge/Electron-40-47848F?style=flat-square&logo=electron&logoColor=white)](https://electronjs.org)
[![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-24%20passed-brightgreen?style=flat-square)](desktop/python_backend/tests/test_core.py)

**Nexus Shadow-Quant** is a self-contained desktop application that ingests live BTC market data, computes quant diagnostics, produces calibrated ML probabilities, and runs a fully local **autonomous paper trader** — all on your machine.

</div>

---

## 🔥 TL;DR

- **Predictor goal (code-verified):** probability that BTC will be **up at least +0.30% within 15 minutes**
- **Models:** XGBoost (primary) + optional Transformer sequence model (earns weight only if it performs)
- **Retraining:** every **6 hours**, from scratch, on the most recent **500,000** 1-minute candles (~1 year)
- **Champion-Challenger gate:** newly trained models must beat the current production model (logloss + accuracy) before promotion
- **Drift monitoring:** PSI-based feature drift, prediction distribution shift, and calibration quality (Brier + ECE) tracked every 30 min
- **Dashboard:** drag-and-drop grid layout with saveable presets, light/dark theme, World Clock + Swiss Weather widgets
- **Real-time:** WebSocket push for price, predictions, positions, and quant data (~1s latency)
- **16-model Quant Intelligence:** HMM regime, GJR-GARCH, Heston, Rough Vol, Merton Jump, Bates SVJ, EMD, HHT, Wavelets, MF-DFA, RQA, TDA, PPO RL, Almgren-Chriss, OFI
- **Everything local:** Electron + React + FastAPI (localhost) + Python quant/ML core
- **Dr. Nexus AI:** branded analytical reports with rich markdown, multi-provider LLM (OpenAI → Gemini → Ollama → embedded fallback)
- **Trading:** paper-only simulation (long/short, configurable leverage) with confidence gating + risk controls

---

## 📸 Screenshots

<div align="center">

<img src="assets/demo.gif" alt="Nexus Shadow-Quant Dashboard" width="100%">

<p><em>Platform Overview — Real-time BTC forecasting, 16-model Quant Intelligence, and Dr. Nexus AI Analyst</em></p>

</div>

<details>
<summary><strong>🖼️ Click to see more screenshots</strong></summary>

<br>

| | |
|:---:|:---:|
| <img src="assets/screenshots/dr-nexus-chat.png" width="100%"> | <img src="assets/screenshots/quant-intelligence.png" width="100%"> |
| **Dr. Nexus AI Chat** — branded analysis cards with provider badge | **Quant Intelligence** — 16-model diagnostic panel |
| <img src="assets/screenshots/trading-view.png" width="100%"> | <img src="assets/screenshots/settings.png" width="100%"> |
| **Trading View** — live chart with positions overlay | **Settings** — API keys, LLM provider, model architecture |

</details>

---

## 🧬 What This Actually Predicts (Important)

This is **not** a naive "next candle UP/DOWN" model.

✅ **Label definition (predictor.py):**
- **Horizon:** 15 minutes (close-to-close)
- **Target = 1 (UP):** `close[t+15] > close[t] × 1.003`  → **+0.30% move**
- **Target = 0 (DOWN / NO-EDGE):** anything else (including small up moves < +0.30%)

The +0.30% hurdle is intentionally baked into the label as a "**must-be-worth-trading**" filter (approx. fees + slippage). In plain terms:

> The model estimates the probability that BTC will be **up at least +0.30% within 15 minutes**.

The UI surfaces this as **UP/DOWN + confidence %** (probability output).

---

## ⚙️ How the System Works (End-to-End Pipeline)

Below is the exact runtime loop as implemented (high-level), including what runs every minute vs what runs on the 6-hour retrain cycle.

### 1) Live Data Ingestion (every ~60s)
- BTC/USDT 1m candles via Binance (REST + WebSocket)
- WebSocket adds **live microstructure signals** (spread bps, trades/sec, buy/sell ratio)

### 2) Feature Engineering (same codepath for train & predict)
- **42 scale-invariant features**: returns/ratios/z-scores — never raw price levels
- Identical `_engineer_features()` pipeline is used for:
  - training data matrix
  - live inference vector

### 3) Prediction (every 60s)
- **XGBoost** outputs `P(target=1)`
- **Calibration:** Platt scaling (sigmoid) via `CalibratedClassifierCV`
- **Transformer (optional):**
  - sequence length: 30 timesteps
  - starts with weight = 0
  - only contributes if validation accuracy > 52%
  - ensemble weight increases as performance improves

### 4) Quant Overlay (16-model engine)
A `QuantEngine` runs 16 institutional-grade models organized in tiers:

| Tier | Models | Purpose |
|:-----|:-------|:--------|
| **Core** | HMM Regime, GJR-GARCH, OFI, EMD | Regime detection, volatility, order flow, cycles |
| **Tier 1** | Merton Jump, Rough Vol, HHT, Wavelets | Jump detection, roughness, time-frequency |
| **Tier 2** | Bates SVJ, RQA, MF-DFA, TDA, Heston | Stochastic vol, recurrence, fractals, topology |
| **Tier 3** | PPO RL Agent, Almgren-Chriss | Adaptive trading, optimal execution |

All results are streamed to the UI via WebSocket in real-time.

### 5) Dashboard (Electron + React)
- **Drag-and-drop grid:** 11 resizable cards (react-grid-layout)
- **Layout presets:** 3 saveable slots + reset to default
- **Light / Dark theme:** toggle with localStorage persistence
- **World Clock:** 6 financial hubs (NYSE, LSE, SIX, MOEX, TSE, SSE) with market open/close status
- **Swiss Weather:** live conditions for Zürich
- **TradingView chart:** live BTC/USDT with multiple timeframes + indicators
- **16-model Quant Intelligence panel:** collapsible sections with gauges, badges, active signals
- **Hardware Monitor:** live GPU/CPU utilization, VRAM, temperature
- **Paper Trading Stats:** equity curve, win/loss, PnL breakdown

### 6) Paper Trader (simulation)
- Confidence gate (default `PAPER_MIN_CONFIDENCE = 30`)
- Simulated market fills
- Auto exits: prediction flip OR max hold OR risk rules

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Electron Shell                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │          React Frontend (TypeScript + Vite + RGL)      │  │
│  │  Dashboard (grid) │ Trading │ Dr. Nexus │ Settings     │  │
│  │  ├── PriceCard     ├── Quant Intelligence (16 models)  │  │
│  │  ├── SignalBadge    ├── TradingView Chart              │  │
│  │  ├── WorldClock     ├── News Feed                      │  │
│  │  └── SwissWeather   └── System Health                  │  │
│  └──────────────────────┬─────────────────────────────────┘  │
│                         │                                     │
│            WebSocket (push) + REST (localhost:8420)           │
│                         │                                     │
│  ┌──────────────────────┴─────────────────────────────────┐  │
│  │            Python Backend (FastAPI + Uvicorn)           │  │
│  │  DataCollector → FeatureEngine → Predictor              │  │
│  │        │                 │            │                 │  │
│  │        └──── QuantEngine (16 models) ──┘                 │  │
│  │              PaperTrader (simulation)                    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

Everything runs locally. External calls:
- Binance public market data endpoints (REST + WebSocket)
- OpenAI / Gemini API for Dr. Nexus AI chat (optional — embedded Qwen 0.5B fallback included)

---

## 🧩 Feature Set (42 Total — Scale-Invariant)

All features are designed to be price-level agnostic: **returns/ratios/z-scores** rather than raw prices.

### Feature families
- **Returns & momentum:** 1/5/15m, 1h, 4h returns; RSI (Kalman-smoothed), volume momentum
- **Candle geometry:** high-low range ratio, close-open ratio
- **Trend context:** SMA distance ratios + multi-timeframe trend flags (5m / 15m / 1h)
- **Volatility & risk:** rolling realized vol; GJR-GARCH asymmetry proxy; vol regime ratio
- **Cycles & fractals:** FFT dominant periods; rolling Hurst
- **Microstructure proxies:** tick volatility, "whale" ratio, buy/sell ratio
- **Drift & jumps:** Wasserstein drift; Hawkes-like self-exciting intensity
- **Cross-asset:** ETH returns/vol, ETH/BTC trend, PAXG returns
- **Live-only WS inputs:** trades/sec, WS buy/sell ratio, spread (bps)

✅ Train and live inference share the same feature engineering path.

---

## 🧠 Quant Models: "Used by Predictor" vs "UI-Only"

### Included in the predictor feature vector (vectorized)
- Regime id + confidence (Hurst-based)
- GJR-GARCH volatility (vectorized)
- Hawkes intensity proxy (vectorized)
- Wasserstein drift (vectorized)

✅ These are computed row-by-row using past-only information (no lookahead).

### UI-only diagnostics (Quant Intelligence panel)
The 16-model `QuantEngine` provides real-time institutional-grade diagnostics:

| # | Model | Output | Category |
|:--|:------|:-------|:---------|
| 1 | **HMM Regime** | BULL/SIDEWAYS/BEAR + confidence + state probabilities | Market Regime |
| 2 | **GJR-GARCH** | Forecast vol, asymmetry (γ), conditional vol | Volatility |
| 3 | **Heston SV** | Current/mean vol, leverage ρ | Volatility |
| 4 | **Rough Vol** | Hurst H, roughness score, interpretation | Volatility |
| 5 | **OFI** | Buy/Sell pressure strength, normalized | Order Flow |
| 6 | **EMD** | Top-3 cycle strengths | Cycles |
| 7 | **HHT** | Dominant frequency, period (minutes) | Cycles |
| 8 | **Wavelets** | Trend strength, signal vs noise | Frequency |
| 9 | **Merton Jump** | Detected, probability, direction, risk level | Jumps |
| 10 | **Bates SVJ** | Jump intensity, risk score | Jumps |
| 11 | **MF-DFA** | Delta H, spectral width, interpretation | Fractals |
| 12 | **TDA** | Persistence, complexity, topology score | Topology |
| 13 | **RQA** | Determinism, recurrence rate, interpretation | Patterns |
| 14 | **Almgren-Chriss** | Optimal execution trajectory, market impact | Execution |
| 15 | **PPO RL Agent** | Action distribution (HOLD/BUY/SELL), value | Deep RL |
| 16 | **Basic Metrics** | RSI, Momentum, Sharpe, VWAP distance | Basics |

⚠️ UI-only models may fit on "whatever history is available" and can look optimistic visually. This does **not** change model accuracy because they are not part of the feature vector.

---

## 🧠 ML Models

### XGBoost (Primary)
Hardcoded parameters (current):
- objective: `binary:logistic`
- eval_metric: `logloss`
- n_estimators: 500
- max_depth: 6
- learning_rate: 0.03
- subsample: 0.8
- colsample_bytree: 0.7
- min_child_weight: 5
- gamma: 0.1
- reg_alpha: 0.1 (L1)
- reg_lambda: 1.5 (L2)
- tree_method: `hist`

Training mechanics:
- window: last **500,000** 1m candles
- temporal split: 80% train / 20% test
- sample weights: exponential recency bias (oldest ~5% weight → newest 100%)
- calibration: Platt scaling via `CalibratedClassifierCV`

### Transformer (Optional)
- encoder: d_model=1024, 16 heads, 12 layers, FFN=4096, dropout=0.15
- seq length: 30 timesteps
- training: 30 epochs, AdamW, BCEWithLogitsLoss (AMP-safe)
- GPU: CUDA if available

Ensembling:
- starts at weight 0
- only participates if validation accuracy > 52%
- weight increases with performance (capped)

---

## 💰 Paper Trading Engine (Simulation)

Current implementation (config-driven):
- **Mode:** paper only (no real exchange execution)
- **Direction:** long/short with configurable leverage (default 10x)
- **Multi-position:** up to 3 concurrent positions (`MAX_CONCURRENT = 3`)
- **Entry gate:** adaptive confidence threshold (starts at `PAPER_MIN_CONFIDENCE = 30%`, self-adjusts based on recent performance)
- **Position sizing:** half-Kelly criterion on available balance
- **Execution:** simulated market fills at current price
- **Fee deduction:** Binance taker (0.04%) + slippage (0.01%) charged at both open and close
- **Exits:**
  - TP/SL based on ATR-scaled volatility bands
  - trailing stop-loss (ratchets in profit direction)
  - prediction flip (opposing signal)
  - max hold time: `PAPER_MAX_HOLD_SEC` (default 7200 sec / 2 hours)
  - liquidation if price reaches margin threshold
- **Risk controls:**
  - circuit breaker: halts trading if drawdown exceeds `PAPER_MAX_DRAWDOWN` (default 20%)
  - cooldown: minimum `PAPER_COOLDOWN_SEC` (default 60s) between trades
- **Feedback loop:** trade outcomes (PnL, regime, confidence, hold time) logged for adaptive threshold tuning
- **Net-PnL accounting:** trade records include `gross_pnl_usd`, `pnl_usd` (net), `entry_fee`, `exit_fee`, `total_fee`; stats expose `total_fees`, `net_sharpe_ratio`

---

## 📈 Evaluation Status (Real vs Missing)

### Offline audit backtest (separate script)
A walk-forward backtest on ~3.15M candles reported:
- Accuracy: **50.71%**
- Sharpe: **0.88**

> Note: "Sharpe" is only meaningful when tied to a defined strategy + cost model. Treat this as "promising but modest edge" until fully replicated inside the live system with net-PnL accounting.

### Live system evaluation (current)
- single temporal split (80/20) for calibration
- accuracy validation logged after the 15m horizon passes
- fee-adjusted net-PnL tracked per trade (gross/net/fee breakdown in CSV)
- **rolling walk-forward evaluation** (K=5 expanding-window folds) runs after each retrain; per-fold accuracy/logloss + aggregates logged in retrain history

---

## 🚀 Installation

### Prerequisites

| Software | Version | Required? |
|:---------|:--------|:----------|
| [Python](https://python.org) | 3.12 (3.10+ works) | ✅ Yes |
| [Node.js](https://nodejs.org) | 20 LTS+ | ✅ Yes |
| [Git](https://git-scm.com) | any | Optional |
| NVIDIA GPU + CUDA | RTX 3060+ | ❌ Optional (only for Transformer) |

> **Note:** XGBoost (the primary model) runs on CPU. GPU is only needed for the optional Transformer/LSTM module.

### Option A: Clone with Git

```powershell
git clone https://github.com/lukeedIII/Predictor.git
cd Predictor
```

### Option B: Download ZIP

1. Go to [github.com/lukeedIII/Predictor](https://github.com/lukeedIII/Predictor)
2. Click **Code → Download ZIP**
3. Extract and open PowerShell in the extracted folder

### Step-by-Step Setup

```powershell
# ── 1. Python backend ──
cd desktop\python_backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# ── 2. API keys (create .env file) ──
# Binance keys are required (free, read-only is enough)
# OpenAI key is optional (only for Dr. Nexus AI chat)
New-Item .env -ItemType File
# Edit .env and add:
#   BINANCE_API_KEY=your_key_here
#   BINANCE_SECRET_KEY=your_secret_here
#   OPENAI_API_KEY=your_key_here    (optional)

# ── 3. Verify backend ──
python -c "import predictor; print('Backend OK')"

# ── 4. Frontend (Electron + React) ──
cd ..
npm install

# ── 5. Launch ──
npm run dev
```

### Quick One-Liner (after prerequisites are installed)

```powershell
# From the project root:
cd desktop\python_backend && python -m venv venv && .\venv\Scripts\Activate.ps1 && pip install -r requirements.txt && cd .. && npm install && npm run dev
```

### Running Tests

```powershell
cd desktop\python_backend
.\venv\Scripts\Activate.ps1
python -m pytest tests/ -v
```

### Backtest / Audit Scripts

To reproduce the walk-forward backtest metrics:

```powershell
cd desktop\python_backend
.\venv\Scripts\Activate.ps1
python run_backtest_parallel.py    # uses all cores
# or
python run_backtest.py             # single-threaded
```

> Requires historical BTC/USDT 1m candles in the data directory. Horizon is set via `config.PREDICTION_HORIZON_MINUTES` (default: 15 min).

---

## ✅ What's Done Right (Code-Verified)
- Scale-invariant features (returns/ratios; avoids raw price leakage)
- Same feature computation path for training and live inference
- Label includes +0.30% hurdle (fees/slippage awareness)
- Temporal split (no shuffle)
- Exponential recency weighting for adaptation
- Probability calibration (Platt scaling)
- Live prediction validation after 15 minutes
- Transformer must earn inclusion before contributing
- **Champion-Challenger deployment gate** — new models must match or beat production model on logloss + accuracy before promotion (with configurable grace period for cold start)
- **Drift monitoring** — 3-channel detection: feature PSI, prediction distribution shift, and calibration drift (Brier score + ECE); runs every 30 min with `OK / WARNING / CRITICAL` severity levels
- **Fee-adjusted net-PnL** — Binance taker (0.04%) + slippage (0.01%) deducted at both open and close; trade records include gross/net/fee breakdown; stats expose total_fees and net Sharpe
- **Rolling walk-forward evaluation** — K=5 expanding-window folds after each retrain; logs per-fold accuracy/logloss + aggregates (mean/std/min/max) in retrain history for regime-stability analysis
- **XGBoost early stopping** — stops building trees when eval-set logloss stalls for 30 rounds; typically saves 50-70% training time while preserving (or improving) generalization
- **Regime-based trade gating** — 3-layer filter: (1) Hurst chaos filter (H ≈ 0.5 = random walk), (2) vol-regime bounds (skip extreme volatility > 3x and dead markets < 0.15x), (3) regime win-rate gate blocks trading when a regime's recent win rate drops below 35% over 5+ trades
- **Dynamic class-imbalance correction** — `scale_pos_weight` set to neg/pos ratio at each training call; compensates for label skew caused by the +0.30% hurdle (UP is typically the minority class)
- **Gap detection / quarantine** — detects time gaps > 5 min between consecutive candles, marks gap rows + 3-row buffer as quarantined; quarantined rows excluded from training while features still forward-fill for continuity
- **Pinned dependency versions** — all packages in `requirements.txt` use compatible-range constraints (`>=min,<next-major`) for reproducibility
- **Semantic HMM state ordering** — after fit, HMM states are sorted by mean return (highest → BULL, lowest → BEAR); eliminates label instability across refits
- **RQA/TDA computational guardrails** — both capped at 200-point windows and use `scipy.spatial.distance.cdist` (vectorized) instead of O(n²) Python loops
- **Backtest horizon alignment** — all backtest runners use `config.PREDICTION_HORIZON_MINUTES` (15 min), matching the predictor's label horizon exactly
- **Thread-safe model access** — `threading.RLock` protects model swap during retrain and model read during prediction; eliminates race conditions
- **Dynamic CPU pinning** — XGBoost `n_jobs` set to `min(8, cpu_count-1)`, leaving one core for UI responsiveness
- **GPU is optional** — system check downgrades GPU absence to a warning (XGBoost runs on CPU); GPU only needed for optional Transformer/LSTM
- **Test suite** — 24 pytest tests across 8 areas: label creation, causal integrity, gap detection, HMM ordering, PSI drift, Sharpe annualization, champion-challenger config, RQA/TDA guardrails
- **Clean MIT license** — pure MIT with no contradictory additional terms
- **Drag-and-drop dashboard** — 11 resizable, repositionable cards using react-grid-layout with JSON-serialized layout persistence
- **Layout presets** — 3 saveable preset slots + reset; active slot tracked in localStorage
- **Light / Dark theme** — scoped CSS variable overrides (`.dashboard-light` scope) with toggle button and persistence
- **World Clock widget** — 6 financial hub clocks (NYSE/LSE/SIX/MOEX/TSE/SSE) with live market open/close detection
- **Swiss Weather widget** — live conditions for Zürich (temperature, wind, precipitation)
- **WebSocket real-time push** — price, predictions, positions, accuracy, quant data pushed every ~1s; REST used for larger payloads
- **Quant data integrity fixes** — GARCH current vol falls back to forecast; MF-DFA NOT_COMPUTED handled gracefully; TDA/Bates SVJ key mappings corrected
- **Dr. Nexus branded output** — dual-mode system prompt: Analysis Card format (`# 🔮 Dr. Nexus | [Title]`) for market questions, conversational for casual chat; rich markdown rendering via react-markdown + remark-gfm
- **Provider badge** — every Dr. Nexus response shows which LLM generated it (`via gpt-4.1-mini`, `via embedded:qwen2.5-0.5b`, etc.) for full transparency
- **Embedded fallback LLM** — built-in Qwen2.5-0.5B-Instruct (~1GB) as last-resort provider; lazy-loaded, GPU/CPU auto-detect, selectable in Settings
- **Hardware Monitor widget** — live GPU/CPU metrics (utilization, VRAM, temperature) from nvidia-smi
- **Paper Trading Stats widget** — equity curve, win/loss ratio, cumulative PnL breakdown

---

## ❌ Known Gaps (Ranked by Impact)
1) ~~No **champion–challenger** deployment~~ → ✅ **Implemented v6.2.0**
2) ~~No **drift monitoring**~~ → ✅ **Implemented v6.2.0** (feature PSI + prediction drift + calibration Brier/ECE)
3) ~~No fee-adjusted **net-PnL accounting**~~ → ✅ **Implemented v6.2.0** (taker 0.04% + slippage 0.01% per fill)
4) ~~No rolling **walk-forward evaluation**~~ → ✅ **Implemented v6.2.0** (K=5 expanding-window folds, logged in retrain history)
5) ~~No **early stopping** for XGBoost~~ → ✅ **Implemented v6.2.0** (stops at best iteration via eval-set logloss, `early_stopping_rounds=30`)
6) ~~No regime-specific models or regime-based trade gating~~ → ✅ **Implemented v6.2.0** (3-layer gating: Hurst + vol-regime bounds + win-rate gate)
7) ~~No explicit class-imbalance handling~~ → ✅ **Implemented v6.2.0** (dynamic `scale_pos_weight` = neg/pos ratio per training call)
8) ~~Missing gap detection/quarantine~~ → ✅ **Implemented v6.2.0** (gaps > 5min detected, quarantine rows + 3-row buffer, excluded from training)

---

## 🗺️ Roadmap (High-Impact Next Steps)
- [x] Champion–Challenger + promotion rules (logloss + accuracy gate, configurable thresholds)
- [x] Drift monitoring: PSI + calibration drift + prediction drift (3-channel `DriftMonitor`)
- [x] Fee/slippage-aware paper fills + net-PnL tracking + fee breakdown in trade records
- [x] Rolling walk-forward evaluation (K=5 folds, accuracy/logloss aggregates in retrain history)
- [x] XGBoost early stopping (eval-set logloss, `early_stopping_rounds=30`, logs trees used)
- [x] Regime-based trade gating (Hurst filter + vol-regime bounds + regime win-rate gate)
- [ ] Time-of-day / day-of-week features (optional)

---

## 🛡️ Security Notes
- No secrets committed (loaded via `.env`)
- Paper trading only (no real orders)
- Everything local except Binance market data + optional OpenAI chat calls

---

## ⚠️ Disclaimer
Nexus Shadow-Quant is an educational and research tool. It is not financial advice. Cryptocurrency markets are volatile. You are responsible for any decisions you make.

---

<div align="center">

**v6.4.0 Beta Stable** · Built locally with ⚡ by **G-luc**

</div>
