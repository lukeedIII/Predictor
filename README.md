<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-banner.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/logo-banner.svg">
  <img alt="Nexus Shadow-Quant" src="assets/logo-banner.svg" width="100%">
</picture>

<br/>

[![Version](https://img.shields.io/badge/version-6.2.0-blue?style=flat-square)](https://github.com/lukeedIII/Predictor)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Electron](https://img.shields.io/badge/Electron-40-47848F?style=flat-square&logo=electron&logoColor=white)](https://electronjs.org)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-Private-red?style=flat-square)]()

**Nexus Shadow-Quant** is a self-contained desktop application that ingests live BTC market data, computes institutional-style quant diagnostics, produces calibrated ML probabilities, and runs a fully local **autonomous paper trader** — all on your machine.

</div>

---

## 🔥 TL;DR

- **Predictor goal (code-verified):** probability that BTC will be **up at least +0.30% within 15 minutes**
- **Models:** XGBoost (primary) + optional Transformer sequence model (earns weight only if it performs)
- **Retraining:** every **6 hours**, from scratch, on the most recent **500,000** 1-minute candles (~1 year)
- **Champion-Challenger gate:** newly trained models must beat the current production model (logloss + accuracy) before promotion
- **Drift monitoring:** PSI-based feature drift, prediction distribution shift, and calibration quality (Brier + ECE) tracked every 30 min
- **Everything local:** Electron + React + FastAPI (localhost) + Python quant/ML core
- **Trading:** paper-only simulation (long/short, configurable leverage) with confidence gating + risk controls

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

### 4) Quant Overlay (context & UI)
A QuantEngine runs diagnostics (some are UI-only), e.g. HMM/GARCH/entropy/TDA panels, plus drift/vol/jump proxies.

### 5) Paper Trader (simulation)
- Confidence gate (default `PAPER_MIN_CONFIDENCE = 30`)
- Simulated market fills
- Auto exits: prediction flip OR max hold OR risk rules

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Electron Shell                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              React Frontend (TypeScript + Vite)         │  │
│  │  Dashboard  │  Paper Trading  │  Dr. Nexus  │ Settings  │  │
│  └──────────────────────┬─────────────────────────────────┘  │
│                         │                                     │
│               REST + WebSocket (localhost:8420)               │
│                         │                                     │
│  ┌──────────────────────┴─────────────────────────────────┐  │
│  │            Python Backend (FastAPI + Uvicorn)           │  │
│  │  DataCollector → FeatureEngine → Predictor              │  │
│  │        │                 │            │                 │  │
│  │        └──────── QuantEngine (UI) ─────┘                 │  │
│  │              PaperTrader (simulation)                    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

Everything runs locally. External calls:
- Binance public market data endpoints (REST + WebSocket)
- OpenAI API only if Dr. Nexus is enabled (optional)

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

### UI-only diagnostics (not fed into the ML predictor)
- HMM regime panels
- GARCH VaR panels
- entropy / RQA / TDA / EMD style diagnostics
- additional overlays for interpretation

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

## 🧪 Reproducibility

### Development mode
```powershell
git clone https://github.com/lukeedIII/Predictor.git
cd Predictor

cd desktop
npm install

pip install -r requirements.txt

copy ..\.env.example python_backend\.env
# OpenAI key is optional (Dr. Nexus disabled if missing)

npm run dev
```

### Backtest / audit scripts
This repo includes backtest tooling (e.g. `run_backtest_parallel.py`). To reproduce headline metrics:

1) Ensure you have historical BTC/USDT 1m candles available in the expected data location used by the backtest script.
2) Run the backtest script from the python backend context.
3) Compare reported:
   - accuracy
   - Sharpe (confirm definition)
   - regime breakdown (if supported)

> If you want, add a `BACKTEST.md` with exact dataset path expectations and command examples for your machine.

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

---

## ❌ Known Gaps (Ranked by Impact)
1) ~~No **champion–challenger** deployment~~ → ✅ **Implemented v6.2.0**
2) ~~No **drift monitoring**~~ → ✅ **Implemented v6.2.0** (feature PSI + prediction drift + calibration Brier/ECE)
3) ~~No fee-adjusted **net-PnL accounting**~~ → ✅ **Implemented v6.2.0** (taker 0.04% + slippage 0.01% per fill)
4) ~~No rolling **walk-forward evaluation**~~ → ✅ **Implemented v6.2.0** (K=5 expanding-window folds, logged in retrain history)
5) ~~No **early stopping** for XGBoost~~ → ✅ **Implemented v6.2.0** (stops at best iteration via eval-set logloss, `early_stopping_rounds=30`)
6) No regime-specific models or regime-based trade gating
7) No explicit class-imbalance handling (label skew from +0.30% hurdle)
8) Missing gap detection/quarantine (currently forward-fill)

---

## 🗺️ Roadmap (High-Impact Next Steps)
- [x] Champion–Challenger + promotion rules (logloss + accuracy gate, configurable thresholds)
- [x] Drift monitoring: PSI + calibration drift + prediction drift (3-channel `DriftMonitor`)
- [x] Fee/slippage-aware paper fills + net-PnL tracking + fee breakdown in trade records
- [x] Rolling walk-forward evaluation (K=5 folds, accuracy/logloss aggregates in retrain history)
- [x] XGBoost early stopping (eval-set logloss, `early_stopping_rounds=30`, logs trees used)
- [ ] Regime gating or regime-specific models (router via HMM/Hurst/vol regime)
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

**v6.2.0 Beta Stable** · Built locally with ⚡ by **G-luc**

</div>
