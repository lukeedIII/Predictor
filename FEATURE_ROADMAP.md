# 🚀 Nexus Shadow-Quant — Feature Roadmap & Implementation Tracker

> **Last Updated:** 2026-02-13  
> **Current Version:** v6.0 — Microstructure + Base Model  
> **Next Target:** v7.0 — Advanced ML / Reinforcement Learning

---

## Status Legend
- ⬜ Not started
- 🟡 In progress
- ✅ Complete
- 🔴 Blocked

---

## Phase 1: Real-Time Data Infrastructure (Binance WebSocket)
> **Priority: ✅ COMPLETE** — Foundation for accurate pricing and better models  

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 1.1 | Binance WebSocket client (Python) | ✅ | `binance_ws.py` — combined `btcusdt@trade` + `btcusdt@ticker` streams |
| 1.2 | Live trade + 24h ticker stream | ✅ | Real-time price, 24h change, high/low, volume |
| 1.3 | Auto-reconnect with backoff | ✅ | Exponential backoff + jitter, max 30s |
| 1.4 | Internal WS push to frontend | ✅ | `/ws/live` endpoint, 1s broadcast with price + predictions + bot status |
| 1.5 | REST fallback | ✅ | `/api/live-price` endpoint + `_get_live_price()` prefers WS data |

---

## Phase 2: UI Improvements
> **Priority: ✅ COMPLETE** — Real-time feel achieved

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 2.1 | UTC Clock in status bar | ✅ | Ticking HH:MM:SS UTC in StatusBar |
| 2.2 | Live price ticker | ✅ | Real-time BTC price in StatusBar |
| 2.3 | Price flash animation | ✅ | Green flash up, red flash down |
| 2.4 | 24h change display | ✅ | Percentage from WebSocket ticker |
| 2.5 | Live candle growth | ✅ | Current candle updates via `series.update()` |
| 2.6 | TradingView chart | ✅ | Candlestick + Volume + MA(7/25/99) |
| 2.7 | Connection health indicator | ✅ | Triple: API, WS, Binance Feed |
| 2.8 | Bid/Ask spread display | ⬜ | Requires order book WS (future) |

---

## Phase 3: Microstructure Features & Model Upgrade
> **Priority: ✅ COMPLETE** — 5 new features, 35→40 total

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 3.1 | Trade intensity feature | ✅ | `trade_intensity` — proxy from candle range × volume |
| 3.2 | Buy/sell ratio feature | ✅ | `buy_sell_ratio` — close position within candle range |
| 3.3 | VWAP momentum | ✅ | `vwap_momentum` — deviation from volume-weighted avg price |
| 3.4 | Tick volatility | ✅ | `tick_volatility` — high-low range scaled by ATR |
| 3.5 | Large trade ratio | ✅ | `large_trade_ratio` — volume spikes vs median |
| 3.6 | Feature importance dashboard | ✅ | `/api/feature-importance` + `FeatureImportance.tsx` |
| 3.7 | Live trade tracking (WS) | ✅ | 60s rolling deques in `binance_ws.py` |

---

## Phase 4: Quality of Life
> **Priority: ✅ COMPLETE**

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 4.1 | System health panel | ✅ | `/api/system-health` + `SystemHealth.tsx` (GPU, VRAM, model age) |
| 4.2 | Prediction accuracy tracker | ✅ | Visible on Dashboard via WebSocket push |
| 4.3 | Trade notification sounds | ✅ | `useSound.tsx` — Web Audio API (no external files) |
| 4.4 | CSV export | ✅ | `/api/export/trades` — downloadable CSV |
| 4.5 | Dark/light theme toggle | ⬜ | Currently dark-only |
| 4.6 | Keyboard shortcuts help modal | ✅ | Ctrl+1-4 and other shortcuts |

---

## Phase 5: Base Model Training
> **Priority: ✅ COMPLETE** — Ship with pre-trained model for instant-on

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 5.1 | Base model directory (`BASE_MODEL_DIR`) | ✅ | In `config.py` → `models/base/` |
| 5.2 | Training script | ✅ | `train_base_model.py` — 6mo data, XGB+LSTM ensemble |
| 5.3 | Fallback loading in predictor | ✅ | `initialize_models()` loads base → user model priority |
| 5.4 | Audit report | ✅ | JSON audit saved at `base_model_audit.json` |

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-13 | v6.0 | Phase 3-5: Microstructure features, system health, base model training |
| 2026-02-13 | v5.1 | Real-time Binance WebSocket, TradingView chart, StatusBar upgrade |
| 2026-02-13 | v5.0 | Project cleanup, AI instructions created |
| 2026-02-13 | — | Feature roadmap created (this document) |
