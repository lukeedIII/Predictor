# 📖 PROJECT BIBLE — Nexus Shadow-Quant

> **Generated:** 2026-02-18 · **Audited against:** live source code on disk
>
> This is the **single source of truth** for what Nexus Shadow-Quant actually _is_, what it _contains_, and where the existing docs are **wrong**.

---

## 1 · What Is This Project?

**Nexus Shadow-Quant** is a self-contained **desktop application** that:

1. Ingests **live BTC/USDT 1-minute candle data** from Binance (REST + WebSocket).
2. Engineers **42 scale-invariant features** (returns, ratios, z-scores — never raw prices).
3. Runs **XGBoost** (primary) + an optional **Transformer** sequence model to produce a calibrated probability: _"Will BTC be up ≥ threshold within 15 minutes?"_
4. Overlays **16 institutional-grade quantitative models** (HMM regime, GARCH, Heston, RQA, TDA, PPO RL, etc.) for diagnostic context.
5. Operates a fully local **autonomous paper trader** with professional risk management (Kelly sizing, trailing SL, circuit breaker, regime gating).
6. Provides a **Dr. Nexus AI analyst** chatbot (multi-provider: Gemini → OpenAI → embedded Qwen 0.5B fallback).
7. Wraps everything in an **Electron + React + TypeScript** desktop shell with a drag-and-drop dashboard, TradingView chart, world clock, Swiss weather widget, and more.

**Everything runs locally.** External calls are limited to Binance market data, optional LLM APIs, and optional Hugging Face model sync.

---

## 2 · Tech Stack (Code-Verified)

| Layer | Technology | Version (from code) |
|:------|:-----------|:--------------------|
| Desktop Shell | Electron | 40.3.0 |
| Frontend | React (TypeScript) | 19.2.0 |
| Bundler | Vite | 7.3.1 |
| Charting (candlestick) | lightweight-charts | 5.1.0 |
| Charting (stats) | Recharts | 3.7.0 |
| Grid layout | react-grid-layout | 2.2.2 |
| Markdown rendering | react-markdown + remark-gfm | 10.1.0 / 4.0.1 |
| Routing | react-router-dom | 7.13.0 |
| Backend API | FastAPI + Uvicorn (Python 3.12) | ≥0.115 |
| ML (primary) | XGBoost | ≥3.0.0 |
| ML (deep) | PyTorch (Transformer) | ≥2.5.0 |
| Quant math | SciPy, filterpy, hmmlearn | pinned ranges |
| Data | Pandas, NumPy, PyArrow | pinned ranges |
| Exchange data | ccxt + websocket-client | ≥4.0.0 |
| NLP | transformers, feedparser, beautifulsoup4 | pinned ranges |
| Build/install | electron-builder (NSIS) | 26.7.0 |

---

## 3 · Project Structure (True Layout)

```
F:\Predictor\                          ← PROJECT ROOT
│
├── desktop/                           ★ THE ACTIVE APPLICATION
│   ├── electron/                      Electron main process
│   │   ├── main.js                    Window lifecycle, Python spawn, splash, tray, IPC
│   │   ├── preload.js                 contextBridge (minimize/maximize/close)
│   │   ├── splash.html                Boot splash with /api/boot-status polling
│   │   └── splashPreload.js           Splash window preload
│   │
│   ├── src/                           React frontend (TypeScript)
│   │   ├── App.tsx                    Root: routing, Titlebar, Sidebar, StatusBar (8.7KB)
│   │   ├── index.css                  ALL styles: design system + components (50.7KB)
│   │   ├── main.tsx                   Entry point
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx          Main dashboard: grid layout, cards (15KB)
│   │   │   ├── PaperTrading.tsx       Trading interface: positions, equity, history (17.8KB)
│   │   │   ├── NexusAgent.tsx         Dr. Nexus AI chat (17.7KB)
│   │   │   ├── Settings.tsx           API keys, system info, model selector (35.8KB)
│   │   │   ├── FirstRunSetup.tsx      One-time setup wizard (7KB)
│   │   │   └── GpuFarm.tsx            ⚠️ GPU mining game (NOT in any docs) (16.6KB)
│   │   ├── components/
│   │   │   ├── QuantPanel.tsx         16-model Quant Intelligence panel (20.9KB)
│   │   │   ├── PaperStats.tsx         Paper trading stats widget (19.2KB)
│   │   │   ├── SwissWeather.tsx       Swiss weather widget (18KB)
│   │   │   ├── TradingViewChart.tsx   Candlestick chart (18KB)
│   │   │   ├── HardwareMonitor.tsx    GPU/CPU metrics (17.7KB)
│   │   │   ├── TrainingLog.tsx        Training progress log (15.7KB)
│   │   │   ├── Icons.tsx              SVG icon library (11.7KB)
│   │   │   ├── SystemHealth.tsx       System health panel (10.1KB)
│   │   │   ├── ModelRegistry.tsx      Model registry display (10KB)
│   │   │   ├── WorldClock.tsx         Financial hub clocks (6.6KB)
│   │   │   ├── SignalBadge.tsx        UP/DOWN signal badge (3.5KB)
│   │   │   ├── NewsFeed.tsx           Crypto news feed (2.9KB)
│   │   │   └── MetricCard.tsx         Reusable metric card (2KB)
│   │   ├── hooks/
│   │   │   ├── useApi.ts             REST polling hook (3KB)
│   │   │   └── useKeyboardShortcuts.ts  Ctrl+1-4, Ctrl+B etc. (2.6KB)
│   │   ├── stores/
│   │   │   ├── liveStore.ts           ⚠️ WebSocket state (replaces useWebSocket.tsx) (8.8KB)
│   │   │   └── toastStore.tsx         Toast notification store (3.3KB)
│   │   └── types/
│   │       ├── electron.d.ts          Electron API types
│   │       └── fontsource.d.ts        Font types
│   │
│   ├── python_backend/                ★ ALL PYTHON SOURCE (43 files)
│   │   ├── api_server.py              FastAPI server — 3175 lines / 129KB
│   │   ├── predictor.py               ML engine — 2129 lines / 104KB
│   │   ├── quant_models.py            16-model quant engine — 2473 lines / 94KB
│   │   ├── paper_trader.py            Trading engine — 1248 lines / 56KB
│   │   ├── nexus_agent.py             Dr. Nexus AI — 46KB
│   │   ├── pretrain_multi_arch.py     Multi-arch pretraining — 35KB
│   │   ├── derivatives_feed.py        Binance Futures data — 30KB
│   │   ├── pretrain_transformer.py    Transformer pretraining — 30KB
│   │   ├── alt_data.py                Free alternative data — 25KB
│   │   ├── app.py                     Alternate entry point? — 26KB
│   │   ├── telegram_notifier.py       Telegram notifications — 23KB
│   │   ├── binance_ws.py              WebSocket client — 17KB
│   │   ├── gpu_game.py                ⚠️ GPU mining game backend — 17KB
│   │   ├── backtester.py              Historical backtesting — 17KB
│   │   ├── drift_monitor.py           Drift detection — 16KB
│   │   ├── first_run.py               First-run setup — 15KB
│   │   ├── probability_calibrator.py  Platt scaling + EV — 12KB
│   │   ├── config.py                  Central configuration — 12KB
│   │   ├── nexus_logger.py            Logging system — 11KB
│   │   ├── math_core.py               Indicators & math — 10KB
│   │   ├── nexus_memory.py            AI agent memory (SQLite) — 10KB
│   │   ├── baselines.py               Baseline strategies — 10KB
│   │   ├── fintech_theme.py           Plotly theme — 9KB
│   │   ├── sentiment_engine.py        Sentiment analysis — 9KB
│   │   ├── dr_nexus_system_prompt.txt Dr. Nexus prompt — 8.6KB
│   │   ├── twitter_scraper.py         ⚠️ Twitter/X scraper — 7.8KB
│   │   ├── train_base_model.py        Base model training — 7.6KB
│   │   ├── download_historical.py     Historical data downloader — 6.7KB
│   │   ├── data_collector.py          Binance data fetcher — 6.7KB
│   │   ├── embedded_llm.py            Embedded Qwen 0.5B — 6.6KB
│   │   ├── bootstrap.py               App bootstrap — 5.5KB
│   │   ├── backtest_utils.py          Backtest utilities — 5.3KB
│   │   ├── system_check.py            Hardware check — 5.3KB
│   │   ├── whale_monitor.py           ⚠️ Whale detector — 4.8KB
│   │   ├── run_backtest_parallel.py   Parallel backtest — 4.8KB
│   │   ├── main.py                    Uvicorn launcher — 5.9KB
│   │   ├── hf_sync.py                 Hugging Face sync — 3.1KB
│   │   ├── run_backtest.py            Single-thread backtest — 2.9KB
│   │   ├── notifications.py           Desktop notifications — 2.3KB
│   │   ├── hardware_profiler.py       HW profiling — 1KB
│   │   ├── requirements.txt           Python deps (pinned ranges)
│   │   └── tests/
│   │       └── test_core.py           24 pytest tests (19.9KB)
│   │
│   ├── src_old/                       ⚠️ OLD frontend code (pre-rewrite)
│   ├── python_embedded/               Embedded CPython for installed .exe
│   ├── release/                       Built installer output
│   ├── dist/                          Vite build output
│   └── package.json                   npm config (v6.4.2)
│
├── training_kit/                      Standalone Transformer training suite
│   ├── train_server.py                Flask training server (43.7KB)
│   ├── models.py                      Model architectures (18.2KB)
│   ├── templates/                     Web UI
│   ├── README.md                      Well-documented
│   └── ...
│
├── OLD-Stuff/                         ⚠️ Archived old code (9 subdirs)
│   ├── brainstorming/                 Early research notes
│   ├── root_python_files/             Old Python files from root
│   ├── streamlit/                     Old Streamlit UI (pre-Electron)
│   ├── routers/ + routers_dir/        Old FastAPI router attempts
│   ├── old_builds/ + old_tests/       Old artifacts
│   └── misc/                          Misc old files
│
├── NexusSQ-v6.1.2/                   ⚠️ DEAD COPY — old version snapshot
│   └── (duplicate frontend + backend + node_modules)
│
├── Beta Tester/                      ⚠️ DEAD COPY — beta tester build
│   └── (duplicate frontend + backend, identical to NexusSQ-v6.1.2)
│
├── Picture and Video/                Demo videos (54MB total)
├── assets/                           Logo, demo GIFs, HF model card
├── build_scripts/                    PowerShell build scripts
├── data/                             Runtime data (89MB CSV + 49MB Parquet)
├── models/                           Trained models (.joblib, .pth, .pkl)
├── logs/                             App logs
│
├── README.md                         Main project README (v6.4.2)
├── FEATURE_ROADMAP.md                Feature tracker (STALE — says v6.0)
├── PROJECT_INSTRUCTIONS.json         AI agent instructions (STALE — says v5.0)
├── requirements.txt                  Root-level deps (NO version pins!)
├── .env.example                      API key template
├── .gitignore                        Git exclusions
├── LICENSE                           MIT
├── INSTALL.bat                       First-time setup script
├── START.bat                         Launch script (Python + Vite, no Electron)
│
├── NexusSQ-Full.zip                  ⚠️ 6.8 GB archive (in repo root!)
├── NexusSQ-Latest.zip                350KB archive
├── NexusTrainingKit.zip              20KB archive
└── v6.1.2.zip                        41MB archive
```

---

## 4 · 🚨 Documentation Discrepancies (README vs Reality)

These are concrete, code-verified differences between what `README.md` claims and what the source code actually does.

### 4.1 — Version Chaos (4 Different Versions)

| Source | Claims |
|:-------|:-------|
| `package.json` | `6.4.2` |
| `README.md` footer | `v6.4.2 Beta Stable` |
| `config.py` → `VERSION` | `v6.2.1 Beta Stable` |
| `PROJECT_INSTRUCTIONS.json` | `v5.0 Stable Beta Testing` |
| `FEATURE_ROADMAP.md` | `v6.0` |

> **Verdict:** `package.json` is the authoritative source. `config.py` is behind. `PROJECT_INSTRUCTIONS.json` and `FEATURE_ROADMAP.md` are severely outdated.

### 4.2 — Prediction Threshold

| Source | Value |
|:-------|:------|
| `README.md` | +0.30% hurdle (`1.003`) |
| `config.py` → `PREDICTION_THRESHOLD` | `0.001` (0.1%) |
| `config.py` comment | _"Previous: 0.003 (0.3%) caused 120:1 class imbalance"_ |

> **Verdict:** The threshold was **changed from 0.3% to 0.1%** but the README was never updated. The entire README section "What This Actually Predicts" is now inaccurate.

### 4.3 — Deep Model: Transformer, NOT LSTM

| Source | Claims |
|:-------|:-------|
| `README.md` line 235 | "Transformer (Optional)" — ✅ correct |
| `PROJECT_INSTRUCTIONS.json` | "PyTorch LSTM (3-layer, 512 hidden)" — ❌ WRONG |
| `predictor.py` | `NexusLSTM` is a **deprecated alias** that redirects to `NexusTransformer` |

> **Verdict:** The deep model is a **Transformer** (12-layer, d_model=1024, 16 heads, ~152M params). `PROJECT_INSTRUCTIONS.json` still describes the old LSTM.

### 4.4 — Paper Trading Parameters

| Parameter | README Says | `config.py` Actual | Delta |
|:----------|:-----------|:-------------------|:------|
| `PAPER_MIN_CONFIDENCE` | 30% | **40%** | +10 |
| `PAPER_COOLDOWN_SEC` | 60s | **120s** | ×2 |
| `PAPER_MAX_HOLD_SEC` | 7200s (2h) | **5400s (90min)** | -25% |
| Max same-direction | not mentioned | **3** (pyramiding) | new |
| Dynamic leverage | not mentioned | **3x–20x range** | new |

### 4.5 — api_server.py Size

| Source | Claims |
|:-------|:-------|
| `PROJECT_INSTRUCTIONS.json` | "42KB, 1156 lines, 59 endpoints" |
| Actual file | **129KB, 3175 lines** |

> **Verdict:** The file has **tripled** in size since the JSON was written.

### 4.6 — Frontend Components (Missing from Docs)

| Component / Page | In README? | In PROJECT_INSTRUCTIONS? | Actually Exists? |
|:-----------------|:-----------|:------------------------|:-----------------|
| `GpuFarm.tsx` | ❌ | ❌ | ✅ 16.6KB page |
| `gpu_game.py` | ❌ | ❌ | ✅ 17KB backend |
| `liveStore.ts` | ❌ | ❌ | ✅ 8.8KB (WebSocket state) |
| `toastStore.tsx` | ❌ | ❌ | ✅ 3.3KB |
| `useWebSocket.tsx` | ✅ (in JSON) | ✅ | ❌ **DOES NOT EXIST** |
| `ModelRegistry.tsx` | ❌ | ❌ | ✅ 10KB |
| `TrainingLog.tsx` | ❌ | ❌ | ✅ 15.7KB |
| `HardwareMonitor.tsx` | partial | ❌ | ✅ 17.7KB |
| `PaperStats.tsx` | partial | ❌ | ✅ 19.2KB |

### 4.7 — Backend Files (Missing from All Docs)

These Python files exist but appear in **zero** documentation:

| File | Size | Purpose |
|:-----|:-----|:--------|
| `derivatives_feed.py` | 30KB | Binance Futures funding/OI/basis data |
| `gpu_game.py` | 17KB | GPU mining mini-game |
| `telegram_notifier.py` | 23KB | Telegram trade notifications |
| `twitter_scraper.py` | 7.8KB | Twitter/X data scraper |
| `whale_monitor.py` | 4.8KB | Large transaction detector |
| `embedded_llm.py` | 6.6KB | Local Qwen 0.5B LLM |
| `probability_calibrator.py` | 12KB | Platt scaling + expected value |
| `pretrain_multi_arch.py` | 35KB | Multi-architecture pretraining |
| `pretrain_transformer.py` | 30KB | Transformer pretraining |
| `bootstrap.py` | 5.5KB | Application bootstrap |
| `nexus_logger.py` | 11KB | Structured logging |
| `fintech_theme.py` | 9KB | Plotly dark theme |
| `baselines.py` | 10KB | Baseline strategy comparisons |
| `notifications.py` | 2.3KB | Desktop notifications (plyer) |
| `download_historical.py` | 6.7KB | Bulk historical data download |
| `hardware_profiler.py` | 1KB | Hardware profiling |
| `app.py` | 26KB | Alternative entry point (unclear purpose) |

### 4.8 — Requirements Discrepancy

| File | Pins Versions? | Location |
|:-----|:--------------|:---------|
| `desktop/python_backend/requirements.txt` | ✅ Yes (range-pinned) | Active, correct |
| Root `requirements.txt` | ❌ **No pins at all** | Outdated/duplicate, missing packages |

> The root `requirements.txt` is missing: `websocket-client` (listed but not in backend's), and the backend's file is missing `websocket-client` too (it's only in root). Neither file includes `telegram` dependencies.

---

## 5 · 🗑️ Dead Weight & Cleanup Opportunities

### 5.1 — Duplicate/Archive Folders (All .gitignored)

| Path | Size | What It Is |
|:-----|:-----|:-----------|
| `NexusSQ-v6.1.2/` | ~41MB + node_modules | Old version snapshot (including full `node_modules`) |
| `Beta Tester/` | Similar | Identical to NexusSQ-v6.1.2 (same `App.tsx`, same `index.css`) |
| `OLD-Stuff/` | varies | 9 subdirs of archived code (Streamlit UI, brainstorming, old tests) |
| `desktop/src_old/` | varies | Pre-rewrite frontend (has `App.css` that current version doesn't) |

> All these are `.gitignored` so they won't be pushed, but they eat local disk.

### 5.2 — Giant ZIP Files in Root

| File | Size |
|:-----|:-----|
| `NexusSQ-Full.zip` | **6.8 GB** |
| `v6.1.2.zip` | 41 MB |
| `NexusSQ-Latest.zip` | 350 KB |
| `NexusTrainingKit.zip` | 20 KB |

> All `.gitignored`, but `NexusSQ-Full.zip` alone is **6.8 GB** sitting in the project root.

### 5.3 — `desktop/README.md` = Vite Boilerplate

The file `desktop/README.md` is the **default Vite template README** (React + TypeScript + Vite ESLint guide). It has nothing to do with the project. It should be deleted or replaced.

### 5.4 — PDF in Root

`Come rendere Predictor "snappy" e reattiva in stile app da trading (es. Binance).pdf` (93KB) — a personal brainstorming PDF in Italian sitting in the project root. `.gitignored` via `*.pdf`.

---

## 6 · What's Actually Working (Code-Verified ✅)

These features are **confirmed present in source code** with real implementations:

| Feature | Key File(s) | Status |
|:--------|:-----------|:-------|
| XGBoost training with 500K 1m candles | `predictor.py` L483-495 | ✅ Real |
| 42 scale-invariant features | `predictor.py` L532-757 | ✅ Real |
| Same feature path train/predict | `_engineer_features()` used in both | ✅ Real |
| Transformer (optional, earn-to-play) | `NexusTransformer` class, L102-201 | ✅ Real |
| Multi-arch selection (4 sizes) | `config.py` MODEL_ARCHITECTURES | ✅ Real |
| Platt calibration | `probability_calibrator.py` | ✅ Real |
| 6-hour auto-retrain | `api_server.py` `_auto_retrain_loop` | ✅ Real |
| Champion-Challenger gate | `api_server.py` `_do_retrain` | ✅ Real |
| Drift monitoring (PSI + Brier + ECE) | `drift_monitor.py` (16KB) | ✅ Real |
| 16-model QuantEngine | `quant_models.py` (2473 lines, 16 classes) | ✅ Real |
| Paper trader with Kelly, trailing SL | `paper_trader.py` (1248 lines) | ✅ Real |
| Regime-based trade gating | `paper_trader.py` `evaluate_signal()` | ✅ Real |
| Fee-adjusted net PnL | `Position.unrealized_pnl(net=True)` | ✅ Real |
| WebSocket push (5 ticks/sec price) | `api_server.py` `_ws_push_loop` | ✅ Real |
| Dr. Nexus AI (multi-provider) | `nexus_agent.py` + `embedded_llm.py` | ✅ Real |
| Drag-and-drop grid | react-grid-layout in Dashboard.tsx | ✅ Real |
| Light/Dark theme | `index.css` `.dashboard-light` scope | ✅ Real |
| World Clock (6 hubs) | `WorldClock.tsx` | ✅ Real |
| Swiss Weather | `SwissWeather.tsx` | ✅ Real |
| TradingView chart | `TradingViewChart.tsx` (lightweight-charts) | ✅ Real |
| Quant Intelligence panel | `QuantPanel.tsx` | ✅ Real |
| Hardware monitor | `HardwareMonitor.tsx` | ✅ Real |
| Paper trading stats | `PaperStats.tsx` | ✅ Real |
| Training log viewer | `TrainingLog.tsx` | ✅ Real |
| Model registry | `ModelRegistry.tsx` | ✅ Real |
| GPU mining game | `GpuFarm.tsx` + `gpu_game.py` | ✅ Real (undocumented) |
| Telegram notifications | `telegram_notifier.py` (23KB) | ✅ Real (undocumented) |
| Derivatives data (funding/OI) | `derivatives_feed.py` (30KB) | ✅ Real (undocumented) |
| Cross-asset features (ETH, PAXG) | `config.py` + `predictor.py` | ✅ Real |
| Gap detection / quarantine | `config.py` GAP_* constants | ✅ Real |
| Walk-forward evaluation (K=5) | `config.py` WALK_FORWARD_FOLDS | ✅ Real |
| XGBoost early stopping | `config.py` XGB_EARLY_STOPPING_ROUNDS=30 | ✅ Real |
| HF model sync | `hf_sync.py` | ✅ Real |
| Standalone training kit | `training_kit/` (Flask + web UI) | ✅ Real |
| INSTALL.bat / START.bat | Root batch files | ✅ Functional |
| NSIS installer build | `build_scripts/` + electron-builder | ✅ Real |
| Test suite (24 tests) | `tests/test_core.py` | ✅ Real |

---

## 7 · Configuration Reference (from `config.py`)

### Prediction

| Constant | Value | Notes |
|:---------|:------|:------|
| `PREDICTION_HORIZON_MINUTES` | 15 | Minutes into the future |
| `PREDICTION_THRESHOLD` | 0.001 | ⚠️ 0.1% — NOT 0.3% as README says |
| `PREDICTION_MIN_CLASS_RATIO` | 0.15 | Skip training if minority < 15% |

### Paper Trading

| Constant | Value |
|:---------|:------|
| `PAPER_STARTING_BALANCE` | $10,000 |
| `PAPER_DEFAULT_LEVERAGE` | 10x |
| `PAPER_LEVERAGE_MIN / MAX` | 3x – 20x (dynamic) |
| `PAPER_MIN_CONFIDENCE` | 40% |
| `PAPER_COOLDOWN_SEC` | 120s |
| `PAPER_MAX_HOLD_SEC` | 5400s (90 min) |
| `PAPER_MAX_DRAWDOWN` | 20% |
| `PAPER_MAX_SAME_DIRECTION` | 3 (pyramid limit) |
| `PAPER_FEE_TAKER_PCT` | 0.04% |
| `PAPER_SLIPPAGE_PCT` | 0.01% |

### Champion-Challenger

| Constant | Value |
|:---------|:------|
| `CHALLENGER_MIN_LOGLOSS_IMPROVEMENT` | 0.0 (must be ≤ champion) |
| `CHALLENGER_MIN_ACCURACY_PCT` | 49% min |
| `CHALLENGER_GRACE_RETRAINS` | 2 (cold-start grace) |

### Drift Monitoring

| Constant | Value |
|:---------|:------|
| `DRIFT_PSI_WARNING / CRITICAL` | 0.10 / 0.25 |
| `DRIFT_BRIER_WARNING / CRITICAL` | 0.30 / 0.35 |
| `DRIFT_CHECK_INTERVAL_MIN` | 30 min |

### System

| Constant | Value |
|:---------|:------|
| `API_PORT` (main.py) | 8420 |
| `XGBOOST_N_JOBS` | min(8, cpu_count - 1) |
| `RETRAIN_INTERVAL_HOURS` | 6 |
| `DERIVATIVES_ENABLED` | True |

---

## 8 · API Endpoints (from `api_server.py`)

The API runs on `localhost:8420`. Major endpoint groups:

- **System:** `/api/boot-status`, `/api/status`, `/api/system-check`, `/api/shutdown`
- **Prediction:** `/api/prediction`, `/api/market-data`, `/api/cycles`
- **Trading:** `/api/positions`, `/api/trade-history`, `/api/equity-history`, `/api/stats`, `/api/trade`, `/api/close`, `/api/close-all`, `/api/bot/start`, `/api/bot/stop`
- **Training:** `/api/train`, `/api/retrain-status`
- **Agent (Dr. Nexus):** `/api/agent/chat`, `/api/agent/state`, `/api/agent/history`, `/api/agent/knowledge`, `/api/agent/memory-stats`, `/api/agent/new-session`
- **Settings:** `/api/settings`, `/api/settings/validate`
- **First Run:** `/api/first-run/status`, `/api/first-run/trigger`
- **WebSocket:** `/ws/live` (push: price, predictions, positions, quant data)
- **GPU Game:** endpoints for the mining mini-game
- **Derivatives:** endpoints for funding/OI/basis data
- **News:** `/api/news`
- **Export:** `/api/export/trades`
- **Hardware:** `/api/hardware`

> Total endpoint count has grown well beyond the 59 documented in `PROJECT_INSTRUCTIONS.json`.

---

## 9 · How to Run (Verified)

### Development Mode
```powershell
# Terminal 1: Install (one-time)
cd F:\Predictor\desktop\python_backend
pip install -r requirements.txt
cd F:\Predictor\desktop
npm install

# Terminal 2: Launch
cd F:\Predictor\desktop
npm run dev
# → Starts Vite dev server (port 5173) + Electron + Python backend (port 8420)
```

### Quick Launch (Without Electron)
```powershell
# Double-click START.bat — launches Python backend + Vite in separate windows
# Frontend: http://localhost:5173
# Backend:  http://localhost:8420
```

### Tests
```powershell
cd F:\Predictor\desktop\python_backend
python -m pytest tests/ -v    # 24 tests
```

### Training Kit (Standalone)
```powershell
cd F:\Predictor\training_kit
pip install -r requirements.txt
python train_server.py         # → http://localhost:5555
```

---

## 10 · Overall Assessment

### The Good ✅
- **The core engine is real and substantial.** This is not a toy — it's 2129 lines of ML pipeline (`predictor.py`) + 2473 lines of quant math (`quant_models.py`) + 1248 lines of trading logic (`paper_trader.py`).
- **Professional risk management** is actually implemented (Kelly, trailing SL, circuit breaker, regime gating, fee accounting).
- **Feature engineering is correct** — scale-invariant, same train/predict path, no lookahead.
- **Tests exist** — 24 pytest tests covering core logic.
- **The UI is rich** — 13 components, 6 pages, drag-and-drop grid, real-time WebSocket, multiple data visualizations.
- **.gitignore is well-configured** — dead folders and binaries won't be pushed to GitHub.

### The Messy 🟡
- **4 different version strings** across 4 files — needs a single source of truth.
- **README has at least 6 factual inaccuracies** (threshold, confidence, cooldown, hold time, file sizes, LSTM vs Transformer references in the JSON).
- **`PROJECT_INSTRUCTIONS.json` is ~2 major versions behind** — describes architecture from v5.0 while the app is at v6.4.2.
- **`FEATURE_ROADMAP.md` is frozen at v6.0** — doesn't track anything added after that.
- **17+ Python files are undocumented** in any project doc.
- **2 requirements.txt files** (root vs backend), one unpinned.
- The `desktop/README.md` is default Vite boilerplate.

### The Bloat 🔴
- **6.8 GB zip** in project root.
- **Two dead folder copies** (`NexusSQ-v6.1.2/` and `Beta Tester/`) with duplicated `node_modules`.
- **`src_old/`** pre-rewrite code still sitting in the desktop folder.
- **Demo videos** (54MB) in `Picture and Video/`.

---

> **Bottom line:** The software itself is impressive and well-engineered. The documentation is stale and inconsistent. The file system is cluttered with old versions and large archives. A cleanup pass + doc refresh would bring everything into alignment.
