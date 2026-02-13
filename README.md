<div align="center">

# ⚡ Nexus Shadow-Quant

### Institutional-Grade Bitcoin Intelligence & Autonomous Trading Platform

[![Version](https://img.shields.io/badge/version-6.0.1-blue?style=flat-square)](https://github.com/lukeedIII/Predictor)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Electron](https://img.shields.io/badge/Electron-40-47848F?style=flat-square&logo=electron&logoColor=white)](https://electronjs.org)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-Private-red?style=flat-square)]()

---

*A desktop application that combines machine learning, quantitative finance, and real-time market data to analyze Bitcoin price movements — built for research, education, and paper trading.*

</div>

---

## 🧬 What Is This?

Nexus Shadow-Quant is a **self-contained desktop app** that watches Bitcoin in real-time, runs ML predictions every 60 seconds, and lets you paper trade based on those signals — all locally on your machine.

It's not a cloud service. It's not a SaaS product. It's a genuine research tool that sits on your desktop, trains its own models on your GPU, and gives you institutional-grade analytics that would normally require a Bloomberg terminal and a quant team.

**The core idea:** Take 6+ years of BTC minute-level data (3.15M candles), engineer 35 scale-invariant features, train an XGBoost + LSTM ensemble, and use that to predict 15-minute price direction — then wrap the whole thing in a beautiful Electron app with real-time charts, risk management, and an AI assistant.

---

## ✨ Key Features

### 🧠 ML Prediction Engine
- **XGBoost + LSTM Ensemble** — Two complementary models voting together. XGBoost handles tabular features brilliantly; the LSTM captures sequential patterns that gradient boosting misses.
- **35 Engineered Features** — Every single feature is scale-invariant (returns, ratios, z-scores — never raw prices). This means the model trained on $20K BTC works just as well at $100K.
- **Automatic Retraining** — The engine retrains periodically on fresh data. You don't have to touch it.
- **GPU Accelerated** — If you have an NVIDIA GPU (RTX 3060+), training and inference use CUDA automatically. Falls back gracefully to CPU.

### 📊 Real-Time Dashboard
- **Live Candlestick Chart** — Professional trading chart (lightweight-charts) with real-time price updates via Binance WebSocket.
- **Quant HUD** — At a glance: market regime (trending/mean-reverting/volatile/chaotic), FFT cycle analysis, Hurst exponent, order flow pressure, and jump risk.
- **Prediction Cards** — Current direction call, confidence level, target prices, and accuracy tracking.
- **Feature Importance** — See which of the 35 features are driving the current prediction.

### 💰 Paper Trading Engine
- **Fully Autonomous** — Set it and forget it. The bot opens and closes positions based on prediction signals.
- **Institutional Risk Management:**
  - Kelly Criterion position sizing (half-Kelly, capped at 25%)
  - Trailing stop-loss (activates at +0.3%, locks 50% of gains)
  - Circuit breaker at 20% drawdown
  - Signal confirmation (2 consecutive same-direction predictions required)
  - 1.5:1 reward-to-risk ratio on every position
  - Max 3 concurrent positions with margin allocation
  - 1-hour maximum hold time
- **Full Trade History** — Every trade logged with entry/exit prices, PnL, hold time, regime at entry, and close reason.

### 🤖 Dr. Nexus — AI Quant Analyst
- **GPT-4o Powered Chat** — Ask questions about your portfolio, the current market regime, why a prediction was made, or general quant finance topics.
- **Live State Injection** — Every message includes a real-time JSON snapshot of the entire platform state (price, predictions, positions, quant metrics). The AI sees what you see.
- **Persistent Memory** — Dr. Nexus remembers key insights across conversations and learns from trading outcomes.

### 📈 Quant Analysis Suite
| Module | What It Does |
|---|---|
| **Regime Detection** | Hidden Markov Model classifying market into 4 states — adapts trading behavior to current conditions |
| **FFT Cycles** | Fourier decomposition identifying dominant short/mid/long market cycles |
| **Hurst Exponent** | Measures market persistence — trending (H>0.5), random (H≈0.5), or mean-reverting (H<0.5) |
| **Order Flow** | Synthetic microstructure analysis modeling buy/sell pressure imbalances |
| **Hawkes Process** | Self-exciting point process for volatility clustering and jump risk estimation |
| **Wasserstein Drift** | Optimal transport metric measuring distribution shift in recent vs. historical returns |

### 🌍 Cross-Asset Intelligence
- **ETH/USDT** — Ethereum often leads Bitcoin by 1-5 minutes. The model watches for this.
- **ETH/BTC** — Ratio movements signal BTC dominance shifts.
- **PAXG/USDT** — Gold proxy for macro fear/flight-to-safety detection.
- **Fear & Greed Index** — Crowd sentiment as a contrarian indicator.
- **Google Trends** — "Bitcoin" search volume as a retail interest gauge.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Electron Shell                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              React Frontend (TypeScript + Vite)         │  │
│  │                                                         │  │
│  │   Dashboard  │  Paper Trading  │  Dr. Nexus  │ Settings │  │
│  │   (Charts)   │  (Autonomous)   │  (AI Chat)  │ (Config) │  │
│  └──────────────────────┬─────────────────────────────────┘  │
│                         │                                     │
│              REST API + WebSocket (port 8420)                 │
│                         │                                     │
│  ┌──────────────────────┴─────────────────────────────────┐  │
│  │            Python Backend (FastAPI + Uvicorn)           │  │
│  │                                                         │  │
│  │   NexusPredictor    PaperTrader      DataCollector      │  │
│  │   (XGBoost+LSTM)    (Risk Engine)    (Binance WS)       │  │
│  │                                                         │  │
│  │   QuantModels       NexusAgent       SentimentEngine    │  │
│  │   (HMM, FFT, etc)  (GPT-4o Chat)   (Alt Data)          │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

The frontend talks to the backend over localhost. The backend handles all data collection, ML inference, and trading logic. Everything runs locally — no cloud dependencies except Binance for price data and OpenAI for the AI chat (optional).

---

## 📂 Project Structure

```
Predictor/
├── .env.example                 # API key template — copy to .env
├── .gitignore
├── README.md                    # You are here
├── FEATURE_ROADMAP.md           # Planned features and priorities
├── PROJECT_INSTRUCTIONS.json    # Full project blueprint (for AI agents)
├── requirements.txt             # Python dependencies
│
├── build_scripts/               # PowerShell scripts for building installers
│
├── desktop/                     # ★ THE APPLICATION
│   ├── electron/                # Electron main process, preload, splash screen
│   │   ├── main.js              # Window management, Python process spawning
│   │   ├── preload.js           # IPC bridge
│   │   └── splash.html          # Animated loading screen
│   │
│   ├── src/                     # React frontend (TypeScript)
│   │   ├── pages/               # Dashboard, PaperTrading, NexusAgent, Settings
│   │   ├── components/          # TradingViewChart, QuantHUD, PredictionCard, etc.
│   │   ├── hooks/               # useWebSocket, useApi, useSound
│   │   └── index.css            # Global design system
│   │
│   ├── python_backend/          # ★ ALL PYTHON SOURCE CODE
│   │   ├── api_server.py        # FastAPI REST API (30+ endpoints)
│   │   ├── predictor.py         # XGBoost + LSTM ensemble predictor
│   │   ├── paper_trader.py      # Autonomous paper trading engine
│   │   ├── quant_models.py      # HMM regime detection, FFT, Hawkes, etc.
│   │   ├── math_core.py         # Hurst exponent, statistical utilities
│   │   ├── nexus_agent.py       # Dr. Nexus AI chat (GPT-4o integration)
│   │   ├── data_collector.py    # Binance API data ingestion
│   │   ├── binance_ws.py        # WebSocket price feed
│   │   ├── alt_data.py          # Fear & Greed, Google Trends
│   │   ├── config.py            # Centralized configuration
│   │   └── ...                  # Additional modules
│   │
│   ├── package.json             # Node dependencies + build scripts
│   └── vite.config.ts           # Vite bundler configuration
│
└── OLD-Stuff/                   # Archived files (do not use)
```

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** 18+ and **npm**
- **Python** 3.12 (3.11+ should work)
- **NVIDIA GPU** with CUDA (optional but recommended for training speed)
- A **Binance account** is NOT required — the app uses public market data endpoints

### Development Mode

```powershell
# 1. Clone the repo
git clone https://github.com/lukeedIII/Predictor.git
cd Predictor

# 2. Install Node dependencies
cd desktop
npm install

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Set up your environment
copy ..\.env.example python_backend\.env
# Edit .env with your API keys (OpenAI key is optional — only needed for Dr. Nexus chat)

# 5. Run in dev mode
npm run dev
```

This will:
1. Start the Vite dev server (frontend hot-reload)
2. Launch Electron
3. Spawn the Python backend on port 8420
4. Open the app with DevTools available

### Building the Installer

```powershell
cd desktop

# Build the frontend
npm run build

# Package with Electron Builder
npx electron-builder --win dir

# The output will be in desktop/release/win-unpacked/
```

---

## 🧪 How the Prediction Works

1. **Data Collection** — Every 60 seconds, the app fetches the latest BTC/USDT candle from Binance, plus ETH, PAXG, and ETH/BTC data.

2. **Feature Engineering** — 35 features are computed from raw OHLCV data. Every feature is a return, ratio, or z-score — never a raw price. This makes the model price-level agnostic.

3. **Ensemble Prediction** — XGBoost produces a probability; the LSTM (if trained and validated above 52% accuracy) produces another. These are weighted-averaged into a final UP/DOWN call with a confidence score.

4. **Quant Overlay** — The regime detector, Hurst exponent, and cycle analysis provide context. A prediction in a trending regime with high Hurst is more reliable than one in a chaotic regime.

5. **Signal Validation** — The paper trader requires 2 consecutive same-direction predictions before opening a position. This filters out noise.

### Statistical Validation

The prediction engine has been audited on **3.15M candles** spanning 6+ years:

| Metric | Value |
|---|---|
| **Accuracy** | 50.71% |
| **Sharpe Ratio** | 0.88 (positive — the only model variant to achieve this) |
| **Feature Count** | 35 (all scale-invariant) |
| **Prediction Horizon** | 15 minutes |
| **Training Data** | 2018–2024 BTC/USDT 1-minute candles |

> **50.71% might not sound impressive**, but in financial markets, a consistent edge above 50% with a positive Sharpe ratio is what institutional traders spend millions trying to achieve. Combined with proper risk management (Kelly sizing, trailing stops), even a small edge compounds over thousands of trades.

---

## 🔧 Configuration

All settings are managed through `config.py` and the `.env` file:

| Key | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Optional | Powers Dr. Nexus AI chat. Without it, the chat feature is disabled but everything else works. |
| `BINANCE_API_KEY` | No | Public endpoints are used by default. Only needed for authenticated endpoints. |

The app auto-detects whether it's running in development mode or as an installed application, and adjusts all paths accordingly.

---

## 🛡️ Security Notes

- **No API keys are stored in the codebase.** All secrets are loaded from environment variables or `.env` files.
- **All trading is paper-only.** The app never places real orders on any exchange.
- **All data stays local.** Your predictions, trades, and chat history are stored on your machine. Nothing is sent to external servers except Binance price queries and OpenAI chat requests (when you use Dr. Nexus).

---

## 📋 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Desktop Shell** | Electron 40 | Cross-platform, frameless window, system tray |
| **Frontend** | React 18 + TypeScript | Type-safe, component-based UI |
| **Bundler** | Vite 7 | Fast HMR, optimized production builds |
| **Charts** | lightweight-charts | Performant financial charting (TradingView engine) |
| **Backend** | FastAPI + Uvicorn | Async Python, automatic OpenAPI docs |
| **ML** | XGBoost + PyTorch | Gradient boosting + deep learning ensemble |
| **GPU** | CUDA (PyTorch) | GPU-accelerated training and inference |
| **Data** | Pandas, NumPy, SciPy | Feature engineering and statistical analysis |
| **Market Data** | Binance REST + WebSocket | Real-time and historical BTC/USDT data |
| **AI Chat** | OpenAI GPT-4o | Context-aware quant analyst assistant |

---

## 📌 Roadmap

See [FEATURE_ROADMAP.md](FEATURE_ROADMAP.md) for the full prioritized list. Highlights:

- [ ] Live Binance order book depth visualization
- [ ] Multi-timeframe prediction (5m, 15m, 1h, 4h)
- [ ] Backtest visualization with equity curves
- [ ] Portfolio analytics and performance attribution
- [ ] Support for additional trading pairs (ETH, SOL, etc.)

---

## ⚠️ Disclaimer

Nexus Shadow-Quant is an **educational and research tool**. It is not financial advice. All predictions are generated by statistical models and machine learning algorithms — they do not guarantee future performance or profits. Cryptocurrency markets are highly volatile and speculative.

**You are fully responsible for any trading decisions you make.** This software performs paper trading only and does not interact with real exchange accounts.

---

<div align="center">

**v6.0.1 Beta Stable** · Built with ⚡ by **G-luc**

</div>
