# ⚡ Nexus Shadow-Quant

> **Institutional-Grade Bitcoin Intelligence & Autonomous Paper Trading Platform**

---

## 🤖 AI Agent — READ THIS FIRST

> **If you are an AI agent, coding assistant, or LLM working on this project:**
>
> 1. **Read `PROJECT_INSTRUCTIONS.json`** in the project root — it contains the COMPLETE project blueprint:
>    architecture, all modules, every API endpoint, class/method documentation, code patterns, known gotchas, and development instructions.
>
> 2. **The canonical source code lives in `desktop/python_backend/`** — NOT the root directory.
>    The root only has config files, tests, and data. All Python logic is inside the desktop app.
>
> 3. **Do NOT use files from `OLD-Stuff/`** — those are archived duplicates from the old Streamlit era.
>
> 4. **After making changes, always update `PROJECT_INSTRUCTIONS.json`** with any new modules, endpoints, or architectural changes.
>
> 5. **Key gotchas** to avoid:
>    - Features MUST be computed identically in `train()` and `predict()` — any divergence causes model accuracy collapse
>    - CSS `backdrop-filter` and `transform` create stacking contexts — never extend beyond `.main-content`
>    - `config.py` auto-detects dev vs installed mode — never hardcode paths
>    - The API server is a monolith (`api_server.py`, 1156 lines) — keep it that way for simplicity

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Electron Shell                     │
│  ┌─────────────────────────────────────────────────┐ │
│  │              React Frontend (Vite)               │ │
│  │  Dashboard │ Paper Trading │ Dr. Nexus │ Settings│ │
│  │            │    Engine     │ AI Chat   │         │ │
│  └──────────────────┬──────────────────────────────┘ │
│                     │ REST API + WebSocket            │
│  ┌──────────────────┴──────────────────────────────┐ │
│  │           Python Backend (FastAPI)               │ │
│  │  NexusPredictor │ PaperTrader │ DataCollector    │ │
│  │  (XGBoost+LSTM) │ (Risk Mgmt) │ (Binance API)   │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## 📂 Project Structure

```
F:\Predictor\
├── PROJECT_INSTRUCTIONS.json   ← AI agent blueprint (READ FIRST)
├── .env.example                ← API key template
├── .gitignore
├── README.md                   ← This file
├── requirements.txt            ← Python dependencies
├── build_scripts/              ← PowerShell build helpers
├── data/                       ← Runtime market data (generated)
├── models/                     ← Trained ML models (generated)
├── logs/                       ← Application logs
├── tests/                      ← Python unit tests (pytest)
├── desktop/                    ← ★ THE APPLICATION
│   ├── electron/               ← Main process, preload, splash
│   ├── src/                    ← React frontend (TypeScript)
│   ├── python_backend/         ← ★ ALL Python source code
│   ├── release/                ← Built .exe installer
│   └── package.json            ← Node dependencies + scripts
└── OLD-Stuff/                  ← Archived files (do not use)
```

## 🚀 Quick Start (Dev Mode)

```bash
# 1. Install Node dependencies
cd desktop && npm install

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example desktop/python_backend/.env
# Edit .env with your Binance/Gemini/OpenAI keys

# 4. Run in development mode
cd desktop && npm run dev
```

## 🔨 Build Installer

```bash
cd desktop && npm run dist
# Output: desktop/release/Nexus Shadow-Quant Setup 5.0.0.exe (~2GB)
```

## 🧠 Core Components

| Component | File | Purpose |
|---|---|---|
| **AI Predictor** | `predictor.py` | XGBoost + LSTM ensemble, 60+ features, 15-min horizon |
| **Paper Trader** | `paper_trader.py` | Autonomous trading, Kelly sizing, trailing SL, 3 concurrent positions |
| **API Server** | `api_server.py` | FastAPI REST API, 30+ endpoints, auto-retrain scheduler |
| **Math Engine** | `math_core.py` | Hurst exponent, FFT cycles, regime detection |
| **AI Agent** | `nexus_agent.py` | Dr. Nexus — context-aware quant analyst chat |
| **Config** | `config.py` | Centralized settings, path resolution, API keys |

## 📊 Tech Stack

- **Frontend**: React 18 + TypeScript + Vite 6
- **Desktop**: Electron 40 (frameless, custom titlebar)
- **Backend**: Python 3.12 + FastAPI + Uvicorn
- **ML**: XGBoost + PyTorch LSTM (CUDA-accelerated)
- **Data**: Binance REST API, Pandas, NumPy, SciPy
- **Charts**: Plotly.js (interactive candlesticks)
- **Build**: electron-builder (NSIS Windows installer)

## ⚠️ Disclaimer

Nexus Shadow-Quant is an educational and research tool. It is NOT financial advice.
All predictions are statistical models and do NOT guarantee profits.
You are fully responsible for any trading decisions.

---

*v5.0 Stable Beta Testing — Built with ⚡ by G-luc*
