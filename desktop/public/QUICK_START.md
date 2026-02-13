# ⚡ Nexus Shadow-Quant — Quick Start Guide

## 🚀 First Launch
1. **Install** the app and launch it
2. The **First-Run Wizard** starts automatically:
   - System check (GPU, RAM, disk)
   - Market data sync (~30 seconds)
   - AI model training (~2-5 minutes)
3. Once done, the **Dashboard** loads automatically

---

## 📊 Dashboard (Main Screen)
- **AI Prediction** — Shows UP/DOWN direction with confidence %
- **Live Chart** — BTC/USDT with candlesticks, indicators (SMA, EMA, Bollinger, VWAP)
- **Quant Panel** — Market regime, FFT cycles, Hurst exponent, Order Flow, Jump Risk

> The chart updates every 10 seconds with live Binance data.

---

## 💹 Paper Trading
This is a **simulated** trading bot — no real money, no risk.

### Start the Bot
1. Go to **Paper Trading** (second icon in sidebar)
2. Click **▶️ Start Bot** — the AI opens positions automatically
3. Watch trades appear in real-time

### Manual Trading
- **📈 Manual Long** — Open a buy position
- **📉 Manual Short** — Open a sell position
- **Close** — Close individual positions
- **✖️ Close All** — Close everything at once

### What You'll See
- **Position Cards** — Entry price, PnL, TP/SL levels
- **Performance Stats** — Win rate, Sharpe ratio, drawdown, Kelly fraction
- **Equity Curve** — Your portfolio value over time
- **Trade History** — All closed trades with PnL

> Starting balance: **$10,000** (simulated)
> The bot uses 10x leverage with AI-driven entry signals.

---

## ⚙️ Settings
- **API Keys** are **optional** — the app works without them
- **Google Gemini / OpenAI** — Enables AI market commentary (nice to have)
- **Binance API Key** — Not needed for paper trading (public data is free)
- All keys are stored **locally** on your machine only

---

## ❓ FAQ

**Q: Do I need a GPU?**
A: Recommended (NVIDIA with CUDA). The app works on CPU but training is slower.

**Q: Is this real trading?**
A: No! Paper trading only. No real money is involved.

**Q: How does the AI predict?**
A: XGBoost + LSTM neural network trained on 1.5M+ candles of BTC data, plus alternative data (Fear & Greed, Google Trends).

**Q: Where is my data stored?**
A: `C:\Users\<you>\AppData\Local\nexus-shadow-quant\` — models, market data, and settings.

---

⚠️ **Disclaimer**: This is an educational research tool. Not financial advice. All predictions are statistical models and do NOT guarantee profits.
