---
license: mit
tags:
  - finance
  - trading
  - bitcoin
  - cryptocurrency
  - quantitative-analysis
  - ensemble
  - xgboost
  - pytorch
  - transformer
  - lstm
  - time-series
  - forecasting
language:
  - en
pipeline_tag: tabular-classification
library_name: pytorch
---

<div align="center">

# 🔮 Nexus Shadow-Quant — Trained Models

### Institutional-Grade Crypto Intelligence Engine

[![GitHub](https://img.shields.io/badge/Source-lukeedIII%2FPredictor-181717?style=for-the-badge&logo=github)](https://github.com/lukeedIII/Predictor)
[![Version](https://img.shields.io/badge/Version-v6.4.2-6366f1?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)]()

</div>

---

## 📋 Overview

This repository contains the **pre-trained model artifacts** for [Nexus Shadow-Quant](https://github.com/lukeedIII/Predictor) — a 16-model ensemble engine for BTC/USDT directional forecasting.

**Why this exists:** Training the full model stack from scratch takes ~6 hours on a modern GPU. By hosting the trained weights here, new installations can pull them instantly and skip the initial training phase entirely.

---

## 🏗️ Model Architecture

| Model | Type | Parameters | Trained | Purpose |
|:---|:---|:---|:---|:---|
| `predictor_v3.joblib` | XGBoost Ensemble | ~500 trees | 15 Feb 2026, 02:31 | Primary directional classifier |
| `nexus_lstm_v3.pth` | Bi-LSTM | ~2M | 14 Feb 2026, 11:45 | Sequence pattern recognition |
| `nexus_transformer_v2.pth` | Transformer (152M) | 5 epochs | 15 Feb 2026, 04:44 | Long-range dependency modeling |
| `nexus_medium_transformer_v1.pth` | Transformer (Medium) | 5 epochs | 15 Feb 2026, 05:49 | Balanced capacity/speed |
| `nexus_small_transformer_v1.pth` | Transformer (Small) | 10 epochs | 15 Feb 2026, 05:24 | Fast inference, high accuracy |
| `nexus_transformer_pretrained.pth` | Pretrained base | — | 14 Feb 2026, 07:22 | Foundation weights |
| `feature_scaler_v3.pkl` | StandardScaler | — | 15 Feb 2026, 02:31 | Feature normalization state |

### Supporting Models (16-Model Quant Panel)
- **GARCH(1,1)** — Volatility regime detection
- **MF-DFA** — Multi-fractal detrended fluctuation analysis
- **TDA** — Topological Data Analysis (persistent homology)
- **Bates SVJ** — Stochastic volatility with jumps
- **HMM (3-state)** — Hidden Markov Model for regime classification
- **RQA** — Recurrence Quantification Analysis
- + 10 more statistical models

---

## 📊 Performance (Audited)

| Metric | Value |
|:---|:---|
| **Audit Size** | 105,031 predictions on 3.15M candles |
| **Accuracy** | 50.71% (statistically significant above 50%) |
| **Sharpe Ratio** | 0.88 (annualized, fee-adjusted) |
| **Prediction Horizon** | 15 minutes |
| **Features** | 42 scale-invariant (returns/ratios/z-scores) |
| **Fee Model** | Binance taker 0.04% + slippage 0.01% |

---

## 🕐 Training Log

<details>
<summary><strong>📈 Small Transformer — 10 epochs (15 Feb 2026)</strong></summary>

| Epoch | Accuracy | Timestamp |
|:---|:---|:---|
| 1 | 60.0% | 05:09 |
| 2 | 69.7% | 05:10 |
| 3 | 72.6% | 05:12 |
| 4 | 74.5% | 05:14 |
| 5 | 75.2% | 05:15 |
| 6 | 76.0% | 05:17 |
| 7 | 76.8% | 05:19 |
| 8 | 76.8% | 05:20 |
| 9 | 76.9% | 05:22 |
| **10** | **76.9%** ✅ | **05:24** |

</details>

<details>
<summary><strong>📈 Medium Transformer — 5 epochs (15 Feb 2026)</strong></summary>

| Epoch | Accuracy | Timestamp |
|:---|:---|:---|
| 1 | 58.1% | 05:34 |
| 2 | 69.8% | 05:37 |
| 3 | 72.7% | 05:41 |
| 4 | 74.8% | 05:45 |
| **5** | **76.2%** ✅ | **05:49** |

</details>

<details>
<summary><strong>📈 Nexus Transformer (152M) — 9 epochs (15 Feb 2026)</strong></summary>

| Epoch | Accuracy | Timestamp |
|:---|:---|:---|
| 1 | 51.3% | 06:30 |
| 2 | 52.4% | 06:51 |
| 3 | 52.4% | 07:12 |
| 4 | 53.1% | 07:32 |
| 5 | 54.6% | 07:52 |
| 6 | 55.3% | 08:13 |
| 7 | 57.3% | 08:33 |
| 8 | 58.1% | 08:54 |
| **9** | **58.7%** ✅ | **09:14** |

*Epoch 10 failed — weights from epoch 9 preserved.*

</details>

---

## ⚡ Quick Start

### Automatic (Recommended)
The Nexus Shadow-Quant app will **auto-pull** these models on first startup if no local models are found. Simply:
1. Set your `HUGGINGFACE_TOKEN` and `HF_REPO_ID` in Settings.
2. Restart the backend.
3. Models are downloaded and the predictor is ready instantly.

### Manual
```bash
pip install huggingface_hub
huggingface-cli download Lukeed/Predictor-Models --local-dir ./models
```

---

## 🔄 Sync Protocol

| Action | What happens |
|:---|:---|
| **Push to Hub** | Uploads all files from `models/` folder to this repo |
| **Pull from Hub** | Downloads latest weights, re-initializes the predictor |
| **Auto-Pull** | On startup, if no local models found, pulls automatically |

---

## ⚠️ Disclaimer

These models are trained on historical BTC/USDT data and are provided for **educational and research purposes only**. They are not financial advice. Cryptocurrency markets are volatile. Past performance does not guarantee future results.

---

<div align="center">

**Dr. Nexus** · *Quantitative intelligence, engineered locally.*

</div>
