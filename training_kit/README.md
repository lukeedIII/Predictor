# ⚡ Nexus Training Kit

> **BTC Transformer Model Trainer** — Train, tune, and deploy neural networks for the Nexus Shadow-Quant prediction engine.

---

## What Is This?

A standalone training suite that downloads real Bitcoin data (4M+ candles, 2017–2025) from HuggingFace and trains transformer neural networks to predict price direction — all from a premium web dashboard.

### Built-in Architectures

| Architecture | Parameters | VRAM | Description |
|:---|:---|:---|:---|
| SmallTransformer | 3.2M | 0.2 GB | 4-layer, d256, 8 heads — fast, low VRAM |
| MediumTransformer | 19M | 0.6 GB | 6-layer, d512, 8 heads — balanced |
| MidLargeTransformer | 32M | 1.5 GB | 8-layer, d768, 12 heads — higher capacity |
| NexusTransformer | 152M | 2.5 GB | 12-layer, d1024, 16 heads — maximum capacity |

### V2 Features

- 🏗️ **Custom Architecture Builder** — design your own transformer with sliders (d_model, layers, heads, FFN multiplier, dropout, sequence length) with live parameter estimation
- 🚀 **Push to App** — deploy a trained model directly into the main Nexus Shadow-Quant app with one click (auto-backups the old model)
- 🧹 **Clear VRAM** — force PyTorch to release cached GPU memory between runs
- 🔄 **Reset State** — clear all logs, stats, and training history (checkpoints are preserved)
- 📊 **Live Dashboard** — real-time accuracy curves, VRAM monitor, batch progress, training queue

---

## Requirements

- Python 3.10+ with pip
- NVIDIA GPU with CUDA support
- Internet connection (first run only, to download BTC data)

---

## Quick Start

```bash
# 1. Install dependencies (one-time)
pip install -r requirements.txt

# 2. Start the training server
python train_server.py

# 3. Open in browser
# http://localhost:5555
```

Or use the batch files:
1. Double-click `INSTALL.bat` (one-time setup)
2. Double-click `START_TRAINING.bat`
3. Open [localhost:5555](http://localhost:5555)

---

## How to Use

### Training
1. **Select architectures** from the preset list (default: all 4)
2. **Set epochs** and **learning rate** in the controls bar
3. Click **▶ Start** — data auto-downloads on first run
4. Click **⏸ Pause** to safely stop (saves checkpoint)
5. Click **▶ Continue** to resume from last checkpoint
6. Close the browser — training keeps running in the terminal
7. `Ctrl+C` in terminal — saves checkpoint and exits gracefully

### Custom Architecture Builder
1. Name your model in the text field
2. Adjust sliders — live parameter count + VRAM estimate updates in real-time
3. Click **+ Create & Add to Queue** — the custom arch appears in presets
4. Start training as usual

### Push to App
After training, deploy your best model to the main Nexus Shadow-Quant app:
1. Select a trained model from the dropdown
2. Set the target filename (e.g. `nexus_small_transformer_v1.pth`)
3. Click **🚀 Push as Main Model**
4. The old model is automatically backed up before replacing

> **Destination:** `%LOCALAPPDATA%\nexus-shadow-quant\models\`

### Utility Buttons
- **🧹 Clear VRAM** — releases cached GPU memory (disabled during active training)
- **🔄 Reset** — clears all state, logs, and history (checkpoints are kept)

---

## Files & Folders

```
training_kit/
├── train_server.py       # Main server + training engine
├── models.py             # All model architectures + custom builder
├── templates/
│   └── index.html        # Premium web UI (Flask-served)
├── requirements.txt      # Python dependencies
├── START_TRAINING.bat     # Windows launcher
├── README.md             # This file
│
├── data/                 # Auto-downloaded BTC datasets (~400MB)
├── checkpoints/          # Per-epoch checkpoints (auto-saved)
└── models/               # Best model weights (training output)
```

---

## API Endpoints

| Endpoint | Method | Description |
|:---|:---|:---|
| `/api/state` | GET | Current training state (polled every 500ms) |
| `/api/architectures` | GET | Available model architectures |
| `/api/start` | POST | Start training with selected archs |
| `/api/stop` | POST | Pause training (saves checkpoint) |
| `/api/continue` | POST | Resume from last checkpoint |
| `/api/checkpoints` | GET | List saved checkpoints |
| `/api/trained_models` | GET | List trained `.pth` models |
| `/api/push_to_app` | POST | Copy model to main app's model dir |
| `/api/clear_vram` | POST | Force GPU VRAM release |
| `/api/reset_state` | POST | Reset all training state to defaults |
| `/api/estimate` | POST | Estimate params for custom architecture |
| `/api/custom_arch` | POST | Register a custom architecture |

---

## Tips

- Batch size auto-scales based on available VRAM
- Each epoch auto-saves a checkpoint — you can always resume
- Early stopping kicks in after 8 epochs with no improvement
- The dashboard shows real-time accuracy curves (train vs. val)
- Models are saved in the same format as the main system

---

## Troubleshooting

**"CUDA not available"**
```bash
# Update NVIDIA drivers, then install PyTorch with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**"Port 5555 in use"**
```bash
python train_server.py --port 6666
```

**"Module not found"**
```bash
pip install -r requirements.txt
```
