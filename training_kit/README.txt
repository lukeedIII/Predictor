╔══════════════════════════════════════════════════════╗
║              NEXUS TRAINING KIT                      ║
║         BTC Transformer Model Trainer                ║
╚══════════════════════════════════════════════════════╝

WHAT IS THIS?
═════════════
This is a standalone training kit for the Nexus BTC prediction 
models. It downloads real Bitcoin data (4M+ candles from 2017-2025)
and trains transformer neural networks to predict price direction.

4 Model Architectures (all ready to train):
  • SmallTransformer    —  3.2M params  (0.2 GB VRAM)
  • MediumTransformer   —  19M params   (0.6 GB VRAM)
  • MidLargeTransformer —  32M params   (1.5 GB VRAM)
  • NexusTransformer    —  152M params  (2.5 GB VRAM)

With your RTX 3090 (24GB), you can train ALL of these easily!


REQUIREMENTS
════════════
  • Python 3.10+ (with pip)
  • NVIDIA GPU with CUDA support (RTX 3090 recommended)
  • Internet connection (first time only, to download data)


QUICK START (3 steps)
═════════════════════

  1. Double-click INSTALL.bat  (one-time setup)
     → Installs Python packages (torch, flask, etc.)

  2. Double-click START_TRAINING.bat
     → Starts the training server

  3. Open http://localhost:5555 in your browser
     → You'll see a beautiful dashboard to control training!


HOW TO USE
══════════
  • Select which architectures to train (default: all 4)
  • Set epochs (50 is a good default for 2-3 day runs)
  • Click "Start Training" → it auto-downloads data first time
  • Click "Pause" to safely stop (saves a checkpoint)
  • Click "Continue" to resume from where you left off
  • Close browser = training keeps running in terminal!
  • Ctrl+C in terminal = saves checkpoint and exits


FILES & FOLDERS
═══════════════
  train_server.py     — Main server + training engine
  models.py           — All 4 model architectures
  templates/          — Web UI (auto-served by Flask)
  requirements.txt    — Python dependencies
  
  Created automatically:
  data/               — Downloaded BTC datasets (~400MB)
  checkpoints/        — Per-epoch checkpoints (auto-saved)
  models/             — Best model weights (final output)


OUTPUT
══════
After training, the best model weights will be in:
  models/nexus_small_transformer_pretrained.pth
  models/nexus_medium_transformer_pretrained.pth
  models/nexus_midlarge_transformer_pretrained.pth
  models/nexus_nexus_transformer_pretrained.pth

Send these .pth files back and they'll be integrated 
into the main Nexus Shadow-Quant system!


TIPS
════
  • With 24GB VRAM, batch size auto-scales to 4096 (fast!)
  • Training runs in background even if you close the browser
  • Each epoch saves a checkpoint — you can ALWAYS resume
  • The dashboard shows real-time accuracy curves
  • Early stopping kicks in after 8 epochs with no improvement
  • Models are saved in the same format as the main system
  • If Python isn't found, install from https://python.org


TROUBLESHOOTING
═══════════════
  "CUDA not available"
    → Make sure NVIDIA drivers are up to date
    → Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121

  "Port 5555 in use"
    → Change port: python train_server.py --port 6666

  "Module not found"
    → Run INSTALL.bat again or: pip install -r requirements.txt
