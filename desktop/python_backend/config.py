"""
Nexus Shadow-Quant — Configuration
====================================
Central configuration for all modules.
Supports both development mode (local dirs) and installed app mode (AppData).
"""

import os
import sys
from dotenv import load_dotenv


# ── Detect Runtime Mode ───────────────────────────────
def _is_installed():
    """Check if running as installed app (embedded Python) vs dev mode."""
    return getattr(sys, 'frozen', False) or 'python_embedded' in sys.executable


def _get_data_root():
    """
    Installed app → C:/Users/<user>/AppData/Local/nexus-shadow-quant/
    Dev mode      → F:/Predictor/ (project directory)
    """
    if _is_installed():
        try:
            import platformdirs
            return platformdirs.user_data_dir("nexus-shadow-quant", "Nexus")
        except ImportError:
            # Fallback: use AppData directly
            appdata = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            return os.path.join(appdata, "nexus-shadow-quant")
    return os.path.dirname(os.path.abspath(__file__))


# ── Paths ─────────────────────────────────────────────
DATA_ROOT = _get_data_root()
DATA_DIR = os.path.join(DATA_ROOT, "data")
MODEL_DIR = os.path.join(DATA_ROOT, "models")
LOG_DIR = os.path.join(DATA_ROOT, "logs")
DB_PATH = os.path.join(DATA_DIR, "nexus_brain.db")
MARKET_DATA_PATH = os.path.join(DATA_DIR, "market_data.csv")
MARKET_DATA_PARQUET_PATH = os.path.join(DATA_DIR, "market_data.parquet")

# Cross-asset data paths (Phase 3: inter-market correlation)
CROSS_ASSET_PAIRS = ['ETH/USDT', 'PAXG/USDT', 'ETH/BTC']
ETH_DATA_PATH = os.path.join(DATA_DIR, "eth_usdt.parquet")
PAXG_DATA_PATH = os.path.join(DATA_DIR, "paxg_usdt.parquet")
ETHBTC_DATA_PATH = os.path.join(DATA_DIR, "eth_btc.parquet")

# Ensure directories exist
for _d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

# Load .env from data root (API keys are stored here)
_env_path = os.path.join(DATA_ROOT, ".env")
if os.path.exists(_env_path):
    try:
        load_dotenv(_env_path)
    except (UnicodeDecodeError, Exception) as _e:
        # .env may have non-UTF-8 bytes (e.g. Windows-1252) — re-encode and retry
        import logging as _log
        _log.warning(f"load_dotenv failed ({_e}), re-encoding .env to UTF-8")
        try:
            with open(_env_path, "rb") as _f:
                _raw = _f.read()
            _text = _raw.decode("utf-8", errors="replace")
            with open(_env_path, "w", encoding="utf-8") as _f:
                _f.write(_text)
            load_dotenv(_env_path)
        except Exception:
            _log.warning("Could not recover .env — continuing without it")
else:
    load_dotenv()  # fallback: check project dir

# ── System Settings ───────────────────────────────────
DEFAULT_DEVICE = "cuda"
VERSION = "v6.0.1 Beta Stable"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
IS_INSTALLED = _is_installed()

# ── Base Model (ships with app for instant-on predictions) ──
BASE_MODEL_DIR = os.path.join(DATA_ROOT, "models", "base")

# ── API Keys ──────────────────────────────────────────
# Exchange
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

# AI Providers (configurable via Settings page)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Notifications
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Intervals ─────────────────────────────────────────
UPDATE_INTERVAL_SEC = 60
NEWS_INTERVAL_MIN = 30
HARDWARE_LOG_MIN = 10
RETRAIN_INTERVAL_HOURS = 6

# ── Paper Trading ─────────────────────────────────────
PAPER_STARTING_BALANCE = 10000
PAPER_TRADES_PATH = os.path.join(DATA_DIR, "paper_trades.csv")
PAPER_EQUITY_PATH = os.path.join(DATA_DIR, "paper_equity.csv")
PAPER_POSITIONS_PATH = os.path.join(DATA_DIR, "paper_positions.json")
PAPER_DEFAULT_LEVERAGE = 10
PAPER_MIN_CONFIDENCE = 30
PAPER_MAX_DRAWDOWN = 0.20
PAPER_COOLDOWN_SEC = 60
PAPER_MAX_HOLD_SEC = 7200

# ── News Sources ──────────────────────────────────────
NEWS_SOURCES = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://cryptonews.com/news/feed/",
    "https://decrypt.co/feed",
    "https://www.newsbtc.com/feed/",
]
