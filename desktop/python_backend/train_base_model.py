"""
Base Model Training Script — Nexus Shadow-Quant v6.0
=====================================================
Downloads 6 months of historical BTC/USDT 1-minute data + cross-asset data,
engineers all 40 features, trains the XGBoost+LSTM ensemble, and saves the
"base model" that ships embedded in the desktop application.

The base model gives new users instant-on predictions without waiting for
local data collection + training.

Usage:
    python train_base_model.py                  # Full 6-month pipeline
    python train_base_model.py --days 90        # Custom window
    python train_base_model.py --skip-download  # Use existing data
    python train_base_model.py --dry-run        # Validate pipeline only
"""

import os
import sys
import time
import logging
import argparse
import json
from datetime import datetime

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [BASE-TRAIN] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_all_data(days: int = 180):
    """Download BTC + cross-asset data for training."""
    from download_historical import download_historical_data
    
    # === 1. BTC/USDT (primary) ===
    logger.info(f"📥 Downloading BTC/USDT 1m data ({days} days)...")
    btc_df = download_historical_data(
        symbol="BTC/USDT", timeframe="1m", days=days,
        output_path=config.MARKET_DATA_PATH
    )
    logger.info(f"   → {len(btc_df):,} BTC candles downloaded")
    
    # === 2. Cross-asset pairs ===
    cross_pairs = [
        ("ETH/USDT", config.ETH_DATA_PATH),
        ("PAXG/USDT", config.PAXG_DATA_PATH),
        ("ETH/BTC", config.ETHBTC_DATA_PATH),
    ]
    
    for symbol, path in cross_pairs:
        try:
            logger.info(f"📥 Downloading {symbol} 1m data ({days} days)...")
            df = download_historical_data(
                symbol=symbol, timeframe="1m", days=days,
                output_path=path.replace('.parquet', '.csv')
            )
            # Save as parquet too
            if not path.endswith('.parquet'):
                df.to_parquet(path, index=False)
            logger.info(f"   → {len(df):,} {symbol} candles")
        except Exception as e:
            logger.warning(f"   ⚠️ {symbol} download failed: {e} (continuing without)")
    
    return btc_df


def train_base_model(skip_download: bool = False, days: int = 180, dry_run: bool = False):
    """
    Full base model training pipeline:
    1. Download/validate data
    2. Engineer features
    3. Train XGBoost + LSTM ensemble
    4. Validate & audit
    5. Save to base model directory
    """
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("  NEXUS BASE MODEL TRAINER v6.0")
    logger.info("=" * 60)
    logger.info(f"  Days:          {days}")
    logger.info(f"  Skip download: {skip_download}")
    logger.info(f"  Dry run:       {dry_run}")
    logger.info(f"  Output dir:    {config.BASE_MODEL_DIR}")
    logger.info("=" * 60)
    
    # === STEP 1: Data ===
    if dry_run:
        logger.info("📂 Dry-run: skipping data download, validating pipeline...")
    elif not skip_download:
        download_all_data(days=days)
    else:
        logger.info("📂 Using existing data (--skip-download)")
        if not os.path.exists(config.MARKET_DATA_PARQUET_PATH) and not os.path.exists(config.MARKET_DATA_PATH):
            logger.error("❌ No data found! Run without --skip-download first.")
            return False
    
    # === STEP 2: Initialize predictor and engineer features ===
    logger.info("🔧 Initializing predictor & engineering features...")
    from predictor import NexusPredictor
    predictor = NexusPredictor()
    
    logger.info(f"   Feature count: {len(predictor.features)}")
    logger.info(f"   Features: {', '.join(predictor.features[:10])}... (+{len(predictor.features)-10} more)")
    
    if dry_run:
        logger.info("✅ Dry run: pipeline validated successfully")
        logger.info(f"   Feature count: {len(predictor.features)}")
        logger.info(f"   Data exists: {os.path.exists(config.MARKET_DATA_PARQUET_PATH)}")
        return True
    
    # === STEP 3: Train ===
    logger.info("🧠 Training base model...")
    is_trained, progress = predictor.train()
    
    if not is_trained:
        logger.error(f"❌ Training failed at {progress:.0f}%")
        return False
    
    # === STEP 4: Save to base model directory ===
    os.makedirs(config.BASE_MODEL_DIR, exist_ok=True)
    
    import shutil
    import joblib
    import pickle
    
    # Copy trained models to base directory
    base_files = {
        predictor.model_path: os.path.join(config.BASE_MODEL_DIR, "base_xgboost.joblib"),
        predictor.lstm_path: os.path.join(config.BASE_MODEL_DIR, "base_lstm.pth"),
        predictor.scaler_path: os.path.join(config.BASE_MODEL_DIR, "base_scaler.pkl"),
        predictor.ensemble_state_path: os.path.join(config.BASE_MODEL_DIR, "base_ensemble_state.pkl"),
    }
    
    for src, dst in base_files.items():
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"   📦 Saved: {os.path.basename(dst)}")
    
    # === STEP 5: Audit Report ===
    elapsed = time.time() - start_time
    
    audit = {
        "version": "v6.0",
        "trained_at": datetime.now().isoformat(),
        "training_days": days,
        "feature_count": len(predictor.features),
        "features": predictor.features,
        "xgb_accuracy": predictor.last_validation_accuracy,
        "lstm_accuracy": predictor.lstm_validation_acc,
        "ensemble_weights": {
            "xgb": predictor.xgb_weight,
            "lstm": predictor.lstm_weight,
        },
        "training_time_sec": round(elapsed, 1),
        "model_files": list(base_files.values()),
    }
    
    audit_path = os.path.join(config.BASE_MODEL_DIR, "base_model_audit.json")
    with open(audit_path, 'w') as f:
        json.dump(audit, f, indent=2)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("  ✅ BASE MODEL TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  XGBoost Accuracy:  {predictor.last_validation_accuracy:.1f}%")
    logger.info(f"  LSTM Accuracy:     {predictor.lstm_validation_acc:.1f}%")
    logger.info(f"  Ensemble Weights:  XGB={predictor.xgb_weight:.2f} LSTM={predictor.lstm_weight:.2f}")
    logger.info(f"  Features:          {len(predictor.features)}")
    logger.info(f"  Training Time:     {elapsed:.0f}s")
    logger.info(f"  Output:            {config.BASE_MODEL_DIR}")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Nexus base model for desktop embedding")
    parser.add_argument("--days", type=int, default=180, help="Days of historical data (default: 180)")
    parser.add_argument("--skip-download", action="store_true", help="Use existing data")
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline only")
    args = parser.parse_args()
    
    success = train_base_model(
        skip_download=args.skip_download,
        days=args.days,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if success else 1)
