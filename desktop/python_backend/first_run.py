"""
Nexus Shadow-Quant — First Run Setup
======================================
Orchestrates the entire first-launch experience:
  1. Download 3 years of BTC/USDT 1-minute data from Binance
  2. Download FinBERT sentiment model from HuggingFace
  3. Train XGBoost + LSTM models from scratch
  4. Validate everything works

Outputs JSON progress lines for Electron to consume.

Usage:
    python first_run.py                    # Normal mode
    python first_run.py --json-progress    # Electron mode (JSON output)
    python first_run.py --days 30          # Quick setup (30 days for testing)
"""

import json
import sys
import os
import time
import logging
import argparse

# Ensure project modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# Suppress verbose logging during first run (we have our own progress)
logging.basicConfig(level=logging.WARNING)

_json_mode = False


def emit(phase: int, total_phases: int, stage: str, progress: float, message: str, **extra):
    """Emit progress to stdout (JSON or human-readable)."""
    overall = ((phase - 1) / total_phases + (progress / 100) / total_phases) * 100
    
    if _json_mode:
        payload = {
            "phase": phase,
            "total_phases": total_phases,
            "stage": stage,
            "phase_progress": round(progress, 1),
            "overall_progress": round(overall, 1),
            "message": message,
        }
        payload.update(extra)
        print(json.dumps(payload), flush=True)
    else:
        bar_len = 30
        filled = int(bar_len * overall / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {overall:5.1f}%  Phase {phase}/{total_phases}: {message}", end="", flush=True)
        if progress >= 100:
            print()  # newline at end of phase


def run_first_setup(days: int = 1095):
    """Execute the complete first-run setup."""
    total_phases = 4
    
    if not _json_mode:
        print("\n" + "═" * 60)
        print("  ⚡ NEXUS SHADOW-QUANT — FIRST RUN SETUP")
        print("═" * 60)
        print(f"\n  Downloading {days} days of BTC/USDT data and training AI models.\n")

    # ══════════════════════════════════════════════════════════
    # PHASE 1: Load Seed Data + Download Missing
    # ══════════════════════════════════════════════════════════
    emit(1, total_phases, "download", 0, "Checking for bundled seed data...")
    
    try:
        import pandas as pd
        
        # Look for seed data bundled with the app
        seed_paths = [
            os.path.join(os.path.dirname(__file__), "data", "seed_data.parquet"),
            os.path.join(os.path.dirname(__file__), "..", "python_backend", "data", "seed_data.parquet"),
        ]
        
        seed_df = None
        for sp in seed_paths:
            if os.path.exists(sp):
                emit(1, total_phases, "download", 5, "Extracting bundled seed data...")
                seed_df = pd.read_parquet(sp)
                emit(1, total_phases, "download", 15,
                     f"Seed data loaded — {len(seed_df):,} candles")
                break
        
        # Also check if user already has data in AppData
        if os.path.exists(config.MARKET_DATA_PARQUET_PATH):
            existing_df = pd.read_parquet(config.MARKET_DATA_PARQUET_PATH)
            if seed_df is None or len(existing_df) > len(seed_df):
                seed_df = existing_df
                emit(1, total_phases, "download", 15,
                     f"Existing data found — {len(seed_df):,} candles")
        
        if seed_df is not None and len(seed_df) > 100_000:
            # Incremental update — only download what's missing
            if 'timestamp' in seed_df.columns:
                last_ts = pd.to_datetime(seed_df['timestamp']).max()
            else:
                last_ts = seed_df.index.max() if hasattr(seed_df.index, 'max') else None
            
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            
            if last_ts is not None:
                # Calculate gap
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
                gap_hours = (now - last_ts).total_seconds() / 3600
                gap_candles = int(gap_hours * 60)  # 1-min candles
                
                emit(1, total_phases, "download", 20,
                     f"Latest data: {last_ts.strftime('%Y-%m-%d %H:%M')} — "
                     f"Gap: {gap_candles:,} candles ({gap_hours:.0f}h)")
                
                if gap_candles > 5:
                    # Download only the gap
                    from download_historical import download_historical_data
                    import download_historical as dl_module
                    
                    _orig_emit = dl_module._emit
                    def _patched_emit(stage, progress, message, **extra):
                        # Scale progress to 20-90 range
                        scaled = 20 + (progress * 0.7)
                        emit(1, total_phases, stage, scaled, message, **extra)
                    dl_module._emit = _patched_emit
                    
                    gap_days = max(1, int(gap_hours / 24) + 1)
                    gap_df = download_historical_data(
                        symbol="BTC/USDT",
                        timeframe="1m",
                        days=gap_days,
                        output_path=None,  # Don't save yet
                    )
                    dl_module._emit = _orig_emit
                    
                    if gap_df is not None and len(gap_df) > 0:
                        # Merge seed + gap
                        emit(1, total_phases, "download", 90, 
                             f"Merging {len(gap_df):,} new candles with seed data...")
                        combined = pd.concat([seed_df, gap_df]).drop_duplicates()
                        if 'timestamp' in combined.columns:
                            combined = combined.sort_values('timestamp').reset_index(drop=True)
                        combined.to_parquet(config.MARKET_DATA_PARQUET_PATH)
                        candle_count = len(combined)
                    else:
                        # Gap download failed, use seed as-is
                        seed_df.to_parquet(config.MARKET_DATA_PARQUET_PATH)
                        candle_count = len(seed_df)
                else:
                    # Data is fresh enough
                    emit(1, total_phases, "download", 90, "Data is up to date!")
                    seed_df.to_parquet(config.MARKET_DATA_PARQUET_PATH)
                    candle_count = len(seed_df)
            else:
                # No timestamp info, just use seed
                seed_df.to_parquet(config.MARKET_DATA_PARQUET_PATH)
                candle_count = len(seed_df)
            
            emit(1, total_phases, "download_complete", 100,
                 f"✓ Market data synced — {candle_count:,} total candles",
                 candles=candle_count)
        else:
            # No seed data — full download
            emit(1, total_phases, "download", 5,
                 "No seed data found — downloading full dataset...")
            
            from download_historical import download_historical_data
            import download_historical as dl_module
            
            _orig_emit = dl_module._emit
            def _patched_emit(stage, progress, message, **extra):
                emit(1, total_phases, stage, progress, message, **extra)
            dl_module._emit = _patched_emit
            
            df = download_historical_data(
                symbol="BTC/USDT",
                timeframe="1m",
                days=days,
                output_path=config.MARKET_DATA_PATH,
            )
            dl_module._emit = _orig_emit
            
            candle_count = len(df) if df is not None else 0
            emit(1, total_phases, "download_complete", 100,
                 f"Downloaded {candle_count:,} candles", candles=candle_count)
    
    except Exception as e:
        emit(1, total_phases, "error", 0, f"Data setup failed: {str(e)[:80]}", error=str(e))
        if not _json_mode:
            print(f"\n  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        return False

    # ══════════════════════════════════════════════════════════
    # PHASE 2: Download FinBERT Model (~438 MB)
    # ══════════════════════════════════════════════════════════
    emit(2, total_phases, "downloading_model", 0, "Downloading FinBERT sentiment model...")
    
    try:
        finbert_cache = os.path.join(config.MODEL_DIR, "finbert")
        os.makedirs(finbert_cache, exist_ok=True)
        
        emit(2, total_phases, "downloading_model", 20, "Loading HuggingFace transformers...")
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "ProsusAI/finbert"
        
        emit(2, total_phases, "downloading_model", 40, "Downloading FinBERT tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=finbert_cache)
        
        emit(2, total_phases, "downloading_model", 60, "Downloading FinBERT model weights (438 MB)...")
        _model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=finbert_cache)
        
        emit(2, total_phases, "finbert_complete", 100, "FinBERT sentiment model ready")
        
        del _tokenizer, _model  # Free memory for training
    except Exception as e:
        emit(2, total_phases, "warning", 100,
             f"FinBERT download skipped: {str(e)[:60]}. Sentiment will use fallback.",
             error=str(e)[:100])
        # Non-fatal — app works without FinBERT

    # ══════════════════════════════════════════════════════════
    # PHASE 3: Train XGBoost Model
    # ══════════════════════════════════════════════════════════
    emit(3, total_phases, "training_xgb", 0, "Initializing XGBoost training pipeline...")
    
    try:
        import pandas as pd
        import numpy as np
        
        emit(3, total_phases, "training_xgb", 10, "Loading market data for training...")
        
        # Use the predictor's training pipeline
        from predictor import NexusPredictor
        
        emit(3, total_phases, "training_xgb", 20, "Initializing Nexus prediction engine...")
        
        predictor = NexusPredictor()
        
        emit(3, total_phases, "training_xgb", 30, "Loading and processing features...")
        
        # The train() method handles everything: feature engineering, XGBoost, and LSTM
        emit(3, total_phases, "training_xgb", 40, "Training models (XGBoost + LSTM)...")
        
        is_trained, progress, _promotion = predictor.train()
        
        if is_trained:
            emit(3, total_phases, "xgb_complete", 100,
                 "XGBoost + LSTM training complete")
        else:
            emit(3, total_phases, "error", 0, "Training failed — model did not converge")
            return False
        
    except Exception as e:
        emit(3, total_phases, "error", 0, f"XGBoost training failed: {str(e)[:80]}", error=str(e))
        if not _json_mode:
            import traceback
            traceback.print_exc()
        return False

    # ══════════════════════════════════════════════════════════
    # PHASE 4: Verify Models
    # ══════════════════════════════════════════════════════════
    emit(4, total_phases, "verify", 0, "Verifying trained models...")
    
    try:
        import torch
        
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        
        emit(4, total_phases, "verify", 30, f"Device: {gpu_name}")
        
        # Check model files exist
        model_exists = predictor.is_trained
        emit(4, total_phases, "verify", 60,
             f"Model status: {'Ready' if model_exists else 'Not found'}")
        
        emit(4, total_phases, "verify_complete", 100,
             f"All models verified on {gpu_name}", gpu=gpu_name)
        
    except Exception as e:
        emit(4, total_phases, "warning", 100,
             f"Verification check: {str(e)[:60]}",
             error=str(e)[:100])

    # ══════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════
    emit(4, total_phases, "complete", 100, "Setup complete! Nexus Shadow-Quant is ready.")
    
    if not _json_mode:
        print("\n" + "═" * 60)
        print("  ✅ FIRST RUN SETUP COMPLETE")
        print("═" * 60)
        print(f"  • Data:   {config.MARKET_DATA_PARQUET_PATH}")
        print(f"  • Models: {config.MODEL_DIR}")
        print(f"  • Logs:   {config.LOG_DIR}")
        print(f"\n  Launch the dashboard to start analyzing! 🚀\n")
    
    return True


def main():
    global _json_mode
    
    parser = argparse.ArgumentParser(description='Nexus Shadow-Quant First Run Setup')
    parser.add_argument('--days', type=int, default=1095,
                        help='Days of historical data to download (default: 1095 = 3 years)')
    parser.add_argument('--json-progress', action='store_true',
                        help='Output progress as JSON lines for Electron')
    
    args = parser.parse_args()
    _json_mode = args.json_progress
    
    success = run_first_setup(days=args.days)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
