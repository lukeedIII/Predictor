"""
Historical Data Downloader
==========================
Downloads years of BTC/USDT 1-minute data from Binance.

Usage:
    python download_historical.py [--days 1095] [--start 2020-01-01]
    python download_historical.py --days 1095 --json-progress   # For Electron UI
"""

import ccxt
import pandas as pd
import time
import os
import sys
import json
from datetime import datetime, timedelta
import argparse
import config

# Global flag for JSON progress mode
_json_progress = False


def _emit(stage: str, progress: float, message: str, **extra):
    """Emit progress either as JSON (for Electron) or human-readable text."""
    if _json_progress:
        payload = {"stage": stage, "progress": round(progress, 1), "message": message}
        payload.update(extra)
        print(json.dumps(payload), flush=True)
    else:
        print(f"   {message}")


def download_historical_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    start_date: str = None,
    days: int = 1095,
    output_path: str = None
) -> pd.DataFrame:
    """
    Download historical OHLCV data from Binance.
    
    Args:
        symbol: Trading pair (default BTC/USDT)
        timeframe: Candle timeframe (1m, 5m, 1h, 1d)
        start_date: Start date (YYYY-MM-DD) or None for 'days ago'
        days: Number of days to download if start_date is None (default 3 years)
        output_path: Where to save CSV
    
    Returns:
        DataFrame with OHLCV data
    """
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    # Calculate start timestamp
    if start_date:
        start_ts = exchange.parse8601(f"{start_date}T00:00:00Z")
    else:
        start_ts = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
    
    end_ts = exchange.milliseconds()
    
    _emit("downloading", 0, f"Downloading {symbol} {timeframe} data ({days} days)...",
          symbol=symbol, timeframe=timeframe, days=days)
    
    if not _json_progress:
        print(f"   From: {exchange.iso8601(start_ts)}")
        print(f"   To:   {exchange.iso8601(end_ts)}")
    
    all_data = []
    current_ts = start_ts
    
    # Binance limit is 1000 candles per request
    limit = 1000
    
    # Estimate total requests needed
    total_requests = max(1, ((end_ts - start_ts) / (limit * 60 * 1000)) + 1)
    request_count = 0
    last_progress_emit = 0
    
    while current_ts < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_ts, limit=limit)
            
            if not ohlcv:
                break
            
            all_data.extend(ohlcv)
            
            # Move to next batch
            current_ts = ohlcv[-1][0] + 1
            
            request_count += 1
            progress = min(99, (request_count / total_requests) * 100)
            
            # Emit progress every 2% or every 25 requests
            if progress - last_progress_emit >= 2 or request_count % 25 == 0:
                _emit("downloading", progress,
                      f"Downloading... {len(all_data):,} candles ({progress:.0f}%)",
                      candles=len(all_data), requests=request_count)
                last_progress_emit = progress
            
            # Respect rate limits
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            _emit("downloading", progress if 'progress' in dir() else 0,
                  f"Retrying... ({str(e)[:60]})", error=str(e)[:100])
            time.sleep(5)
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    _emit("saving", 95, f"Saving {len(df):,} candles...", candles=len(df))
    
    # Save to CSV + Parquet
    if output_path is None:
        output_path = config.MARKET_DATA_PATH
    
    # Backup existing data first
    if os.path.exists(output_path):
        backup_path = output_path.replace('.csv', '_backup.csv')
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(output_path, backup_path)
    
    df.to_csv(output_path, index=False)
    
    # Also save as parquet for faster loading
    parquet_path = output_path.replace('.csv', '.parquet')
    df.to_parquet(parquet_path, index=False)
    
    _emit("complete", 100,
          f"Downloaded {len(df):,} candles ({df['timestamp'].min().strftime('%Y-%m-%d')} → {df['timestamp'].max().strftime('%Y-%m-%d')})",
          candles=len(df),
          date_from=str(df['timestamp'].min()),
          date_to=str(df['timestamp'].max()))
    
    return df


def download_multi_timeframe(days: int = 1095):
    """Download data for multiple timeframes."""
    
    timeframes = ['1m', '5m', '1h', '1d']
    
    for tf in timeframes:
        output_path = os.path.join(config.DATA_DIR, f"market_data_{tf}.csv")
        download_historical_data(timeframe=tf, days=days, output_path=output_path)
        print()


def main():
    global _json_progress
    
    parser = argparse.ArgumentParser(description='Download historical BTC/USDT data')
    parser.add_argument('--days', type=int, default=1095, help='Number of days to download (default: 3 years)')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default='1m', help='Timeframe (1m, 5m, 1h, 1d)')
    parser.add_argument('--multi', action='store_true', help='Download multiple timeframes')
    parser.add_argument('--json-progress', action='store_true', help='Output progress as JSON lines (for Electron)')
    
    args = parser.parse_args()
    _json_progress = args.json_progress
    
    if not _json_progress:
        print("\n" + "=" * 60)
        print("  NEXUS HISTORICAL DATA DOWNLOADER")
        print("=" * 60 + "\n")
    
    if args.multi:
        download_multi_timeframe(args.days)
    else:
        download_historical_data(
            timeframe=args.timeframe,
            start_date=args.start,
            days=args.days
        )
    
    if not _json_progress:
        print("\n✅ Download complete!")
        print("   Run `python run_backtest.py` to test with new data.\n")


if __name__ == "__main__":
    main()
