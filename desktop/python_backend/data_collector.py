import ccxt
import pandas as pd
import time
import os
from datetime import datetime
import config
import logging

# Setup Logging
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, "data_collector.log")),
        logging.StreamHandler()
    ]
)

class DataCollector:
    def __init__(self, symbol=config.SYMBOL, timeframe=config.TIMEFRAME):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_SECRET_KEY,
        })
        self.csv_path = config.MARKET_DATA_PATH

    def fetch_ohlcv(self, limit=100):
        """Fetches last N candles from Binance with retry logic."""
        retries = 5
        for i in range(retries):
            try:
                logging.info(f"Fetching {self.timeframe} candles for {self.symbol}...")
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                # Drop the last candle — it's almost always still open/unfinalised
                if len(df) > 1:
                    df = df.iloc[:-1]
                return df
            except Exception as e:
                wait_time = (i + 1) * 2
                logging.error(f"Error fetching data: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        return None

    def update_csv(self, df):
        """Appends new data to CSV if it doesn't already exist."""
        if df is None or df.empty:
            return

        if not os.path.exists(self.csv_path):
            df.to_csv(self.csv_path, index=False)
            logging.info(f"Created new file: {self.csv_path}")
        else:
            try:
                existing_df = pd.read_csv(self.csv_path)
                if existing_df.empty or 'timestamp' not in existing_df.columns:
                    raise ValueError("CSV is empty or missing columns")
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            except (pd.errors.EmptyDataError, ValueError, Exception) as e:
                logging.warning(f"CSV corrupt/empty ({e}), overwriting with fresh data.")
                df.to_csv(self.csv_path, index=False)
                parquet_path = self.csv_path.replace('.csv', '.parquet')
                df.to_parquet(parquet_path, index=False)
                return
            
            # Combine and remove duplicates
            combined = pd.concat([existing_df, df]).drop_duplicates(subset=['timestamp'], keep='last')
            combined.sort_values('timestamp', inplace=True)
            combined.to_csv(self.csv_path, index=False)
            # Also save parquet for fast loading (50x faster than CSV)
            parquet_path = self.csv_path.replace('.csv', '.parquet')
            combined.to_parquet(parquet_path, index=False)
            logging.info(f"Updated {self.csv_path} with latest data.")

    def collect_and_save(self, limit=5):
        """Fetches and saves data once."""
        df = self.fetch_ohlcv(limit=limit)
        if df is not None:
            self.update_csv(df)
            # Also fetch cross-asset pairs (ETH, Gold, ETH/BTC)
            self.collect_cross_assets(limit=limit)
            return True
        return False

    def collect_cross_assets(self, limit=5):
        """Fetch cross-asset pairs for inter-market correlation features.
        Each pair is stored in its own Parquet file, aligned by timestamp."""
        pair_paths = {
            'ETH/USDT': config.ETH_DATA_PATH,
            'PAXG/USDT': config.PAXG_DATA_PATH,
            'ETH/BTC': config.ETHBTC_DATA_PATH,
        }
        
        for symbol, path in pair_paths.items():
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                if len(df) > 1:
                    df = df.iloc[:-1]  # Drop unfinalised candle
                
                # Append to existing or create new
                if os.path.exists(path):
                    existing = pd.read_parquet(path)
                    combined = pd.concat([existing, df]).drop_duplicates(subset=['timestamp'], keep='last')
                    combined.sort_values('timestamp', inplace=True)
                    combined.to_parquet(path, index=False)
                else:
                    df.to_parquet(path, index=False)
                
            except Exception as e:
                logging.debug(f"Cross-asset {symbol}: {e}")  # Non-blocking

    def get_latest_price(self):
        """Returns the last closed price from CSV."""
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                if not df.empty and 'close' in df.columns:
                    return df['close'].iloc[-1]
            except (pd.errors.EmptyDataError, Exception):
                pass
        return 0.0

    def run(self):
        """Main loop to collect data every minute."""
        logging.info("Starting Nexus Data Collector...")
        while True:
            try:
                # Fetch recent candles
                df = self.fetch_ohlcv(limit=5)
                if df is not None:
                    # Filter for closed candles (previous minute)
                    # Note: The last candle in Binance OHLCV is often still "open"
                    self.update_csv(df)
                
                # Wait until next minute
                now = datetime.now()
                seconds_to_wait = 60 - now.second
                logging.info(f"Waiting {seconds_to_wait}s for next cycle...")
                time.sleep(seconds_to_wait)
                
            except KeyboardInterrupt:
                logging.info("Stopping Data Collector...")
                break
            except Exception as e:
                logging.critical(f"Fatal error in main loop: {e}")
                time.sleep(10)

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
