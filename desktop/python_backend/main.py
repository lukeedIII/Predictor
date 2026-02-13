import logging, time, os
import pandas as pd
from datetime import datetime, timedelta
import config
from data_collector import DataCollector
from sentiment_engine import SentimentEngine
from predictor import NexusPredictor
from whale_monitor import WhaleMonitor
from notifications import NotificationHub
from hardware_profiler import MemoryProfiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_nexus():
    logging.info(f"[[ NEXUS SHADOW-QUANT {config.VERSION} ACTIVE ]]")
    collector = DataCollector()
    sentiment = SentimentEngine()
    predictor = NexusPredictor()
    whale_monitor = WhaleMonitor()
    hub = NotificationHub()
    profiler = MemoryProfiler()
    
    last_sentiment = datetime.now() - timedelta(minutes=10)  # Force initial sentiment fetch
    last_hw = datetime.now()
    last_train = datetime.now() - timedelta(hours=1)  # Force initial training
    current_sentiment = 0.0  # Neutral = 0, not 0.5
    
    # Elon alert cooldown
    last_elon_alert = datetime.now() - timedelta(hours=1)
    
    while True:
        try:
            logging.info("--- New Cycle Started ---")
            
            # 1. Market Data Collection
            collector.collect_and_save()
            price = collector.get_latest_price()
            logging.info(f"Current BTC Price: ${price:,.2f}")

            # 2. Combined Sentiment Analysis (every 5 minutes)
            if datetime.now() - last_sentiment > timedelta(minutes=5):
                logging.info("Fetching combined sentiment (RSS + Twitter)...")
                
                # Combined sentiment from RSS and Twitter
                current_sentiment = sentiment.get_combined_sentiment(max_twitter_accounts=20)
                last_sentiment = datetime.now()
                
                # Check for Elon BTC tweets
                if datetime.now() - last_elon_alert > timedelta(minutes=30):
                    elon_alert = sentiment.check_elon_alert()
                    if elon_alert:
                        logging.warning(f"🚨 ELON BTC TWEET DETECTED: {elon_alert['text'][:100]}...")
                        hub.send_desktop_notify(
                            "🚨 ELON BTC ALERT",
                            f"{elon_alert['text'][:200]}..."
                        )
                        last_elon_alert = datetime.now()
                
                logging.info(f"Combined Sentiment: {current_sentiment:+.2f}")
            
            # 3. On-Chain Metrics with Pressure Differentiation
            mempool = whale_monitor.fetch_mempool_stats()
            whales = whale_monitor.fetch_recent_whales(price)
            whale_pressure = whale_monitor.calculate_whale_pressure(whales, mempool)
            logging.info(f"Whale Pressure: {whale_pressure:+.2f} | Mempool: {mempool:+.2f}")

            # 4. Persist Insights for Training
            t_path = os.path.join(config.DATA_DIR, "training_dataset.csv")
            new_row = pd.DataFrame([{
                "timestamp": datetime.now(),
                "btc_price": price,
                "sentiment_score": current_sentiment,
                "whale_pressure": whale_pressure,
                "mempool_factor": mempool
            }])
            new_row.to_csv(t_path, mode='a', header=not os.path.exists(t_path), index=False)

            # 5. Training (only every hour to prevent overfitting/instability)
            if datetime.now() - last_train > timedelta(hours=1):
                logging.info("Starting scheduled model training...")
                is_trained, progress = predictor.train()
                last_train = datetime.now()
                if is_trained:
                    logging.info("Model training complete.")
                else:
                    logging.info(f"Training progress: {progress:.1f}%")
            else:
                # Just check if model exists
                is_trained = predictor.is_statistically_verified

            # 6. Prediction & Alerts
            if is_trained:
                pred = predictor.get_prediction()
                logging.info(f"Prediction: {pred['direction']} @ {pred['confidence']:.1f}% confidence")
                
                if pred['confidence'] >= 75 and pred.get('verified', False):
                    hub.send_desktop_notify(
                        "Nexus Alpha Signal",
                        f"{pred['direction']} {pred['confidence']:.0f}% (Verified Model)"
                    )
            
            # 7. Hardware Monitoring
            if datetime.now() - last_hw > timedelta(minutes=config.HARDWARE_LOG_MIN):
                profiler.log_vram_usage()
                last_hw = datetime.now()
            
            # 8. Data collection progress (lightweight count — don't load 206MB CSV!)
            df_len = 0
            if os.path.exists(config.MARKET_DATA_PARQUET_PATH):
                df_len = len(pd.read_parquet(config.MARKET_DATA_PARQUET_PATH, columns=['close']))
            elif os.path.exists(config.MARKET_DATA_PATH):
                # Count lines without loading entire file into memory
                with open(config.MARKET_DATA_PATH, 'r') as f:
                    df_len = sum(1 for _ in f) - 1  # -1 for header
            logging.info(f"Data Points: {df_len} | Training Ready: {is_trained} | Sentiment: {current_sentiment:+.2f}")
            
            time.sleep(config.UPDATE_INTERVAL_SEC)
            
        except Exception as e:
            logging.error(f"Loop error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_nexus()
