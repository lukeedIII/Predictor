import os
import sys
import time
import json
import logging
import threading
from datetime import datetime

# Adjust sys.path to ensure we can import parent directory modules without issue
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SentimentWorker')

STATE_FILE = os.path.join(os.path.dirname(__file__), 'latest_sentiment.json')

class SentimentWorker:
    """
    Dedicated background worker for heavy NLP inference.
    Executes FinBERT on a longer cadence (e.g., 5-15 mins) and writes
    results safely to a JSON state file for Predictor.py to read instantly
    during real-time WebSocket ticks.
    """
    def __init__(self, interval_minutes: int = 15):
        self.interval_minutes = interval_minutes
        self.stop_event = threading.Event()
        self.thread = None
        self.engine = None
        
    def start(self):
        """Start the background worker thread."""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Sentiment worker already running.")
            return
            
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, name="SentimentWorker", daemon=True)
        self.thread.start()
        logger.info(f"Sentiment worker started (interval: {self.interval_minutes}m)")
        
    def stop(self):
        """Signal the worker to stop and wait for it to join."""
        if self.thread is not None:
            logger.info("Stopping Sentiment worker...")
            self.stop_event.set()
            self.thread.join(timeout=5)
            self.thread = None
            
    def _initialize_engine(self):
        """Initialize FinBERT only when the thread starts to save main thread memory."""
        if self.engine is not None:
            return
            
        try:
            from sentiment_engine import SentimentEngine
            self.engine = SentimentEngine()
        except Exception as e:
            logger.error(f"Failed to load Sentiment Engine: {e}")
            
    def _run_loop(self):
        """Main loop fetching sentiment text and running FinBERT analysis."""
        self._initialize_engine()
        
        while not self.stop_event.is_set():
            # Run sentiment pass immediately
            self._execute_sentiment_pass()
            
            # Wait for interval, but respond to stop_event quickly
            # Break the wait into 1-second chunks
            for _ in range(self.interval_minutes * 60):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
                
    def _execute_sentiment_pass(self):
        """Executes a single pass of reading news/tweets and inferring sentiment."""
        if self.engine is None:
            return
            
        try:
            # We cap twitter accounts to control scraping time
            combined_score = self.engine.get_combined_sentiment(max_twitter_accounts=10)
            
            # Additional targeted fetch: check for active Elon Musk market-moving warnings
            elon_alert = self.engine.check_elon_alert()
            
            payload = {
                "timestamp": datetime.now().isoformat(),
                "combined_score": float(combined_score),
                "elon_alert_active": True if elon_alert else False,
                "status": "active"
            }
            if elon_alert:
                 payload["elon_alert_text"] = elon_alert.get('text', "")
            
            self._save_state_atomically(payload)
            logger.info(f"Sentiment pass complete. Score saved: {combined_score:+.2f}")
            
        except Exception as e:
            logger.error(f"Error during sentiment calculation: {e}", exc_info=True)
            
    def _save_state_atomically(self, payload: dict):
        """Write to a tmp file first, then os.replace for atomic safety."""
        tmp_file = STATE_FILE + ".tmp"
        try:
            with open(tmp_file, 'w') as f:
                json.dump(payload, f)
            os.replace(tmp_file, STATE_FILE)
        except Exception as e:
            logger.error(f"Failed to atomically save state to {STATE_FILE}: {e}")

if __name__ == "__main__":
    # Test execution
    worker = SentimentWorker(interval_minutes=1)
    worker.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()
