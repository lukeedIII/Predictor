import requests
import logging
import os
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WhaleMonitor:
    def __init__(self):
        self.base_url = "https://mempool.space/api"
        self.whale_threshold_btc = 100  # Lowered for more signals
        self.recent_whales = []

    def fetch_mempool_stats(self):
        """Fetches general mempool stats to calculate network activity factor."""
        try:
            response = requests.get(f"{self.base_url}/mempool", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            vsize = data.get("vsize", 0)
            count = data.get("count", 0)
            
            # Normalize: typical mempool vsize is 50-200 MB
            # Return value between -1 and 1 (negative = low activity, positive = high)
            normalized = (vsize / 100_000_000) - 1.0  # Center around 100MB
            return max(-1.0, min(1.0, normalized))
            
        except Exception as e:
            logging.error(f"Error fetching mempool stats: {e}")
            return 0.0

    def fetch_recent_whales(self, current_btc_price):
        """Scans recent transactions for large BTC moves."""
        try:
            response = requests.get(f"{self.base_url}/mempool/recent", timeout=10)
            response.raise_for_status()
            transactions = response.json()
            
            new_whales = []
            for tx in transactions:
                btc_value = tx.get("value", 0) / 100_000_000
                
                if btc_value >= self.whale_threshold_btc:
                    whale_entry = {
                        "txid": tx.get("txid"),
                        "amount_btc": round(btc_value, 2),
                        "amount_usd": round(btc_value * current_btc_price, 2),
                        "fee_rate": tx.get("feeRate", 0),
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    new_whales.append(whale_entry)
            
            # Keep unique whales, limit to last 10
            for nw in new_whales:
                if nw["txid"] not in [w["txid"] for w in self.recent_whales]:
                    self.recent_whales.insert(0, nw)
            
            self.recent_whales = self.recent_whales[:10]
            return self.recent_whales
            
        except Exception as e:
            logging.error(f"Error fetching recent mempool transactions: {e}")
            return self.recent_whales

    def calculate_whale_pressure(self, whales, mempool_factor):
        """
        Calculate net whale pressure.
        
        Without L2 order book data, we estimate pressure using:
        - High fee rate = urgency = likely selling (need quick confirmation)
        - Low fee rate = patience = likely accumulating
        - High mempool = congestion = selling pressure
        
        Returns: float between -1 (buy pressure) and +1 (sell pressure)
        """
        if not whales:
            return 0.0
        
        try:
            # Average fee rate of whale transactions
            avg_fee = sum(w.get("fee_rate", 10) for w in whales) / len(whales)
            
            # High fee (>50 sat/vB) suggests urgency = sell pressure
            # Low fee (<10 sat/vB) suggests patience = buy pressure
            fee_pressure = (avg_fee - 30) / 100  # Centered around 30 sat/vB
            fee_pressure = max(-1.0, min(1.0, fee_pressure))
            
            # Combine with mempool congestion
            # High congestion + high whales = sell pressure
            combined_pressure = 0.6 * fee_pressure + 0.4 * mempool_factor
            
            return max(-1.0, min(1.0, combined_pressure))
            
        except Exception as e:
            logging.error(f"Error calculating whale pressure: {e}")
            return 0.0

    def get_whale_alert_status(self):
        """Check if API is reachable."""
        try:
            response = requests.get(f"{self.base_url}/mempool", timeout=5)
            return response.status_code == 200
        except:
            return False

if __name__ == "__main__":
    monitor = WhaleMonitor()
    stats = monitor.fetch_mempool_stats()
    whales = monitor.fetch_recent_whales(78000)
    pressure = monitor.calculate_whale_pressure(whales, stats)
    print(f"Mempool Factor: {stats:.2f}")
    print(f"Recent Whales: {len(whales)}")
    print(f"Whale Pressure: {pressure:.2f}")
