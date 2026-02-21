import os
import sys
import time
import json
import logging
import threading
import requests
from datetime import datetime, timezone

# Adjust sys.path to ensure we can import parent directory modules without issue
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MacroCalendar')

STATE_FILE = os.path.join(os.path.dirname(__file__), 'latest_macro_events.json')

# We only care about USD High-impact events for BTC volatility
TARGET_CURRENCY = "USD"
TARGET_IMPACT = "High"

# How many minutes before/after an event should the circuit breaker trip?
CIRCUIT_BREAKER_PRE_MINUTES = 30
CIRCUIT_BREAKER_POST_MINUTES = 15

class MacroCalendarWorker:
    """
    Dedicated background worker for querying economic calendars.
    Fetches the weekly calendar periodically (e.g. every 4 hours)
    and evaluates minute-by-minute if we are in a 'volatility circuit breaker'
    zone due to high-impact macro announcements (CPI, FOMC, NFP).
    """
    def __init__(self, fetch_interval_hours: int = 4):
        self.fetch_interval_hours = max(1, fetch_interval_hours)
        self.stop_event = threading.Event()
        self.thread = None
        self.events = []
        
    def start(self):
        """Start the background worker thread."""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Macro worker already running.")
            return
            
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, name="MacroCalendarWorker", daemon=True)
        self.thread.start()
        logger.info(f"Macro calendar worker started (fetch interval: {self.fetch_interval_hours}h)")
        
    def stop(self):
        """Signal the worker to stop and wait for it to join."""
        if self.thread is not None:
            logger.info("Stopping Macro worker...")
            self.stop_event.set()
            self.thread.join(timeout=5)
            self.thread = None
            
    def _run_loop(self):
        """Main loop: occasionally fetch calendar, continuously evaluate proximity."""
        
        # Initial fetch
        self._fetch_calendar()
        
        last_fetch_time = time.time()
        
        while not self.stop_event.is_set():
            now_time = time.time()
            
            # Fetch calendar every X hours
            if now_time - last_fetch_time > self.fetch_interval_hours * 3600:
                self._fetch_calendar()
                last_fetch_time = now_time
                
            # Evaluate proximity to events every 15 seconds
            self._evaluate_circuit_breaker()
            
            # Wait 15 seconds
            for _ in range(15):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
                
    def _fetch_calendar(self):
        """Fetch weekly FF calendar JSON."""
        try:
            logger.info("Fetching economic calendar...")
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                parsed_events = []
                
                for item in data:
                    country = item.get("country", "")
                    impact = item.get("impact", "")
                    
                    if country == TARGET_CURRENCY and impact == TARGET_IMPACT:
                        # Date comes in ISO-ish format: "2024-03-12T08:30:00-04:00"
                        # We need it robustly parsed to UTC timestamp
                        try:
                            date_str = item.get("date", "")
                            # Python 3.11+ handles the colon in timezone. For universal compat, strip it or use fromisoformat
                            event_time = datetime.fromisoformat(date_str)
                            # Convert to local timestamp for easy comparison
                            event_ts = event_time.timestamp()
                            
                            parsed_events.append({
                                "title": item.get("title", ""),
                                "time": event_ts,
                                "date_str": date_str
                            })
                        except Exception as dt_e:
                            logger.debug(f"Date parse error: {dt_e}")
                
                self.events = sorted(parsed_events, key=lambda x: x["time"])
                logger.info(f"Loaded {len(self.events)} high-impact USD events for the week.")
            else:
                logger.warning(f"Calendar fetch failed with status {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching calendar: {e}")
            
    def _evaluate_circuit_breaker(self):
        """Check if current time is within [event - pre_mins, event + post_mins]."""
        now = time.time()
        
        closest_event = None
        min_diff = float("inf")
        circuit_breaker = False
        circuit_breaker_reason = ""
        
        for event in self.events:
            event_time = event["time"]
            diff_minutes = (event_time - now) / 60.0
            
            # Find the closest event in the future (or very recently past)
            if diff_minutes > -CIRCUIT_BREAKER_POST_MINUTES and diff_minutes < min_diff:
                min_diff = diff_minutes
                closest_event = event
                
            # Check if circuit breaker is tripped
            if -CIRCUIT_BREAKER_POST_MINUTES <= diff_minutes <= CIRCUIT_BREAKER_PRE_MINUTES:
                circuit_breaker = True
                circuit_breaker_reason = f"High Impact: {event['title']}"
                
        # Calculate minutes until next major event (cap at 9999 for neutral)
        event_proximity = min_diff if closest_event and min_diff > 0 else 9999.0
        
        payload = {
            "timestamp": datetime.now().isoformat(),
            "volatility_circuit_breaker": circuit_breaker,
            "circuit_breaker_reason": circuit_breaker_reason,
            "event_proximity_minutes": round(float(event_proximity), 1),
            "next_event_title": closest_event["title"] if closest_event else "None",
            "status": "active"
        }
        
        self._save_state_atomically(payload)
        
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
    worker = MacroCalendarWorker()
    worker._fetch_calendar()
    worker._evaluate_circuit_breaker()
    print("MACRO CALENDAR TEST RUN COMPLETE. Check latest_macro_events.json")
