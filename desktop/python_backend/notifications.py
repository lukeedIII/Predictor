from plyer import notification
import logging
import config
import os
import requests

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NotificationHub:
    def __init__(self):
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    def send_desktop_notify(self, title, message):
        """Sends a Windows Toast notification."""
        try:
            notification.notify(
                title=title,
                message=message,
                app_name='Nexus-Prediction',
                timeout=10
            )
            logging.info(f"Desktop Notification sent: {title}")
        except Exception as e:
            logging.error(f"Failed to send desktop notification: {e}")

    def send_telegram(self, message):
        """Sends a telegram message if token is provided."""
        if not self.telegram_token or not self.telegram_chat_id:
            return
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {"chat_id": self.telegram_chat_id, "text": message}
            requests.post(url, json=payload, timeout=5)
            logging.info("Telegram notification sent.")
        except Exception as e:
            logging.error(f"Failed to send Telegram message: {e}")

    def alert_high_confidence(self, direction, confidence):
        """Alerts when prediction confidence > 80%."""
        title = "🔥 High Confidence Prediction"
        msg = f"Nexus predicts {direction} movement with {confidence}% confidence!"
        self.send_desktop_notify(title, msg)
        self.send_telegram(f"Nexus Alert: {msg}")

    def alert_mega_whale(self, btc_amount, usd_value, txid):
        """Alerts when a whale move > 1000 BTC is detected."""
        title = "🐋 MEGA WHALE DETECTED"
        msg = f"Alert: {btc_amount} BTC (${usd_value:,}) moved in Mempool!"
        self.send_desktop_notify(title, msg)
        self.send_telegram(f"Whale Alert: {msg}\nCheck: https://mempool.space/tx/{txid}")

if __name__ == "__main__":
    hub = NotificationHub()
    hub.send_desktop_notify("Nexus Init", "Notification system is active.")
