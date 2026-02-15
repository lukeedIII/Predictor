"""
Telegram Notification Service for Nexus Shadow-Quant Paper Trader
=================================================================
Sends real-time trade alerts and hourly P&L summaries via Telegram Bot API.

Setup:
1. Create a bot via @BotFather on Telegram → get Bot Token
2. Send a message to the bot, then use https://api.telegram.org/bot<TOKEN>/getUpdates to find your Chat ID
3. Enter both in Settings → Telegram section
"""

import os
import logging
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends paper trading notifications to Telegram.
    Uses the Telegram Bot API (no dependencies beyond httpx/urllib).
    """
    
    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"
    
    def __init__(self):
        self._token: Optional[str] = None
        self._chat_id: Optional[str] = None
        self._enabled = False
        self._last_hourly: Optional[datetime] = None
        self._hourly_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stats_fn = None  # Callback to get current stats
        self._reload_credentials()
    
    def _reload_credentials(self):
        """Reload Telegram credentials from environment."""
        self._token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
        self._chat_id = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
        self._enabled = bool(self._token and self._chat_id and len(self._token) > 10)
        if self._enabled:
            logger.info("Telegram notifier: ENABLED ✅")
        else:
            logger.info("Telegram notifier: disabled (no token/chat_id configured)")
    
    @property
    def is_enabled(self) -> bool:
        """Check if Telegram is configured and ready."""
        # Re-check env vars in case they were updated via Settings
        token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
        chat_id = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
        if token and chat_id and len(token) > 10:
            self._token = token
            self._chat_id = chat_id
            self._enabled = True
        return self._enabled
    
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured Telegram chat."""
        if not self.is_enabled:
            return False
        
        url = self.BASE_URL.format(token=self._token)
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        
        try:
            if HAS_HTTPX:
                with httpx.Client(timeout=10) as client:
                    r = client.post(url, json=payload)
                    if r.status_code == 200:
                        logger.debug("Telegram message sent successfully")
                        return True
                    else:
                        logger.warning(f"Telegram API error {r.status_code}: {r.text[:200]}")
                        return False
            else:
                # Fallback to urllib (no external deps)
                import urllib.request
                import json
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        return True
                    return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
    
    # ========== TRADE NOTIFICATIONS ==========
    
    def notify_trade_open(self, direction: str, price: float, size_usd: float,
                          leverage: int, confidence: float, regime: str,
                          tp: float, sl: float, tp1: float = None):
        """Send a trade open notification."""
        emoji = "🟢" if direction == "LONG" else "🔴"
        tp1_line = f"\n├ TP1: <code>${tp1:,.2f}</code> (partial exit)" if tp1 else ""
        
        msg = (
            f"{emoji} <b>TRADE OPENED: {direction}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"├ Entry: <code>${price:,.2f}</code>\n"
            f"├ Size: <code>${size_usd:,.2f}</code> ({leverage}x)\n"
            f"├ Confidence: <code>{confidence:.0f}%</code>\n"
            f"├ Regime: <code>{regime}</code>\n"
            f"├ TP: <code>${tp:,.2f}</code>{tp1_line}\n"
            f"└ SL: <code>${sl:,.2f}</code>\n"
            f"\n⏱ {datetime.now().strftime('%H:%M:%S')}"
        )
        
        # Send in background to not block the trading loop
        threading.Thread(target=self.send_message, args=(msg,), daemon=True).start()
    
    def notify_trade_close(self, direction: str, entry_price: float, exit_price: float,
                           pnl_usd: float, pnl_pct: float, reason: str,
                           balance_after: float, size_usd: float = 0):
        """Send a trade close notification."""
        won = pnl_usd >= 0
        emoji = "✅" if won else "❌"
        pnl_emoji = "💰" if won else "💸"
        
        msg = (
            f"{emoji} <b>TRADE CLOSED: {direction}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"├ Entry: <code>${entry_price:,.2f}</code>\n"
            f"├ Exit: <code>${exit_price:,.2f}</code>\n"
            f"├ {pnl_emoji} PnL: <code>${pnl_usd:+,.2f}</code> ({pnl_pct:+.2f}%)\n"
            f"├ Reason: <code>{reason}</code>\n"
            f"└ Balance: <code>${balance_after:,.2f}</code>\n"
            f"\n⏱ {datetime.now().strftime('%H:%M:%S')}"
        )
        
        threading.Thread(target=self.send_message, args=(msg,), daemon=True).start()
    
    def notify_partial_close(self, direction: str, price: float, 
                             closed_size: float, pnl: float, remaining: float):
        """Send a partial exit notification."""
        msg = (
            f"⚡ <b>PARTIAL EXIT: {direction}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"├ Price: <code>${price:,.2f}</code>\n"
            f"├ Closed: <code>${closed_size:,.2f}</code>\n"
            f"├ PnL: <code>${pnl:+,.2f}</code>\n"
            f"└ Remaining: <code>${remaining:,.2f}</code>\n"
            f"\n⏱ {datetime.now().strftime('%H:%M:%S')}"
        )
        
        threading.Thread(target=self.send_message, args=(msg,), daemon=True).start()
    
    def send_hourly_summary(self, stats: Dict):
        """Send hourly P&L summary."""
        balance = stats.get('balance', 0)
        starting = stats.get('starting_balance', 10000)
        total_pnl = stats.get('total_pnl', 0)
        total_pnl_pct = stats.get('total_pnl_pct', 0)
        win_rate = stats.get('win_rate', 0)
        total_trades = stats.get('total_trades', 0)
        winning = stats.get('winning_trades', 0)
        losing = stats.get('losing_trades', 0)
        drawdown = stats.get('max_drawdown_pct', 0)
        positions = stats.get('positions_count', 0)
        unrealized = stats.get('unrealized_pnl', 0)
        sharpe = stats.get('net_sharpe_ratio', 0)
        fees = stats.get('total_fees', 0)
        pf = stats.get('profit_factor', 0)
        
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        health = "🟢" if total_pnl >= 0 else ("🟡" if total_pnl > -100 else "🔴")
        
        msg = (
            f"📊 <b>HOURLY SUMMARY</b> {health}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"├ {pnl_emoji} Balance: <code>${balance:,.2f}</code>\n"
            f"├ Total PnL: <code>${total_pnl:+,.2f}</code> ({total_pnl_pct:+.2f}%)\n"
            f"├ Unrealized: <code>${unrealized:+,.2f}</code>\n"
            f"├ Open Positions: <code>{positions}</code>\n"
            f"├ ──────────────\n"
            f"├ Trades: <code>{total_trades}</code> (W: {winning} / L: {losing})\n"
            f"├ Win Rate: <code>{win_rate:.1f}%</code>\n"
            f"├ Profit Factor: <code>{pf:.2f}</code>\n"
            f"├ Sharpe: <code>{sharpe:.2f}</code>\n"
            f"├ Drawdown: <code>{drawdown:.2f}%</code>\n"
            f"└ Fees Paid: <code>${fees:.2f}</code>\n"
            f"\n⏱ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        self.send_message(msg)
    
    # ========== HOURLY SCHEDULER ==========
    
    def start_hourly_loop(self, stats_callback):
        """Start background thread that sends hourly summaries."""
        self._stats_fn = stats_callback
        if self._hourly_thread and self._hourly_thread.is_alive():
            return  # Already running
        
        self._stop_event.clear()
        self._hourly_thread = threading.Thread(target=self._hourly_worker, daemon=True)
        self._hourly_thread.start()
        logger.info("Telegram hourly summary scheduler started")
    
    def stop_hourly_loop(self):
        """Stop the hourly summary scheduler."""
        self._stop_event.set()
    
    def _hourly_worker(self):
        """Background worker that sends hourly summaries."""
        while not self._stop_event.is_set():
            # Wait until the next hour mark (or 60 minutes if can't align)
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            wait_secs = (next_hour - now).total_seconds()
            
            # Wait (but check stop_event every 30s)
            waited = 0
            while waited < wait_secs and not self._stop_event.is_set():
                time.sleep(min(30, wait_secs - waited))
                waited += 30
            
            if self._stop_event.is_set():
                break
            
            # Send hourly summary if enabled
            if self.is_enabled and self._stats_fn:
                try:
                    stats = self._stats_fn()
                    self.send_hourly_summary(stats)
                    logger.info("Telegram hourly summary sent")
                except Exception as e:
                    logger.error(f"Telegram hourly summary failed: {e}")
    
    def test_connection(self) -> Dict:
        """Test the Telegram bot connection. Returns status dict."""
        if not self.is_enabled:
            return {"ok": False, "error": "Telegram not configured (missing token or chat_id)"}
        
        test_msg = (
            "🤖 <b>Nexus Shadow-Quant</b>\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "✅ Telegram connection test successful!\n"
            "You will receive trade alerts and hourly summaries here.\n"
            f"\n⏱ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        ok = self.send_message(test_msg)
        if ok:
            return {"ok": True, "message": "Test message sent! Check your Telegram."}
        else:
            return {"ok": False, "error": "Failed to send — check Bot Token and Chat ID"}


# Singleton instance
telegram = TelegramNotifier()
