"""
Telegram Notification Service for Nexus Shadow-Quant Paper Trader
=================================================================
Premium-grade trade alerts, partial exits, and hourly P&L summaries
with integrated news headlines and market links.

Setup:
1. Create a bot via @BotFather on Telegram → get Bot Token
2. Send a message to the bot, then use https://api.telegram.org/bot<TOKEN>/getUpdates to find your Chat ID
3. Enter both in Settings → Telegram section
"""

import os
import logging
import threading
import time
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logger = logging.getLogger(__name__)

# ── Branding ──
BRAND = "⚡ Nexus Shadow-Quant"
CHART_LINK = "https://www.tradingview.com/chart/?symbol=BINANCE:BTCUSDT"
FEAR_GREED_LINK = "https://alternative.me/crypto/fear-and-greed-index/"

# ── Emoji maps ──
REGIME_EMOJI = {
    "BULL": "🐂", "BEAR": "🐻", "SIDEWAYS": "〰️", "UNKNOWN": "❓",
    "VOLATILE": "🌪️", "BREAKOUT": "🚀",
}
REASON_EMOJI = {
    "TAKE_PROFIT": "🎯", "STOP_LOSS": "🛑", "TRAILING_SL": "🎢",
    "MAX_HOLD": "⏰", "LIQUIDATED": "💀", "MANUAL": "🖐️",
    "CIRCUIT_BREAKER": "⚡", "TP1_PARTIAL": "✂️",
}
CONFIDENCE_BARS = {
    range(0, 20): "░░░░░", range(20, 40): "▓░░░░",
    range(40, 60): "▓▓░░░", range(60, 80): "▓▓▓░░",
    range(80, 101): "▓▓▓▓▓",
}

# ── Motivational lines ──
WIN_QUOTES = [
    "Printing money! 💸", "Clean execution 🎯", "The bot eats 🍽️",
    "Alpha captured 📡", "Institutional vibes 🏦", "Cash register goes brrr 🖨️",
    "EZ clap 👏", "Nailed it 🔨",
]
LOSS_QUOTES = [
    "Cost of doing business 📉", "Risk managed ✅", "Small hit, move on 💪",
    "Learning the market 📚", "Controlled loss, stay focused 🧠",
    "Drawdown absorbed 🛡️",
]


def _conf_bars(conf: float) -> str:
    """Return visual bars for confidence level."""
    c = int(conf)
    for rng, bars in CONFIDENCE_BARS.items():
        if c in rng:
            return bars
    return "▓▓▓░░"


def _fetch_latest_news(max_items: int = 3) -> List[Dict]:
    """Fetch latest news headlines from the running API server."""
    try:
        url = "http://127.0.0.1:8420/api/news"
        if HAS_HTTPX:
            with httpx.Client(timeout=5) as client:
                r = client.get(url)
                if r.status_code == 200:
                    data = r.json()
                    items = data.get('items', [])
                    # Filter to only real news (not market signals)
                    news = [i for i in items if i.get('source') not in ('Market Signal', 'Market', None)]
                    return news[:max_items]
        else:
            import urllib.request
            import json
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                items = data.get('items', [])
                news = [i for i in items if i.get('source') not in ('Market Signal', 'Market', None)]
                return news[:max_items]
    except Exception as e:
        logger.debug(f"News fetch for Telegram failed: {e}")
    return []


def _format_news_block(news: List[Dict]) -> str:
    """Format news headlines into Telegram-friendly block."""
    if not news:
        return ""
    
    sentiment_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
    lines = ["\n📰 <b>Latest Headlines</b>"]
    for item in news:
        headline = item.get('headline', '')[:80]
        source = item.get('source', '')
        url = item.get('url', '')
        sent = item.get('sentiment', 'neutral')
        t = item.get('time', '')
        
        se = sentiment_emoji.get(sent, "⚪")
        time_str = f" · {t}" if t else ""
        
        if url:
            lines.append(f"{se} <a href=\"{url}\">{headline}</a> <i>— {source}{time_str}</i>")
        else:
            lines.append(f"{se} {headline} <i>— {source}{time_str}</i>")
    
    return "\n".join(lines)


def _price_move_emoji(entry: float, exit: float, direction: str) -> str:
    """Get emoji showing price movement magnitude."""
    pct = abs(exit - entry) / entry * 100
    if pct > 1.0:
        return "🔥🔥🔥"
    elif pct > 0.5:
        return "🔥🔥"
    elif pct > 0.2:
        return "🔥"
    return "📊"


class TelegramNotifier:
    """
    Premium-grade Telegram notification service.
    Sends beautifully formatted trade alerts with news + market context.
    """
    
    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"
    
    def __init__(self):
        self._token: Optional[str] = None
        self._chat_id: Optional[str] = None
        self._enabled = False
        self._last_hourly: Optional[datetime] = None
        self._hourly_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._stats_fn = None
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
        token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
        chat_id = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
        if token and chat_id and len(token) > 10:
            self._token = token
            self._chat_id = chat_id
            self._enabled = True
        return self._enabled
    
    def send_message(self, text: str, parse_mode: str = "HTML",
                     disable_preview: bool = False) -> bool:
        """Send a message to the configured Telegram chat."""
        if not self.is_enabled:
            return False
        
        url = self.BASE_URL.format(token=self._token)
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_preview,
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
                import urllib.request
                import json
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(url, data=data,
                                            headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
    
    # ========== TRADE OPEN ==========
    
    def notify_trade_open(self, direction: str, price: float, size_usd: float,
                          leverage: int, confidence: float, regime: str,
                          tp: float, sl: float, tp1: float = None):
        """Send a premium trade open notification with news context."""
        
        dir_emoji = "🟢 📈" if direction == "LONG" else "🔴 📉"
        regime_e = REGIME_EMOJI.get(regime, "❓")
        bars = _conf_bars(confidence)
        
        # Risk/Reward calculation
        if direction == "LONG":
            risk = abs(price - sl)
            reward = abs(tp - price)
        else:
            risk = abs(sl - price)
            reward = abs(price - tp)
        rr = reward / risk if risk > 0 else 0
        
        tp1_line = ""
        if tp1:
            tp1_line = f"│  ├ TP1 (partial):  <code>${tp1:,.2f}</code>\n"
        
        msg = (
            f"{dir_emoji}  <b>NEW {direction} POSITION</b>\n"
            f"{'═' * 26}\n"
            f"\n"
            f"💲 <b>Entry Price</b>\n"
            f"│  └ <code>${price:,.2f}</code>\n"
            f"\n"
            f"📐 <b>Position Details</b>\n"
            f"│  ├ Size:      <code>${size_usd:,.2f}</code>\n"
            f"│  ├ Leverage:  <code>{leverage}x</code>\n"
            f"│  └ R:R Ratio: <code>{rr:.1f}:1</code>\n"
            f"\n"
            f"🧠 <b>Signal Intelligence</b>\n"
            f"│  ├ Confidence: <code>{confidence:.0f}%</code> [{bars}]\n"
            f"│  └ Regime:     {regime_e} <code>{regime}</code>\n"
            f"\n"
            f"🎯 <b>Targets</b>\n"
            f"│  ├ Take Profit: <code>${tp:,.2f}</code>\n"
            f"{tp1_line}"
            f"│  └ Stop Loss:   <code>${sl:,.2f}</code>\n"
            f"\n"
            f"<a href=\"{CHART_LINK}\">📊 View Live Chart</a>\n"
            f"\n"
            f"⏱ <i>{datetime.now().strftime('%b %d, %H:%M:%S')}</i>  ·  {BRAND}"
        )
        
        # Fetch news in the same thread (it's already background)
        def _send():
            news = _fetch_latest_news(2)
            full_msg = msg
            if news:
                full_msg += "\n" + _format_news_block(news)
            self.send_message(full_msg)
        
        threading.Thread(target=_send, daemon=True).start()
    
    # ========== TRADE CLOSE ==========
    
    def notify_trade_close(self, direction: str, entry_price: float, exit_price: float,
                           pnl_usd: float, pnl_pct: float, reason: str,
                           balance_after: float, size_usd: float = 0):
        """Send a premium trade close notification."""
        
        won = pnl_usd >= 0
        header_emoji = "✅ 💰" if won else "❌ 💸"
        result_word = "WIN" if won else "LOSS"
        pnl_color = "+" if won else ""
        reason_e = REASON_EMOJI.get(reason, "📌")
        move_e = _price_move_emoji(entry_price, exit_price, direction)
        quote = random.choice(WIN_QUOTES if won else LOSS_QUOTES)
        
        # Price movement
        price_delta = exit_price - entry_price
        price_pct = (price_delta / entry_price) * 100
        
        msg = (
            f"{header_emoji}  <b>TRADE CLOSED — {result_word}</b>\n"
            f"{'═' * 26}\n"
            f"\n"
            f"<i>\"{quote}\"</i>\n"
            f"\n"
            f"📋 <b>Trade Summary</b>\n"
            f"│  ├ Direction: <code>{direction}</code>\n"
            f"│  ├ Entry:     <code>${entry_price:,.2f}</code>\n"
            f"│  ├ Exit:      <code>${exit_price:,.2f}</code>  {move_e}\n"
            f"│  └ Move:      <code>{price_pct:+.3f}%</code>\n"
            f"\n"
            f"💰 <b>Result</b>\n"
            f"│  ├ PnL:       <code>${pnl_usd:+,.2f}</code>\n"
            f"│  ├ Return:    <code>{pnl_pct:+.2f}%</code>\n"
            f"│  └ Reason:    {reason_e} <code>{reason}</code>\n"
            f"\n"
            f"🏦 <b>Account</b>\n"
            f"│  └ Balance:   <code>${balance_after:,.2f}</code>\n"
            f"\n"
            f"<a href=\"{CHART_LINK}\">📊 View Chart</a>\n"
            f"\n"
            f"⏱ <i>{datetime.now().strftime('%b %d, %H:%M:%S')}</i>  ·  {BRAND}"
        )
        
        threading.Thread(target=self.send_message, args=(msg,), daemon=True).start()
    
    # ========== PARTIAL EXIT ==========
    
    def notify_partial_close(self, direction: str, price: float,
                             closed_size: float, pnl: float, remaining: float):
        """Send a partial exit (TP1) notification."""
        
        won = pnl >= 0
        pnl_emoji = "💰" if won else "💸"
        
        msg = (
            f"✂️ ⚡ <b>PARTIAL EXIT — {direction}</b>\n"
            f"{'═' * 26}\n"
            f"\n"
            f"│  ├ Price:     <code>${price:,.2f}</code>\n"
            f"│  ├ Closed:    <code>${closed_size:,.2f}</code>\n"
            f"│  ├ {pnl_emoji} PnL:    <code>${pnl:+,.2f}</code>\n"
            f"│  ├ Remaining: <code>${remaining:,.2f}</code>\n"
            f"│  └ SL moved:  <b>→ Breakeven</b> 🛡️\n"
            f"\n"
            f"<i>50% secured, letting the rest ride! 🏄</i>\n"
            f"\n"
            f"⏱ <i>{datetime.now().strftime('%b %d, %H:%M:%S')}</i>  ·  {BRAND}"
        )
        
        threading.Thread(target=self.send_message, args=(msg,), daemon=True).start()
    
    # ========== HOURLY SUMMARY ==========
    
    def send_hourly_summary(self, stats: Dict):
        """Send premium hourly P&L summary with news digest."""
        
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
        streak = stats.get('current_streak', 0)
        best_trade = stats.get('best_trade_pnl', 0)
        worst_trade = stats.get('worst_trade_pnl', 0)
        
        # Health indicator
        if total_pnl > 0:
            health = "🟢"
            performance = "Profitable"
        elif total_pnl > -100:
            health = "🟡"
            performance = "Neutral"
        else:
            health = "🔴"
            performance = "In Drawdown"
        
        # Win rate grade
        if win_rate >= 60:
            wr_grade = "🔥 Excellent"
        elif win_rate >= 50:
            wr_grade = "✅ Good"
        elif win_rate >= 40:
            wr_grade = "⚠️ Fair"
        else:
            wr_grade = "🚨 Needs review"
        
        # Streak display
        if streak > 0:
            streak_str = f"🔥 {streak}W streak"
        elif streak < 0:
            streak_str = f"❄️ {abs(streak)}L streak"
        else:
            streak_str = "—"
        
        # ROI from starting balance
        roi = ((balance - starting) / starting * 100) if starting else 0
        
        # Progress bar for balance (relative to starting)
        progress_pct = min(max(balance / starting, 0.5), 1.5)
        filled = int((progress_pct - 0.5) / 1.0 * 10)
        bar = "▓" * max(filled, 0) + "░" * max(10 - filled, 0)
        
        msg = (
            f"📊  <b>HOURLY REPORT</b>  {health}\n"
            f"{'═' * 26}\n"
            f"\n"
            f"🏦 <b>Account Overview</b>\n"
            f"│  ├ Balance:     <code>${balance:,.2f}</code>\n"
            f"│  ├ Total PnL:   <code>${total_pnl:+,.2f}</code> ({total_pnl_pct:+.2f}%)\n"
            f"│  ├ ROI:         <code>{roi:+.2f}%</code>\n"
            f"│  ├ Unrealized:  <code>${unrealized:+,.2f}</code>\n"
            f"│  └ [{bar}] {performance}\n"
            f"\n"
            f"📈 <b>Performance</b>\n"
            f"│  ├ Trades:      <code>{total_trades}</code> (✅{winning} / ❌{losing})\n"
            f"│  ├ Win Rate:    <code>{win_rate:.1f}%</code>  {wr_grade}\n"
            f"│  ├ Profit F:    <code>{pf:.2f}x</code>\n"
            f"│  ├ Sharpe:      <code>{sharpe:.2f}</code>\n"
            f"│  ├ Streak:      {streak_str}\n"
            f"│  └ Drawdown:    <code>{drawdown:.2f}%</code>\n"
            f"\n"
            f"💎 <b>Highlights</b>\n"
            f"│  ├ Best Trade:  <code>${best_trade:+,.2f}</code>\n"
            f"│  ├ Worst Trade: <code>${worst_trade:+,.2f}</code>\n"
            f"│  ├ Fees Paid:   <code>${fees:.2f}</code>\n"
            f"│  └ Open Pos:    <code>{positions}</code>\n"
            f"\n"
            f"<a href=\"{CHART_LINK}\">📊 BTC Chart</a>  ·  "
            f"<a href=\"{FEAR_GREED_LINK}\">😱 Fear/Greed</a>\n"
        )
        
        # Add news digest
        news = _fetch_latest_news(3)
        if news:
            msg += _format_news_block(news)
            msg += "\n"
        
        msg += (
            f"\n⏱ <i>{datetime.now().strftime('%b %d, %Y — %H:%M')}</i>\n"
            f"{BRAND}"
        )
        
        self.send_message(msg)
    
    # ========== CIRCUIT BREAKER ==========
    
    def notify_circuit_breaker(self, balance: float, peak: float, drawdown_pct: float):
        """Alert when circuit breaker activates."""
        msg = (
            f"🚨🚨🚨 <b>CIRCUIT BREAKER ACTIVATED</b> 🚨🚨🚨\n"
            f"{'═' * 26}\n"
            f"\n"
            f"The bot has stopped trading due to excessive drawdown.\n"
            f"\n"
            f"│  ├ Balance:  <code>${balance:,.2f}</code>\n"
            f"│  ├ Peak:     <code>${peak:,.2f}</code>\n"
            f"│  └ Drawdown: <code>{drawdown_pct:.1f}%</code> ⚠️\n"
            f"\n"
            f"<i>Trading will resume when balance recovers above 90% of peak.</i>\n"
            f"\n"
            f"⏱ <i>{datetime.now().strftime('%b %d, %H:%M:%S')}</i>  ·  {BRAND}"
        )
        threading.Thread(target=self.send_message, args=(msg,), daemon=True).start()
    
    # ========== DAILY SUMMARY ==========
    
    def send_daily_summary(self, stats: Dict):
        """Send end-of-day recap (called at midnight)."""
        
        balance = stats.get('balance', 0)
        total_pnl = stats.get('total_pnl', 0)
        total_trades = stats.get('total_trades', 0)
        winning = stats.get('winning_trades', 0)
        losing = stats.get('losing_trades', 0)
        win_rate = stats.get('win_rate', 0)
        fees = stats.get('total_fees', 0)
        
        grade = "A+" if win_rate >= 65 else "A" if win_rate >= 55 else "B" if win_rate >= 45 else "C" if win_rate >= 35 else "F"
        
        msg = (
            f"🌙 <b>DAILY RECAP</b>  ·  Grade: <b>{grade}</b>\n"
            f"{'═' * 26}\n"
            f"\n"
            f"│  ├ Balance:   <code>${balance:,.2f}</code>\n"
            f"│  ├ Total PnL: <code>${total_pnl:+,.2f}</code>\n"
            f"│  ├ Trades:    <code>{total_trades}</code> (✅{winning} / ❌{losing})\n"
            f"│  ├ Win Rate:  <code>{win_rate:.1f}%</code>\n"
            f"│  └ Fees:      <code>${fees:.2f}</code>\n"
            f"\n"
            f"<i>See you tomorrow! 💤</i>\n"
            f"\n"
            f"⏱ <i>{datetime.now().strftime('%b %d, %Y')}</i>  ·  {BRAND}"
        )
        
        self.send_message(msg)
    
    # ========== HOURLY SCHEDULER ==========
    
    def start_hourly_loop(self, stats_callback):
        """Start background thread that sends hourly summaries."""
        self._stats_fn = stats_callback
        if self._hourly_thread and self._hourly_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._hourly_thread = threading.Thread(target=self._hourly_worker, daemon=True)
        self._hourly_thread.start()
        logger.info("Telegram hourly summary scheduler started")
    
    def stop_hourly_loop(self):
        """Stop the hourly summary scheduler."""
        self._stop_event.set()
    
    def _hourly_worker(self):
        """Background worker: hourly summaries + midnight daily recap."""
        while not self._stop_event.is_set():
            now = datetime.now()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            wait_secs = (next_hour - now).total_seconds()
            
            # Wait (check stop_event every 30s)
            waited = 0
            while waited < wait_secs and not self._stop_event.is_set():
                time.sleep(min(30, wait_secs - waited))
                waited += 30
            
            if self._stop_event.is_set():
                break
            
            if self.is_enabled and self._stats_fn:
                try:
                    stats = self._stats_fn()
                    
                    # Send daily recap at midnight
                    if datetime.now().hour == 0:
                        self.send_daily_summary(stats)
                    else:
                        self.send_hourly_summary(stats)
                    
                    logger.info("Telegram summary sent")
                except Exception as e:
                    logger.error(f"Telegram summary failed: {e}")
    
    # ========== TEST ==========
    
    def test_connection(self) -> Dict:
        """Test the Telegram bot connection. Returns status dict."""
        if not self.is_enabled:
            return {"ok": False, "error": "Telegram not configured (missing token or chat_id)"}
        
        test_msg = (
            f"🤖  <b>{BRAND}</b>\n"
            f"{'═' * 26}\n"
            f"\n"
            f"✅ <b>Connection Successful!</b>\n"
            f"\n"
            f"You will receive:\n"
            f"│  ├ 🟢 Trade open alerts\n"
            f"│  ├ ✅ Trade close alerts\n"
            f"│  ├ ✂️  Partial exit alerts\n"
            f"│  ├ 📊 Hourly P&L summaries\n"
            f"│  ├ 🌙 Daily recaps at midnight\n"
            f"│  ├ 🚨 Circuit breaker warnings\n"
            f"│  └ 📰 News headlines with links\n"
            f"\n"
            f"<a href=\"{CHART_LINK}\">📊 BTC/USDT Chart</a>  ·  "
            f"<a href=\"{FEAR_GREED_LINK}\">😱 Fear & Greed Index</a>\n"
            f"\n"
            f"⏱ <i>{datetime.now().strftime('%b %d, %Y — %H:%M:%S')}</i>"
        )
        
        ok = self.send_message(test_msg)
        if ok:
            return {"ok": True, "message": "Test message sent! Check your Telegram."}
        else:
            return {"ok": False, "error": "Failed to send — check Bot Token and Chat ID"}


# Singleton instance
telegram = TelegramNotifier()
