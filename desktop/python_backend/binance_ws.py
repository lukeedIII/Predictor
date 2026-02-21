"""
Binance WebSocket Client — Real-Time Market Data
==================================================
Connects to Binance public WebSocket streams for sub-second BTC price updates.
No API key required — these are public data streams.

Streams used (combined):
  - btcusdt@trade   → individual trades (price ticks)
  - btcusdt@ticker  → 24h rolling window stats (high/low/volume/change)

Requires: pip install websocket-client
"""

import json
import math
import time
import threading
import logging
import pandas as pd
from collections import deque
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────
COMBINED_STREAM_URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@trade/btcusdt@ticker/btcusdt@depth20@100ms"
MAX_BACKOFF_SEC = 30
BASE_DELAY_SEC = 1


class BinanceWSClient:
    """Real-time Binance market data via WebSocket.
    
    Thread-safe — runs in a daemon thread using `websocket-client` library.
    Access latest data via properties from any thread (GIL-safe reads).
    
    Usage:
        client = BinanceWSClient()
        client.start()
        print(client.price)       # Latest BTC/USDT price
        print(client.change_pct)  # 24h change %
        print(client.snapshot)    # Full data dict for API responses
    """

    def __init__(self, on_price_update: Optional[Callable[[float], None]] = None):
        # ─── Latest price data ────────────────────────
        self._price: float = 0.0
        self._bid: float = 0.0
        self._ask: float = 0.0
        self._last_trade_time: float = 0.0

        # ─── 24h ticker stats ─────────────────────────
        self._change_24h: float = 0.0
        self._change_pct: float = 0.0
        self._high_24h: float = 0.0
        self._low_24h: float = 0.0
        self._volume_btc: float = 0.0
        self._volume_usdt: float = 0.0
        self._open_24h: float = 0.0

        # ─── Connection state ─────────────────────────
        self._connected: bool = False
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._ws_app = None
        self._backoff: float = BASE_DELAY_SEC
        self._reconnect_count: int = 0
        self._last_message_time: float = 0.0
        self._trade_count: int = 0  # trades received since start

        # ─── Phase 3: Microstructure tracking ─────────
        self._trade_timestamps: deque = deque(maxlen=600)  # last 60s of trades (capped)
        self._buy_volume_60s: deque = deque(maxlen=600)    # buy volume per trade
        self._sell_volume_60s: deque = deque(maxlen=600)   # sell volume per trade
        self._trade_ts_60s: deque = deque(maxlen=600)      # timestamps for volume deques

        # ─── Phase 4: 1-second OHLCV candle aggregation ──
        self._1s_candles: deque = deque(maxlen=7200)       # 2h of 1s candles
        self._current_1s: dict = {}                         # candle being built
        self._current_1s_sec: int = 0                       # epoch second of current candle

        # ─── Phase 4: L2 Order Book depth ─────────────
        self._bvd_ratio: float = 1.0       # Buy Volume Delta (Bid volume / Ask volume)
        self._wall_bids: float = 0.0       # Total volume of limit buy walls
        self._wall_asks: float = 0.0       # Total volume of limit sell walls
        self._bids_top5: float = 0.0       # Volume in top 5 bid levels
        self._asks_top5: float = 0.0       # Volume in top 5 ask levels

        # Callback for price updates (used by api_server to push to frontend WS)
        self._on_price_update = on_price_update

    # ═══════════════════════════════════════════════════
    #  PUBLIC PROPERTIES (thread-safe reads via GIL)
    # ═══════════════════════════════════════════════════

    @property
    def price(self) -> float:
        return self._price

    @property
    def bid(self) -> float:
        return self._bid

    @property
    def ask(self) -> float:
        return self._ask

    @property
    def change_24h(self) -> float:
        return self._change_24h

    @property
    def change_pct(self) -> float:
        return self._change_pct

    @property
    def high_24h(self) -> float:
        return self._high_24h

    @property
    def low_24h(self) -> float:
        return self._low_24h

    @property
    def volume_btc(self) -> float:
        return self._volume_btc

    @property
    def volume_usdt(self) -> float:
        return self._volume_usdt

    @property
    def open_24h(self) -> float:
        return self._open_24h

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def snapshot(self) -> dict:
        """Full market data snapshot for API responses."""
        # Compute live microstructure metrics
        now = time.time()
        cutoff = now - 60.0
        
        # Trades per second (60s window)
        recent_trades = sum(1 for t in self._trade_timestamps if t > cutoff)
        trades_per_sec = round(recent_trades / 60.0, 1)
        
        # Buy/sell ratio (60s window)
        buy_vol = sum(v for v, t in zip(self._buy_volume_60s, self._trade_ts_60s) if t > cutoff)
        sell_vol = sum(v for v, t in zip(self._sell_volume_60s, self._trade_ts_60s) if t > cutoff)
        buy_sell = round(buy_vol / (sell_vol + 1e-9), 2) if (buy_vol + sell_vol) > 0 else 1.0
        
        return {
            "price": self._price,
            "bid": self._bid,
            "ask": self._ask,
            "change_24h": self._change_24h,
            "change_pct": self._change_pct,
            "high_24h": self._high_24h,
            "low_24h": self._low_24h,
            "volume_btc": round(self._volume_btc, 2),
            "volume_usdt": round(self._volume_usdt, 2),
            "open_24h": self._open_24h,
            "ws_connected": self._connected,
            "last_update": self._last_message_time,
            "trades_received": self._trade_count,
            # Phase 3: Live microstructure
            "trades_per_sec": trades_per_sec,
            "buy_sell_ratio": buy_sell,
            "buy_volume_60s": round(buy_vol, 4),
            "sell_volume_60s": round(sell_vol, 4),
            # Phase 4: L2 Order Book snapshot
            "bvd_ratio": round(self._bvd_ratio, 3),
            "bids_top5": round(self._bids_top5, 2),
            "asks_top5": round(self._asks_top5, 2),
            "wall_bids": round(self._wall_bids, 2),
            "wall_asks": round(self._wall_asks, 2),
        }

    def get_1s_candles(self) -> 'pd.DataFrame':
        """Return accumulated 1-second OHLCV candles as a DataFrame.
        
        Thread-safe: copies the deque under GIL.
        Returns empty DataFrame if no candles yet.
        """
        candles = list(self._1s_candles)  # snapshot under GIL
        if not candles:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return pd.DataFrame(candles)

    # ═══════════════════════════════════════════════════
    #  LIFECYCLE
    # ═══════════════════════════════════════════════════

    def start(self):
        """Start the WebSocket client in a background daemon thread."""
        if self._running:
            logger.warning("BinanceWS already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="BinanceWS"
        )
        self._thread.start()
        logger.info("🔌 BinanceWS client started")

    def stop(self):
        """Stop the WebSocket client gracefully."""
        self._running = False
        self._connected = False
        if getattr(self, '_ws_app', None):
            try:
                self._ws_app.close()
            except Exception:
                pass
        logger.info("🔌 BinanceWS client stopped")

    # ═══════════════════════════════════════════════════
    #  CONNECTION LOOP (runs in daemon thread)
    # ═══════════════════════════════════════════════════

    def _run_loop(self):
        """Main reconnection loop — runs forever until stop() is called."""
        import random
        
        current_t = threading.current_thread()

        while self._running and current_t == self._thread:
            try:
                self._connect()
            except Exception as e:
                logger.warning(f"BinanceWS connection failed: {e}")

            if not self._running or current_t != self._thread:
                break

            # Exponential backoff with jitter
            jitter = random.uniform(0, 1)
            delay = min(self._backoff + jitter, MAX_BACKOFF_SEC)
            self._backoff = min(self._backoff * 2, MAX_BACKOFF_SEC)
            self._reconnect_count += 1
            logger.info(
                f"BinanceWS reconnecting in {delay:.1f}s "
                f"(attempt #{self._reconnect_count})"
            )

            # Sleep in small chunks so stop() works quickly
            sleep_until = time.time() + delay
            while time.time() < sleep_until and self._running and current_t == self._thread:
                time.sleep(0.5)

    def _connect(self):
        """Connect to Binance combined stream using websocket-client."""
        try:
            import websocket as ws_lib
        except ImportError:
            logger.error(
                "websocket-client not installed! "
                "Run: pip install websocket-client"
            )
            time.sleep(10)
            return

        logger.info(f"BinanceWS connecting to combined stream...")

        self._ws_app = ws_lib.WebSocketApp(
            COMBINED_STREAM_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        # run_forever blocks until disconnect
        # ping_interval handles Binance's 20s ping requirement
        self._ws_app.run_forever(
            ping_interval=20,
            ping_timeout=10,
            reconnect=0,  # we handle reconnect ourselves
        )

    # ═══════════════════════════════════════════════════
    #  WEBSOCKET CALLBACKS
    # ═══════════════════════════════════════════════════

    def _on_open(self, ws):
        self._connected = True
        self._backoff = BASE_DELAY_SEC  # reset backoff on success
        logger.info("✅ BinanceWS connected — receiving live trades")

    def _on_message(self, ws, message: str):
        try:
            data = json.loads(message)
            self._process_message(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"BinanceWS parse error: {e}")

    def _on_error(self, ws, error):
        logger.warning(f"BinanceWS error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self._connected = False
        logger.info(
            f"BinanceWS disconnected: status={close_status_code} msg={close_msg}"
        )

    # ═══════════════════════════════════════════════════
    #  MESSAGE PROCESSING
    # ═══════════════════════════════════════════════════

    def _process_message(self, data: dict):
        """Route combined stream messages to appropriate handlers."""
        self._last_message_time = time.time()

        # Combined stream wraps data in {"stream": "...", "data": {...}}
        if "stream" in data:
            stream_name = data["stream"]
            payload = data.get("data", {})
        else:
            # Direct stream message (non-combined fallback)
            payload = data
            stream_name = payload.get("e", "")

        if "trade" in stream_name:
            self._handle_trade(payload)
        elif "ticker" in stream_name:
            self._handle_ticker(payload)
        elif "depth" in stream_name:
            self._handle_depth(payload)

    def _handle_trade(self, data: dict):
        """Handle individual trade event.
        
        All Binance prices are STRINGS — must parse with float().
        Timestamps are in MILLISECONDS.
        """
        try:
            new_price = float(data.get("p", 0))
            if new_price <= 0:
                return

            self._price = new_price
            self._last_trade_time = float(data.get("T", 0)) / 1000.0
            self._trade_count += 1

            # Phase 3: Track microstructure (trade timestamps + buy/sell volume)
            now = time.time()
            self._trade_timestamps.append(now)
            trade_qty = float(data.get("q", 0))  # quantity in BTC
            is_buyer_maker = data.get("m", False)  # True = seller-initiated
            if is_buyer_maker:
                self._sell_volume_60s.append(trade_qty)
                self._buy_volume_60s.append(0.0)
            else:
                self._buy_volume_60s.append(trade_qty)
                self._sell_volume_60s.append(0.0)
            self._trade_ts_60s.append(now)

            # ── 1-second candle aggregation ──
            epoch_sec = int(now)
            if epoch_sec != self._current_1s_sec:
                # Finalize previous candle (if exists)
                if self._current_1s:
                    self._1s_candles.append(self._current_1s.copy())
                # Start new candle
                self._current_1s_sec = epoch_sec
                self._current_1s = {
                    'timestamp': pd.Timestamp.utcfromtimestamp(epoch_sec),
                    'open': new_price,
                    'high': new_price,
                    'low': new_price,
                    'close': new_price,
                    'volume': trade_qty,
                }
            else:
                # Update current candle
                c = self._current_1s
                if c:
                    c['high'] = max(c['high'], new_price)
                    c['low'] = min(c['low'], new_price)
                    c['close'] = new_price
                    c['volume'] += trade_qty

            # Fire callback (used by api_server to push to frontend)
            if self._on_price_update:
                self._on_price_update(new_price)
        except (ValueError, TypeError):
            pass

    def _handle_ticker(self, data: dict):
        """Handle 24h rolling ticker event.
        
        Updates: price change, high/low, volume, bid/ask.
        All values are STRINGS from Binance.
        """
        try:
            self._change_24h = float(data.get("p", 0))
            self._change_pct = float(data.get("P", 0))
            self._high_24h = float(data.get("h", 0))
            self._low_24h = float(data.get("l", 0))
            self._volume_btc = float(data.get("v", 0))
            self._volume_usdt = float(data.get("q", 0))
            self._open_24h = float(data.get("o", 0))
            self._bid = float(data.get("b", 0))
            self._ask = float(data.get("a", 0))

            # If no trade has arrived yet, use ticker's last price
            if self._price == 0:
                last = data.get("c")
                if last:
                    self._price = float(last)
        except (ValueError, TypeError):
            pass

    def _handle_depth(self, data: dict):
        """Handle L2 Order Book depth updates.
        
        Calculates Bid/Ask Imbalance (BVD) and detects liquidity walls.
        data["b"] = Bids [[price, qty], ...]
        data["a"] = Asks [[price, qty], ...]
        """
        try:
            bids = data.get("b", [])
            asks = data.get("a", [])
            
            # Sum volume for top 5 levels (immediate liquidity)
            bids_top5 = sum(float(qty) for price, qty in bids[:5])
            asks_top5 = sum(float(qty) for price, qty in asks[:5])
            
            self._bids_top5 = bids_top5
            self._asks_top5 = asks_top5
            
            # Total volume in the 20-level book
            total_bids = sum(float(qty) for price, qty in bids)
            total_asks = sum(float(qty) for price, qty in asks)
            
            # Buy Volume Delta (Ratio of Bid volume to Ask volume)
            self._bvd_ratio = total_bids / (total_asks + 1e-9) if (total_bids + total_asks) > 0 else 1.0
            
            # Detect Liquidity Walls (single price level with > 15 BTC)
            self._wall_bids = sum(float(qty) for price, qty in bids if float(qty) > 15.0)
            self._wall_asks = sum(float(qty) for price, qty in asks if float(qty) > 15.0)
            
        except (ValueError, TypeError, IndexError):
            pass


# ═══════════════════════════════════════════════════════
#  Standalone test — run directly to verify connection
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    def on_price(p: float):
        print(f"  💰 ${p:,.2f}", end="\r")

    client = BinanceWSClient(on_price_update=on_price)
    client.start()

    try:
        while True:
            time.sleep(5)
            s = client.snapshot
            print(f"\n{'='*60}")
            print(f"  BTC/USDT:    ${s['price']:,.2f}")
            print(f"  24h Change:  ${s['change_24h']:+,.2f} ({s['change_pct']:+.2f}%)")
            print(f"  24h Range:   ${s['low_24h']:,.2f} — ${s['high_24h']:,.2f}")
            print(f"  Volume:      {s['volume_btc']:,.2f} BTC")
            print(f"  Bid/Ask:     ${s['bid']:,.2f} / ${s['ask']:,.2f}")
            print(f"  Connected:   {s['ws_connected']}")
            print(f"  Trades:      {s['trades_received']:,}")
    except KeyboardInterrupt:
        client.stop()
        print("\nStopped.")
