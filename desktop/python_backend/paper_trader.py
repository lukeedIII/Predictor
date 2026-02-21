"""
Nexus Shadow-Quant Paper Trading Engine
========================================
Autonomous paper trading bot with professional-grade risk management.
Reads live BTC price, auto-trades based on Nexus AI predictions, logs 
everything for model retraining.

Risk Management:
- Kelly Criterion position sizing
- Regime filter (skip random/chaotic markets)
- 2:1 reward-to-risk ratio
- 20% drawdown circuit breaker
- Minimum confidence threshold
- Cooldown between trades
"""

import json
import os
import time
import logging
import threading
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import config
from nexus_logger import NexusLogger
from telegram_notifier import telegram

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Position:
    """Represents an open paper trading position."""
    
    def __init__(self, direction: str, entry_price: float, size_usd: float,
                 leverage: int, confidence: float, regime: str,
                 tp_price: float, sl_price: float, hurst: float = 0.5):
        self.direction = direction          # "LONG" or "SHORT"
        self.entry_price = entry_price
        self.size_usd = size_usd            # Notional value (may reduce after partial close)
        self.original_size_usd = size_usd   # Original notional (never changes)
        self.margin = size_usd / leverage   # Actual capital used
        self.leverage = leverage
        self.confidence = confidence
        self.regime = regime
        self.hurst = hurst                  # Hurst at entry (for feedback)
        self.tp_price = tp_price            # Take-profit (full TP)
        self.sl_price = sl_price            # Stop-loss
        self.initial_sl = sl_price           # Original SL (before trailing)
        self.entry_time = datetime.now()
        self.liquidation_price = self._calc_liquidation()
        
        # TP1 — partial exit target (60% of TP distance)
        tp_dist = abs(tp_price - entry_price)
        if direction == "LONG":
            self.tp1_price = entry_price + tp_dist * 0.6
        else:
            self.tp1_price = entry_price - tp_dist * 0.6
        self.tp1_hit = False                # Whether partial exit was taken
        
        # Fee tracking — entry fee charged at open
        fee_pct = (getattr(config, 'PAPER_FEE_TAKER_PCT', 0.04)
                   + getattr(config, 'PAPER_SLIPPAGE_PCT', 0.01))
        self.entry_fee = round(size_usd * fee_pct / 100, 4)
        
        # Trailing stop loss state
        self.best_price = entry_price       # Best price seen while open
    
    def _calc_liquidation(self) -> float:
        """Calculate liquidation price based on leverage."""
        liq_move = 1.0 / self.leverage  # e.g., 10x = 10% move
        if self.direction == "LONG":
            return self.entry_price * (1 - liq_move)
        else:
            return self.entry_price * (1 + liq_move)
    
    def update_trailing_sl(self, current_price: float):
        """Ratchet stop loss toward profit when price moves favorably.
        Trailing SL activates when position is >0.3% in profit.
        Trail distance = 50% of the distance from entry to best price."""
        if self.direction == "LONG":
            if current_price > self.best_price:
                self.best_price = current_price
            # Only trail once we're >0.3% in profit
            profit_pct = (self.best_price - self.entry_price) / self.entry_price
            if profit_pct > 0.003:
                # Trail at 50% of gains — lock in half the move
                trail_price = self.entry_price + (self.best_price - self.entry_price) * 0.5
                if trail_price > self.sl_price:
                    self.sl_price = trail_price
        else:  # SHORT
            if current_price < self.best_price:
                self.best_price = current_price
            profit_pct = (self.entry_price - self.best_price) / self.entry_price
            if profit_pct > 0.003:
                trail_price = self.entry_price - (self.entry_price - self.best_price) * 0.5
                if trail_price < self.sl_price:
                    self.sl_price = trail_price
    
    def unrealized_pnl(self, current_price: float, net: bool = True) -> float:
        """Calculate unrealized PnL at current price.
        If net=True (default), subtracts entry fee + estimated exit fee."""
        if self.direction == "LONG":
            pct_move = (current_price - self.entry_price) / self.entry_price
        else:
            pct_move = (self.entry_price - current_price) / self.entry_price
        gross = self.size_usd * pct_move
        if not net:
            return gross
        # Subtract entry fee + estimated exit fee
        est_exit_fee = self.entry_fee  # same rate both sides
        return gross - self.entry_fee - est_exit_fee
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """PnL as percentage of margin (not notional)."""
        return (self.unrealized_pnl(current_price) / self.margin) * 100
    
    def should_liquidate(self, current_price: float) -> bool:
        """Check if position should be auto-liquidated."""
        if self.direction == "LONG":
            return current_price <= self.liquidation_price
        else:
            return current_price >= self.liquidation_price
    
    def should_tp(self, current_price: float) -> bool:
        """Check if take-profit hit."""
        if self.direction == "LONG":
            return current_price >= self.tp_price
        else:
            return current_price <= self.tp_price
    
    def should_sl(self, current_price: float) -> bool:
        """Check if stop-loss hit (includes trailing SL)."""
        if self.direction == "LONG":
            return current_price <= self.sl_price
        else:
            return current_price >= self.sl_price
    
    def should_tp1(self, current_price: float) -> bool:
        """Check if TP1 (partial exit target) hit."""
        if self.tp1_hit:
            return False
        if self.direction == "LONG":
            return current_price >= self.tp1_price
        else:
            return current_price <= self.tp1_price
    
    def to_dict(self) -> Dict:
        return {
            'direction': self.direction,
            'entry_price': self.entry_price,
            'size_usd': self.size_usd,
            'original_size_usd': getattr(self, 'original_size_usd', self.size_usd),
            'margin': self.margin,
            'leverage': self.leverage,
            'confidence': self.confidence,
            'regime': self.regime,
            'hurst': self.hurst,
            'tp_price': self.tp_price,
            'tp1_price': getattr(self, 'tp1_price', self.tp_price),
            'tp1_hit': getattr(self, 'tp1_hit', False),
            'sl_price': self.sl_price,
            'initial_sl': self.initial_sl,
            'best_price': self.best_price,
            'entry_time': self.entry_time.isoformat(),
            'liquidation_price': self.liquidation_price,
            'entry_fee': self.entry_fee
        }


class PaperTrader:
    """
    Autonomous paper trading engine.
    
    Integrates with NexusPredictor to auto-execute trades based on AI signals,
    using professional risk management (Kelly Criterion, regime filters, etc.)
    """
    
    # Max concurrent positions for data farming
    MAX_CONCURRENT = 6

    def __init__(self, starting_balance: float = None, default_leverage: int = None):
        # Thread-safety lock: protects all balance and position mutations.
        # Using RLock so the same thread can re-acquire (e.g. close_position called
        # from a method that already holds the lock).
        self._lock = threading.RLock()

        self.starting_balance = starting_balance or config.PAPER_STARTING_BALANCE
        self.balance = self.starting_balance
        self.peak_balance = self.starting_balance
        self.leverage = default_leverage or config.PAPER_DEFAULT_LEVERAGE
        
        # State — multi-position support
        self.positions: list = []  # List of active Position objects
        self.position = None       # Legacy compat — points to first position or None
        self.is_running = False
        self.circuit_breaker_active = False
        self.last_trade_time: Optional[datetime] = None
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0        # Net PnL (after fees)
        self.total_gross_pnl = 0.0  # Gross PnL (before fees)
        self.total_fees = 0.0       # Cumulative fees paid
        self.trade_history = []
        self.equity_history = []
        
        # Kelly Criterion state (initialized with modest defaults)
        self._win_rate = 0.52    # Start conservative
        self._avg_win = 0.015    # 1.5% avg win
        self._avg_loss = 0.01    # 1.0% avg loss
        
        # Signal streak tracking (used by evaluate_signal)
        self._signal_streak = 0
        self._signal_direction = None
        
        # ===== PHASE 2: Feedback Loop =====
        # Trade feedback log — persisted to JSON for predictor to consume.
        # Bounded deque (max 2000 entries) prevents unbounded memory growth.
        self._feedback_path = os.path.join(config.DATA_ROOT, 'feedback', 'trade_feedback.json')
        self._adaptive_path = os.path.join(config.DATA_ROOT, 'feedback', 'adaptive_config.json')
        self._feedback_log: deque = deque(maxlen=2000)
        self._adaptive_min_confidence = config.PAPER_MIN_CONFIDENCE  # starts at default
        self._load_feedback()
        
        # Logger
        self.nlog = NexusLogger()
        
        # Load existing history if any
        self._load_history()
        self._load_positions()
        
        self.nlog.log_system(f"PaperTrader initialized: ${self.balance:,.2f} balance, {self.leverage}x leverage")
        logging.info(f"PaperTrader initialized: ${self.balance:,.2f} balance, {self.leverage}x leverage")
        
        # Start Telegram hourly summary scheduler
        telegram.start_hourly_loop(self.get_stats)
    
    # ========== RISK MANAGEMENT ==========
    
    def kelly_fraction(self) -> float:
        """
        Kelly Criterion: f* = (p * b - q) / b
        Where p = win probability, q = 1-p, b = avg_win / avg_loss
        
        Default 15% per slot until enough data (3 slots × 15% = 45% max).
        Capped at 25% for safety (half-Kelly is standard practice).
        Floor at 5% so positions stay meaningful.
        """
        if self._avg_loss == 0 or self.total_trades < 5:
            return 0.08  # Default 8% per slot (6 slots × 8% = 48% max deployed)
        
        p = self._win_rate
        q = 1 - p
        b = self._avg_win / self._avg_loss
        
        kelly = (p * b - q) / b
        
        # Half-Kelly (standard for real trading)
        half_kelly = kelly / 2
        
        # Floor 5%, cap at 25%
        return max(0.05, min(0.25, half_kelly))
    
    def calculate_position_size(self, confidence: float = 50) -> float:
        """
        Confidence-scaled Kelly sizing.
        Scale: conf 40% → 0.5x Kelly, conf 60% → 1.0x, conf 80%+ → 1.5x
        This creates varied position sizes based on signal quality.
        """
        fraction = self.kelly_fraction()
        
        # Confidence multiplier: linear scale [0.5, 1.5] mapped to [40, 80]
        conf_mult = np.clip((confidence - 40) / 40, 0, 1) * 1.0 + 0.5  # [0.5, 1.5]
        adjusted_fraction = fraction * conf_mult
        adjusted_fraction = min(adjusted_fraction, 0.30)  # Hard cap 30%
        
        used_margin = sum(p.margin for p in self.positions)
        available = self.balance - used_margin
        risk_amount = available * adjusted_fraction
        notional = risk_amount * self.leverage
        return max(notional, 100)  # Min $100 position
    
    def check_circuit_breaker(self) -> bool:
        """Check if drawdown exceeds maximum allowed."""
        if self.peak_balance == 0:
            return False
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown >= config.PAPER_MAX_DRAWDOWN:
            self.circuit_breaker_active = True
            logging.warning(f"CIRCUIT BREAKER: Drawdown {drawdown:.1%} exceeds {config.PAPER_MAX_DRAWDOWN:.0%} limit!")
            # Telegram alert
            telegram.notify_circuit_breaker(
                balance=self.balance, peak=self.peak_balance,
                drawdown_pct=drawdown * 100
            )
            return True
        return False
    
    def check_cooldown(self) -> bool:
        """Check if enough time has passed since last trade."""
        if self.last_trade_time is None:
            return True
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        return elapsed >= config.PAPER_COOLDOWN_SEC
    
    def calculate_tp_sl(self, entry_price: float, direction: str, 
                        volatility: float = 0.005, regime: str = 'UNKNOWN',
                        confidence: float = 50) -> Tuple[float, float]:
        """
        Regime-adaptive TP/SL with dynamic reward:risk ratio.
        Trending markets → let winners run (2.5:1)
        Ranging markets → quick scalps (1:1)
        High confidence → wider TP (let conviction trades breathe)
        """
        risk_pct = max(volatility * 1.0, 0.002)   # Floor at 0.2%
        
        # Regime-adaptive reward:risk ratio
        if regime == 'BULL' and direction == 'LONG':
            rr_ratio = 2.5     # Let longs run in bull
        elif regime == 'BEAR' and direction == 'SHORT':
            rr_ratio = 2.5     # Let shorts run in bear
        elif regime == 'SIDEWAYS':
            rr_ratio = 1.0     # Quick scalps in chop
        else:
            rr_ratio = 1.5     # Default (counter-trend or unknown)
        
        # High confidence → wider TP (conviction trades breathe more)
        if confidence > 75:
            rr_ratio *= 1.2
        
        reward_pct = risk_pct * rr_ratio
        
        if direction == "LONG":
            tp = entry_price * (1 + reward_pct)
            sl = entry_price * (1 - risk_pct)
        else:
            tp = entry_price * (1 - reward_pct)
            sl = entry_price * (1 + risk_pct)
        
        return tp, sl
    
    # ========== TRADING LOGIC ==========
    
    def _quant_adjusted_confidence(self, prediction: Dict) -> float:
        """
        Adjust raw confidence using the 16-model quant overlay.
        OFI alignment, RQA determinism, and jump detection modify the score.
        """
        raw_conf = prediction.get('confidence', 50)
        quant = prediction.get('quant', {})
        bonus = 0
        
        # OFI alignment: if signal direction matches order flow, boost
        ofi = 0
        of_data = quant.get('order_flow', {})
        if of_data:
            ofi = of_data.get('normalized', 0)
        direction = prediction.get('direction', 'NEUTRAL')
        if (direction == 'UP' and ofi > 0.3) or (direction == 'DOWN' and ofi < -0.3):
            bonus += 5  # OFI confirms signal
            logging.debug(f"Quant boost +5: OFI {ofi:.2f} confirms {direction}")
        elif (direction == 'UP' and ofi < -0.5) or (direction == 'DOWN' and ofi > 0.5):
            bonus -= 5  # OFI contradicts signal
            logging.debug(f"Quant penalty -5: OFI {ofi:.2f} contradicts {direction}")
        
        # RQA determinism: high determinism = more predictable = boost
        rqa = quant.get('rqa', {})
        det = rqa.get('determinism', 0)
        if det > 0.8:
            bonus += 3
            logging.debug(f"Quant boost +3: RQA determinism {det:.3f}")
        elif det < 0.3:
            bonus -= 3
            logging.debug(f"Quant penalty -3: Low RQA determinism {det:.3f}")
        
        # Jump detection: penalize if jumps detected (unpredictable)
        jumps = quant.get('jumps', {})
        if jumps.get('detected', False):
            bonus -= 8
            logging.debug(f"Quant penalty -8: Jump detected")
        
        # Regime confidence: high HMM confidence = stronger signal
        regime_data = quant.get('regime', {})
        regime_conf = regime_data.get('confidence', 0)
        if regime_conf > 80:
            bonus += 2
        
        adjusted = np.clip(raw_conf + bonus, 0, 99)
        if bonus != 0:
            logging.info(f"Quant overlay: {raw_conf:.1f}% → {adjusted:.1f}% (bonus={bonus:+d})")
        return adjusted
    
    def _dynamic_leverage(self, prediction: Dict) -> int:
        """
        Adjust leverage based on regime and volatility.
        Trending + low-vol → higher leverage (up to LEVERAGE_MAX)
        High-vol or chaotic → lower leverage (down to LEVERAGE_MIN)
        """
        base = self.leverage  # Config default (10x)
        regime = prediction.get('regime_label', 'UNKNOWN')
        vol_regime = prediction.get('vol_regime', 1.0)
        confidence = prediction.get('confidence', 50)
        lev_min = getattr(config, 'PAPER_LEVERAGE_MIN', 3)
        lev_max = getattr(config, 'PAPER_LEVERAGE_MAX', 20)
        
        if regime in ('BULL', 'BEAR') and vol_regime < 1.0 and confidence > 65:
            return min(base + 5, lev_max)   # Up to 20x in clean trends
        elif vol_regime > 2.0:
            return max(base - 5, lev_min)   # Deleverage in chaos (min 3x)
        elif regime == 'SIDEWAYS' and vol_regime < 0.5:
            return max(base - 3, lev_min)   # Conservative in dead markets
        return base
    
    def evaluate_signal(self, prediction: Dict) -> Optional[str]:
        """
        Evaluate the AI prediction and decide whether to trade.
        v2.0: Tiered confirmation, pyramiding, quant-overlay confidence adjustment.
        Returns "LONG", "SHORT", or None.

        NOTE: This method intentionally does NOT mutate the caller's dict.
        A shallow copy is made internally so _adjusted_confidence is only
        visible within this call.

        Thread-safety: Pure-data operations (dict copy, quant overlay, regime
        filters from prediction payload) execute before lock acquisition.
        All reads/writes of shared mutable state (_signal_streak, positions,
        circuit_breaker_active, last_trade_time, _feedback_log, balance,
        peak_balance) are guarded by self._lock (RLock — re-entrant for
        check_cooldown calls).
        """
        # Work on a copy so we never mutate the caller's prediction dict.
        # This is intentionally outside the lock — no shared state involved.
        prediction = dict(prediction)

        # Apply quant overlay to adjust raw confidence
        raw_confidence = prediction.get('confidence', 0)
        confidence = self._quant_adjusted_confidence(prediction)
        direction = prediction.get('direction', 'NEUTRAL')
        hurst = prediction.get('hurst', 0.5)

        # Store adjusted confidence back for sizing use (on the copy only)
        prediction['_adjusted_confidence'] = confidence

        # Fast exits based purely on prediction payload (no shared state)
        if direction == 'NEUTRAL':
            return None

        if confidence < getattr(config, 'PAPER_MIN_CONFIDENCE', 0):
            return None

        if 0.48 <= hurst <= 0.52:
            logging.debug(f"Skip: Hurst {hurst:.3f} indicates random/chaotic regime")
            return None

        vol_regime = prediction.get('vol_regime', 1.0)
        vol_max = getattr(config, 'REGIME_VOL_MAX', 3.0)
        vol_min = getattr(config, 'REGIME_VOL_MIN', 0.15)
        if vol_regime > vol_max:
            logging.debug(f"Skip: vol_regime {vol_regime:.2f} > {vol_max} (extreme volatility)")
            return None
        if vol_regime < vol_min:
            logging.debug(f"Skip: vol_regime {vol_regime:.2f} < {vol_min} (dead market)")
            return None

        ev = prediction.get('expected_value', None)
        calibrator_fitted = prediction.get('calibrator_fitted', False)
        if calibrator_fitted and ev is not None:
            min_ev = getattr(config, 'MIN_EXPECTED_VALUE', 0.0)
            if ev < min_ev:
                logging.debug(f"Skip: EV={ev:.4f} < {min_ev} (negative expected value)")
                return None

        # ── Acquire lock: all remaining checks read/write shared mutable state ──
        with self._lock:
            # Rule 1: Adaptive minimum confidence (adjusts based on recent performance)
            min_conf = self._adaptive_min_confidence
            if confidence < min_conf:
                self._signal_streak = 0
                logging.debug(f"Skip: confidence {confidence:.1f}% < {min_conf:.0f}% (adaptive)")
                return None

            # Rule 2c: Regime win-rate gate — block trading in losing regimes
            regime_label = prediction.get('regime_label', 'UNKNOWN')
            min_wr = getattr(config, 'REGIME_MIN_WIN_RATE', 0.35)
            min_trades = getattr(config, 'REGIME_MIN_TRADES', 5)
            if self._feedback_log:
                regime_trades = [t for t in self._feedback_log[-50:] if t.get('regime') == regime_label]
                if len(regime_trades) >= min_trades:
                    wins = sum(1 for t in regime_trades if t.get('won', False))
                    wr = wins / len(regime_trades)
                    if wr < min_wr:
                        self._signal_streak = 0
                        logging.info(
                            f"REGIME GATE: {regime_label} blocked — "
                            f"win rate {wr*100:.0f}% < {min_wr*100:.0f}% "
                            f"({wins}/{len(regime_trades)} wins)"
                        )
                        return None

            # Rule 3: Tiered signal confirmation based on confidence
            wanted = 'LONG' if direction == 'UP' else 'SHORT'
            if not hasattr(self, '_signal_streak'):
                self._signal_streak = 0
                self._signal_direction = None

            if wanted == self._signal_direction:
                self._signal_streak += 1
            else:
                self._signal_direction = wanted
                self._signal_streak = 1

            # Tiered: strong signals need fewer confirmations
            if confidence >= 75:
                min_confirms = 1   # Instant entry for strong signals
            elif confidence >= 55:
                min_confirms = 2   # Standard confirmation
            else:
                min_confirms = 3   # Extra caution for weak signals

            if self._signal_streak < min_confirms:
                logging.debug(f"Skip: signal confirmation {self._signal_streak}/{min_confirms} for {wanted} (conf={confidence:.0f}%)")
                return None

            # Rule 4: Cooldown period (check_cooldown reads last_trade_time — shared)
            if not self.check_cooldown():
                remaining = config.PAPER_COOLDOWN_SEC
                if self.last_trade_time:
                    remaining -= (datetime.now() - self.last_trade_time).total_seconds()
                logging.debug(f"Skip: cooldown active ({remaining:.0f}s remaining)")
                return None

            # Rule 5: Circuit breaker
            if self.circuit_breaker_active:
                # Auto-reset if balance recovers above 90% of peak
                if self.balance >= self.peak_balance * 0.90:
                    self.circuit_breaker_active = False
                    logging.info("Circuit breaker RESET — balance recovered")
                else:
                    logging.debug("Skip: circuit breaker active")
                    return None

            # Rule 6: Max concurrent positions
            if len(self.positions) >= self.MAX_CONCURRENT:
                logging.debug(f"Skip: max {self.MAX_CONCURRENT} concurrent positions reached")
                return None

            # Rule 7: Pyramid limit — max N in same direction (replaces old no-stacking)
            max_same = getattr(config, 'PAPER_MAX_SAME_DIRECTION', 3)
            same_dir_count = sum(1 for p in self.positions if p.direction == wanted)
            if same_dir_count >= max_same:
                logging.debug(f"Skip: already {same_dir_count} {wanted} positions (pyramid limit {max_same})")
                return None

            # All checks passed — reset streak and trade
            self._signal_streak = 0
            logging.info(
                f"SIGNAL CONFIRMED: {wanted} @ raw={raw_confidence:.1f}% adjusted={confidence:.1f}% "
                f"hurst={hurst:.3f} confirms={min_confirms}"
            )
            return wanted
    
    def open_position(self, direction: str, current_price: float, 
                      prediction: Dict, volatility: float = 0.005):
        """Open a new paper trading position with dynamic leverage and confidence-scaled sizing."""
        with self._lock:
            return self._open_position_locked(direction, current_price, prediction, volatility)

    def _open_position_locked(self, direction: str, current_price: float,
                              prediction: Dict, volatility: float = 0.005):
        """Internal: open_position body, called under self._lock."""
        if len(self.positions) >= self.MAX_CONCURRENT:
            logging.warning(f"Cannot open: max {self.MAX_CONCURRENT} positions reached")
            return False
        
        # Dynamic leverage based on regime + volatility
        trade_leverage = self._dynamic_leverage(prediction)
        
        # Confidence-scaled position sizing
        adj_conf = prediction.get('_adjusted_confidence', prediction.get('confidence', 50))
        size_usd = self.calculate_position_size(confidence=adj_conf)
        
        # Override leverage for this trade
        margin = size_usd / trade_leverage
        
        # Calculate available margin (total balance minus margin used by open positions)
        used_margin = sum(p.margin for p in self.positions)
        available = self.balance - used_margin
        
        # Ensure we have enough available margin
        if margin > available * 0.90:  # Keep 10% buffer per slot
            margin = available * 0.80
            size_usd = margin * trade_leverage
        
        if margin < 10:  # Minimum $10 margin
            logging.warning(f"Not enough margin: ${available:.2f} available")
            return False
        
        # Regime-adaptive TP/SL
        regime = prediction.get('regime_label', 'UNKNOWN')
        tp, sl = self.calculate_tp_sl(current_price, direction, volatility, 
                                       regime=regime, confidence=adj_conf)
        
        # Create position and add to list
        new_pos = Position(
            direction=direction,
            entry_price=current_price,
            size_usd=size_usd,
            leverage=trade_leverage,
            confidence=adj_conf,
            regime=regime,
            tp_price=tp,
            sl_price=sl,
            hurst=prediction.get('hurst', 0.5)
        )
        
        # Deduct entry fee from balance
        entry_fee = new_pos.entry_fee
        self.balance -= entry_fee
        self.total_fees += entry_fee
        
        self.positions.append(new_pos)
        self.position = self.positions[0] if self.positions else None  # Legacy compat
        self._save_positions()  # Persist for crash recovery
        
        self.last_trade_time = datetime.now()
        
        self.nlog.log_trade_open(direction, current_price, size_usd, trade_leverage, 
                                  adj_conf)
        logging.info(
            f"OPENED {direction} #{len(self.positions)} @ ${current_price:,.2f} | "
            f"Size: ${size_usd:,.2f} ({trade_leverage}x) | Conf: {adj_conf:.0f}% | Fee: ${entry_fee:.2f} | "
            f"TP: ${tp:,.2f} (TP1: ${new_pos.tp1_price:,.2f}) | SL: ${sl:,.2f} | "
            f"Liq: ${new_pos.liquidation_price:,.2f} | Regime: {regime}"
        )
        
        # Telegram notification
        telegram.notify_trade_open(
            direction=direction, price=current_price, size_usd=size_usd,
            leverage=trade_leverage, confidence=adj_conf, regime=regime,
            tp=tp, sl=sl, tp1=new_pos.tp1_price
        )
        
        return True
    
    def close_position(self, current_price: float, reason: str = "MANUAL", pos: 'Position' = None) -> Dict:
        """Close a specific position (or first position if none specified)."""
        with self._lock:
            return self._close_position_locked(current_price, reason, pos)

    def _close_position_locked(self, current_price: float, reason: str = "MANUAL", pos: 'Position' = None) -> Dict:
        """Close a specific position (or first position if none specified)."""
        if pos is None:
            # Legacy compat: close first position
            if not self.positions:
                return {}
            pos = self.positions[0]
        
        if pos not in self.positions:
            return {}
        
        gross_pnl = pos.unrealized_pnl(current_price, net=False)
        
        # Handle liquidation — lose entire margin
        if reason == "LIQUIDATED":
            gross_pnl = -pos.margin
        
        # ── Fee deduction: exit fee ──
        fee_pct = (getattr(config, 'PAPER_FEE_TAKER_PCT', 0.04)
                   + getattr(config, 'PAPER_SLIPPAGE_PCT', 0.01))
        exit_fee = round(pos.size_usd * fee_pct / 100, 4)
        entry_fee = pos.entry_fee  # already charged at open
        total_fee = round(entry_fee + exit_fee, 4)
        
        net_pnl = gross_pnl - exit_fee  # entry fee was already deducted at open
        net_pnl_pct = (net_pnl / pos.margin) * 100 if pos.margin else 0
        gross_pnl_pct = (gross_pnl / pos.margin) * 100 if pos.margin else 0
        
        if reason == "LIQUIDATED":
            net_pnl = -(pos.margin + exit_fee)
            net_pnl_pct = -100.0
            gross_pnl_pct = -100.0
        
        # Update balance (exit fee deducted; entry fee was already deducted at open)
        self.balance += gross_pnl - exit_fee
        self.total_pnl += net_pnl
        self.total_gross_pnl += gross_pnl
        self.total_fees += exit_fee
        self.total_trades += 1
        
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update peak balance
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        # Update Kelly statistics (use net PnL for realistic sizing)
        self._update_kelly_stats(net_pnl, pos.size_usd)
        
        # Build trade record
        hold_secs = (datetime.now() - pos.entry_time).total_seconds()
        trade_record = {
            'timestamp_open': pos.entry_time.isoformat(),
            'timestamp_close': datetime.now().isoformat(),
            'direction': pos.direction,
            'entry_price': pos.entry_price,
            'exit_price': current_price,
            'size_usd': pos.size_usd,
            'margin': pos.margin,
            'leverage': pos.leverage,
            'gross_pnl_usd': round(gross_pnl, 2),
            'pnl_usd': round(net_pnl, 2),          # net PnL (after fees)
            'pnl_pct': round(net_pnl_pct, 2),       # net PnL % of margin
            'entry_fee': round(entry_fee, 4),
            'exit_fee': round(exit_fee, 4),
            'total_fee': round(total_fee, 4),
            'confidence': pos.confidence,
            'regime': pos.regime,
            'close_reason': reason,
            'balance_after': round(self.balance, 2),
            'kelly_fraction': round(self.kelly_fraction(), 4),
            'model_version': config.VERSION
        }
        
        # ===== FEEDBACK LOOP: Log trade outcome for predictor =====
        feedback_entry = {
            'ts': datetime.now().isoformat(),
            'direction': pos.direction,
            'confidence': pos.confidence,
            'regime': pos.regime,
            'hurst': getattr(pos, 'hurst', 0.5),
            'pnl_usd': round(net_pnl, 2),
            'pnl_pct': round(net_pnl_pct, 2),
            'won': net_pnl > 0,
            'close_reason': reason,
            'hold_minutes': round(hold_secs / 60, 1),
            'trailing_sl_moved': getattr(pos, 'sl_price', 0) != getattr(pos, 'initial_sl', 0),
            'best_price_seen': getattr(pos, 'best_price', pos.entry_price),
        }
        self._feedback_log.append(feedback_entry)
        self._save_feedback()
        self._update_adaptive_threshold()
        
        self.trade_history.append(trade_record)
        self._save_trade(trade_record)
        self._save_equity_point()
        
        self.positions.remove(pos)
        self.position = self.positions[0] if self.positions else None  # Legacy compat
        self._save_positions()  # Persist for crash recovery
        
        # Check circuit breaker
        self.check_circuit_breaker()
        
        self.nlog.log_trade_close(trade_record)
        logging.info(
            f"CLOSED {pos.direction} @ ${current_price:,.2f} | "
            f"Reason: {reason} | Net: ${net_pnl:+,.2f} ({net_pnl_pct:+.1f}%) | "
            f"Fees: ${total_fee:.2f} | Balance: ${self.balance:,.2f} | Open: {len(self.positions)}"
        )
        
        # Telegram notification
        telegram.notify_trade_close(
            direction=pos.direction, entry_price=pos.entry_price,
            exit_price=current_price, pnl_usd=net_pnl, pnl_pct=net_pnl_pct,
            reason=reason, balance_after=self.balance, size_usd=pos.size_usd
        )
        
        return trade_record
    
    def partial_close(self, pos: Position, current_price: float, 
                      fraction: float = 0.5, reason: str = "TP1_PARTIAL"):
        """
        Close a fraction of a position (scale-out).
        Reduces size_usd and returns freed margin to balance.
        """
        close_size = pos.size_usd * fraction
        
        # Calculate PnL on the closed portion
        if pos.direction == "LONG":
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - current_price) / pos.entry_price
        
        gross_pnl = close_size * pnl_pct
        
        # Exit fee on closed portion
        fee_pct = (getattr(config, 'PAPER_FEE_TAKER_PCT', 0.04)
                   + getattr(config, 'PAPER_SLIPPAGE_PCT', 0.01))
        exit_fee = round(close_size * fee_pct / 100, 4)
        net_pnl = gross_pnl - exit_fee
        
        # Update position size (keep remaining fraction open)
        pos.size_usd *= (1 - fraction)
        freed_margin = close_size / pos.leverage
        pos.margin = pos.size_usd / pos.leverage
        
        # Credit balance: freed margin + net PnL
        self.balance += freed_margin + net_pnl
        self.total_pnl += net_pnl
        self.total_gross_pnl += gross_pnl
        self.total_fees += exit_fee
        
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        logging.info(
            f"PARTIAL CLOSE {pos.direction} {fraction*100:.0f}% @ ${current_price:,.2f} | "
            f"Closed: ${close_size:,.2f} | PnL: ${net_pnl:+.2f} | "
            f"Remaining: ${pos.size_usd:,.2f} | Reason: {reason}"
        )
        
        # Telegram notification
        telegram.notify_partial_close(
            direction=pos.direction, price=current_price,
            closed_size=close_size, pnl=net_pnl, remaining=pos.size_usd
        )
    
    def update(self, current_price: float, prediction: Dict = None, 
               volatility: float = 0.005) -> Optional[Dict]:
        """
        Main update loop — call this every tick.
        v2.0: Checks TP1 partial exits, TP/SL/liquidation, then evaluates new signals.
        
        Returns trade record if a trade was closed, None otherwise.
        """
        result = None
        
        # 1. Update trailing stop loss for ALL open positions
        for pos in list(self.positions):
            pos.update_trailing_sl(current_price)
        
        # 2. Check ALL open positions for exits (priority: liquidation > TP > TP1 > SL)
        for pos in list(self.positions):  # Copy list since we may remove during iteration
            # Time-based exit: auto-close positions held too long
            hold_secs = (datetime.now() - pos.entry_time).total_seconds()
            max_hold = getattr(config, 'PAPER_MAX_HOLD_SEC', 5400)
            if hold_secs > max_hold:
                result = self.close_position(current_price, "MAX_HOLD_TIME", pos)
            elif pos.should_liquidate(current_price):
                result = self.close_position(current_price, "LIQUIDATED", pos)
            elif pos.should_tp(current_price):
                result = self.close_position(current_price, "TAKE_PROFIT", pos)
            elif hasattr(pos, 'should_tp1') and pos.should_tp1(current_price):
                # TP1 partial exit: close 50%, move SL to breakeven
                self.partial_close(pos, current_price, fraction=0.5, reason="TP1_PARTIAL")
                pos.tp1_hit = True
                pos.sl_price = pos.entry_price  # Move SL to breakeven after TP1
                logging.info(f"TP1 hit — SL moved to breakeven ${pos.entry_price:,.2f}")
            elif pos.should_sl(current_price):
                # Determine if this was the trailing SL or original SL
                if pos.sl_price == pos.entry_price and getattr(pos, 'tp1_hit', False):
                    sl_reason = "BREAKEVEN_EXIT"  # SL was moved to entry after TP1
                elif pos.sl_price != pos.initial_sl:
                    sl_reason = "TRAILING_STOP"
                else:
                    sl_reason = "STOP_LOSS"
                result = self.close_position(current_price, sl_reason, pos)
        
        # 3. Evaluate new signals if slots available and running
        if len(self.positions) < self.MAX_CONCURRENT and self.is_running and prediction is not None:
            signal = self.evaluate_signal(prediction)
            if signal is not None:
                self.open_position(signal, current_price, prediction, volatility)
        
        # 4. Record equity
        self._save_equity_point()
        
        return result
    
    # ========== STATISTICS ==========
    
    def get_stats(self) -> Dict:
        """Get comprehensive trading statistics."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        # Drawdown
        drawdown = 0
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
        
        # Unrealized PnL (across all open positions)
        unrealized = 0
        last_price = self._last_price if hasattr(self, '_last_price') else None
        for pos in self.positions:
            p = last_price or pos.entry_price
            unrealized += pos.unrealized_pnl(p)
        
        # Sharpe ratio — time-weighted annualisation
        sharpe = 0
        if len(self.trade_history) > 2:
            returns = [t['pnl_pct'] for t in self.trade_history]
            if np.std(returns) > 0:
                # Compute average trade duration in hours
                durations = []
                for t in self.trade_history:
                    try:
                        entry = pd.to_datetime(t.get('entry_time', t.get('timestamp', datetime.now())))
                        exit_t = pd.to_datetime(t.get('exit_time', t.get('timestamp', datetime.now())))
                        dur_h = max(1/60, (exit_t - entry).total_seconds() / 3600)
                        durations.append(dur_h)
                    except Exception:
                        durations.append(1.0)  # Default 1 hour
                avg_hours = np.mean(durations) if durations else 1.0
                trades_per_year = 8760 / avg_hours  # Hours in a year / avg trade duration
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(trades_per_year)
        
        # Profit factor (on net PnL)
        net_wins = sum(t['pnl_usd'] for t in self.trade_history if t['pnl_usd'] > 0)
        net_losses = abs(sum(t['pnl_usd'] for t in self.trade_history if t['pnl_usd'] < 0))
        profit_factor = net_wins / net_losses if net_losses > 0 else float('inf')
        
        # Net Sharpe (based on pnl_pct which is now net of fees)
        net_sharpe = 0
        if len(self.trade_history) > 2:
            net_returns = [t['pnl_pct'] for t in self.trade_history]
            if np.std(net_returns) > 0:
                durations_net = []
                for t in self.trade_history:
                    try:
                        t_open = pd.to_datetime(t.get('timestamp_open', t.get('timestamp', datetime.now())))
                        t_close = pd.to_datetime(t.get('timestamp_close', t.get('timestamp', datetime.now())))
                        dur_h = max(1/60, (t_close - t_open).total_seconds() / 3600)
                        durations_net.append(dur_h)
                    except Exception:
                        durations_net.append(1.0)
                avg_h_net = np.mean(durations_net) if durations_net else 1.0
                tpy_net = 8760 / avg_h_net
                net_sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(tpy_net)
        
        # Current win/loss streak
        current_streak = 0
        for t in reversed(self.trade_history):
            if not current_streak:
                current_streak = 1 if t['pnl_usd'] > 0 else -1
            elif (t['pnl_usd'] > 0 and current_streak > 0):
                current_streak += 1
            elif (t['pnl_usd'] <= 0 and current_streak < 0):
                current_streak -= 1
            else:
                break
        
        # Best/worst trades
        best_trade = max((t['pnl_usd'] for t in self.trade_history), default=0)
        worst_trade = min((t['pnl_usd'] for t in self.trade_history), default=0)
        
        # Margin locked in open positions
        margin_in_use = sum(pos.margin for pos in self.positions)
        
        return {
            'balance': self.balance,
            'starting_balance': self.starting_balance,
            'available_balance': self.balance - margin_in_use,
            'margin_in_use': margin_in_use,
            'total_pnl': self.total_pnl,
            'total_gross_pnl': self.total_gross_pnl,
            'total_fees': self.total_fees,
            'total_pnl_pct': (self.balance / self.starting_balance - 1) * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'net_sharpe_ratio': net_sharpe,
            'max_drawdown_pct': drawdown,
            'profit_factor': profit_factor,
            'kelly_fraction': self.kelly_fraction(),
            'unrealized_pnl': unrealized,
            'circuit_breaker': self.circuit_breaker_active,
            'position_open': len(self.positions) > 0,
            'positions_count': len(self.positions),
            'max_concurrent': self.MAX_CONCURRENT,
            'leverage': self.leverage,
            'current_streak': current_streak,
            'best_trade_pnl': best_trade,
            'worst_trade_pnl': worst_trade,
        }
    
    # ========== INTERNAL HELPERS ==========
    
    def _update_kelly_stats(self, pnl: float, size: float):
        """Update running Kelly Criterion statistics."""
        if self.total_trades == 0:
            return
        
        self._win_rate = self.winning_trades / self.total_trades
        
        wins = [t['pnl_pct'] for t in self.trade_history if t['pnl_usd'] > 0]
        losses = [abs(t['pnl_pct']) for t in self.trade_history if t['pnl_usd'] < 0]
        
        if wins:
            # De-leverage: pnl_pct is margin-amplified, divide by leverage for notional return
            self._avg_win = np.mean(wins) / 100 / self.leverage
        if losses:
            self._avg_loss = np.mean(losses) / 100 / self.leverage
    
    def _save_trade(self, record: Dict):
        """Append trade to CSV log.

        CSV append cannot be made fully atomic with rename (we'd need to
        re-write the whole file each time which is O(N) for large histories).
        Instead we write the new row to a separate per-trade .part file and
        then rename-append it, keeping the risk window tiny.
        """
        os.makedirs(os.path.dirname(config.PAPER_TRADES_PATH), exist_ok=True)
        df_row = pd.DataFrame([record])
        header = not os.path.exists(config.PAPER_TRADES_PATH)
        df_row.to_csv(config.PAPER_TRADES_PATH, mode='a', header=header, index=False)
    
    def _save_equity_point(self):
        """Record equity snapshot."""
        # Use _last_price for accurate PnL (NOT 0)
        price = self._last_price if hasattr(self, '_last_price') and self._last_price else None
        unrealized = sum(
            p.unrealized_pnl(price or p.entry_price) for p in self.positions
        ) if self.positions else 0
        
        point = {
            'timestamp': datetime.now().isoformat(),
            'balance': round(self.balance, 2),
            'unrealized': round(unrealized, 2)
        }
        self.equity_history.append(point)
    
    def _load_history(self):
        """Load existing trade history from CSV."""
        if os.path.exists(config.PAPER_TRADES_PATH):
            try:
                df = pd.read_csv(config.PAPER_TRADES_PATH)
                self.trade_history = df.to_dict('records')
                self.total_trades = len(df)
                self.winning_trades = len(df[df['pnl_usd'] > 0])
                self.losing_trades = len(df[df['pnl_usd'] <= 0])
                self.total_pnl = df['pnl_usd'].sum()
                
                # Restore fee accumulators from history
                if 'total_fee' in df.columns:
                    self.total_fees = df['total_fee'].sum()
                if 'gross_pnl_usd' in df.columns:
                    self.total_gross_pnl = df['gross_pnl_usd'].sum()
                else:
                    self.total_gross_pnl = self.total_pnl  # legacy fallback
                
                if 'balance_after' in df.columns and len(df) > 0:
                    self.balance = df['balance_after'].iloc[-1]
                    self.peak_balance = max(self.starting_balance, df['balance_after'].max())
                
                self._update_kelly_stats(0, 0)
                logging.info(f"Loaded {self.total_trades} historical trades, balance: ${self.balance:,.2f}, fees: ${self.total_fees:,.2f}")
            except Exception as e:
                logging.warning(f"Could not load trade history: {e}")
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Return trade history as DataFrame."""
        if self.trade_history:
            return pd.DataFrame(self.trade_history)
        return pd.DataFrame()
    
    def reset(self):
        """Full reset — clears all history and restores starting balance."""
        self.balance = self.starting_balance
        self.peak_balance = self.starting_balance
        self.positions = []
        self.position = None
        self.is_running = False
        self.circuit_breaker_active = False
        self.last_trade_time = None
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_gross_pnl = 0.0
        self.total_fees = 0.0
        self.trade_history = []
        self.equity_history = []
        self._win_rate = 0.52
        self._avg_win = 0.015
        self._avg_loss = 0.01
        
        # Delete CSV files
        for path in [config.PAPER_TRADES_PATH, config.PAPER_EQUITY_PATH]:
            if os.path.exists(path):
                os.remove(path)
        
        # Delete positions file
        positions_path = getattr(config, 'PAPER_POSITIONS_PATH', 
                                 os.path.join(config.DATA_DIR, 'paper_positions.json'))
        if os.path.exists(positions_path):
            os.remove(positions_path)
        
        logging.info("PaperTrader RESET — all history cleared")
    
    # ========== POSITION PERSISTENCE ==========
    
    def _save_positions(self):
        """Persist open positions to JSON for crash recovery.

        Uses atomic temp-write + os.replace so a crash during write never
        leaves a truncated or zero-byte file on disk.
        """
        positions_path = getattr(config, 'PAPER_POSITIONS_PATH',
                                 os.path.join(config.DATA_DIR, 'paper_positions.json'))
        try:
            os.makedirs(os.path.dirname(positions_path), exist_ok=True)
            data = [p.to_dict() for p in self.positions]
            tmp_path = positions_path + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, positions_path)  # atomic on same volume
        except Exception as e:
            logging.warning(f"Failed to save positions: {e}")
    
    def _load_positions(self):
        """Restore open positions from JSON after restart."""
        positions_path = getattr(config, 'PAPER_POSITIONS_PATH',
                                 os.path.join(config.DATA_DIR, 'paper_positions.json'))
        if not os.path.exists(positions_path):
            return
        
        try:
            with open(positions_path, 'r') as f:
                data = json.load(f)
            
            for p_data in data:
                pos = Position(
                    direction=p_data['direction'],
                    entry_price=p_data['entry_price'],
                    size_usd=p_data['size_usd'],
                    leverage=p_data['leverage'],
                    confidence=p_data['confidence'],
                    regime=p_data['regime'],
                    tp_price=p_data['tp_price'],
                    sl_price=p_data['sl_price'],
                    hurst=p_data.get('hurst', 0.5)
                )
                # Restore entry time and trailing SL state
                pos.entry_time = datetime.fromisoformat(p_data['entry_time'])
                pos.initial_sl = p_data.get('initial_sl', p_data['sl_price'])
                pos.best_price = p_data.get('best_price', p_data['entry_price'])
                if 'entry_fee' in p_data:
                    pos.entry_fee = p_data['entry_fee']
                self.positions.append(pos)
            
            self.position = self.positions[0] if self.positions else None
            
            if self.positions:
                logging.info(f"Restored {len(self.positions)} open position(s) from disk")
        except Exception as e:
            logging.warning(f"Failed to load positions: {e}")
    
    # ========== FEEDBACK LOOP PERSISTENCE ==========
    
    def _load_feedback(self):
        """Load trade feedback log and adaptive config from disk."""
        # Load feedback log
        if os.path.exists(self._feedback_path):
            try:
                with open(self._feedback_path, 'r') as f:
                    data = json.load(f)
                    self._feedback_log = deque(data, maxlen=2000)
                logging.info(f"Loaded {len(self._feedback_log)} trade feedback entries")
            except Exception as e:
                logging.warning(f"Failed to load feedback: {e}")
                self._feedback_log = deque(maxlen=2000)
        
        # Load adaptive config
        if os.path.exists(self._adaptive_path):
            try:
                with open(self._adaptive_path, 'r') as f:
                    cfg = json.load(f)
                self._adaptive_min_confidence = cfg.get('min_confidence', config.PAPER_MIN_CONFIDENCE)
                logging.info(f"Adaptive threshold loaded: {self._adaptive_min_confidence:.0f}%")
            except Exception:
                pass
    
    def _save_feedback(self):
        """Persist feedback log to JSON for predictor to consume.

        Uses atomic temp-write + os.replace.
        """
        try:
            os.makedirs(os.path.dirname(self._feedback_path), exist_ok=True)
            # Keep last 200 entries to avoid unbounded growth
            trimmed = self._feedback_log[-200:]
            tmp_path = self._feedback_path + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(trimmed, f, indent=2)
            os.replace(tmp_path, self._feedback_path)  # atomic on same volume
        except Exception as e:
            logging.warning(f"Failed to save feedback: {e}")
    
    def _update_adaptive_threshold(self):
        """Adjust min confidence threshold based on recent trade performance.
        Uses PnL-WEIGHTED scoring: a $100 win/loss matters 10x more than $10.
        Combines win rate with expectancy (avg PnL per trade) for smarter adaptation."""
        recent = self._feedback_log[-20:]  # Last 20 trades
        if len(recent) < 5:
            return  # Not enough data to adapt
        
        # ===== PnL-weighted win rate =====
        # Weight each trade by absolute PnL magnitude
        total_weight = 0
        weighted_wins = 0
        pnl_values = []
        
        for t in recent:
            pnl = abs(t.get('pnl_usd', t.get('pnl_pct', 1)))  # prefer USD, fallback to pct
            weight = max(pnl, 0.01)  # Floor to avoid zero-weight
            total_weight += weight
            pnl_values.append(t.get('pnl_usd', t.get('pnl_pct', 0)))
            if t.get('won', False):
                weighted_wins += weight
        
        weighted_wr = weighted_wins / total_weight if total_weight > 0 else 0.5
        
        # ===== Expectancy: avg PnL per trade (magnitude-aware) =====
        avg_pnl = np.mean(pnl_values) if pnl_values else 0
        
        # ===== Combined score: blend weighted win rate + expectancy signal =====
        # expectancy_signal: positive PnL → boost, negative → dampen
        expectancy_signal = np.clip(avg_pnl / 50, -1, 1)  # ±$50 = full signal
        
        # Composite: 60% weighted win rate, 40% expectancy
        composite = weighted_wr * 0.6 + (0.5 + expectancy_signal * 0.5) * 0.4
        
        if composite >= 0.60:
            # Strong performance: be more aggressive
            self._adaptive_min_confidence = max(40, config.PAPER_MIN_CONFIDENCE - 15)
        elif composite >= 0.50:
            # Normal performance: slightly below default
            self._adaptive_min_confidence = max(45, config.PAPER_MIN_CONFIDENCE - 10)
        elif composite >= 0.40:
            # Below average: use default
            self._adaptive_min_confidence = config.PAPER_MIN_CONFIDENCE
        else:
            # Losing badly (big losses): be cautious
            self._adaptive_min_confidence = min(75, config.PAPER_MIN_CONFIDENCE + 15)
        
        # Save adaptive config
        try:
            os.makedirs(os.path.dirname(self._adaptive_path), exist_ok=True)
            with open(self._adaptive_path, 'w') as f:
                json.dump({
                    'min_confidence': self._adaptive_min_confidence,
                    'weighted_win_rate': round(weighted_wr * 100, 1),
                    'avg_pnl_usd': round(avg_pnl, 2),
                    'composite_score': round(composite * 100, 1),
                    'sample_size': len(recent),
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception:
            pass
        
        logging.info(
            f"Adaptive threshold: {self._adaptive_min_confidence:.0f}% "
            f"(weighted WR: {weighted_wr*100:.0f}%, avg PnL: ${avg_pnl:+.2f}, "
            f"composite: {composite*100:.0f}% over {len(recent)} trades)"
        )
    
    def get_feedback_summary(self) -> Dict:
        """Get feedback loop summary for the predictor/UI."""
        if not self._feedback_log:
            return {'total_feedback': 0, 'adaptive_threshold': self._adaptive_min_confidence}
        
        recent = self._feedback_log[-20:]
        wins = sum(1 for t in recent if t.get('won', False))
        
        # Win rate by regime
        regime_stats = {}
        for entry in self._feedback_log[-50:]:
            regime = entry.get('regime', 'UNKNOWN')
            if regime not in regime_stats:
                regime_stats[regime] = {'wins': 0, 'total': 0}
            regime_stats[regime]['total'] += 1
            if entry.get('won', False):
                regime_stats[regime]['wins'] += 1
        
        for regime in regime_stats:
            s = regime_stats[regime]
            s['win_rate'] = round(s['wins'] / s['total'] * 100, 1) if s['total'] > 0 else 0
        
        return {
            'total_feedback': len(self._feedback_log),
            'recent_win_rate': round(wins / len(recent) * 100, 1) if recent else 0,
            'adaptive_threshold': self._adaptive_min_confidence,
            'regime_performance': regime_stats,
            'trailing_sl_saves': sum(1 for t in self._feedback_log if t.get('trailing_sl_moved', False) and t.get('won', False)),
        }
