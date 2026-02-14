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
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import config
from nexus_logger import NexusLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Position:
    """Represents an open paper trading position."""
    
    def __init__(self, direction: str, entry_price: float, size_usd: float,
                 leverage: int, confidence: float, regime: str,
                 tp_price: float, sl_price: float, hurst: float = 0.5):
        self.direction = direction          # "LONG" or "SHORT"
        self.entry_price = entry_price
        self.size_usd = size_usd            # Notional value
        self.margin = size_usd / leverage   # Actual capital used
        self.leverage = leverage
        self.confidence = confidence
        self.regime = regime
        self.hurst = hurst                  # Hurst at entry (for feedback)
        self.tp_price = tp_price            # Take-profit
        self.sl_price = sl_price            # Stop-loss
        self.initial_sl = sl_price           # Original SL (before trailing)
        self.entry_time = datetime.now()
        self.liquidation_price = self._calc_liquidation()
        
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
    
    def to_dict(self) -> Dict:
        return {
            'direction': self.direction,
            'entry_price': self.entry_price,
            'size_usd': self.size_usd,
            'margin': self.margin,
            'leverage': self.leverage,
            'confidence': self.confidence,
            'regime': self.regime,
            'hurst': self.hurst,
            'tp_price': self.tp_price,
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
    MAX_CONCURRENT = 3

    def __init__(self, starting_balance: float = None, default_leverage: int = None):
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
        
        # ===== PHASE 2: Feedback Loop =====
        # Trade feedback log — persisted to JSON for predictor to consume
        self._feedback_path = os.path.join(config.DATA_ROOT, 'feedback', 'trade_feedback.json')
        self._adaptive_path = os.path.join(config.DATA_ROOT, 'feedback', 'adaptive_config.json')
        self._feedback_log = []
        self._adaptive_min_confidence = config.PAPER_MIN_CONFIDENCE  # starts at default
        self._load_feedback()
        
        # Logger
        self.nlog = NexusLogger()
        
        # Load existing history if any
        self._load_history()
        self._load_positions()
        
        self.nlog.log_system(f"PaperTrader initialized: ${self.balance:,.2f} balance, {self.leverage}x leverage")
        logging.info(f"PaperTrader initialized: ${self.balance:,.2f} balance, {self.leverage}x leverage")
    
    # ========== RISK MANAGEMENT ==========
    
    def kelly_fraction(self) -> float:
        """
        Kelly Criterion: f* = (p * b - q) / b
        Where p = win probability, q = 1-p, b = avg_win / avg_loss
        Capped at 25% for safety (half-Kelly is standard practice).
        """
        if self._avg_loss == 0 or self.total_trades < 5:
            return 0.02  # Default 2% until enough data
        
        p = self._win_rate
        q = 1 - p
        b = self._avg_win / self._avg_loss
        
        kelly = (p * b - q) / b
        
        # Half-Kelly (standard for real trading)
        half_kelly = kelly / 2
        
        # Floor 1%, cap at 25%
        return max(0.01, min(0.25, half_kelly))
    
    def calculate_position_size(self) -> float:
        """Calculate position size in USD based on Kelly fraction and leverage."""
        fraction = self.kelly_fraction()
        risk_amount = self.balance * fraction
        notional = risk_amount * self.leverage
        return notional
    
    def check_circuit_breaker(self) -> bool:
        """Check if drawdown exceeds maximum allowed."""
        if self.peak_balance == 0:
            return False
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown >= config.PAPER_MAX_DRAWDOWN:
            self.circuit_breaker_active = True
            logging.warning(f"CIRCUIT BREAKER: Drawdown {drawdown:.1%} exceeds {config.PAPER_MAX_DRAWDOWN:.0%} limit!")
            return True
        return False
    
    def check_cooldown(self) -> bool:
        """Check if enough time has passed since last trade."""
        if self.last_trade_time is None:
            return True
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        return elapsed >= config.PAPER_COOLDOWN_SEC
    
    def calculate_tp_sl(self, entry_price: float, direction: str, 
                        volatility: float = 0.005) -> Tuple[float, float]:
        """
        Calculate TP/SL with professional 2:1 reward-to-risk ratio.
        Uses ATR-like volatility for dynamic levels.
        """
        # Risk = 1x volatility, Reward = 1.5x volatility (tighter scalp levels)
        risk_pct = max(volatility * 1.0, 0.002)   # Floor at 0.2%
        reward_pct = risk_pct * 1.5               # 1.5:1 R:R
        
        if direction == "LONG":
            tp = entry_price * (1 + reward_pct)
            sl = entry_price * (1 - risk_pct)
        else:
            tp = entry_price * (1 - reward_pct)
            sl = entry_price * (1 + risk_pct)
        
        return tp, sl
    
    # ========== TRADING LOGIC ==========
    
    def evaluate_signal(self, prediction: Dict) -> Optional[str]:
        """
        Evaluate the AI prediction and decide whether to trade.
        Uses signal confirmation: requires consecutive predictions in same
        direction before committing capital.
        Returns "LONG", "SHORT", or None.
        """
        # Extract prediction data
        confidence = prediction.get('confidence', 0)
        direction = prediction.get('direction', 'NEUTRAL')
        hurst = prediction.get('hurst', 0.5)
        
        # Rule 0: NEUTRAL = no signal
        if direction == 'NEUTRAL':
            self._signal_streak = 0
            return None
        
        # Rule 1: Adaptive minimum confidence (adjusts based on recent performance)
        min_conf = self._adaptive_min_confidence
        if confidence < min_conf:
            self._signal_streak = 0
            logging.debug(f"Skip: confidence {confidence:.1f}% < {min_conf:.0f}% (adaptive)")
            return None
        
        # Rule 2: Regime filter — skip chaotic markets (Hurst near 0.5)
        if 0.48 <= hurst <= 0.52:
            self._signal_streak = 0
            logging.debug(f"Skip: Hurst {hurst:.3f} indicates random/chaotic regime")
            return None
        
        # Rule 2b: Vol-regime filter — skip extreme volatility (whipsaw) and dead markets
        vol_regime = prediction.get('vol_regime', 1.0)
        vol_max = getattr(config, 'REGIME_VOL_MAX', 3.0)
        vol_min = getattr(config, 'REGIME_VOL_MIN', 0.15)
        if vol_regime > vol_max:
            self._signal_streak = 0
            logging.debug(f"Skip: vol_regime {vol_regime:.2f} > {vol_max} (extreme volatility)")
            return None
        if vol_regime < vol_min:
            self._signal_streak = 0
            logging.debug(f"Skip: vol_regime {vol_regime:.2f} < {vol_min} (dead market)")
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
        
        # Rule 3: Signal confirmation — need 2+ consecutive predictions in same direction
        wanted = 'LONG' if direction == 'UP' else 'SHORT'
        if not hasattr(self, '_signal_streak'):
            self._signal_streak = 0
            self._signal_direction = None
        
        if wanted == self._signal_direction:
            self._signal_streak += 1
        else:
            self._signal_direction = wanted
            self._signal_streak = 1
        
        min_confirms = 2  # Need 2 consecutive same-direction signals
        if self._signal_streak < min_confirms:
            logging.debug(f"Skip: signal confirmation {self._signal_streak}/{min_confirms} for {wanted}")
            return None
        
        # Rule 4: Cooldown period
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
        
        # Rule 7: No stacking same direction — diversify exposure
        open_dirs = [p.direction for p in self.positions]
        if wanted in open_dirs:
            logging.debug(f"Skip: already have {wanted} position open — no stacking")
            return None
        
        # All checks passed — reset streak and trade
        self._signal_streak = 0
        logging.info(f"SIGNAL CONFIRMED: {wanted} @ confidence={confidence:.1f}%, hurst={hurst:.3f}")
        return wanted
    
    def open_position(self, direction: str, current_price: float, 
                      prediction: Dict, volatility: float = 0.005):
        """Open a new paper trading position (supports multiple concurrent)."""
        if len(self.positions) >= self.MAX_CONCURRENT:
            logging.warning(f"Cannot open: max {self.MAX_CONCURRENT} positions reached")
            return False
        
        # Calculate available margin (total balance minus margin used by open positions)
        used_margin = sum(p.margin for p in self.positions)
        available = self.balance - used_margin
        
        # Position sizing via Kelly — fraction of AVAILABLE balance
        fraction = self.kelly_fraction()
        risk_amount = available * fraction
        size_usd = risk_amount * self.leverage
        margin = size_usd / self.leverage
        
        # Ensure we have enough available margin
        if margin > available * 0.90:  # Keep 10% buffer per slot
            margin = available * 0.80
            size_usd = margin * self.leverage
        
        if margin < 10:  # Minimum $10 margin
            logging.warning(f"Not enough margin: ${available:.2f} available")
            return False
        
        # Calculate TP/SL
        tp, sl = self.calculate_tp_sl(current_price, direction, volatility)
        
        # Create position and add to list
        new_pos = Position(
            direction=direction,
            entry_price=current_price,
            size_usd=size_usd,
            leverage=self.leverage,
            confidence=prediction.get('confidence', 0),
            regime=prediction.get('regime_label', 'UNKNOWN'),
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
        
        self.nlog.log_trade_open(direction, current_price, size_usd, self.leverage, 
                                  prediction.get('confidence', 0))
        logging.info(
            f"OPENED {direction} #{len(self.positions)} @ ${current_price:,.2f} | "
            f"Size: ${size_usd:,.2f} ({self.leverage}x) | Fee: ${entry_fee:.2f} | "
            f"TP: ${tp:,.2f} | SL: ${sl:,.2f} | "
            f"Liq: ${new_pos.liquidation_price:,.2f}"
        )
        
        return True
    
    def close_position(self, current_price: float, reason: str = "MANUAL", pos: 'Position' = None) -> Dict:
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
        
        return trade_record
    
    def update(self, current_price: float, prediction: Dict = None, 
               volatility: float = 0.005) -> Optional[Dict]:
        """
        Main update loop — call this every tick.
        Checks open position for TP/SL/liquidation, 
        then evaluates new signals if flat.
        
        Returns trade record if a trade was closed, None otherwise.
        """
        result = None
        
        # 1. Update trailing stop loss for ALL open positions
        for pos in list(self.positions):
            pos.update_trailing_sl(current_price)
        
        # 2. Check ALL open positions for exits
        for pos in list(self.positions):  # Copy list since we may remove during iteration
            # Time-based exit: auto-close positions held too long
            hold_secs = (datetime.now() - pos.entry_time).total_seconds()
            max_hold = getattr(config, 'PAPER_MAX_HOLD_SEC', 3600)
            if hold_secs > max_hold:
                result = self.close_position(current_price, "MAX_HOLD_TIME", pos)
            elif pos.should_liquidate(current_price):
                result = self.close_position(current_price, "LIQUIDATED", pos)
            elif pos.should_tp(current_price):
                result = self.close_position(current_price, "TAKE_PROFIT", pos)
            elif pos.should_sl(current_price):
                # Determine if this was the trailing SL or original SL
                sl_reason = "TRAILING_STOP" if pos.sl_price != pos.initial_sl else "STOP_LOSS"
                result = self.close_position(current_price, sl_reason, pos)
        
        # 2. Evaluate new signals if slots available and running
        if len(self.positions) < self.MAX_CONCURRENT and self.is_running and prediction is not None:
            signal = self.evaluate_signal(prediction)
            if signal is not None:
                self.open_position(signal, current_price, prediction, volatility)
        
        # 3. Record equity
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
            'leverage': self.leverage
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
        """Append trade to CSV log."""
        os.makedirs(os.path.dirname(config.PAPER_TRADES_PATH), exist_ok=True)
        df = pd.DataFrame([record])
        header = not os.path.exists(config.PAPER_TRADES_PATH)
        df.to_csv(config.PAPER_TRADES_PATH, mode='a', header=header, index=False)
    
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
        """Persist open positions to JSON for crash recovery."""
        positions_path = getattr(config, 'PAPER_POSITIONS_PATH',
                                 os.path.join(config.DATA_DIR, 'paper_positions.json'))
        try:
            os.makedirs(os.path.dirname(positions_path), exist_ok=True)
            data = [p.to_dict() for p in self.positions]
            with open(positions_path, 'w') as f:
                json.dump(data, f, indent=2)
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
                    self._feedback_log = json.load(f)
                logging.info(f"Loaded {len(self._feedback_log)} trade feedback entries")
            except Exception as e:
                logging.warning(f"Failed to load feedback: {e}")
                self._feedback_log = []
        
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
        """Persist feedback log to JSON for predictor to consume."""
        try:
            os.makedirs(os.path.dirname(self._feedback_path), exist_ok=True)
            # Keep last 200 entries to avoid unbounded growth
            trimmed = self._feedback_log[-200:]
            with open(self._feedback_path, 'w') as f:
                json.dump(trimmed, f, indent=2)
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
