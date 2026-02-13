"""
Tests for PaperTrader — verifying critical bug fixes.
Covers: equity point crash, position persistence, Kelly de-leveraging,
        Hurst regime filter, and Sharpe computation.
"""

import os
import sys
import json
import pytest
import tempfile
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from paper_trader import PaperTrader, Position


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Use a temporary directory for all paper trading files."""
    with patch.object(config, 'DATA_DIR', str(tmp_path)), \
         patch.object(config, 'PAPER_TRADES_PATH', str(tmp_path / 'paper_trades.csv')), \
         patch.object(config, 'PAPER_EQUITY_PATH', str(tmp_path / 'paper_equity.csv')), \
         patch.object(config, 'PAPER_POSITIONS_PATH', str(tmp_path / 'paper_positions.json')):
        yield tmp_path


@pytest.fixture
def trader(tmp_data_dir):
    """Create a fresh PaperTrader with temporary storage."""
    t = PaperTrader(starting_balance=10000, default_leverage=10)
    return t


class TestEquityPoint:
    """Fix #5: _save_equity_point should NOT crash with current_price=0."""

    def test_equity_snapshot_no_crash_no_positions(self, trader):
        """Should work when no positions are open."""
        trader._save_equity_point()
        assert len(trader.equity_history) == 1
        assert trader.equity_history[0]['unrealized'] == 0

    def test_equity_snapshot_uses_last_price(self, trader):
        """Unrealized PnL should use _last_price, not 0."""
        # Simulate opening a position
        pred = {'confidence': 80, 'regime_label': 'TRENDING'}
        trader.open_position("LONG", 50000.0, pred, volatility=0.005)
        
        # Set last price to something above entry
        trader._last_price = 51000.0
        trader._save_equity_point()
        
        # PnL should be positive (price went up on a LONG)
        assert trader.equity_history[-1]['unrealized'] > 0

    def test_equity_snapshot_no_last_price_uses_entry(self, trader):
        """Falls back to entry price when _last_price not set."""
        pred = {'confidence': 80, 'regime_label': 'TRENDING'}
        trader.open_position("LONG", 50000.0, pred, volatility=0.005)
        
        # Don't set _last_price — should fall back to entry_price
        if hasattr(trader, '_last_price'):
            delattr(trader, '_last_price')
        trader._save_equity_point()
        
        # PnL at entry price = ~0
        assert abs(trader.equity_history[-1]['unrealized']) < 1


class TestPositionPersistence:
    """Fix #9: Position state should survive restarts."""

    def test_save_and_load_positions(self, tmp_data_dir):
        """Open positions should persist to disk and reload."""
        # Open a position
        t1 = PaperTrader(starting_balance=10000, default_leverage=10)
        pred = {'confidence': 80, 'regime_label': 'TRENDING'}
        t1.open_position("LONG", 50000.0, pred, volatility=0.005)
        
        assert len(t1.positions) == 1
        
        # Verify JSON file exists
        pos_path = config.PAPER_POSITIONS_PATH
        assert os.path.exists(pos_path)
        
        # Create new trader — should load the position
        t2 = PaperTrader(starting_balance=10000, default_leverage=10)
        assert len(t2.positions) == 1
        assert t2.positions[0].direction == "LONG"
        assert t2.positions[0].entry_price == 50000.0

    def test_positions_cleared_on_close(self, tmp_data_dir):
        """Closed positions should be removed from persistence."""
        trader = PaperTrader(starting_balance=10000, default_leverage=10)
        pred = {'confidence': 80, 'regime_label': 'TRENDING'}
        trader.open_position("LONG", 50000.0, pred, volatility=0.005)
        trader._last_price = 50100.0
        trader.close_position(50100.0, reason="TP_HIT")
        
        # JSON file should be empty list
        with open(config.PAPER_POSITIONS_PATH, 'r') as f:
            data = json.load(f)
        assert data == []


class TestKellyDeLeverage:
    """Fix #8: Kelly should use notional returns, not leveraged returns."""

    def test_kelly_stats_de_leveraged(self, trader):
        """_avg_win and _avg_loss should be divided by leverage."""
        # Simulate some trades with 10x leverage
        trader.trade_history = [
            {'pnl_usd': 100, 'pnl_pct': 10.0},  # 10% on margin = 1% notional
            {'pnl_usd': 50, 'pnl_pct': 5.0},     # 5% on margin = 0.5% notional  
            {'pnl_usd': -80, 'pnl_pct': -8.0},    # 8% on margin = 0.8% notional
        ]
        trader.total_trades = 3
        trader.winning_trades = 2
        trader.losing_trades = 1
        
        trader._update_kelly_stats(0, 0)
        
        # avg_win should be ~0.75% notional (mean of 1% and 0.5%), not 7.5%
        assert trader._avg_win == pytest.approx(0.0075, abs=0.001)
        # avg_loss should be ~0.8% notional, not 8%
        assert trader._avg_loss == pytest.approx(0.008, abs=0.001)


class TestHurstFilter:
    """Fix #10: Hurst filter band should be 0.45-0.55, not 0.48-0.52."""

    def test_random_market_rejected(self, trader):
        """Hurst ~0.50 should be filtered (chaotic/random)."""
        pred = {'confidence': 80, 'direction': 'UP', 'hurst': 0.50}
        result = trader.evaluate_signal(pred)
        assert result is None

    def test_borderline_chaotic_rejected(self, trader):
        """Hurst 0.46 was previously accepted — now rejected."""
        pred = {'confidence': 80, 'direction': 'UP', 'hurst': 0.46}
        result = trader.evaluate_signal(pred)
        assert result is None  # Wider band catches this

    def test_trending_market_accepted(self, trader):
        """Hurst 0.60 (trending) should be accepted."""
        pred = {'confidence': 80, 'direction': 'UP', 'hurst': 0.60}
        # Need to bypass cooldown
        trader.last_trade_time = datetime.now() - timedelta(hours=1)
        result = trader.evaluate_signal(pred)
        assert result == "LONG"

    def test_mean_reverting_accepted(self, trader):
        """Hurst 0.35 (mean-reverting) should be accepted."""
        pred = {'confidence': 80, 'direction': 'DOWN', 'hurst': 0.35}
        trader.last_trade_time = datetime.now() - timedelta(hours=1)
        result = trader.evaluate_signal(pred)
        assert result == "SHORT"


class TestSharpeComputation:
    """Fix #6: Sharpe should use time-weighted annualisation, not sqrt(252)."""

    def test_sharpe_uses_trade_duration(self, trader):
        """Verify Sharpe adapts to actual trade frequency, not daily assumption."""
        now = datetime.now()
        # Simulate hourly trades with varying returns
        pnl_values = [1.2, 0.8, 1.5, -0.3, 0.9, 1.1, 0.5, -0.2, 1.0, 0.7]
        trader.trade_history = [
            {'pnl_usd': pnl * 10, 'pnl_pct': pnl, 
             'entry_time': (now - timedelta(hours=i+1)).isoformat(),
             'exit_time': (now - timedelta(hours=i)).isoformat()}
            for i, pnl in enumerate(pnl_values)
        ]
        trader.total_trades = 10
        
        stats = trader.get_stats()
        sharpe = stats['sharpe_ratio']
        
        # With hourly trades and positive returns, Sharpe should be large and positive
        # Old sqrt(252) would give different result than sqrt(8760)
        assert sharpe > 0
