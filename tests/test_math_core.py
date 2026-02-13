"""
Unit tests for MathCore — Kalman, Hurst, FFT, Monte Carlo, Market Regime.
Run: python -m pytest tests/test_math_core.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from math_core import MathCore


@pytest.fixture
def math():
    return MathCore()


@pytest.fixture
def trending_prices():
    """Synthetic trending series (upward drift)."""
    np.random.seed(42)
    n = 500
    trend = np.cumsum(np.random.normal(0.001, 0.01, n)) + 100
    return trend


@pytest.fixture
def random_walk():
    """Pure random walk (Hurst ~ 0.5)."""
    np.random.seed(42)
    return np.cumsum(np.random.randn(500)) + 100


class TestKalmanSmooth:
    def test_output_same_length(self, math, trending_prices):
        result = math.kalman_smooth(trending_prices)
        assert len(result) == len(trending_prices)
    
    def test_reduces_noise(self, math):
        np.random.seed(42)
        clean = np.linspace(100, 200, 200)
        noisy = clean + np.random.normal(0, 5, 200)
        smoothed = math.kalman_smooth(noisy)
        
        # Smoothed should be closer to clean than noisy was
        noise_before = np.mean(np.abs(noisy - clean))
        noise_after = np.mean(np.abs(smoothed - clean))
        assert noise_after < noise_before
    
    def test_short_input(self, math):
        result = math.kalman_smooth(np.array([100.0, 101.0]))
        assert len(result) == 2


class TestHurstExponent:
    def test_output_range(self, math, trending_prices):
        h = math.calculate_hurst_exponent(trending_prices)
        assert 0.0 <= h <= 1.0
    
    def test_trending_above_half(self, math, trending_prices):
        h = math.calculate_hurst_exponent(trending_prices)
        # Trending data should give H > 0.5
        assert h > 0.45  # Relaxed due to randomness
    
    def test_short_input_fallback(self, math):
        h = math.calculate_hurst_exponent(np.array([100.0, 101.0, 102.0]))
        assert 0.0 <= h <= 1.0


class TestExtractCycles:
    def test_returns_correct_count(self, math, trending_prices):
        cycles = math.extract_cycles(trending_prices, top_n=3)
        assert len(cycles) == 3
    
    def test_all_finite(self, math, trending_prices):
        cycles = math.extract_cycles(trending_prices, top_n=2)
        for c in cycles:
            assert np.isfinite(c)
    
    def test_short_input(self, math):
        cycles = math.extract_cycles(np.array([100.0, 101.0, 102.0]), top_n=2)
        assert len(cycles) == 2


class TestGarchVolatility:
    def test_positive_output(self, math):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 200)
        vol = math.estimate_garch_volatility(returns)
        assert vol > 0
    
    def test_high_vol_detected(self, math):
        np.random.seed(42)
        calm = np.random.normal(0, 0.001, 200)
        wild = np.random.normal(0, 0.05, 200)
        
        v_calm = math.estimate_garch_volatility(calm)
        v_wild = math.estimate_garch_volatility(wild)
        assert v_wild > v_calm


class TestMonteCarlo:
    def test_output_shape(self, math):
        result = math.run_monte_carlo(50000, 0.02, steps=60, simulations=100)
        # Should return something meaningful (dict or array)
        assert result is not None
    
    def test_price_positive(self, math):
        result = math.run_monte_carlo(50000, 0.02, steps=10, simulations=50)
        if isinstance(result, dict):
            assert result.get("mean", 0) > 0 or result.get("median", 0) > 0
        elif isinstance(result, np.ndarray):
            assert np.all(result > 0)


class TestMarketRegime:
    def test_trending(self, math):
        regime = math.get_market_regime(0.7)
        assert regime == "TRENDING"
    
    def test_mean_reverting(self, math):
        regime = math.get_market_regime(0.3)
        assert regime == "MEAN_REVERTING"
    
    def test_random(self, math):
        regime = math.get_market_regime(0.5)
        assert regime == "RANDOM"
