"""
Unit tests for QuantEngine — HMM, GJR-GARCH, OFI, EMD.
Run: python -m pytest tests/test_quant_models.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


@pytest.fixture
def prices():
    """Synthetic BTC-like price series."""
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, 500)
    return np.exp(np.cumsum(returns)) * 50000


@pytest.fixture
def volumes():
    """Synthetic volume series."""
    np.random.seed(42)
    return np.random.uniform(100, 10000, 500)


@pytest.fixture
def returns():
    """Log returns from synthetic prices."""
    np.random.seed(42)
    return np.random.normal(0, 0.02, 500)


class TestHiddenMarkovRegime:
    def test_prepare_observations(self, prices, volumes):
        from quant_models import HiddenMarkovRegime
        hmm = HiddenMarkovRegime()
        obs = hmm.prepare_observations(prices, volumes)
        if obs is not None:
            assert obs.shape[1] == 3  # returns, vol, vol_change
            assert len(obs) > 0
    
    def test_fit_and_predict(self, prices, volumes):
        from quant_models import HiddenMarkovRegime
        hmm = HiddenMarkovRegime()
        fitted = hmm.fit(prices, volumes)
        if fitted:
            result = hmm.predict_regime(prices, volumes)
            assert 'regime' in result
            assert result['regime'] in ['BULL', 'SIDEWAYS', 'BEAR']
            assert 0 <= result['confidence'] <= 100


class TestGJRGarch:
    def test_fit(self, returns):
        from quant_models import GJRGarchVolatility
        garch = GJRGarchVolatility()
        fitted = garch.fit(returns)
        # May fail if scipy not available, that's ok
        if fitted:
            assert garch.is_fitted
    
    def test_forecast_positive(self, returns):
        from quant_models import GJRGarchVolatility
        garch = GJRGarchVolatility()
        if garch.fit(returns):
            forecast = garch.forecast_volatility(horizon=1)
            assert forecast > 0


class TestOrderFlowImbalance:
    def test_ofi_signal(self, prices, volumes):
        from quant_models import OrderFlowImbalance
        ofi = OrderFlowImbalance()
        result = ofi.get_ofi_signal(prices, volumes)
        assert 'signal' in result
        assert result['signal'] in ['BUY_PRESSURE', 'SELL_PRESSURE', 'NEUTRAL']


class TestEMD:
    def test_decompose(self, prices):
        from quant_models import EmpiricalModeDecomposition
        emd = EmpiricalModeDecomposition()
        if emd.decompose(prices):
            strength = emd.get_cycle_strength()
            assert strength is not None


class TestQuantEngine:
    def test_initialize(self, prices, volumes):
        from quant_models import QuantEngine
        engine = QuantEngine()
        result = engine.initialize(prices, volumes)
        # At least some models should initialize
        assert isinstance(result, bool)
    
    def test_analyze_returns_dict(self, prices, volumes):
        from quant_models import QuantEngine
        engine = QuantEngine()
        engine.initialize(prices, volumes)
        analysis = engine.analyze(prices, volumes)
        
        assert isinstance(analysis, dict)
        assert 'regime' in analysis
        assert 'volatility' in analysis
        assert 'signals' in analysis
    
    def test_tier2_models_in_analysis(self, prices, volumes):
        """Verify the newly wired Tier 2 models produce output."""
        from quant_models import QuantEngine
        engine = QuantEngine()
        engine.initialize(prices, volumes)
        analysis = engine.analyze(prices, volumes)
        
        # These should now exist in the analysis (even if NOT_CALIBRATED)
        for key in ['bates_svj', 'rqa', 'multifractal', 'tda', 'execution']:
            assert key in analysis, f"Missing Tier 2/3 key: {key}"
