"""
Unit tests for NexusPredictor — feature engineering, training, prediction, save/load.
Run: python -m pytest tests/test_predictor.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import tempfile
import shutil


@pytest.fixture
def mock_data_dir(tmp_path):
    """Set up a temp data dir with synthetic market data."""
    import config
    
    # Override config paths to use temp dir
    original_data_dir = config.DATA_DIR
    original_model_dir = config.MODEL_DIR
    original_csv = config.MARKET_DATA_PATH
    original_parquet = config.MARKET_DATA_PARQUET_PATH
    
    data_dir = str(tmp_path / "data")
    model_dir = str(tmp_path / "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    config.DATA_DIR = data_dir
    config.MODEL_DIR = model_dir
    config.MARKET_DATA_PATH = os.path.join(data_dir, "market_data.csv")
    config.MARKET_DATA_PARQUET_PATH = os.path.join(data_dir, "market_data.parquet")
    
    # Create synthetic market data (200 candles)
    np.random.seed(42)
    n = 200
    timestamps = pd.date_range("2025-01-01", periods=n, freq="1min")
    close = np.cumsum(np.random.normal(0, 50, n)) + 50000
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": close - np.random.uniform(0, 30, n),
        "high": close + np.random.uniform(0, 50, n),
        "low": close - np.random.uniform(0, 50, n),
        "close": close,
        "volume": np.random.uniform(10, 1000, n)
    })
    df.to_csv(config.MARKET_DATA_PATH, index=False)
    df.to_parquet(config.MARKET_DATA_PARQUET_PATH, index=False)
    
    yield config
    
    # Restore
    config.DATA_DIR = original_data_dir
    config.MODEL_DIR = original_model_dir
    config.MARKET_DATA_PATH = original_csv
    config.MARKET_DATA_PARQUET_PATH = original_parquet


class TestDataLoading:
    def test_load_parquet_preferred(self, mock_data_dir):
        from predictor import NexusPredictor
        p = NexusPredictor()
        df = p._load_market_data()
        assert df is not None
        assert len(df) == 200
    
    def test_load_csv_fallback(self, mock_data_dir):
        # Remove parquet to force CSV fallback
        if os.path.exists(mock_data_dir.MARKET_DATA_PARQUET_PATH):
            os.remove(mock_data_dir.MARKET_DATA_PARQUET_PATH)
        
        from predictor import NexusPredictor
        p = NexusPredictor()
        df = p._load_market_data()
        assert df is not None
        assert len(df) == 200
    
    def test_load_returns_none_no_data(self, mock_data_dir):
        # Remove both files
        for path in [mock_data_dir.MARKET_DATA_PATH, mock_data_dir.MARKET_DATA_PARQUET_PATH]:
            if os.path.exists(path):
                os.remove(path)
        
        from predictor import NexusPredictor
        p = NexusPredictor()
        df = p._load_market_data()
        assert df is None


class TestFeatureEngineering:
    def test_features_created(self, mock_data_dir):
        from predictor import NexusPredictor
        p = NexusPredictor()
        df = p.load_and_engineer_features()
        assert df is not None
        # Check key features exist
        for col in ['sma_20', 'sma_50', 'rsi', 'volatility', 'hurst']:
            assert col in df.columns, f"Missing feature: {col}"
    
    def test_rsi_range(self, mock_data_dir):
        from predictor import NexusPredictor
        p = NexusPredictor()
        df = p.load_and_engineer_features()
        rsi = df['rsi'].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100


class TestTemporalSplit:
    def test_no_overlap(self, mock_data_dir):
        from predictor import NexusPredictor
        p = NexusPredictor()
        df = p.load_and_engineer_features()
        if df is not None and len(df) > 50:
            train, test = p.create_temporal_split(df)
            assert len(train) + len(test) == len(df)
            # Train should be chronologically before test
            assert train.index[-1] < test.index[0]


class TestTargetVariable:
    def test_target_binary(self, mock_data_dir):
        from predictor import NexusPredictor
        p = NexusPredictor()
        df = p.load_and_engineer_features()
        if df is not None:
            df_with_target = p.create_target_variable(df, for_training=True)
            assert 'target' in df_with_target.columns
            assert set(df_with_target['target'].unique()).issubset({0, 1})
    
    def test_live_mode_no_target(self, mock_data_dir):
        from predictor import NexusPredictor
        p = NexusPredictor()
        df = p.load_and_engineer_features()
        if df is not None:
            df_live = p.create_target_variable(df, for_training=False)
            assert 'target' not in df_live.columns


class TestPrediction:
    def test_prediction_structure(self, mock_data_dir):
        from predictor import NexusPredictor
        p = NexusPredictor()
        # Train first
        p.train()
        pred = p.get_prediction()
        
        assert 'direction' in pred
        assert pred['direction'] in ['UP', 'DOWN', 'NEUTRAL']
        assert 'confidence' in pred
        assert 0 <= pred['confidence'] <= 100
