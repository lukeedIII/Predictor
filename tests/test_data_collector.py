"""
Unit tests for DataCollector — fetch, append, dedup.
Run: python -m pytest tests/test_data_collector.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime


@pytest.fixture
def sample_df():
    """Create a sample OHLCV DataFrame."""
    return pd.DataFrame({
        'timestamp': pd.date_range("2025-01-01", periods=5, freq="1min"),
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [101.0, 102.0, 103.0, 104.0, 105.0],
        'low': [99.0, 100.0, 101.0, 102.0, 103.0],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })


@pytest.fixture
def overlapping_df():
    """Create data that overlaps with sample_df."""
    return pd.DataFrame({
        'timestamp': pd.date_range("2025-01-01 00:03", periods=5, freq="1min"),
        'open': [103.0, 104.0, 105.0, 106.0, 107.0],
        'high': [104.0, 105.0, 106.0, 107.0, 108.0],
        'low': [102.0, 103.0, 104.0, 105.0, 106.0],
        'close': [103.5, 104.5, 105.5, 106.5, 107.5],
        'volume': [1300, 1400, 1500, 1600, 1700]
    })


class TestUpdateCSV:
    def test_creates_new_file(self, tmp_path, sample_df):
        import config
        original = config.MARKET_DATA_PATH
        csv_path = str(tmp_path / "market_data.csv")
        config.MARKET_DATA_PATH = csv_path
        
        from data_collector import DataCollector
        collector = DataCollector()
        collector.csv_path = csv_path
        collector.update_csv(sample_df)
        
        assert os.path.exists(csv_path)
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 5
        
        config.MARKET_DATA_PATH = original
    
    def test_appends_new_data(self, tmp_path, sample_df, overlapping_df):
        import config
        original = config.MARKET_DATA_PATH
        csv_path = str(tmp_path / "market_data.csv")
        config.MARKET_DATA_PATH = csv_path
        
        from data_collector import DataCollector
        collector = DataCollector()
        collector.csv_path = csv_path
        
        # Write initial data
        collector.update_csv(sample_df)
        # Append overlapping data
        collector.update_csv(overlapping_df)
        
        loaded = pd.read_csv(csv_path)
        # 5 original + 5 overlap, but 2 overlap (min 3 and min 4) = 8 unique
        assert len(loaded) == 8  # Deduplication by timestamp
        
        config.MARKET_DATA_PATH = original
    
    def test_parquet_also_created(self, tmp_path, sample_df, overlapping_df):
        import config
        original = config.MARKET_DATA_PATH
        csv_path = str(tmp_path / "market_data.csv")
        config.MARKET_DATA_PATH = csv_path
        
        from data_collector import DataCollector
        collector = DataCollector()
        collector.csv_path = csv_path
        
        # First write creates the file (no parquet yet since it goes to else branch)
        collector.update_csv(sample_df)
        # Second write triggers parquet creation
        collector.update_csv(overlapping_df)
        
        parquet_path = csv_path.replace('.csv', '.parquet')
        assert os.path.exists(parquet_path), "Parquet file should be created on update"
        
        config.MARKET_DATA_PATH = original
    
    def test_handles_none(self, tmp_path):
        from data_collector import DataCollector
        collector = DataCollector()
        collector.csv_path = str(tmp_path / "market_data.csv")
        # Should not crash on None input
        collector.update_csv(None)
        assert not os.path.exists(collector.csv_path)
    
    def test_handles_empty_df(self, tmp_path):
        from data_collector import DataCollector
        collector = DataCollector()
        collector.csv_path = str(tmp_path / "market_data.csv")
        collector.update_csv(pd.DataFrame())
        assert not os.path.exists(collector.csv_path)


class TestGetLatestPrice:
    def test_returns_last_close(self, tmp_path, sample_df):
        import config
        original = config.MARKET_DATA_PATH
        csv_path = str(tmp_path / "market_data.csv")
        config.MARKET_DATA_PATH = csv_path
        sample_df.to_csv(csv_path, index=False)
        
        from data_collector import DataCollector
        collector = DataCollector()
        collector.csv_path = csv_path
        price = collector.get_latest_price()
        assert price == 104.5
        
        config.MARKET_DATA_PATH = original
    
    def test_returns_zero_no_file(self, tmp_path):
        from data_collector import DataCollector
        collector = DataCollector()
        collector.csv_path = str(tmp_path / "nonexistent.csv")
        assert collector.get_latest_price() == 0.0
