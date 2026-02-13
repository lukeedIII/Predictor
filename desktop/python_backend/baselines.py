"""
Nexus Baseline Models
=====================
Simple baseline strategies for comparison.
The main system MUST beat these to be credible.

Usage:
    from baselines import RandomBaseline, PersistenceBaseline, MABaseline
    baseline = MABaseline()
    direction, confidence = baseline.predict(data)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaselineModel:
    """Base class for all baselines."""
    
    name: str = "BaselineModel"
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        """
        Make a prediction.
        
        Args:
            data: DataFrame with OHLCV, up to current time (no future data)
            
        Returns:
            (direction, confidence) where:
            - direction: 1 = UP, 0 = DOWN
            - confidence: 0.0 to 1.0
        """
        raise NotImplementedError
    
    def train(self, data: pd.DataFrame) -> None:
        """Optional training method."""
        pass


class RandomBaseline(BaselineModel):
    """
    Random guessing baseline.
    Expected accuracy: ~50%
    """
    
    name = "Random"
    
    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        direction = self.rng.choice([0, 1])
        confidence = 0.5  # Always 50% confidence
        return direction, confidence


class PersistenceBaseline(BaselineModel):
    """
    Persistence (naive) baseline.
    Predict that the next move = last move.
    Expected accuracy: ~50-51%
    """
    
    name = "Persistence"
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        if len(data) < 2:
            return 1, 0.5  # Default to UP
        
        last_return = data['close'].iloc[-1] / data['close'].iloc[-2] - 1
        direction = 1 if last_return > 0 else 0
        
        # Confidence based on magnitude of last return
        confidence = min(0.5 + abs(last_return) * 50, 0.7)
        
        return direction, confidence


class MABaseline(BaselineModel):
    """
    Moving Average Crossover baseline.
    SMA(fast) > SMA(slow) = bullish.
    Expected accuracy: ~51-52%
    """
    
    name = "MA_Crossover"
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        if len(data) < self.slow_period:
            return 1, 0.5
        
        close = data['close']
        sma_fast = close.rolling(self.fast_period).mean().iloc[-1]
        sma_slow = close.rolling(self.slow_period).mean().iloc[-1]
        
        direction = 1 if sma_fast > sma_slow else 0
        
        # Confidence based on distance between MAs
        spread = (sma_fast - sma_slow) / sma_slow
        confidence = min(0.5 + abs(spread) * 100, 0.75)
        
        return direction, confidence


class MomentumBaseline(BaselineModel):
    """
    Simple momentum baseline.
    Look at returns over lookback period.
    Expected accuracy: ~50-52%
    """
    
    name = "Momentum"
    
    def __init__(self, lookback: int = 10):
        self.lookback = lookback
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        if len(data) < self.lookback + 1:
            return 1, 0.5
        
        returns = data['close'].pct_change(self.lookback).iloc[-1]
        direction = 1 if returns > 0 else 0
        
        confidence = min(0.5 + abs(returns) * 20, 0.75)
        
        return direction, confidence


class MeanReversionBaseline(BaselineModel):
    """
    Mean reversion baseline.
    If price is above SMA, predict DOWN. If below, predict UP.
    Expected accuracy: ~51%
    """
    
    name = "MeanReversion"
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        if len(data) < self.lookback:
            return 1, 0.5
        
        current_price = data['close'].iloc[-1]
        sma = data['close'].rolling(self.lookback).mean().iloc[-1]
        
        # If above mean, predict reversion DOWN
        direction = 0 if current_price > sma else 1
        
        deviation = (current_price - sma) / sma
        confidence = min(0.5 + abs(deviation) * 50, 0.70)
        
        return direction, confidence


class RSIBaseline(BaselineModel):
    """
    RSI-based baseline.
    RSI > 70 = overbought (predict DOWN)
    RSI < 30 = oversold (predict UP)
    """
    
    name = "RSI"
    
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def _compute_rsi(self, prices: pd.Series) -> float:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        
        if loss.iloc[-1] == 0:
            return 100 if gain.iloc[-1] > 0 else 50
        
        rs = gain.iloc[-1] / loss.iloc[-1]
        return 100 - (100 / (1 + rs))
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        if len(data) < self.period + 1:
            return 1, 0.5
        
        rsi = self._compute_rsi(data['close'])
        
        if rsi > self.overbought:
            direction = 0  # Expect reversal down
            confidence = min(0.5 + (rsi - 70) / 60, 0.8)
        elif rsi < self.oversold:
            direction = 1  # Expect reversal up
            confidence = min(0.5 + (30 - rsi) / 60, 0.8)
        else:
            # Neutral zone - use momentum
            direction = 1 if rsi > 50 else 0
            confidence = 0.5 + abs(rsi - 50) / 100
        
        return direction, confidence


class BollingerBaseline(BaselineModel):
    """
    Bollinger Bands baseline.
    Price at upper band = predict DOWN
    Price at lower band = predict UP
    """
    
    name = "Bollinger"
    
    def __init__(self, period: int = 20, std_mult: float = 2.0):
        self.period = period
        self.std_mult = std_mult
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        if len(data) < self.period:
            return 1, 0.5
        
        close = data['close']
        sma = close.rolling(self.period).mean().iloc[-1]
        std = close.rolling(self.period).std().iloc[-1]
        
        upper = sma + self.std_mult * std
        lower = sma - self.std_mult * std
        current = close.iloc[-1]
        
        # Position within bands (0 = lower, 1 = upper)
        if upper == lower:
            position = 0.5
        else:
            position = (current - lower) / (upper - lower)
        
        if position > 0.8:
            direction = 0  # Near upper band, expect down
            confidence = min(0.5 + (position - 0.8) * 2, 0.75)
        elif position < 0.2:
            direction = 1  # Near lower band, expect up
            confidence = min(0.5 + (0.2 - position) * 2, 0.75)
        else:
            # In middle - slight momentum bias
            direction = 1 if position > 0.5 else 0
            confidence = 0.5 + abs(position - 0.5) * 0.2
        
        return direction, confidence


class AR1Baseline(BaselineModel):
    """
    AR(1) autoregressive baseline.
    Fits r_t = α + β * r_{t-1} + ε
    """
    
    name = "AR1"
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.alpha = 0.0
        self.beta = 0.0
    
    def train(self, data: pd.DataFrame) -> None:
        if len(data) < self.lookback + 2:
            return
        
        returns = data['close'].pct_change().dropna()
        if len(returns) < 10:
            return
        
        # Simple OLS for AR(1)
        y = returns.values[1:]
        x = returns.values[:-1]
        
        if len(x) < 2:
            return
        
        # Fit: y = alpha + beta * x
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator > 0:
            self.beta = numerator / denominator
            self.alpha = y_mean - self.beta * x_mean
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        if len(data) < 2:
            return 1, 0.5
        
        last_return = data['close'].pct_change().iloc[-1]
        
        # AR(1) prediction
        predicted_return = self.alpha + self.beta * last_return
        
        direction = 1 if predicted_return > 0 else 0
        confidence = min(0.5 + abs(predicted_return) * 50, 0.65)
        
        return direction, confidence


def get_all_baselines() -> list:
    """Return list of all baseline instances."""
    return [
        RandomBaseline(seed=42),
        PersistenceBaseline(),
        MABaseline(fast_period=10, slow_period=30),
        MomentumBaseline(lookback=10),
        MeanReversionBaseline(lookback=20),
        RSIBaseline(period=14),
        BollingerBaseline(period=20),
        AR1Baseline(lookback=60),
    ]


if __name__ == "__main__":
    # Quick test
    print("Testing baselines...")
    
    # Create synthetic data
    np.random.seed(42)
    n = 100
    prices = 100000 * np.cumprod(1 + np.random.randn(n) * 0.001)
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n, freq='1min'),
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.rand(n) * 1000
    })
    
    for baseline in get_all_baselines():
        baseline.train(data)
        direction, conf = baseline.predict(data)
        print(f"  {baseline.name:15s}: direction={direction}, confidence={conf:.2f}")
