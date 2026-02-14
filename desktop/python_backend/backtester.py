"""
Nexus Walk-Forward Backtester
=============================
Provides reproducible evaluation with strict time splits.
No look-ahead bias. Proper cost modeling.

Usage:
    from backtester import WalkForwardBacktester
    bt = WalkForwardBacktester()
    results = bt.run(data, predictor)
    bt.print_report()
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Tuple
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WalkForwardBacktester:
    """
    Walk-forward backtesting engine with strict temporal separation.
    
    Key features:
    - No future data leakage
    - Frozen features at each prediction
    - Cost modeling (slippage + fees)
    - Multiple metric computation
    """
    
    def __init__(
        self,
        train_window: int = 1440,  # 24 hours of 1m candles
        test_window: int = 60,     # 1 hour OOS
        step_size: int = 60,       # Slide by 1 hour
        slippage_pct: float = 0.001,  # 0.1% per trade
        fee_pct: float = 0.0004,   # 0.04% maker fee
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.slippage_pct = slippage_pct
        self.fee_pct = fee_pct
        
        # Results storage
        self.predictions: List[Dict] = []
        self.actuals: List[int] = []
        self.returns: List[float] = []
        self.timestamps: List[datetime] = []
        self.confidences: List[float] = []
        
        # Run metadata
        self.run_id: str = ""
        self.start_time: datetime = None
        self.end_time: datetime = None
        
    def run(
        self,
        data: pd.DataFrame,
        model_fn: Callable,
        feature_fn: Optional[Callable] = None,
        horizon_minutes: int = 60,
        verbose: bool = True
    ) -> Dict:
        """
        Execute walk-forward backtest.
        
        Args:
            data: DataFrame with ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            model_fn: Function that takes (train_data) and returns fitted model
            feature_fn: Optional function to engineer features
            horizon_minutes: Prediction horizon (default 60 = 1H)
            verbose: Print progress
            
        Returns:
            Dict with all metrics and predictions
        """
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        
        # Reset state
        self.predictions = []
        self.actuals = []
        self.returns = []
        self.timestamps = []
        self.confidences = []
        
        # Ensure data is sorted
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        total_rows = len(data)
        min_required = self.train_window + self.test_window + horizon_minutes
        
        if total_rows < min_required:
            raise ValueError(f"Need at least {min_required} rows, got {total_rows}")
        
        # Walk-forward loop
        n_folds = 0
        start_idx = 0
        
        while start_idx + self.train_window + self.test_window + horizon_minutes <= total_rows:
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window
            
            # Extract train/test splits
            train_data = data.iloc[start_idx:train_end].copy()
            test_data = data.iloc[train_end:test_end].copy()
            
            # Train model on train_data
            try:
                model = model_fn(train_data)
            except Exception as e:
                logging.warning(f"Model training failed at fold {n_folds}: {e}")
                start_idx += self.step_size
                continue
            
            # Generate predictions for test_data
            for i in range(len(test_data) - horizon_minutes):
                pred_idx = train_end + i
                target_idx = pred_idx + horizon_minutes
                
                if target_idx >= total_rows:
                    break
                
                # Data available at prediction time (no future leakage)
                available_data = data.iloc[:pred_idx + 1].copy()
                
                # Apply feature engineering if provided
                if feature_fn:
                    available_data = feature_fn(available_data)
                
                # Make prediction
                try:
                    pred_result = model.predict(available_data)
                    if isinstance(pred_result, dict):
                        pred_direction = 1 if pred_result.get('direction', 'UP') == 'UP' else 0
                        confidence = pred_result.get('confidence', 50) / 100.0
                    else:
                        pred_direction = int(pred_result > 0.5)
                        confidence = float(pred_result) if pred_result <= 1 else pred_result / 100.0
                except Exception as e:
                    logging.debug(f"Prediction failed at idx {pred_idx}: {e}")
                    continue
                
                # Actual outcome
                current_price = data.iloc[pred_idx]['close']
                future_price = data.iloc[target_idx]['close']
                actual_direction = 1 if future_price > current_price * 1.003 else 0
                actual_return = (future_price - current_price) / current_price
                
                # Store results
                self.predictions.append(pred_direction)
                self.actuals.append(actual_direction)
                self.returns.append(actual_return)
                self.confidences.append(confidence)
                self.timestamps.append(data.iloc[pred_idx]['timestamp'])
            
            n_folds += 1
            start_idx += self.step_size
            
            if verbose and n_folds % 10 == 0:
                logging.info(f"Completed {n_folds} folds, {len(self.predictions)} predictions")
        
        self.end_time = datetime.now()
        
        if verbose:
            logging.info(f"Backtest complete: {n_folds} folds, {len(self.predictions)} predictions")
        
        return self.compute_metrics()
    
    def run_simple(
        self,
        data: pd.DataFrame,
        predict_fn: Callable[[pd.DataFrame], Tuple[int, float]],
        train_fn: Optional[Callable[[pd.DataFrame], None]] = None,
        horizon_minutes: int = 60,
        verbose: bool = True
    ) -> Dict:
        """
        Simplified run interface for baseline models.
        
        Args:
            data: DataFrame with OHLCV
            predict_fn: Function that takes available_data, returns (direction, confidence)
            train_fn: Optional function to train/fit on training window
            horizon_minutes: Prediction horizon
        """
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        
        # Reset
        self.predictions = []
        self.actuals = []
        self.returns = []
        self.timestamps = []
        self.confidences = []
        
        data = data.sort_values('timestamp').reset_index(drop=True)
        total_rows = len(data)
        
        # Start after minimum lookback
        start_idx = max(100, self.train_window)  # Need some history
        
        n_preds = 0
        for pred_idx in range(start_idx, total_rows - horizon_minutes, self.step_size):
            target_idx = pred_idx + horizon_minutes
            
            # Optional training
            if train_fn and (n_preds % 60 == 0):  # Retrain every hour
                train_data = data.iloc[max(0, pred_idx - self.train_window):pred_idx]
                try:
                    train_fn(train_data)
                except Exception as e:
                    logging.warning(f"Train function failed at prediction {n_preds}: {e}")
            
            # Available data at prediction time
            available_data = data.iloc[:pred_idx + 1].copy()
            
            # Predict
            try:
                pred_direction, confidence = predict_fn(available_data)
            except Exception as e:
                logging.debug(f"Prediction failed: {e}")
                continue
            
            # Actual
            current_price = data.iloc[pred_idx]['close']
            future_price = data.iloc[target_idx]['close']
            actual_direction = 1 if future_price > current_price * 1.003 else 0
            actual_return = (future_price - current_price) / current_price
            
            # Store
            self.predictions.append(pred_direction)
            self.actuals.append(actual_direction)
            self.returns.append(actual_return)
            self.confidences.append(confidence)
            self.timestamps.append(data.iloc[pred_idx]['timestamp'])
            
            n_preds += 1
        
        self.end_time = datetime.now()
        
        if verbose:
            logging.info(f"Backtest complete: {n_preds} predictions")
        
        return self.compute_metrics()
    
    def compute_metrics(self) -> Dict:
        """Compute all evaluation metrics."""
        if not self.predictions:
            return {'error': 'No predictions made'}
        
        preds = np.array(self.predictions)
        actuals = np.array(self.actuals)
        returns = np.array(self.returns)
        confs = np.array(self.confidences)
        
        n = len(preds)
        
        # Accuracy
        accuracy = np.mean(preds == actuals)
        
        # Brier Score (lower is better)
        # For binary: (forecast_prob - actual)^2
        # We use confidence as forecast prob for direction
        forecast_probs = np.where(preds == 1, confs, 1 - confs)
        brier_score = np.mean((forecast_probs - actuals) ** 2)
        
        # Calibration (Expected Calibration Error)
        ece = self._compute_ece(confs, preds, actuals)
        
        # Trading metrics (with costs)
        strategy_returns = self._compute_strategy_returns(preds, returns)
        
        if len(strategy_returns) > 1:
            sharpe = self._compute_sharpe(strategy_returns)
            max_dd = self._compute_max_drawdown(strategy_returns)
            total_return = np.prod(1 + strategy_returns) - 1
        else:
            sharpe = 0.0
            max_dd = 0.0
            total_return = 0.0
        
        # Win rate
        wins = np.sum((preds == 1) & (returns > 0)) + np.sum((preds == 0) & (returns < 0))
        win_rate = wins / n if n > 0 else 0
        
        # Confusion matrix
        tp = np.sum((preds == 1) & (actuals == 1))
        tn = np.sum((preds == 0) & (actuals == 0))
        fp = np.sum((preds == 1) & (actuals == 0))
        fn = np.sum((preds == 0) & (actuals == 1))
        
        results = {
            'run_id': self.run_id,
            'n_predictions': n,
            'accuracy': round(accuracy, 4),
            'brier_score': round(brier_score, 4),
            'ece': round(ece, 4),
            'sharpe_ratio': round(sharpe, 4),
            'max_drawdown': round(max_dd, 4),
            'total_return': round(total_return, 4),
            'win_rate': round(win_rate, 4),
            'confusion_matrix': {
                'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
            },
            'test_period': {
                'start': str(self.timestamps[0]) if self.timestamps else None,
                'end': str(self.timestamps[-1]) if self.timestamps else None
            },
            'runtime_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        }
        
        return results
    
    def _compute_strategy_returns(self, preds: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Compute strategy returns with costs."""
        # Strategy: go long if pred=1, go short if pred=0 (uses full signal)
        position_returns = np.where(preds == 1, returns, -returns)
        
        # Apply costs (slippage + fees) on position changes
        position_changes = np.abs(np.diff(preds, prepend=0))
        costs = position_changes * (self.slippage_pct + self.fee_pct)
        
        net_returns = position_returns - costs
        return net_returns
    
    def _compute_sharpe(self, returns: np.ndarray, periods_per_year: int = None) -> float:
        """Compute annualized Sharpe ratio, deriving periods from step_size."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        
        # Derive periods_per_year from step_size (in minutes) if not specified
        if periods_per_year is None:
            minutes_per_year = 525600  # 365.25 * 24 * 60
            periods_per_year = minutes_per_year // max(1, self.step_size)
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year)
        return sharpe
    
    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        """Compute maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    def _compute_ece(self, confs: np.ndarray, preds: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                avg_confidence = np.mean(confs[in_bin])
                avg_accuracy = np.mean(preds[in_bin] == actuals[in_bin])
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
        
        return ece
    
    def print_report(self):
        """Print formatted results."""
        metrics = self.compute_metrics()
        
        print("\n" + "=" * 60)
        print("  WALK-FORWARD BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Run ID:          {metrics['run_id']}")
        print(f"  Predictions:     {metrics['n_predictions']}")
        print(f"  Test Period:     {metrics['test_period']['start']} → {metrics['test_period']['end']}")
        print("-" * 60)
        print(f"  ACCURACY:        {metrics['accuracy']:.2%}")
        print(f"  Brier Score:     {metrics['brier_score']:.4f} (lower is better)")
        print(f"  ECE:             {metrics['ece']:.4f}")
        print("-" * 60)
        print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:    {metrics['max_drawdown']:.2%}")
        print(f"  Total Return:    {metrics['total_return']:.2%}")
        print(f"  Win Rate:        {metrics['win_rate']:.2%}")
        print("-" * 60)
        cm = metrics['confusion_matrix']
        print(f"  Confusion Matrix:")
        print(f"    TP: {cm['tp']:4d}  |  FP: {cm['fp']:4d}")
        print(f"    FN: {cm['fn']:4d}  |  TN: {cm['tn']:4d}")
        print("=" * 60 + "\n")
    
    def save_results(self, filepath: str = None):
        """Save results to JSON."""
        if filepath is None:
            filepath = f"backtest_results_{self.run_id}.json"
        
        results = self.compute_metrics()
        results['all_predictions'] = list(zip(
            [str(t) for t in self.timestamps],
            self.predictions,
            self.actuals,
            self.returns,
            self.confidences
        ))
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to {filepath}")


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing WalkForwardBacktester...")
    
    # Create synthetic price data
    np.random.seed(42)
    n = 3000
    returns = np.random.randn(n) * 0.001
    prices = 100000 * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n, freq='1min'),
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.rand(n) * 1000
    })
    
    # Simple momentum baseline
    def momentum_predict(df):
        recent_ret = df['close'].pct_change(10).iloc[-1]
        direction = 1 if recent_ret > 0 else 0
        confidence = min(0.5 + abs(recent_ret) * 10, 0.8)
        return direction, confidence
    
    bt = WalkForwardBacktester(train_window=500, test_window=60, step_size=60)
    results = bt.run_simple(data, momentum_predict, horizon_minutes=60)
    bt.print_report()
