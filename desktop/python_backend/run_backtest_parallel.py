"""
Nexus Parallel Backtest Runner
==============================
8-core parallel processing for fast backtesting.

Usage:
    python run_backtest_parallel.py
"""

import pandas as pd
import os
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import WalkForwardBacktester
from baselines import get_all_baselines
from backtest_utils import load_data, get_default_params, run_predictor_backtest, print_comparison_table, save_results
import config


def run_single_baseline(baseline_info: tuple, data: pd.DataFrame, train_window: int,
                        step_size: int, horizon_minutes: int) -> dict:
    """Run a single baseline backtest. Designed for parallel execution."""
    baseline_name, baseline_class = baseline_info

    try:
        baseline = baseline_class()
        bt = WalkForwardBacktester(
            train_window=train_window,
            test_window=60,
            step_size=step_size,
            slippage_pct=0.001,
            fee_pct=0.0004
        )
        baseline.train(data.iloc[:train_window])
        metrics = bt.run_simple(
            data=data,
            predict_fn=baseline.predict,
            train_fn=baseline.train if hasattr(baseline, 'train') else None,
            horizon_minutes=horizon_minutes,
            verbose=False
        )
        metrics['model'] = baseline_name
        return metrics
    except Exception as e:
        return {
            'model': baseline_name,
            'accuracy': 0,
            'error': str(e)
        }


def run_baseline_parallel(data: pd.DataFrame, horizon_minutes: int = 60, n_cores: int = None) -> pd.DataFrame:
    """Run all baselines in parallel using multiprocessing."""
    if n_cores is None:
        n_cores = min(cpu_count(), 8)

    params = get_default_params(data)

    print("\n" + "=" * 70)
    print("  PARALLEL BASELINE COMPARISON - Walk-Forward Backtest")
    print("=" * 70)
    print(f"  Data: {len(data):,} rows ({len(data)/60:.1f} hours)")
    print(f"  Train window: {params['train_window']} min")
    print(f"  Horizon: {horizon_minutes} min")
    print(f"  Step size: {params['step_size']} min")
    print(f"  🚀 Using {n_cores} CPU cores (parallel)")
    print("=" * 70 + "\n")

    from baselines import (RandomBaseline, PersistenceBaseline, MABaseline,
                           MomentumBaseline, MeanReversionBaseline, RSIBaseline,
                           BollingerBaseline, AR1Baseline)

    baseline_classes = [
        ('Random', RandomBaseline),
        ('Persistence', PersistenceBaseline),
        ('MA_Crossover', MABaseline),
        ('Momentum', MomentumBaseline),
        ('MeanReversion', MeanReversionBaseline),
        ('RSI', RSIBaseline),
        ('Bollinger', BollingerBaseline),
        ('AR1', AR1Baseline),
    ]

    print(f"Running {len(baseline_classes)} baselines in parallel...", flush=True)
    start_time = datetime.now()

    run_fn = partial(
        run_single_baseline,
        data=data,
        train_window=params['train_window'],
        step_size=params['step_size'],
        horizon_minutes=horizon_minutes
    )

    with Pool(n_cores) as pool:
        results = pool.map(run_fn, baseline_classes)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ All baselines complete in {elapsed:.1f}s (vs ~{elapsed * n_cores:.0f}s sequential)")

    valid_results = [r for r in results if 'error' not in r]
    df_results = pd.DataFrame(valid_results)
    df_results = df_results.sort_values('accuracy', ascending=False)

    for _, row in df_results.iterrows():
        print(f"  {row['model']}: {row['accuracy']:.2%}")

    return df_results


def main():
    """Run full parallel backtest comparison."""
    print("\n🔬 NEXUS CREDIBILITY AUDIT - PARALLEL MODE\n")
    print(f"Available CPU cores: {cpu_count()}")

    data = load_data()

    if len(data) < 200:
        print("⚠️  WARNING: Insufficient data for proper backtesting!")
        print("   Need at least 1440 rows (24h) for reliable results.\n")

    baseline_results = run_baseline_parallel(data, horizon_minutes=60)

    print("\n" + "-" * 70)
    predictor_result = run_predictor_backtest(data, horizon_minutes=60)

    print_comparison_table(baseline_results, predictor_result)
    save_results(baseline_results, predictor_result)

    print("\n✅ Parallel backtest complete!")
    return baseline_results, predictor_result


if __name__ == "__main__":
    main()
