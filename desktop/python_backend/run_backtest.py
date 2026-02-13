"""
Nexus Backtest Runner (Sequential)
===================================
Run baseline comparisons and system evaluation.

Usage:
    python run_backtest.py
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import WalkForwardBacktester
from baselines import get_all_baselines
from backtest_utils import load_data, get_default_params, run_predictor_backtest, print_comparison_table, save_results
import config


def run_baseline_comparison(data: pd.DataFrame, horizon_minutes: int = 60) -> pd.DataFrame:
    """Run all baselines sequentially and compare."""
    results = []
    params = get_default_params(data)
    bt = WalkForwardBacktester(**params)

    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON - Walk-Forward Backtest")
    print("=" * 70)
    print(f"  Data: {len(data):,} rows ({len(data)/60:.1f} hours)")
    print(f"  Train window: {params['train_window']} min")
    print(f"  Horizon: {horizon_minutes} min")
    print(f"  Step size: {params['step_size']} min")
    print("=" * 70 + "\n")

    for baseline in get_all_baselines():
        print(f"Running {baseline.name}...", end=" ", flush=True)
        try:
            baseline.train(data.iloc[:params['train_window']])
            metrics = bt.run_simple(
                data=data,
                predict_fn=baseline.predict,
                train_fn=baseline.train if hasattr(baseline, 'train') else None,
                horizon_minutes=horizon_minutes,
                verbose=False
            )
            metrics['model'] = baseline.name
            results.append(metrics)
            print(f"Accuracy: {metrics['accuracy']:.2%}, Sharpe: {metrics['sharpe_ratio']:.2f}")
        except Exception as e:
            print(f"FAILED: {e}")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('accuracy', ascending=False)
    return df_results


def main():
    """Run full backtest comparison."""
    print("\n🔬 NEXUS CREDIBILITY AUDIT - Phase 1: Baseline Comparison\n")

    data = load_data()

    if len(data) < 200:
        print("⚠️  WARNING: Insufficient data for proper backtesting!")
        print("   Need at least 1440 rows (24h) for reliable results.")
        print("   Current data will give preliminary results only.\n")

    baseline_results = run_baseline_comparison(data, horizon_minutes=60)

    print("\n" + "-" * 70)
    predictor_result = run_predictor_backtest(data, horizon_minutes=60)

    print_comparison_table(baseline_results, predictor_result)
    save_results(baseline_results, predictor_result)

    print("\n✅ Backtest complete!")
    return baseline_results, predictor_result


if __name__ == "__main__":
    main()
