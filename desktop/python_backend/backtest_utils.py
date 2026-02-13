"""
Backtest Utilities
==================
Shared functions for backtest runners (sequential and parallel).
Extracted to eliminate ~150 lines of duplicated code.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

import config
from backtester import WalkForwardBacktester


def load_data() -> pd.DataFrame:
    """Load market data from CSV."""
    df = pd.read_csv(config.MARKET_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Loaded {len(df):,} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def get_default_params(data: pd.DataFrame) -> dict:
    """Calculate default backtest parameters based on available data."""
    train_window = min(300, len(data) // 3)
    return {
        'train_window': train_window,
        'test_window': 60,
        'step_size': 30,
        'slippage_pct': 0.001,
        'fee_pct': 0.0004,
    }


def run_predictor_backtest(data: pd.DataFrame, horizon_minutes: int = 60) -> dict:
    """Run backtest with the actual Nexus predictor."""
    try:
        from predictor import NexusPredictor
        predictor = NexusPredictor()
    except Exception as e:
        print(f"Warning: Could not load NexusPredictor: {e}")
        return None

    params = get_default_params(data)
    bt = WalkForwardBacktester(**params)

    print("\nRunning Nexus Predictor backtest...", flush=True)
    start_time = datetime.now()

    def nexus_predict(df: pd.DataFrame):
        try:
            if not predictor.is_trained:
                predictor.train(df)
            result = predictor.get_prediction(df)
            direction = 1 if result.get('direction', 'UP') == 'UP' else 0
            confidence = result.get('confidence', 50) / 100.0
            return direction, confidence
        except Exception:
            return 1, 0.5

    def nexus_train(df: pd.DataFrame):
        try:
            predictor.is_trained = False
            predictor.train(df)
        except:
            pass

    try:
        metrics = bt.run_simple(
            data=data,
            predict_fn=nexus_predict,
            train_fn=nexus_train,
            horizon_minutes=horizon_minutes,
            verbose=False
        )
        metrics['model'] = 'NexusPredictor'

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ Predictor complete in {elapsed:.1f}s")

        return metrics
    except Exception as e:
        print(f"Predictor backtest failed: {e}")
        return None


def print_comparison_table(df: pd.DataFrame, predictor_result: dict = None):
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    print(f"{'Model':<15} {'Accuracy':>10} {'Brier':>8} {'Sharpe':>8} {'MaxDD':>8} {'Win%':>8} {'N':>8}")
    print("-" * 80)

    for _, row in df.iterrows():
        print(f"{row['model']:<15} {row['accuracy']:>10.2%} {row['brier_score']:>8.4f} "
              f"{row['sharpe_ratio']:>8.2f} {row['max_drawdown']:>8.2%} "
              f"{row['win_rate']:>8.2%} {row['n_predictions']:>8}")

    if predictor_result:
        print("-" * 80)
        r = predictor_result
        print(f"{'NexusPredictor':<15} {r['accuracy']:>10.2%} {r['brier_score']:>8.4f} "
              f"{r['sharpe_ratio']:>8.2f} {r['max_drawdown']:>8.2%} "
              f"{r['win_rate']:>8.2%} {r['n_predictions']:>8}")

    print("=" * 80)

    # Key findings
    best_baseline = df.iloc[0]
    random_baseline = df[df['model'] == 'Random'].iloc[0] if 'Random' in df['model'].values else None

    print("\n📊 KEY FINDINGS:")
    print(f"  • Best baseline: {best_baseline['model']} ({best_baseline['accuracy']:.2%})")

    if random_baseline is not None:
        print(f"  • Random baseline: {random_baseline['accuracy']:.2%}")

    if predictor_result:
        acc = predictor_result['accuracy']
        sharpe = predictor_result['sharpe_ratio']
        random_acc = random_baseline['accuracy'] if random_baseline is not None else 0.5
        best_acc = best_baseline['accuracy']
        best_sharpe = best_baseline['sharpe_ratio']

        if acc > best_acc:
            print(f"  ✅ NexusPredictor BEATS all baselines by {(acc - best_acc)*100:.1f}pp")
        elif acc > random_acc:
            print(f"  ⚠️  NexusPredictor beats random but not best baseline")
        else:
            print(f"  ❌ NexusPredictor does NOT beat random baseline!")

        if sharpe > best_sharpe:
            print(f"  ✅ NexusPredictor has BEST Sharpe ratio: {sharpe:.2f}")


def save_results(df: pd.DataFrame, predictor_result: dict = None, filepath: str = None):
    """Save results to CSV."""
    if filepath is None:
        filepath = os.path.join(config.DATA_DIR, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    if predictor_result:
        pred_df = pd.DataFrame([predictor_result])
        df = pd.concat([df, pred_df], ignore_index=True)

    df.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")
