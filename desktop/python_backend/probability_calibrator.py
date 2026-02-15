"""
Probability Calibrator — Isotonic Regression
=============================================
Converts raw model output probabilities into calibrated probabilities
(actual frequency of occurrence). Fitted on walk-forward OOS predictions.

Key concepts:
- Raw XGBoost P(UP) might be 0.65, but if the model says 0.65 and
  UP actually happens 72% of the time, the calibrated probability is 0.72
- Isotonic regression is non-parametric and preserves rank ordering
  (if raw says A > B, calibrated also says A > B)
- Expected Value (EV) = calibrated_prob × reward - (1-calibrated_prob) × risk

V7 Roadmap: Phase 2
"""

import numpy as np
import os
import json
import pickle
import logging
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

try:
    import config
except ImportError:
    class config:
        MODEL_DIR = "models"
        DATA_ROOT = "."

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Isotonic regression probability calibrator.
    
    Fitted on walk-forward out-of-sample (OOS) probability-label pairs.
    Transforms raw model output P(UP) into calibrated probabilities.
    """
    
    def __init__(self):
        self.calibrator = IsotonicRegression(
            y_min=0.01, y_max=0.99,  # Prevent 0/1 extremes
            out_of_bounds='clip',
            increasing=True,  # P(UP) should increase with raw prob
        )
        self.is_fitted = False
        self.fit_timestamp = None
        self.n_samples = 0
        self.calibration_stats = {}
        
        # Paths
        self.save_path = os.path.join(config.MODEL_DIR, "probability_calibrator.pkl")
        self.pairs_path = os.path.join(config.MODEL_DIR, "calibration_pairs.json")
        
        # Accumulated OOS pairs from walk-forward folds
        self._oos_probs = []  # raw model probabilities
        self._oos_labels = []  # actual outcomes (0/1)
        
        # Try to load existing calibrator
        self._load()
    
    def _load(self):
        """Load previously fitted calibrator from disk."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    state = pickle.load(f)
                self.calibrator = state['calibrator']
                self.is_fitted = state['is_fitted']
                self.fit_timestamp = state.get('fit_timestamp')
                self.n_samples = state.get('n_samples', 0)
                self.calibration_stats = state.get('calibration_stats', {})
                logger.info(
                    f"[CALIB] Loaded calibrator: {self.n_samples} samples, "
                    f"fitted={self.fit_timestamp}"
                )
            except Exception as e:
                logger.warning(f"[CALIB] Failed to load calibrator: {e}")
        
        # Load accumulated pairs
        if os.path.exists(self.pairs_path):
            try:
                with open(self.pairs_path, 'r') as f:
                    data = json.load(f)
                self._oos_probs = data.get('probs', [])
                self._oos_labels = data.get('labels', [])
                logger.info(f"[CALIB] Loaded {len(self._oos_probs)} OOS calibration pairs")
            except Exception as e:
                logger.warning(f"[CALIB] Failed to load pairs: {e}")
    
    def _save(self):
        """Persist calibrator and OOS pairs to disk."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Save calibrator
        try:
            with open(self.save_path, 'wb') as f:
                pickle.dump({
                    'calibrator': self.calibrator,
                    'is_fitted': self.is_fitted,
                    'fit_timestamp': self.fit_timestamp,
                    'n_samples': self.n_samples,
                    'calibration_stats': self.calibration_stats,
                }, f)
        except Exception as e:
            logger.warning(f"[CALIB] Failed to save calibrator: {e}")
        
        # Save OOS pairs
        try:
            with open(self.pairs_path, 'w') as f:
                json.dump({
                    'probs': self._oos_probs[-5000:],  # Keep last 5K
                    'labels': self._oos_labels[-5000:],
                }, f)
        except Exception as e:
            logger.warning(f"[CALIB] Failed to save pairs: {e}")
    
    def add_walk_forward_pairs(self, probs: np.ndarray, labels: np.ndarray):
        """
        Add OOS probability-label pairs from a walk-forward fold.
        
        Args:
            probs: Raw model P(UP) predictions (0-1)
            labels: Actual outcomes (0 or 1)
        """
        if len(probs) != len(labels):
            logger.warning(f"[CALIB] Mismatched arrays: {len(probs)} probs vs {len(labels)} labels")
            return
        
        self._oos_probs.extend(probs.tolist())
        self._oos_labels.extend(labels.tolist())
        
        # Keep max 5000 most recent pairs
        if len(self._oos_probs) > 5000:
            self._oos_probs = self._oos_probs[-5000:]
            self._oos_labels = self._oos_labels[-5000:]
        
        logger.info(f"[CALIB] Added {len(probs)} pairs, total: {len(self._oos_probs)}")
    
    def fit(self, min_samples: int = 100) -> bool:
        """
        Fit isotonic regression on accumulated OOS pairs.
        
        Returns:
            True if fitting succeeded, False otherwise.
        """
        n = len(self._oos_probs)
        if n < min_samples:
            logger.warning(f"[CALIB] Not enough data to fit ({n}/{min_samples})")
            return False
        
        probs = np.array(self._oos_probs)
        labels = np.array(self._oos_labels)
        
        # Check we have both classes
        if len(np.unique(labels)) < 2:
            logger.warning("[CALIB] Only one class in labels, cannot calibrate")
            return False
        
        try:
            self.calibrator.fit(probs, labels)
            self.is_fitted = True
            self.fit_timestamp = datetime.now().isoformat()
            self.n_samples = n
            
            # Compute calibration quality metrics
            self.calibration_stats = self._compute_stats(probs, labels)
            
            self._save()
            
            logger.info(
                f"[CALIB] Isotonic calibrator fitted on {n} samples | "
                f"ECE={self.calibration_stats.get('ece', -1):.4f} | "
                f"Brier={self.calibration_stats.get('brier', -1):.4f}"
            )
            return True
            
        except Exception as e:
            logger.error(f"[CALIB] Fitting failed: {e}")
            return False
    
    def calibrate(self, raw_prob: float) -> float:
        """
        Transform raw model probability to calibrated probability.
        
        Args:
            raw_prob: Raw P(UP) from model (0-1)
            
        Returns:
            Calibrated P(UP) (0-1). Falls back to raw if not fitted.
        """
        if not self.is_fitted:
            return raw_prob
        
        try:
            cal = float(self.calibrator.predict([raw_prob])[0])
            return np.clip(cal, 0.01, 0.99)
        except Exception:
            return raw_prob
    
    def calibrate_batch(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrate an array of probabilities."""
        if not self.is_fitted:
            return raw_probs
        try:
            return np.clip(self.calibrator.predict(raw_probs), 0.01, 0.99)
        except Exception:
            return raw_probs
    
    def expected_value(self, calibrated_prob: float,
                       reward_pct: float = 1.0,
                       risk_pct: float = 1.0) -> float:
        """
        Calculate Expected Value of a trade.
        
        EV = P(win) × reward - P(loss) × risk
        
        Args:
            calibrated_prob: Calibrated P(UP) or P(direction)
            reward_pct: Expected reward if win (% of position)
            risk_pct: Expected risk if loss (% of position)
            
        Returns:
            EV as percentage of position. Positive = edge.
        """
        p_win = calibrated_prob
        p_loss = 1.0 - calibrated_prob
        return p_win * reward_pct - p_loss * risk_pct
    
    def _compute_stats(self, probs: np.ndarray, labels: np.ndarray) -> dict:
        """Compute calibration quality metrics."""
        stats = {}
        
        # Brier Score (lower = better, 0 = perfect)
        stats['brier'] = float(np.mean((probs - labels) ** 2))
        
        # Expected Calibration Error (ECE)
        try:
            n_bins = min(10, len(probs) // 20)
            if n_bins >= 2:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    labels, probs, n_bins=n_bins, strategy='uniform'
                )
                bin_counts = np.histogram(probs, bins=n_bins)[0]
                # ECE = Σ (|bin_positive_fraction - bin_mean_prediction| × bin_count / total)
                bin_weights = bin_counts[:len(fraction_of_positives)] / len(probs)
                ece = float(np.sum(
                    np.abs(fraction_of_positives - mean_predicted_value) * bin_weights
                ))
                stats['ece'] = ece
                stats['fraction_of_positives'] = fraction_of_positives.tolist()
                stats['mean_predicted_value'] = mean_predicted_value.tolist()
            else:
                stats['ece'] = -1.0
        except Exception as e:
            logger.debug(f"[CALIB] ECE computation failed: {e}")
            stats['ece'] = -1.0
        
        # Sharpness (how peaked the calibrated distribution is)
        calibrated = self.calibrate_batch(probs) if self.is_fitted else probs
        stats['sharpness'] = float(np.std(calibrated))
        
        # Class distribution
        stats['positive_ratio'] = float(np.mean(labels))
        stats['n_samples'] = int(len(probs))
        
        # Overconfidence check: when model says P > 0.6, how often is it right?
        high_conf = probs > 0.6
        if high_conf.sum() > 10:
            stats['overconfidence_rate'] = float(np.mean(labels[high_conf]) - np.mean(probs[high_conf]))
        
        return stats
    
    def get_diagnostics(self) -> dict:
        """Return calibration diagnostics for API/dashboard."""
        return {
            'is_fitted': self.is_fitted,
            'fit_timestamp': self.fit_timestamp,
            'n_samples': self.n_samples,
            'n_pending_pairs': len(self._oos_probs),
            'stats': self.calibration_stats,
        }


# ═══════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("=== Probability Calibrator Test ===\n")
    
    # Simulate walk-forward OOS data
    np.random.seed(42)
    n = 500
    
    # Create biased model output (slightly overconfident)
    true_probs = np.random.beta(2, 2, n)
    labels = (np.random.random(n) < true_probs).astype(int)
    # Model outputs are overconfident (stretched away from 0.5)
    raw_probs = 0.5 + (true_probs - 0.5) * 1.3
    raw_probs = np.clip(raw_probs, 0.01, 0.99)
    
    cal = ProbabilityCalibrator()
    cal.add_walk_forward_pairs(raw_probs[:300], labels[:300])
    cal.add_walk_forward_pairs(raw_probs[300:], labels[300:])
    
    print(f"Total pairs: {len(cal._oos_probs)}")
    
    success = cal.fit()
    print(f"Fit success: {success}")
    
    if success:
        # Test calibration
        test_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        print("\nRaw → Calibrated:")
        for v in test_values:
            cal_v = cal.calibrate(v)
            ev = cal.expected_value(cal_v if v > 0.5 else 1-cal_v)
            print(f"  {v:.2f} → {cal_v:.4f}  (EV={ev:+.4f})")
        
        print(f"\nDiagnostics: {json.dumps(cal.get_diagnostics(), indent=2)}")
