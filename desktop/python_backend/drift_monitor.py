"""
Drift Monitor — Feature, Prediction & Calibration Drift Detection
===================================================================
Three detection channels:
  1. Feature Drift (PSI) — input distribution shift vs training reference
  2. Prediction Drift — model probability output distribution shift
  3. Calibration Drift — Brier score + ECE from validated predictions

Usage:
    monitor = DriftMonitor()
    monitor.snapshot_training_distributions(X_train, y_probs_train)
    report = monitor.get_drift_report(X_live, recent_probs, prediction_history)
"""

import numpy as np
import logging
import json
import os
from datetime import datetime

try:
    import config
except ImportError:
    config = None


# ── Default thresholds (overridden by config if available) ──
_PSI_WARNING = getattr(config, 'DRIFT_PSI_WARNING', 0.10)
_PSI_CRITICAL = getattr(config, 'DRIFT_PSI_CRITICAL', 0.25)
_BRIER_WARNING = getattr(config, 'DRIFT_BRIER_WARNING', 0.30)
_BRIER_CRITICAL = getattr(config, 'DRIFT_BRIER_CRITICAL', 0.35)


def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two 1-D distributions.

    PSI < 0.10  → no significant shift
    PSI 0.10–0.25 → moderate shift (WARNING)
    PSI > 0.25  → significant shift (CRITICAL)
    """
    # Quantile-based binning from reference distribution
    edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Remove duplicate edges (constant features)
    if len(np.unique(edges)) < 3:
        return 0.0

    ref_counts = np.histogram(reference, bins=edges)[0].astype(float) + 1e-6
    cur_counts = np.histogram(current, bins=edges)[0].astype(float) + 1e-6

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes."""
    return float(np.mean((probs - outcomes) ** 2))


def _expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 5) -> float:
    """
    Expected Calibration Error: weighted average of |accuracy - confidence| per bin.
    Lower is better. Perfect calibration = 0.
    """
    if len(probs) < n_bins:
        return 0.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:  # include upper bound for last bin
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = outcomes[mask].mean()
        ece += (count / total) * abs(avg_acc - avg_conf)

    return float(ece)


class DriftMonitor:
    """
    Tracks distribution drift across features, predictions, and calibration.

    Lifecycle:
        1. After training, call snapshot_training_distributions() to set reference
        2. Periodically call get_drift_report() during live prediction
        3. Drift results are logged and available for the retrain loop
    """

    def __init__(self):
        self._ref_feature_stats = {}   # {feature_name: np.array of training values}
        self._ref_prob_hist = None     # Training-time probability distribution
        self._snapshot_time = None
        self._last_report = None
        self._last_check_time = None

        # Persistence path
        data_dir = getattr(config, 'DATA_DIR', os.path.join(os.path.dirname(__file__), 'data'))
        self._snapshot_path = os.path.join(data_dir, 'drift_reference.npz')

        # Try to load persisted snapshot
        self._load_snapshot()

    # ── Reference Snapshot ──────────────────────────────────

    def snapshot_training_distributions(self, X_train, y_probs_train=None):
        """
        Capture training-time feature distributions as the reference baseline.

        Args:
            X_train: DataFrame or ndarray of training features
            y_probs_train: (optional) predicted probabilities on training set
        """
        import pandas as pd

        if hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
            X_values = X_train.values
        else:
            feature_names = [f'f{i}' for i in range(X_train.shape[1])]
            X_values = X_train

        # Store up to 10,000 random samples per feature (memory cap)
        n = min(len(X_values), 10_000)
        indices = np.random.choice(len(X_values), n, replace=False) if len(X_values) > n else np.arange(len(X_values))

        self._ref_feature_stats = {}
        for i, name in enumerate(feature_names):
            col = X_values[indices, i].astype(float)
            # Filter out NaN/Inf
            col = col[np.isfinite(col)]
            if len(col) > 10:
                self._ref_feature_stats[name] = col

        # Store training probability distribution
        if y_probs_train is not None:
            probs = np.asarray(y_probs_train).flatten()
            probs = probs[np.isfinite(probs)]
            self._ref_prob_hist = probs
        else:
            self._ref_prob_hist = None

        self._snapshot_time = datetime.now().isoformat()
        self._save_snapshot()

        logging.info(
            f"[DRIFT-MONITOR] Reference snapshot saved — "
            f"{len(self._ref_feature_stats)} features, "
            f"{n} samples"
        )

    def _save_snapshot(self):
        """Persist reference distributions to disk with atomic replace."""
        tmp_path = self._snapshot_path + ".tmp"
        try:
            save_dict = {'_snapshot_time': np.array([self._snapshot_time])}
            for name, values in self._ref_feature_stats.items():
                save_dict[f'feat_{name}'] = values
            if self._ref_prob_hist is not None:
                save_dict['_prob_hist'] = self._ref_prob_hist
            np.savez_compressed(tmp_path, **save_dict)
            os.replace(tmp_path, self._snapshot_path)
        except Exception as e:
            logging.warning(f"[DRIFT-MONITOR] Failed to save snapshot: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

    def _load_snapshot(self):
        """Load persisted reference distributions."""
        if not os.path.exists(self._snapshot_path):
            return
        try:
            data = np.load(self._snapshot_path, allow_pickle=True)
            self._snapshot_time = str(data['_snapshot_time'][0]) if '_snapshot_time' in data else None
            self._ref_prob_hist = data['_prob_hist'] if '_prob_hist' in data else None
            self._ref_feature_stats = {}
            for key in data.files:
                if key.startswith('feat_'):
                    feature_name = key[5:]  # strip 'feat_' prefix
                    self._ref_feature_stats[feature_name] = data[key]
            if self._ref_feature_stats:
                logging.info(
                    f"[DRIFT-MONITOR] Loaded reference snapshot — "
                    f"{len(self._ref_feature_stats)} features "
                    f"(from {self._snapshot_time})"
                )
        except Exception as e:
            logging.warning(f"[DRIFT-MONITOR] Failed to load snapshot: {e}")

    @property
    def has_reference(self) -> bool:
        """Whether a training reference snapshot exists."""
        return len(self._ref_feature_stats) > 0

    # ── Feature Drift (PSI) ─────────────────────────────────

    def check_feature_drift(self, X_live) -> dict:
        """
        Compute PSI for each feature between training reference and live data.

        Args:
            X_live: DataFrame or ndarray of recent live feature vectors

        Returns:
            dict with per-feature PSI, aggregate metrics, and severity
        """
        if not self.has_reference:
            return {'status': 'NO_REFERENCE', 'severity': 'UNKNOWN'}

        import pandas as pd
        if hasattr(X_live, 'columns'):
            feature_names = list(X_live.columns)
            X_values = X_live.values
        else:
            feature_names = list(self._ref_feature_stats.keys())
            X_values = X_live

        per_feature = {}
        psi_values = []

        for i, name in enumerate(feature_names):
            if name not in self._ref_feature_stats:
                continue
            if i >= X_values.shape[1]:
                break

            live_col = X_values[:, i].astype(float)
            live_col = live_col[np.isfinite(live_col)]

            if len(live_col) < 10:
                continue

            psi = _compute_psi(self._ref_feature_stats[name], live_col)
            per_feature[name] = round(psi, 4)
            psi_values.append(psi)

        if not psi_values:
            return {'status': 'INSUFFICIENT_DATA', 'severity': 'UNKNOWN'}

        mean_psi = float(np.mean(psi_values))
        max_psi = float(np.max(psi_values))
        drifted_features = {k: v for k, v in per_feature.items() if v > _PSI_WARNING}

        # Top 5 drifting features
        top_drifters = dict(sorted(per_feature.items(), key=lambda x: x[1], reverse=True)[:5])

        if mean_psi > _PSI_CRITICAL or max_psi > _PSI_CRITICAL:
            severity = 'CRITICAL'
        elif mean_psi > _PSI_WARNING or len(drifted_features) > 3:
            severity = 'WARNING'
        else:
            severity = 'OK'

        return {
            'status': 'OK',
            'severity': severity,
            'mean_psi': round(mean_psi, 4),
            'max_psi': round(max_psi, 4),
            'features_checked': len(psi_values),
            'features_drifted': len(drifted_features),
            'top_drifters': top_drifters,
        }

    # ── Prediction Drift ────────────────────────────────────

    def check_prediction_drift(self, recent_probs: list) -> dict:
        """
        Compare recent prediction probabilities against training distribution.

        Args:
            recent_probs: list of recent P(UP) values from get_prediction()
        """
        if self._ref_prob_hist is None or len(self._ref_prob_hist) < 20:
            return {'status': 'NO_REFERENCE', 'severity': 'UNKNOWN'}

        probs = np.array(recent_probs, dtype=float)
        probs = probs[np.isfinite(probs)]

        if len(probs) < 10:
            return {'status': 'INSUFFICIENT_DATA', 'severity': 'UNKNOWN'}

        psi = _compute_psi(self._ref_prob_hist, probs, bins=8)
        mean_ref = float(np.mean(self._ref_prob_hist))
        mean_live = float(np.mean(probs))
        std_ref = float(np.std(self._ref_prob_hist))
        std_live = float(np.std(probs))

        if psi > _PSI_CRITICAL:
            severity = 'CRITICAL'
        elif psi > _PSI_WARNING:
            severity = 'WARNING'
        else:
            severity = 'OK'

        return {
            'status': 'OK',
            'severity': severity,
            'psi': round(psi, 4),
            'mean_ref': round(mean_ref, 4),
            'mean_live': round(mean_live, 4),
            'std_ref': round(std_ref, 4),
            'std_live': round(std_live, 4),
        }

    # ── Calibration Drift ───────────────────────────────────

    def check_calibration_drift(self, prediction_history: list) -> dict:
        """
        Compute Brier score and ECE from validated predictions.

        Args:
            prediction_history: list of dicts with 'confidence', 'direction',
                                'validated', 'correct' fields
        """
        # Filter to validated predictions only
        validated = [p for p in prediction_history if p.get('validated', False)]

        if len(validated) < 10:
            return {'status': 'INSUFFICIENT_DATA', 'severity': 'UNKNOWN', 'n_validated': len(validated)}

        # Convert confidence + direction into P(UP) and binary outcome
        probs = []
        outcomes = []
        for p in validated:
            conf = p.get('confidence', 50) / 100.0
            direction = p.get('direction', 'UP')
            correct = 1.0 if p.get('correct', False) else 0.0

            # P(UP) from the model's perspective
            if direction == 'UP':
                prob_up = conf
            else:
                prob_up = 1.0 - conf

            probs.append(prob_up)
            outcomes.append(correct if direction == 'UP' else (1.0 - correct))

        probs = np.array(probs)
        outcomes = np.array(outcomes)

        brier = _brier_score(probs, outcomes)
        ece = _expected_calibration_error(probs, outcomes)

        if brier > _BRIER_CRITICAL:
            severity = 'CRITICAL'
        elif brier > _BRIER_WARNING:
            severity = 'WARNING'
        else:
            severity = 'OK'

        return {
            'status': 'OK',
            'severity': severity,
            'brier_score': round(brier, 4),
            'ece': round(ece, 4),
            'n_validated': len(validated),
        }

    # ── Combined Report ─────────────────────────────────────

    def get_drift_report(self, X_live=None, recent_probs=None, prediction_history=None) -> dict:
        """
        Run all drift checks and return a unified report.

        Args:
            X_live: Recent feature matrix (DataFrame or ndarray)
            recent_probs: List of recent P(UP) values
            prediction_history: Prediction history dicts with validation info

        Returns:
            dict with feature_drift, prediction_drift, calibration_drift,
            overall_severity, and timestamp
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'reference_snapshot': self._snapshot_time,
        }

        # Feature drift
        if X_live is not None:
            report['feature_drift'] = self.check_feature_drift(X_live)
        else:
            report['feature_drift'] = {'status': 'SKIPPED', 'severity': 'UNKNOWN'}

        # Prediction drift
        if recent_probs is not None:
            report['prediction_drift'] = self.check_prediction_drift(recent_probs)
        else:
            report['prediction_drift'] = {'status': 'SKIPPED', 'severity': 'UNKNOWN'}

        # Calibration drift
        if prediction_history is not None:
            report['calibration_drift'] = self.check_calibration_drift(prediction_history)
        else:
            report['calibration_drift'] = {'status': 'SKIPPED', 'severity': 'UNKNOWN'}

        # Overall severity = worst of all channels
        severities = [
            report['feature_drift'].get('severity', 'UNKNOWN'),
            report['prediction_drift'].get('severity', 'UNKNOWN'),
            report['calibration_drift'].get('severity', 'UNKNOWN'),
        ]

        severity_order = {'CRITICAL': 3, 'WARNING': 2, 'OK': 1, 'UNKNOWN': 0}
        worst = max(severities, key=lambda s: severity_order.get(s, 0))
        report['overall_severity'] = worst

        self._last_report = report
        self._last_check_time = datetime.now()

        # Log summary
        feat_sev = report['feature_drift'].get('severity', '?')
        pred_sev = report['prediction_drift'].get('severity', '?')
        cal_sev = report['calibration_drift'].get('severity', '?')
        logging.info(
            f"[DRIFT-MONITOR] Report — "
            f"Feature: {feat_sev} | "
            f"Prediction: {pred_sev} | "
            f"Calibration: {cal_sev} | "
            f"Overall: {worst}"
        )

        return report

    @property
    def last_report(self) -> dict:
        """Most recent drift report (None if never checked)."""
        return self._last_report
