"""
Institutional Quantitative Models for Nexus Shadow-Quant
Based on hedge fund-grade mathematical techniques.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HiddenMarkovRegime:
    """
    Hidden Markov Model for Market Regime Detection.
    
    Detects regime changes BEFORE visible price movements.
    States: Bull (0), Sideways (1), Bear (2)
    
    Formula: P(S_t|O_{1:t}) = α P(O_t|S_t) Σ P(S_t|S_{t-1}) P(S_{t-1}|O_{1:t-1})
    """
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model = None
        self.is_fitted = False
        self.state_names = {0: "BULL", 1: "SIDEWAYS", 2: "BEAR"}
        
        try:
            from hmmlearn.hmm import GaussianHMM
            self.HMM = GaussianHMM
            logging.info("HMM module loaded successfully")
        except ImportError:
            self.HMM = None
            logging.warning("hmmlearn not installed. HMM disabled.")
    
    def prepare_observations(self, prices: np.ndarray, volumes: np.ndarray = None) -> np.ndarray:
        """
        Prepare observation matrix for HMM.
        Features: [returns, volatility, volume_change]
        """
        if len(prices) < 20:
            return None
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Rolling volatility (10-period)
        volatility = pd.Series(returns).rolling(10).std().fillna(0).values
        
        # Volume change (if available)
        if volumes is not None and len(volumes) == len(prices):
            vol_change = np.diff(np.log(volumes + 1))
            vol_change = np.nan_to_num(vol_change, nan=0)
        else:
            vol_change = np.zeros(len(returns))
        
        # Stack features
        observations = np.column_stack([
            returns,
            volatility[1:] if len(volatility) > len(returns) else volatility,
            vol_change
        ])
        
        return observations
    
    def fit(self, prices: np.ndarray, volumes: np.ndarray = None) -> bool:
        """Fit HMM to historical data."""
        if self.HMM is None:
            return False
        
        observations = self.prepare_observations(prices, volumes)
        if observations is None or len(observations) < 50:
            return False
        
        try:
            self.model = self.HMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            self.model.fit(observations)
            self.is_fitted = True
            logging.info(f"HMM fitted with {self.n_states} states on {len(observations)} observations")
            return True
        except Exception as e:
            logging.error(f"HMM fitting error: {e}")
            return False
    
    def predict_regime(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict:
        """
        Predict current market regime and probabilities.
        
        Returns:
            {
                'regime': 'BULL'/'SIDEWAYS'/'BEAR',
                'regime_id': 0/1/2,
                'probabilities': [p_bull, p_sideways, p_bear],
                'confidence': float
            }
        """
        if not self.is_fitted or self.model is None:
            return {'regime': 'UNKNOWN', 'regime_id': -1, 'probabilities': [0.33, 0.34, 0.33], 'confidence': 0.0}
        
        try:
            observations = self.prepare_observations(prices, volumes)
            if observations is None:
                return {'regime': 'UNKNOWN', 'regime_id': -1, 'probabilities': [0.33, 0.34, 0.33], 'confidence': 0.0}
            
            # Get state probabilities for last observation
            _, state_sequence = self.model.decode(observations, algorithm="viterbi")
            posteriors = self.model.predict_proba(observations)
            
            current_state = state_sequence[-1]
            current_probs = posteriors[-1]
            confidence = float(np.max(current_probs))
            
            return {
                'regime': self.state_names.get(current_state, 'UNKNOWN'),
                'regime_id': int(current_state),
                'probabilities': current_probs.tolist(),
                'confidence': confidence * 100
            }
        except Exception as e:
            logging.error(f"HMM prediction error: {e}")
            return {'regime': 'UNKNOWN', 'regime_id': -1, 'probabilities': [0.33, 0.34, 0.33], 'confidence': 0.0}
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get regime transition probability matrix."""
        if not self.is_fitted or self.model is None:
            return np.eye(self.n_states)
        return self.model.transmat_


class GJRGarchVolatility:
    """
    GJR-GARCH Model for Asymmetric Volatility.
    
    Captures the leverage effect: negative shocks have larger impact on volatility.
    Formula: σ²_t = ω + αε²_{t-1} + γI_{t-1}ε²_{t-1} + βσ²_{t-1}
    
    Where:
    - γ: Asymmetry parameter (extra weight on negative shocks)
    - I_{t-1}: Indicator function (1 if ε_{t-1} < 0, else 0)
    """
    
    def __init__(self):
        self.model = None
        self.result = None
        self.is_fitted = False
        
        try:
            from arch import arch_model
            self.arch_model = arch_model
            logging.info("ARCH/GARCH module loaded successfully")
        except ImportError:
            self.arch_model = None
            logging.warning("arch package not installed. GJR-GARCH disabled.")
    
    def fit(self, returns: np.ndarray) -> bool:
        """Fit GJR-GARCH(1,1) model to returns."""
        if self.arch_model is None:
            return False
        
        if len(returns) < 100:
            logging.warning("Not enough data for GJR-GARCH (need 100+)")
            return False
        
        try:
            # Scale returns to percentage
            returns_pct = returns * 100
            
            self.model = self.arch_model(
                returns_pct,
                vol='Garch',
                p=1, o=1, q=1,  # o=1 makes it GJR-GARCH
                mean='Constant',
                dist='Normal'
            )
            self.result = self.model.fit(disp='off', show_warning=False)
            self.is_fitted = True
            
            logging.info(f"GJR-GARCH fitted. Asymmetry (gamma): {self.get_asymmetry():.4f}")
            return True
        except Exception as e:
            logging.error(f"GJR-GARCH fitting error: {e}")
            return False
    
    def forecast_volatility(self, horizon: int = 1) -> float:
        """Forecast volatility for next n periods."""
        if not self.is_fitted or self.result is None:
            return 0.01
        
        try:
            forecast = self.result.forecast(horizon=horizon)
            # Convert back from percentage
            vol = np.sqrt(forecast.variance.values[-1, -1]) / 100
            return float(vol)
        except Exception as e:
            logging.error(f"Volatility forecast error: {e}")
            return 0.01
    
    def get_asymmetry(self) -> float:
        """Get asymmetry parameter (gamma)."""
        if not self.is_fitted or self.result is None:
            return 0.0
        
        try:
            params = self.result.params
            if 'gamma[1]' in params.index:
                return float(params['gamma[1]'])
            elif 'o[1]' in params.index:
                return float(params['o[1]'])
            return 0.0
        except:
            return 0.0
    
    def get_conditional_volatility(self) -> np.ndarray:
        """Get time series of conditional volatility."""
        if not self.is_fitted or self.result is None:
            return np.array([])
        
        try:
            return self.result.conditional_volatility.values / 100
        except:
            return np.array([])


class OrderFlowImbalance:
    """
    Order Flow Imbalance (OFI) for Whale Detection.
    
    Detects buy/sell pressure before price movements.
    Formula: OFI = Σ(sign(ΔV_bid)·ΔV_bid - sign(ΔV_ask)·ΔV_ask)
    
    Alternative (without order book): Use volume + price direction proxy.
    """
    
    def __init__(self):
        self.ofi_history = []
        self.history_limit = 1000
    
    def calculate_proxy_ofi(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """
        Calculate OFI proxy without order book data.
        Uses price direction × volume as approximation.
        
        Positive OFI = Buy pressure
        Negative OFI = Sell pressure
        """
        if len(prices) < 2 or len(volumes) < 2:
            return 0.0
        
        try:
            # Price direction
            price_changes = np.diff(prices)
            direction = np.sign(price_changes)
            
            # Volume-weighted direction
            vol_weights = volumes[1:] / np.mean(volumes[1:])
            
            # OFI proxy: sum of (direction * relative_volume)
            ofi_values = direction * vol_weights
            
            # Use recent values weighted more
            weights = np.exp(np.linspace(-1, 0, len(ofi_values)))
            weighted_ofi = np.sum(ofi_values * weights) / np.sum(weights)
            
            self.ofi_history.append(weighted_ofi)
            if len(self.ofi_history) > self.history_limit:
                self.ofi_history = self.ofi_history[-self.history_limit:]
            
            return float(weighted_ofi)
            
        except Exception as e:
            logging.error(f"OFI calculation error: {e}")
            return 0.0
    
    def get_ofi_signal(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """
        Get OFI signal with interpretation.
        
        Returns:
            {
                'ofi': float,
                'signal': 'BUY_PRESSURE'/'SELL_PRESSURE'/'NEUTRAL',
                'strength': 'STRONG'/'MODERATE'/'WEAK',
                'normalized': float (-1 to 1)
            }
        """
        ofi = self.calculate_proxy_ofi(prices, volumes)
        
        # Normalize based on history
        if len(self.ofi_history) > 10:
            ofi_std = np.std(self.ofi_history)
            normalized = ofi / (ofi_std + 1e-9)
            normalized = np.clip(normalized, -3, 3) / 3  # Scale to -1, 1
        else:
            normalized = np.clip(ofi, -1, 1)
        
        # Interpret signal
        if normalized > 0.5:
            signal = "BUY_PRESSURE"
            strength = "STRONG" if normalized > 0.7 else "MODERATE"
        elif normalized < -0.5:
            signal = "SELL_PRESSURE"
            strength = "STRONG" if normalized < -0.7 else "MODERATE"
        else:
            signal = "NEUTRAL"
            strength = "WEAK"
        
        return {
            'ofi': ofi,
            'signal': signal,
            'strength': strength,
            'normalized': float(normalized)
        }


class EmpiricalModeDecomposition:
    """
    Empirical Mode Decomposition (EMD) for Hidden Cycle Detection.
    
    Decomposes price into Intrinsic Mode Functions (IMFs).
    Formula: x(t) = Σ IMF_k(t) + r_n(t)
    
    Each IMF represents a different frequency component.
    """
    
    def __init__(self):
        self.emd = None
        self.imfs = None
        self.residual = None
        
        try:
            from PyEMD import EMD
            self.EMD = EMD
            logging.info("EMD module loaded successfully")
        except ImportError:
            self.EMD = None
            logging.warning("PyEMD not installed. EMD disabled.")
    
    def decompose(self, prices: np.ndarray, max_imfs: int = 5) -> bool:
        """Decompose price series into IMFs."""
        if self.EMD is None:
            return False
        
        if len(prices) < 50:
            return False
        
        try:
            self.emd = self.EMD()
            self.emd.MAX_ITERATION = 1000
            
            # Decompose
            imfs = self.emd.emd(prices, max_imf=max_imfs)
            
            if len(imfs) < 2:
                return False
            
            self.imfs = imfs[:-1]  # All except residual
            self.residual = imfs[-1]  # Trend/residual
            
            logging.info(f"EMD decomposed into {len(self.imfs)} IMFs")
            return True
            
        except Exception as e:
            logging.error(f"EMD decomposition error: {e}")
            return False
    
    def get_dominant_cycles(self, sampling_rate: float = 1.0) -> List[Dict]:
        """
        Extract dominant cycles from IMFs.
        
        Returns list of {period, amplitude, phase} for each IMF.
        """
        if self.imfs is None:
            return []
        
        cycles = []
        for i, imf in enumerate(self.imfs):
            try:
                # Zero crossings to estimate period
                zero_crossings = np.where(np.diff(np.sign(imf)))[0]
                if len(zero_crossings) > 1:
                    avg_half_period = np.mean(np.diff(zero_crossings))
                    period = 2 * avg_half_period * sampling_rate
                else:
                    period = len(imf) * sampling_rate
                
                # Amplitude
                amplitude = np.std(imf) * 2
                
                # Relative importance
                importance = amplitude / (np.std(self.residual) + 1e-9)
                
                cycles.append({
                    'imf_index': i,
                    'period_minutes': round(period, 1),
                    'amplitude': round(amplitude, 4),
                    'importance': round(importance, 2)
                })
            except Exception as e:
                continue
        
        # Sort by importance
        cycles.sort(key=lambda x: x['importance'], reverse=True)
        return cycles
    
    def get_cycle_strength(self) -> List[float]:
        """
        Get relative strength of each cycle for UI visualization.
        Returns normalized strengths [0-1] for top 3 cycles.
        """
        cycles = self.get_dominant_cycles()
        
        if not cycles:
            return [0.0, 0.0, 0.0]
        
        # Normalize importance to 0-1
        max_importance = max(c['importance'] for c in cycles) + 1e-9
        strengths = [min(1.0, c['importance'] / max_importance) for c in cycles[:3]]
        
        # Pad to 3 elements
        while len(strengths) < 3:
            strengths.append(0.0)
        
        return strengths[:3]


class RegimeSwitchingModel:
    """
    Markov Regime Switching Model.
    
    Combines HMM with regime-specific parameters.
    Formula: P(S_t|Y_{1:t}) ∝ P(Y_t|S_t)∫P(S_t|S_{t-1})P(S_{t-1}|Y_{1:t-1})dS_{t-1}
    """
    
    def __init__(self):
        self.hmm = HiddenMarkovRegime(n_states=3)
        self.current_regime = 'UNKNOWN'
        self.regime_history = []
        self.regime_stats = {
            'BULL': {'mean_return': 0, 'volatility': 0, 'duration': 0},
            'SIDEWAYS': {'mean_return': 0, 'volatility': 0, 'duration': 0},
            'BEAR': {'mean_return': 0, 'volatility': 0, 'duration': 0}
        }
    
    def fit_and_analyze(self, prices: np.ndarray, volumes: np.ndarray = None) -> bool:
        """Fit HMM and analyze regime characteristics."""
        if not self.hmm.fit(prices, volumes):
            return False
        
        try:
            # Get regime sequence
            observations = self.hmm.prepare_observations(prices, volumes)
            if observations is None:
                return False
            
            _, states = self.hmm.model.decode(observations, algorithm="viterbi")
            returns = np.diff(np.log(prices))
            
            # Calculate stats for each regime
            for state_id, state_name in self.hmm.state_names.items():
                mask = states[:-1] == state_id if len(states) > len(returns) else states == state_id
                if np.sum(mask) > 0:
                    regime_returns = returns[mask[:len(returns)]]
                    self.regime_stats[state_name] = {
                        'mean_return': float(np.mean(regime_returns) * 100),  # Percentage
                        'volatility': float(np.std(regime_returns) * 100),
                        'duration': int(np.sum(mask))
                    }
            
            return True
            
        except Exception as e:
            logging.error(f"Regime analysis error: {e}")
            return False
    
    def get_regime(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict:
        """Get current regime with detailed analysis."""
        hmm_result = self.hmm.predict_regime(prices, volumes)
        
        self.current_regime = hmm_result['regime']
        self.regime_history.append(self.current_regime)
        
        # Add regime stats to result
        regime_detail = self.regime_stats.get(self.current_regime, {})
        hmm_result['stats'] = regime_detail
        
        # Calculate regime stability (how long in current regime)
        stability = 1
        for regime in reversed(self.regime_history[:-1]):
            if regime == self.current_regime:
                stability += 1
            else:
                break
        hmm_result['stability'] = stability
        
        return hmm_result


# ========== MEDIUM PRIORITY MODELS ==========

class WaveletAnalysis:
    """
    Wavelet Transform for Time-Frequency Analysis.
    
    Decomposes signals across multiple scales simultaneously.
    Formula: W_f(a,b) = 1/√|a| ∫f(t)ψ((t-b)/a)dt
    
    Better than FFT for non-stationary signals like crypto.
    """
    
    def __init__(self, wavelet: str = 'db4'):
        self.wavelet = wavelet
        self.coeffs = None
        self.pywt = None
        
        try:
            import pywt
            self.pywt = pywt
            logging.info("PyWavelets module loaded successfully")
        except ImportError:
            logging.warning("PyWavelets not installed. Wavelet analysis disabled.")
    
    def decompose(self, prices: np.ndarray, level: int = 4) -> bool:
        """
        Perform Discrete Wavelet Transform decomposition.
        
        Level determines how many frequency bands to extract.
        Higher level = more decomposition.
        """
        if self.pywt is None or len(prices) < 32:
            return False
        
        try:
            # DWT decomposition
            self.coeffs = self.pywt.wavedec(prices, self.wavelet, level=level)
            logging.info(f"Wavelet decomposition: {len(self.coeffs)} levels")
            return True
        except Exception as e:
            logging.error(f"Wavelet decomposition error: {e}")
            return False
    
    def get_frequency_power(self) -> Dict:
        """
        Get power in each frequency band.
        
        Returns dict with power at each scale.
        Lower index = higher frequency (short-term noise)
        Higher index = lower frequency (long-term trend)
        """
        if self.coeffs is None:
            return {}
        
        powers = {}
        total_power = 0
        
        for i, coeff in enumerate(self.coeffs):
            power = np.sum(coeff**2)
            total_power += power
            if i == 0:
                powers['trend'] = power
            else:
                powers[f'scale_{i}'] = power
        
        # Normalize
        if total_power > 0:
            powers = {k: v/total_power for k, v in powers.items()}
        
        return powers
    
    def denoise(self, prices: np.ndarray, threshold_mode: str = 'soft') -> np.ndarray:
        """
        Wavelet denoising of price signal.
        Removes high-frequency noise while preserving signal structure.
        """
        if self.pywt is None or len(prices) < 32:
            return prices
        
        try:
            # Decompose
            coeffs = self.pywt.wavedec(prices, self.wavelet, level=4)
            
            # Calculate threshold
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(prices)))
            
            # Apply threshold to detail coefficients (keep approximation)
            new_coeffs = [coeffs[0]]
            for c in coeffs[1:]:
                new_coeffs.append(self.pywt.threshold(c, threshold, mode=threshold_mode))
            
            # Reconstruct
            return self.pywt.waverec(new_coeffs, self.wavelet)[:len(prices)]
        except:
            return prices
    
    def get_trend_strength(self) -> float:
        """Get strength of trend vs noise (0-1)."""
        powers = self.get_frequency_power()
        if 'trend' in powers:
            return float(powers['trend'])
        return 0.5


class GaussianCopula:
    """
    Gaussian Copula for Dependency Analysis.
    
    Models the dependence structure between assets separately from marginals.
    Formula: C(u,v) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v))
    
    Used to predict: How the crash of one asset drags down others.
    """
    
    def __init__(self):
        self.correlation_matrix = None
        self.is_fitted = False
    
    def fit(self, returns_matrix: np.ndarray) -> bool:
        """
        Fit Gaussian copula to multi-asset returns.
        
        returns_matrix: (n_samples, n_assets) array of returns
        """
        if len(returns_matrix) < 30:
            return False
        
        try:
            from scipy import stats
            
            # Transform to uniform using empirical CDF
            n_assets = returns_matrix.shape[1] if len(returns_matrix.shape) > 1 else 1
            
            if n_assets < 2:
                return False
            
            uniform_data = np.zeros_like(returns_matrix)
            for i in range(n_assets):
                uniform_data[:, i] = stats.rankdata(returns_matrix[:, i]) / (len(returns_matrix) + 1)
            
            # Transform to standard normal
            normal_data = stats.norm.ppf(uniform_data)
            normal_data = np.nan_to_num(normal_data, nan=0, posinf=3, neginf=-3)
            
            # Estimate correlation matrix
            self.correlation_matrix = np.corrcoef(normal_data.T)
            self.is_fitted = True
            
            logging.info(f"Gaussian Copula fitted for {n_assets} assets")
            return True
            
        except Exception as e:
            logging.error(f"Copula fitting error: {e}")
            return False
    
    def get_tail_dependence(self) -> float:
        """
        Estimate tail dependence coefficient.
        Gaussian copula has no tail dependence (theoretical = 0),
        but we can estimate empirical extreme co-movements.
        """
        if not self.is_fitted or self.correlation_matrix is None:
            return 0.0
        
        # Average off-diagonal correlation as proxy
        n = len(self.correlation_matrix)
        if n < 2:
            return 0.0
        
        off_diag = self.correlation_matrix[np.triu_indices(n, k=1)]
        return float(np.mean(off_diag))
    
    def get_contagion_risk(self) -> Dict:
        """
        Estimate contagion risk between assets.
        Higher correlation = higher risk of crash spreading.
        """
        if not self.is_fitted or self.correlation_matrix is None:
            return {'risk': 'UNKNOWN', 'avg_correlation': 0}
        
        avg_corr = self.get_tail_dependence()
        
        if avg_corr > 0.7:
            risk = "HIGH"
        elif avg_corr > 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        
        return {
            'risk': risk,
            'avg_correlation': avg_corr,
            'correlation_matrix': self.correlation_matrix.tolist() if isinstance(self.correlation_matrix, np.ndarray) else []
        }


class TransferEntropyAnalysis:
    """
    Transfer Entropy for Information Flow Detection.
    
    Measures directed information flow between time series.
    Formula: T_{X→Y} = Σ p(y_{t+1}, x_t, y_t) log[p(y_{t+1}|x_t,y_t)/p(y_{t+1}|y_t)]
    
    Detects: Which asset leads another (predictive causality).
    """
    
    def __init__(self, lag: int = 1, bins: int = 10):
        self.lag = lag
        self.bins = bins
        self.te_values = {}
    
    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """Discretize continuous values into bins."""
        return np.digitize(x, np.linspace(np.min(x), np.max(x), self.bins))
    
    def _entropy(self, x: np.ndarray) -> float:
        """Calculate Shannon entropy."""
        _, counts = np.unique(x, return_counts=True)
        p = counts / len(x)
        return -np.sum(p * np.log2(p + 1e-10))
    
    def _joint_entropy(self, *args) -> float:
        """Calculate joint entropy of multiple variables."""
        combined = np.column_stack(args)
        # Create unique combined states
        unique_rows, counts = np.unique(combined, axis=0, return_counts=True)
        p = counts / len(combined)
        return -np.sum(p * np.log2(p + 1e-10))
    
    def calculate_transfer_entropy(self, source: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate transfer entropy from source to target.
        
        TE(X→Y) = H(Y_t+1 | Y_t) - H(Y_t+1 | Y_t, X_t)
        
        High TE = source predicts target.
        """
        if len(source) < 50 or len(target) < 50:
            return 0.0
        
        try:
            # Discretize
            x = self._discretize(source[:-self.lag])
            y_past = self._discretize(target[:-self.lag])
            y_future = self._discretize(target[self.lag:])
            
            # Align lengths
            min_len = min(len(x), len(y_past), len(y_future))
            x, y_past, y_future = x[:min_len], y_past[:min_len], y_future[:min_len]
            
            # H(Y_t+1 | Y_t) = H(Y_t+1, Y_t) - H(Y_t)
            h_y_future_given_past = self._joint_entropy(y_future, y_past) - self._entropy(y_past)
            
            # H(Y_t+1 | Y_t, X_t) = H(Y_t+1, Y_t, X_t) - H(Y_t, X_t)
            h_y_future_given_both = self._joint_entropy(y_future, y_past, x) - self._joint_entropy(y_past, x)
            
            te = max(0, h_y_future_given_past - h_y_future_given_both)
            return float(te)
            
        except Exception as e:
            logging.error(f"Transfer entropy error: {e}")
            return 0.0
    
    def analyze_pair(self, series_a: np.ndarray, series_b: np.ndarray, 
                     name_a: str = "A", name_b: str = "B") -> Dict:
        """
        Analyze bidirectional information flow between two series.
        """
        te_a_to_b = self.calculate_transfer_entropy(series_a, series_b)
        te_b_to_a = self.calculate_transfer_entropy(series_b, series_a)
        
        # Determine leader
        if te_a_to_b > te_b_to_a * 1.5:
            leader = name_a
        elif te_b_to_a > te_a_to_b * 1.5:
            leader = name_b
        else:
            leader = "BIDIRECTIONAL"
        
        self.te_values[f"{name_a}→{name_b}"] = te_a_to_b
        self.te_values[f"{name_b}→{name_a}"] = te_b_to_a
        
        return {
            f'te_{name_a}_to_{name_b}': te_a_to_b,
            f'te_{name_b}_to_{name_a}': te_b_to_a,
            'leader': leader,
            'strength': max(te_a_to_b, te_b_to_a)
        }


class HestonVolatility:
    """
    Heston Stochastic Volatility Model.
    
    Models volatility as mean-reverting stochastic process.
    Formula: dv_t = κ(θ - v_t)dt + ξ√v_t dW^v_t
    
    Parameters:
    - κ (kappa): Speed of mean reversion
    - θ (theta): Long-term variance
    - ξ (xi/vol_of_vol): Volatility of volatility
    - ρ (rho): Correlation between price and volatility
    """
    
    def __init__(self):
        self.kappa = 2.0  # Mean reversion speed
        self.theta = 0.02  # Long-term variance (sqrt = 14% annual vol)
        self.xi = 0.3  # Vol of vol
        self.rho = -0.7  # Negative correlation (crashes = high vol)
        self.v0 = 0.02  # Initial variance
        
        self.is_calibrated = False
        self.variance_path = []
    
    def calibrate(self, returns: np.ndarray) -> bool:
        """
        Simple calibration using method of moments.
        More sophisticated would use option prices.
        """
        if len(returns) < 100:
            return False
        
        try:
            # Estimate realized variance over rolling windows
            window = 20
            variances = pd.Series(returns).rolling(window).var().dropna().values
            
            if len(variances) < 30:
                return False
            
            # Long-term variance (theta)
            self.theta = float(np.mean(variances))
            
            # Initial variance
            self.v0 = float(variances[-1])
            
            # Mean reversion speed (kappa) - from autocorrelation
            autocorr = np.corrcoef(variances[:-1], variances[1:])[0, 1]
            self.kappa = max(0.1, -np.log(autocorr) * 252 / window) if autocorr > 0 else 2.0
            
            # Vol of vol (xi)
            self.xi = float(np.std(np.diff(np.sqrt(variances))) * np.sqrt(252))
            
            # Correlation (rho) - between returns and variance changes
            var_changes = np.diff(variances)
            aligned_returns = returns[window:-1][:len(var_changes)]
            if len(aligned_returns) == len(var_changes):
                self.rho = float(np.corrcoef(aligned_returns, var_changes)[0, 1])
            
            self.is_calibrated = True
            logging.info(f"Heston calibrated: κ={self.kappa:.2f}, θ={self.theta:.4f}, ξ={self.xi:.2f}, ρ={self.rho:.2f}")
            return True
            
        except Exception as e:
            logging.error(f"Heston calibration error: {e}")
            return False
    
    def simulate_variance(self, steps: int = 60, dt: float = 1/252) -> np.ndarray:
        """
        Simulate future variance path using Heston dynamics.
        """
        if not self.is_calibrated:
            return np.full(steps, self.theta)
        
        np.random.seed(None)  # Random seed for each simulation
        v = np.zeros(steps)
        v[0] = self.v0
        
        for t in range(1, steps):
            dW = np.random.randn() * np.sqrt(dt)
            # Ensure variance stays positive (truncation scheme)
            v_prev = max(v[t-1], 1e-8)
            v[t] = v_prev + self.kappa * (self.theta - v_prev) * dt + self.xi * np.sqrt(v_prev) * dW
            v[t] = max(v[t], 1e-8)  # Enforce positivity
        
        self.variance_path = v
        return v
    
    def get_volatility_forecast(self, horizon: int = 60) -> Dict:
        """
        Forecast volatility using Heston model.
        """
        var_path = self.simulate_variance(horizon)
        vol_path = np.sqrt(var_path)
        
        return {
            'current_vol': float(np.sqrt(self.v0)) if self.v0 > 0 else 0,
            'mean_vol': float(np.mean(vol_path)),
            'max_vol': float(np.max(vol_path)),
            'long_term_vol': float(np.sqrt(self.theta)),
            'vol_of_vol': self.xi,
            'mean_reversion_speed': self.kappa,
            'leverage_effect': self.rho
        }


# ========== PHASE 3: TIER 1 MODELS ==========

class MertonJumpDiffusion:
    """
    Merton Jump Diffusion Model for Black Swan Detection.
    
    Adds Poisson jumps to standard Brownian motion.
    Formula: dS_t/S_t = (μ-λk)dt + σdW_t + (J_t-1)dN_t
    
    Detects: Sudden price jumps (Elon tweets, exchange crashes).
    """
    
    def __init__(self):
        self.mu = 0.0  # Drift
        self.sigma = 0.02  # Diffusion volatility
        self.lambda_jump = 0.1  # Jump intensity (jumps per unit time)
        self.mu_jump = 0.0  # Mean jump size
        self.sigma_jump = 0.05  # Jump size volatility
        
        self.is_calibrated = False
        self.jump_history = []
        self.last_jump_probability = 0.0
    
    def calibrate(self, returns: np.ndarray, threshold: float = 3.0) -> bool:
        """
        Calibrate jump parameters from historical returns.
        Jumps = returns beyond threshold * std from mean.
        """
        if len(returns) < 50:
            return False
        
        try:
            # Standard parameters
            self.mu = float(np.mean(returns))
            self.sigma = float(np.std(returns))
            
            # Identify jumps (beyond threshold standard deviations)
            z_scores = np.abs((returns - self.mu) / (self.sigma + 1e-9))
            jump_mask = z_scores > threshold
            jump_returns = returns[jump_mask]
            
            if len(jump_returns) > 0:
                self.lambda_jump = len(jump_returns) / len(returns)
                self.mu_jump = float(np.mean(jump_returns))
                self.sigma_jump = float(np.std(jump_returns)) if len(jump_returns) > 1 else self.sigma * 2
            else:
                self.lambda_jump = 0.01
                self.mu_jump = 0.0
                self.sigma_jump = self.sigma * 3
            
            self.is_calibrated = True
            logging.info(f"Merton Jump calibrated: λ={self.lambda_jump:.3f}, μ_J={self.mu_jump:.4f}, σ_J={self.sigma_jump:.4f}")
            return True
            
        except Exception as e:
            logging.error(f"Merton calibration error: {e}")
            return False
    
    def detect_jump(self, current_return: float) -> Dict:
        """
        Detect if current return is a jump.
        """
        if not self.is_calibrated:
            return {'is_jump': False, 'probability': 0, 'magnitude': 0}
        
        z_score = abs((current_return - self.mu) / (self.sigma + 1e-9))
        
        # Probability of jump (simplified Bayesian)
        # P(jump|return) ∝ P(return|jump) * P(jump)
        from scipy import stats
        p_return_given_normal = stats.norm.pdf(current_return, self.mu, self.sigma)
        p_return_given_jump = stats.norm.pdf(current_return, self.mu_jump, self.sigma_jump)
        
        p_jump = self.lambda_jump
        p_normal = 1 - self.lambda_jump
        
        posterior_jump = (p_return_given_jump * p_jump) / (
            p_return_given_jump * p_jump + p_return_given_normal * p_normal + 1e-10
        )
        
        self.last_jump_probability = float(posterior_jump)
        is_jump = posterior_jump > 0.5
        
        if is_jump:
            self.jump_history.append({
                'magnitude': current_return,
                'probability': posterior_jump
            })
        
        return {
            'is_jump': is_jump,
            'probability': float(posterior_jump * 100),
            'magnitude': float(current_return),
            'z_score': float(z_score),
            'direction': 'UP' if current_return > 0 else 'DOWN'
        }
    
    def get_jump_risk(self) -> Dict:
        """
        Get current jump risk assessment.
        """
        risk_level = "LOW"
        if self.lambda_jump > 0.05:
            risk_level = "HIGH"
        elif self.lambda_jump > 0.02:
            risk_level = "MEDIUM"
        
        return {
            'risk_level': risk_level,
            'jump_intensity': self.lambda_jump,
            'expected_jumps_per_day': self.lambda_jump * 1440,  # Assuming minute data
            'last_jump_prob': self.last_jump_probability * 100
        }


class RoughVolatility:
    """
    Rough Volatility Model (Gatheral et al.).
    
    Volatility is driven by fractional Brownian motion with H < 0.5.
    Formula: v_t = v_0 + ∫ dW_t^H; H < 0.5
    
    Key insight: Crypto volatility is "rougher" than classical models assume.
    Typical H for crypto: 0.1-0.3
    """
    
    def __init__(self):
        self.H = 0.5  # Hurst exponent (< 0.5 = rough)
        self.is_calibrated = False
        self.roughness_history = []
    
    def _estimate_roughness(self, volatility_series: np.ndarray, max_lag: int = 20) -> float:
        """
        Estimate Hurst exponent of log-volatility.
        Uses variogram method for rough processes.
        """
        if len(volatility_series) < max_lag * 2:
            return 0.5
        
        log_vol = np.log(volatility_series + 1e-10)
        
        lags = np.arange(1, max_lag + 1)
        m_values = []
        
        for lag in lags:
            increments = log_vol[lag:] - log_vol[:-lag]
            m = np.mean(np.abs(increments))
            m_values.append(m)
        
        m_values = np.array(m_values)
        
        # Linear regression in log-log space
        log_lags = np.log(lags)
        log_m = np.log(m_values + 1e-10)
        
        # H is the slope
        try:
            slope, _ = np.polyfit(log_lags, log_m, 1)
            H = float(slope)
            # Clamp to valid range
            H = max(0.01, min(0.99, H))
            return H
        except:
            return 0.5
    
    def calibrate(self, returns: np.ndarray, window: int = 20) -> bool:
        """
        Calibrate rough volatility from returns.
        """
        if len(returns) < window * 3:
            return False
        
        try:
            # Calculate rolling volatility
            volatility = pd.Series(returns).rolling(window).std().dropna().values
            
            if len(volatility) < 50:
                return False
            
            self.H = self._estimate_roughness(volatility)
            self.is_calibrated = True
            
            logging.info(f"Rough Volatility calibrated: H={self.H:.3f}")
            return True
            
        except Exception as e:
            logging.error(f"Rough Vol calibration error: {e}")
            return False
    
    def get_roughness_analysis(self) -> Dict:
        """
        Analyze roughness characteristics.
        """
        if not self.is_calibrated:
            return {'H': 0.5, 'interpretation': 'UNKNOWN', 'roughness_score': 50}
        
        if self.H < 0.3:
            interpretation = "VERY_ROUGH"
            score = 90
        elif self.H < 0.4:
            interpretation = "ROUGH"
            score = 70
        elif self.H < 0.5:
            interpretation = "SLIGHTLY_ROUGH"
            score = 50
        else:
            interpretation = "SMOOTH"
            score = 30
        
        return {
            'H': self.H,
            'interpretation': interpretation,
            'roughness_score': score,
            'crypto_typical': 0.1 <= self.H <= 0.3
        }


class HilbertHuangTransform:
    """
    Hilbert-Huang Transform (HHT) for instantaneous frequency.
    
    Combines EMD + Hilbert Transform for time-frequency analysis.
    Formula: H(ω,t) = ∫x(τ)e^{-jωτ}dτ · e^{jωt}
    
    Better than FFT for non-stationary signals (crypto).
    """
    
    def __init__(self):
        self.imfs = None
        self.inst_frequencies = None
        self.inst_amplitudes = None
        self.is_decomposed = False
        
        try:
            from PyEMD import EMD
            from scipy.signal import hilbert
            self.EMD = EMD
            self.hilbert = hilbert
            logging.info("HHT modules loaded successfully")
        except ImportError as e:
            self.EMD = None
            self.hilbert = None
            logging.warning(f"HHT dependencies missing: {e}")
    
    def decompose_and_analyze(self, prices: np.ndarray, sampling_rate: float = 1.0) -> bool:
        """
        Perform EMD + Hilbert transform.
        """
        if self.EMD is None or self.hilbert is None:
            return False
        
        if len(prices) < 50:
            return False
        
        try:
            # EMD decomposition
            emd = self.EMD()
            emd.MAX_ITERATION = 500
            imfs = emd.emd(prices, max_imf=5)
            
            if len(imfs) < 2:
                return False
            
            self.imfs = imfs[:-1]  # Exclude residual
            
            # Hilbert transform for each IMF
            self.inst_frequencies = []
            self.inst_amplitudes = []
            
            for imf in self.imfs:
                analytic = self.hilbert(imf)
                amplitude = np.abs(analytic)
                phase = np.unwrap(np.angle(analytic))
                
                # Instantaneous frequency = d(phase)/dt / 2π
                inst_freq = np.diff(phase) / (2 * np.pi) * sampling_rate
                inst_freq = np.append(inst_freq, inst_freq[-1])  # Pad
                
                self.inst_frequencies.append(inst_freq)
                self.inst_amplitudes.append(amplitude)
            
            self.is_decomposed = True
            logging.info(f"HHT decomposed into {len(self.imfs)} IMFs with Hilbert analysis")
            return True
            
        except Exception as e:
            logging.error(f"HHT error: {e}")
            return False
    
    def get_current_frequencies(self) -> Dict:
        """
        Get current dominant instantaneous frequencies.
        """
        if not self.is_decomposed:
            return {'dominant_freq': 0, 'frequencies': [], 'energy_distribution': {}}
        
        # Get latest frequencies weighted by amplitude
        current_freqs = []
        current_amps = []
        
        for i, (freq, amp) in enumerate(zip(self.inst_frequencies, self.inst_amplitudes)):
            current_freqs.append(float(np.abs(freq[-1])))
            current_amps.append(float(amp[-1]))
        
        # Normalize amplitudes for energy distribution
        total_energy = sum(current_amps) + 1e-10
        energy_dist = {f'IMF_{i+1}': a/total_energy for i, a in enumerate(current_amps)}
        
        # Dominant frequency (highest energy)
        dominant_idx = np.argmax(current_amps)
        return {
            'dominant_freq': current_freqs[dominant_idx] if current_freqs else 0,
            'dominant_period_minutes': 1.0 / (current_freqs[dominant_idx] + 1e-10) if current_freqs else 0,
            'frequencies': current_freqs,
            'amplitudes': current_amps,
            'energy_distribution': energy_dist
        }


# ========== PHASE 4: TIER 2 MODELS ==========

class BatesSVJ:
    """
    Bates Stochastic Volatility with Jumps (SVJ) Model.
    
    Combines Heston stochastic volatility + Merton jumps.
    Formula: dS_t = μS_t dt + √v_t S_t dW^1_t + J_t S_t dN_t
    
    The most comprehensive single model for crypto.
    """
    
    def __init__(self):
        # Heston parameters
        self.kappa = 2.0
        self.theta = 0.04
        self.xi = 0.3
        self.rho = -0.7
        self.v0 = 0.04
        
        # Merton jump parameters
        self.lambda_jump = 0.1
        self.mu_jump = 0.0
        self.sigma_jump = 0.1
        
        self.is_calibrated = False
    
    def calibrate(self, returns: np.ndarray, window: int = 20) -> bool:
        """Calibrate Bates model (simplified moment matching)."""
        if len(returns) < 100:
            return False
        
        try:
            # Volatility estimation
            variances = pd.Series(returns).rolling(window).var().dropna().values
            
            if len(variances) < 30:
                return False
            
            # Heston parameters
            self.theta = float(np.mean(variances))
            self.v0 = float(variances[-1])
            
            autocorr = np.corrcoef(variances[:-1], variances[1:])[0, 1]
            self.kappa = max(0.1, -np.log(abs(autocorr) + 1e-10) * 252 / window)
            
            # Jump parameters (3 sigma rule)
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            jumps = returns[np.abs(returns - mean_ret) > 3 * std_ret]
            
            if len(jumps) > 0:
                self.lambda_jump = len(jumps) / len(returns)
                self.mu_jump = float(np.mean(jumps))
                self.sigma_jump = float(np.std(jumps)) if len(jumps) > 1 else std_ret * 2
            
            self.is_calibrated = True
            logging.info(f"Bates SVJ calibrated: κ={self.kappa:.2f}, θ={self.theta:.4f}, λ={self.lambda_jump:.3f}")
            return True
            
        except Exception as e:
            logging.error(f"Bates calibration error: {e}")
            return False
    
    def get_analysis(self) -> Dict:
        if not self.is_calibrated:
            return {'model': 'BATES_SVJ', 'status': 'UNCALIBRATED'}
        
        return {
            'model': 'BATES_SVJ',
            'heston': {
                'kappa': self.kappa,
                'theta': self.theta,
                'long_term_vol': np.sqrt(self.theta)
            },
            'jumps': {
                'intensity': self.lambda_jump,
                'mean_size': self.mu_jump,
                'vol_size': self.sigma_jump
            },
            'risk_score': min(100, int((self.lambda_jump * 1000) + (self.sigma_jump * 500)))
        }


class RecurrenceQuantification:
    """
    Recurrence Quantification Analysis (RQA).
    
    Detects hidden patterns via recurrence plots.
    Metrics: RR (recurrence rate), DET (determinism), L (avg diagonal)
    
    Discovers: Hidden repeating structures before they complete.
    """
    
    def __init__(self, threshold: float = 0.1, embedding_dim: int = 3, delay: int = 1):
        self.threshold = threshold
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.recurrence_matrix = None
        self.metrics = {}
    
    def _embed_time_series(self, x: np.ndarray) -> np.ndarray:
        """Create time-delay embedding."""
        n = len(x) - (self.embedding_dim - 1) * self.delay
        if n <= 0:
            return np.array([])
        
        embedded = np.zeros((n, self.embedding_dim))
        for i in range(self.embedding_dim):
            embedded[:, i] = x[i * self.delay:i * self.delay + n]
        return embedded
    
    def compute_recurrence(self, prices: np.ndarray) -> bool:
        """Compute recurrence matrix and quantification metrics."""
        if len(prices) < 50:
            return False
        
        try:
            # Normalize
            prices_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-10)
            
            # Embed
            embedded = self._embed_time_series(prices_norm)
            if len(embedded) < 10:
                return False
            
            n = len(embedded)
            
            # Distance matrix (use Euclidean distance)
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_matrix[i, j] = np.linalg.norm(embedded[i] - embedded[j])
            
            # Recurrence matrix (1 if distance < threshold)
            self.recurrence_matrix = (dist_matrix < self.threshold * np.max(dist_matrix)).astype(int)
            
            # Compute RQA metrics
            total_points = n * n
            recurrent_points = np.sum(self.recurrence_matrix)
            
            # Recurrence Rate (RR)
            self.metrics['RR'] = recurrent_points / total_points
            
            # Determinism (DET) - ratio of recurrent points in diagonal lines
            diagonal_points = 0
            for k in range(-n+2, n-1):
                diag = np.diag(self.recurrence_matrix, k)
                # Count points in lines of length >= 2
                in_line = False
                line_len = 0
                for point in diag:
                    if point == 1:
                        line_len += 1
                        in_line = True
                    else:
                        if line_len >= 2:
                            diagonal_points += line_len
                        line_len = 0
                        in_line = False
                if line_len >= 2:
                    diagonal_points += line_len
            
            self.metrics['DET'] = diagonal_points / (recurrent_points + 1e-10)
            
            # Average diagonal line length (L)
            line_lengths = []
            for k in range(-n+2, n-1):
                diag = np.diag(self.recurrence_matrix, k)
                line_len = 0
                for point in diag:
                    if point == 1:
                        line_len += 1
                    else:
                        if line_len >= 2:
                            line_lengths.append(line_len)
                        line_len = 0
                if line_len >= 2:
                    line_lengths.append(line_len)
            
            self.metrics['L'] = np.mean(line_lengths) if line_lengths else 0
            self.metrics['Lmax'] = max(line_lengths) if line_lengths else 0
            
            logging.info(f"RQA computed: RR={self.metrics['RR']:.3f}, DET={self.metrics['DET']:.3f}")
            return True
            
        except Exception as e:
            logging.error(f"RQA error: {e}")
            return False
    
    def get_analysis(self) -> Dict:
        if not self.metrics:
            return {'status': 'NOT_COMPUTED'}
        
        # Interpretation
        if self.metrics['DET'] > 0.7:
            structure = 'HIGHLY_DETERMINISTIC'
        elif self.metrics['DET'] > 0.4:
            structure = 'MODERATELY_DETERMINISTIC'
        else:
            structure = 'STOCHASTIC'
        
        return {
            'RR': self.metrics['RR'],
            'DET': self.metrics['DET'],
            'L': self.metrics['L'],
            'Lmax': self.metrics['Lmax'],
            'interpretation': structure,
            'pattern_strength': int(self.metrics['DET'] * 100)
        }


class MultifractalAnalysis:
    """
    Multifractal Detrended Fluctuation Analysis (MF-DFA).
    
    Crypto markets are multifractal, not unifractal.
    Formula: D(h) = dim_H{x: α(x) = h}
    
    Discovers: Multi-scale market structure.
    """
    
    def __init__(self):
        self.hurst_spectrum = {}
        self.singularity_spectrum = {}
        self.is_computed = False
    
    def compute_mfdfa(self, prices: np.ndarray, q_range: tuple = (-5, 5), scales: list = None) -> bool:
        """Compute multifractal spectrum using MFDFA."""
        if len(prices) < 200:
            return False
        
        try:
            # Log returns
            returns = np.diff(np.log(prices + 1e-10))
            
            # Profile (cumulative sum of detrended returns)
            profile = np.cumsum(returns - np.mean(returns))
            
            # Scales (segment sizes)
            if scales is None:
                scales = [16, 32, 64, 128]
            
            q_values = np.arange(q_range[0], q_range[1] + 1)
            
            Fq = {q: [] for q in q_values}
            
            for s in scales:
                n_segments = len(profile) // s
                if n_segments < 2:
                    continue
                
                variance_segments = []
                
                for i in range(n_segments):
                    segment = profile[i*s:(i+1)*s]
                    # Detrend (linear fit)
                    x = np.arange(s)
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    variance_segments.append(np.mean((segment - trend)**2))
                
                variance_segments = np.array(variance_segments)
                variance_segments = variance_segments[variance_segments > 0]
                
                if len(variance_segments) < 2:
                    continue
                
                for q in q_values:
                    if q == 0:
                        Fq[q].append((s, np.exp(0.5 * np.mean(np.log(variance_segments)))))
                    else:
                        Fq[q].append((s, np.power(np.mean(np.power(variance_segments, q/2)), 1/q)))
            
            # Compute H(q) for each q
            for q in q_values:
                if len(Fq[q]) >= 2:
                    scales_q = np.array([f[0] for f in Fq[q]])
                    fq_vals = np.array([f[1] for f in Fq[q]])
                    
                    # Linear regression in log-log space
                    log_s = np.log(scales_q)
                    log_f = np.log(fq_vals + 1e-10)
                    
                    slope, _ = np.polyfit(log_s, log_f, 1)
                    self.hurst_spectrum[q] = slope
            
            # Multifractality measure (delta H)
            if len(self.hurst_spectrum) >= 2:
                h_values = list(self.hurst_spectrum.values())
                self.delta_h = max(h_values) - min(h_values)
            else:
                self.delta_h = 0
            
            self.is_computed = True
            logging.info(f"MFDFA computed. ΔH = {self.delta_h:.3f}")
            return True
            
        except Exception as e:
            logging.error(f"MFDFA error: {e}")
            return False
    
    def get_analysis(self) -> Dict:
        if not self.is_computed:
            return {'status': 'NOT_COMPUTED', 'multifractality': 0}
        
        # Interpretation
        if self.delta_h > 0.3:
            interpretation = 'STRONGLY_MULTIFRACTAL'
        elif self.delta_h > 0.1:
            interpretation = 'MODERATELY_MULTIFRACTAL'
        else:
            interpretation = 'UNIFRACTAL'
        
        h_at_2 = self.hurst_spectrum.get(2, 0.5)
        
        return {
            'delta_h': self.delta_h,
            'H_2': h_at_2,  # Standard Hurst at q=2
            'hurst_spectrum': self.hurst_spectrum,
            'interpretation': interpretation,
            'complexity_score': int(self.delta_h * 100)
        }


class TopologicalDataAnalysis:
    """
    Topological Data Analysis (TDA) via Persistent Homology.
    
    Detects topological features (loops, voids) in price data.
    Formula: β_k = dim(H_k(X, ℤ_2))
    
    Discovers: Crash topology and market structure changes.
    """
    
    def __init__(self, embedding_dim: int = 3, delay: int = 1):
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.betti_numbers = {}
        self.persistence_pairs = []
        self.is_computed = False
    
    def _time_delay_embedding(self, x: np.ndarray) -> np.ndarray:
        """Create time-delay embedding for point cloud."""
        n = len(x) - (self.embedding_dim - 1) * self.delay
        if n <= 0:
            return np.array([])
        
        embedded = np.zeros((n, self.embedding_dim))
        for i in range(self.embedding_dim):
            embedded[:, i] = x[i * self.delay:i * self.delay + n]
        return embedded
    
    def compute_persistent_homology(self, prices: np.ndarray, max_dimension: int = 1) -> bool:
        """
        Compute persistent homology (simplified Vietoris-Rips).
        
        Note: Full TDA requires gudhi/ripser. This is simplified.
        """
        if len(prices) < 50:
            return False
        
        try:
            # Normalize and embed
            prices_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-10)
            point_cloud = self._time_delay_embedding(prices_norm)
            
            if len(point_cloud) < 10:
                return False
            
            # Simplified: count connected components at different scales
            n = len(point_cloud)
            
            # Distance matrix
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(point_cloud[i] - point_cloud[j])
                    distances[i, j] = d
                    distances[j, i] = d
            
            max_dist = np.max(distances)
            thresholds = np.linspace(0, max_dist * 0.5, 20)
            
            betti_0_history = []  # Connected components
            
            for thresh in thresholds:
                # Count connected components (simplified via adjacency)
                adjacency = distances < thresh
                np.fill_diagonal(adjacency, False)
                
                # Simple connected components count
                visited = set()
                components = 0
                
                for i in range(n):
                    if i not in visited:
                        # BFS
                        queue = [i]
                        while queue:
                            node = queue.pop(0)
                            if node not in visited:
                                visited.add(node)
                                neighbors = np.where(adjacency[node])[0]
                                queue.extend([nb for nb in neighbors if nb not in visited])
                        components += 1
                
                betti_0_history.append(components)
            
            # Store Betti numbers
            self.betti_numbers = {
                'b0_initial': betti_0_history[0] if betti_0_history else n,
                'b0_final': betti_0_history[-1] if betti_0_history else 1,
                'b0_history': betti_0_history
            }
            
            # Persistence = how long features persist
            if len(betti_0_history) > 5:
                # Find "death" times of components
                deaths = []
                for i in range(1, len(betti_0_history)):
                    if betti_0_history[i] < betti_0_history[i-1]:
                        deaths.append(i * (max_dist * 0.5 / 20))
                
                self.betti_numbers['avg_persistence'] = np.mean(deaths) if deaths else 0
            
            self.is_computed = True
            logging.info(f"TDA computed. β₀ initial={self.betti_numbers['b0_initial']}, final={self.betti_numbers['b0_final']}")
            return True
            
        except Exception as e:
            logging.error(f"TDA error: {e}")
            return False
    
    def get_analysis(self) -> Dict:
        if not self.is_computed:
            return {'status': 'NOT_COMPUTED'}
        
        # Interpretation based on Betti numbers
        b0_drop = self.betti_numbers['b0_initial'] - self.betti_numbers['b0_final']
        
        if b0_drop > 30:
            structure = 'HIGHLY_CONNECTED'
        elif b0_drop > 15:
            structure = 'MODERATELY_CONNECTED'
        else:
            structure = 'FRAGMENTED'
        
        return {
            'betti_0_initial': self.betti_numbers['b0_initial'],
            'betti_0_final': self.betti_numbers['b0_final'],
            'connectivity_drop': b0_drop,
            'avg_persistence': self.betti_numbers.get('avg_persistence', 0),
            'interpretation': structure,
            'topology_score': min(100, int(b0_drop * 2))
        }


# ========== PHASE 5: TIER 3 MODELS ==========

class PPOTradingAgent:
    """
    Proximal Policy Optimization (PPO) for Adaptive Trading.
    
    Deep RL for continuous action spaces (position sizing).
    Key: Uses clipped surrogate objective for stable learning.
    
    Formula: L^{CLIP}(θ) = E[min(r_t(θ)A_t, clip(r_t, 1-ε, 1+ε)A_t)]
    """
    
    def __init__(self, state_dim: int = 10, action_dim: int = 3):
        """
        Actions: 0=HOLD, 1=BUY, 2=SELL (simplified)
        State: [price_change, volatility, regime, ofi, hurst, ...]
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # PPO hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.2  # Clip ratio
        self.learning_rate = 3e-4
        self.entropy_coef = 0.01
        
        # Neural network weights (simplified linear policy)
        self.policy_weights = np.random.randn(state_dim, action_dim) * 0.1
        self.value_weights = np.random.randn(state_dim, 1) * 0.1
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        self.is_trained = False
        self.training_episodes = 0
        self.cumulative_reward = 0
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-10)
    
    def get_action(self, state: np.ndarray) -> tuple:
        """Sample action from policy."""
        state = np.array(state).flatten()[:self.state_dim]
        
        # Pad if necessary
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        
        # Policy output (action probabilities)
        logits = state @ self.policy_weights
        probs = self._softmax(logits)
        
        # Sample action
        action = np.random.choice(self.action_dim, p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        
        # Value estimate
        value = float(state @ self.value_weights)
        
        return int(action), log_prob, value
    
    def store_transition(self, state, action, reward, value, log_prob):
        """Store experience for training."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def _compute_gae(self, rewards, values, gamma=0.99, lam=0.95) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        
        return advantages
    
    def train_step(self, epochs: int = 4) -> Dict:
        """Perform PPO update."""
        if len(self.states) < 10:
            return {'status': 'INSUFFICIENT_DATA'}
        
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        old_log_probs = np.array(self.log_probs)
        
        # Compute advantages
        advantages = self._compute_gae(rewards, values)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        
        # PPO update (simplified gradient descent)
        for _ in range(epochs):
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                old_log_prob = old_log_probs[i]
                advantage = advantages[i]
                ret = returns[i]
                
                # Current policy
                logits = state @ self.policy_weights
                probs = self._softmax(logits)
                new_log_prob = np.log(probs[action] + 1e-10)
                
                # Ratio
                ratio = np.exp(new_log_prob - old_log_prob)
                
                # Clipped surrogate
                surr1 = ratio * advantage
                surr2 = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                policy_loss = -min(surr1, surr2)
                
                # Value loss
                new_value = float(state @ self.value_weights)
                value_loss = 0.5 * (ret - new_value) ** 2
                
                # Entropy bonus
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                
                # Gradient update (simplified)
                grad_policy = policy_loss * np.outer(state, np.eye(self.action_dim)[action] - probs)
                self.policy_weights -= self.learning_rate * grad_policy
                
                grad_value = (new_value - ret) * state.reshape(-1, 1)
                self.value_weights -= self.learning_rate * grad_value
        
        # Clear buffer
        self.training_episodes += 1
        self.cumulative_reward += np.sum(rewards)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        self.is_trained = True
        
        logging.info(f"PPO training step complete. Episodes: {self.training_episodes}")
        return {
            'status': 'TRAINED',
            'episodes': self.training_episodes,
            'avg_reward': float(np.mean(rewards)),
            'cumulative_reward': self.cumulative_reward
        }
    
    def get_analysis(self) -> Dict:
        """Get current agent status."""
        return {
            'is_trained': self.is_trained,
            'training_episodes': self.training_episodes,
            'cumulative_reward': self.cumulative_reward,
            'action_names': ['HOLD', 'BUY', 'SELL']
        }


class AlmgrenChrissExecution:
    """
    Almgren-Chriss Optimal Execution Model.
    
    Minimizes execution cost with market impact.
    Formula: x*(t) = X · sinh(κ(T-t)) / sinh(κT)
    
    For large position sizing to minimize slippage.
    """
    
    def __init__(self):
        self.sigma = 0.02  # Daily volatility
        self.gamma = 2.5e-7  # Permanent impact
        self.eta = 2.5e-6  # Temporary impact
        self.lambda_risk = 1e-6  # Risk aversion
        
        self.is_calibrated = False
        self.optimal_trajectory = None
    
    def calibrate(self, returns: np.ndarray, volumes: np.ndarray = None) -> bool:
        """Calibrate model from historical data."""
        if len(returns) < 50:
            return False
        
        try:
            self.sigma = float(np.std(returns))
            
            if volumes is not None and len(volumes) > 0:
                avg_volume = np.mean(volumes)
                # Estimate impact from volume
                self.gamma = 0.1 / (avg_volume + 1e-10)
                self.eta = 0.5 / (avg_volume + 1e-10)
            
            self.is_calibrated = True
            logging.info(f"Almgren-Chriss calibrated: σ={self.sigma:.4f}, γ={self.gamma:.2e}")
            return True
            
        except Exception as e:
            logging.error(f"A-C calibration error: {e}")
            return False
    
    def compute_optimal_trajectory(self, X: float, T: int = 10) -> np.ndarray:
        """
        Compute optimal trading trajectory.
        
        X: Total position to liquidate
        T: Time horizon (periods)
        """
        if not self.is_calibrated:
            return np.linspace(X, 0, T)
        
        try:
            # Kappa (urgency parameter)
            kappa = np.sqrt(self.lambda_risk * self.sigma**2 / self.eta)
            
            # Optimal trajectory
            t = np.arange(T + 1)
            x_star = X * np.sinh(kappa * (T - t)) / (np.sinh(kappa * T) + 1e-10)
            
            # Trading rate
            n_star = np.diff(-x_star)  # Negative because liquidating
            
            self.optimal_trajectory = x_star
            
            return x_star
            
        except Exception as e:
            logging.error(f"Trajectory computation error: {e}")
            return np.linspace(X, 0, T)
    
    def get_execution_cost(self, X: float, T: int = 10) -> Dict:
        """Estimate total execution cost."""
        if not self.is_calibrated:
            return {'status': 'UNCALIBRATED', 'cost': 0}
        
        try:
            trajectory = self.compute_optimal_trajectory(X, T)
            trades = np.diff(-trajectory)
            
            # Permanent impact cost
            perm_cost = 0.5 * self.gamma * X**2
            
            # Temporary impact cost
            temp_cost = self.eta * np.sum(trades**2)
            
            # Volatility cost (risk)
            kappa = np.sqrt(self.lambda_risk * self.sigma**2 / self.eta)
            vol_cost = self.lambda_risk * self.sigma**2 * X**2 / (2 * kappa * np.tanh(kappa * T / 2) + 1e-10)
            
            total_cost = perm_cost + temp_cost + vol_cost
            
            return {
                'status': 'COMPUTED',
                'permanent_impact': perm_cost,
                'temporary_impact': temp_cost,
                'volatility_cost': vol_cost,
                'total_cost': total_cost,
                'cost_bps': total_cost / (abs(X) + 1e-10) * 10000,  # In basis points
                'optimal_trades': trades.tolist() if len(trades) < 20 else trades[:20].tolist()
            }
            
        except Exception as e:
            logging.error(f"Cost estimation error: {e}")
            return {'status': 'ERROR', 'cost': 0}
    
    def get_analysis(self) -> Dict:
        """Get execution model analysis."""
        return {
            'is_calibrated': self.is_calibrated,
            'volatility': self.sigma,
            'permanent_impact': self.gamma,
            'temporary_impact': self.eta,
            'risk_aversion': self.lambda_risk
        }


class QuantEngine:




    """
    Unified Quantitative Analysis Engine.
    Combines all institutional-grade models (High + Medium Priority).
    """
    
    def __init__(self):
        # High Priority Models
        self.hmm_regime = RegimeSwitchingModel()
        self.gjr_garch = GJRGarchVolatility()
        self.ofi = OrderFlowImbalance()
        self.emd = EmpiricalModeDecomposition()
        
        # Medium Priority Models
        self.wavelet = WaveletAnalysis()
        self.copula = GaussianCopula()
        self.transfer_entropy = TransferEntropyAnalysis()
        self.heston = HestonVolatility()
        
        # Tier 1 Models (Phase 3)
        self.merton_jump = MertonJumpDiffusion()
        self.rough_vol = RoughVolatility()
        self.hht = HilbertHuangTransform()
        
        # Tier 2 Models (Phase 4)
        self.bates_svj = BatesSVJ()
        self.rqa = RecurrenceQuantification()
        self.multifractal = MultifractalAnalysis()
        self.tda = TopologicalDataAnalysis()
        
        # Tier 3 Models (Execution & RL)
        self.ppo_agent = PPOTradingAgent()
        self.almgren_chriss = AlmgrenChrissExecution()
        
        self.is_initialized = False
        self.last_analysis = {}



    
    def initialize(self, prices: np.ndarray, volumes: np.ndarray = None) -> bool:
        """Initialize all models with historical data."""
        success = True
        
        # High Priority Models
        # Fit HMM regime model
        if not self.hmm_regime.fit_and_analyze(prices, volumes):
            logging.warning("HMM regime model failed to initialize")
            success = False
        
        # Fit GJR-GARCH
        returns = np.diff(np.log(prices))
        if not self.gjr_garch.fit(returns):
            logging.warning("GJR-GARCH failed to initialize")
            success = False
        
        # Decompose with EMD
        if not self.emd.decompose(prices):
            logging.warning("EMD failed to initialize")
            success = False
        
        # Medium Priority Models
        # Wavelet decomposition
        if not self.wavelet.decompose(prices):
            logging.warning("Wavelet analysis failed to initialize")
        
        # Heston volatility calibration
        if not self.heston.calibrate(returns):
            logging.warning("Heston model failed to calibrate")
        
        # Tier 1 Models (Phase 3)
        # Merton Jump Diffusion
        if not self.merton_jump.calibrate(returns):
            logging.warning("Merton Jump model failed to calibrate")
        
        # Rough Volatility
        if not self.rough_vol.calibrate(returns):
            logging.warning("Rough Volatility failed to calibrate")
        
        # Hilbert-Huang Transform
        if not self.hht.decompose_and_analyze(prices):
            logging.warning("HHT failed to decompose")
        
        # Tier 2 Models (Phase 4)
        # Bates SVJ
        if not self.bates_svj.calibrate(returns):
            logging.warning("Bates SVJ failed to calibrate")
        
        # Recurrence Quantification
        if not self.rqa.compute_recurrence(prices[-200:] if len(prices) > 200 else prices):
            logging.warning("RQA failed to compute")
        
        # Multifractal DFA
        if not self.multifractal.compute_mfdfa(prices):
            logging.warning("MF-DFA failed to compute")
        
        # Topological Data Analysis
        if not self.tda.compute_persistent_homology(prices[-100:] if len(prices) > 100 else prices):
            logging.warning("TDA failed to compute")
        
        # Tier 3: Almgren-Chriss Execution Model
        if not self.almgren_chriss.calibrate(returns, volumes):
            logging.warning("Almgren-Chriss failed to calibrate")
        
        self.is_initialized = success or any([
            self.hmm_regime.hmm.is_fitted,
            self.gjr_garch.is_fitted,
            self.wavelet.coeffs is not None,
            self.heston.is_calibrated,
            self.merton_jump.is_calibrated,
            self.rough_vol.is_calibrated
        ])
        
        logging.info(f"QuantEngine initialized: {self.is_initialized}")
        return self.is_initialized


    
    def analyze(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict:
        """
        Run complete quantitative analysis.
        
        Returns comprehensive analysis dictionary.
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'regime': None,
            'volatility': None,
            'order_flow': None,
            'cycles': None,
            'signals': []
        }
        
        try:
            # Regime Analysis
            analysis['regime'] = self.hmm_regime.get_regime(prices, volumes)
            
            # Volatility Forecast
            if self.gjr_garch.is_fitted:
                analysis['volatility'] = {
                    'forecast': self.gjr_garch.forecast_volatility(horizon=1),
                    'asymmetry': self.gjr_garch.get_asymmetry(),
                    'current': float(self.gjr_garch.get_conditional_volatility()[-1]) if len(self.gjr_garch.get_conditional_volatility()) > 0 else 0
                }
            else:
                analysis['volatility'] = {'forecast': 0.01, 'asymmetry': 0, 'current': 0.01}
            
            # Order Flow
            if volumes is not None:
                analysis['order_flow'] = self.ofi.get_ofi_signal(prices, volumes)
            else:
                analysis['order_flow'] = {'ofi': 0, 'signal': 'NEUTRAL', 'strength': 'WEAK', 'normalized': 0}
            
            # Cycles
            analysis['cycles'] = {
                'strengths': self.emd.get_cycle_strength(),
                'details': self.emd.get_dominant_cycles()[:3]
            }
            
            # Medium Priority: Wavelet Analysis
            if self.wavelet.coeffs is not None:
                analysis['wavelet'] = {
                    'trend_strength': self.wavelet.get_trend_strength(),
                    'frequency_power': self.wavelet.get_frequency_power()
                }
            else:
                analysis['wavelet'] = {'trend_strength': 0.5, 'frequency_power': {}}
            
            # Medium Priority: Heston Volatility
            if self.heston.is_calibrated:
                analysis['heston'] = self.heston.get_volatility_forecast()
            else:
                analysis['heston'] = {'current_vol': 0, 'mean_vol': 0, 'leverage_effect': -0.7}
            
            # Generate trading signals based on analysis
            signals = []
            
            # Regime signal
            if analysis['regime']['regime'] == 'BEAR' and analysis['regime']['confidence'] > 70:
                signals.append({'type': 'REGIME', 'action': 'CAUTION', 'reason': 'Bear regime detected'})
            elif analysis['regime']['regime'] == 'BULL' and analysis['regime']['confidence'] > 70:
                signals.append({'type': 'REGIME', 'action': 'BULLISH', 'reason': 'Bull regime detected'})
            
            # Order flow signal
            if analysis['order_flow']['signal'] == 'BUY_PRESSURE' and analysis['order_flow']['strength'] == 'STRONG':
                signals.append({'type': 'OFI', 'action': 'BUY_SIGNAL', 'reason': 'Strong buy pressure'})
            elif analysis['order_flow']['signal'] == 'SELL_PRESSURE' and analysis['order_flow']['strength'] == 'STRONG':
                signals.append({'type': 'OFI', 'action': 'SELL_SIGNAL', 'reason': 'Strong sell pressure'})
            
            # Volatility signal
            if analysis['volatility']['asymmetry'] > 0.1:
                signals.append({'type': 'VOL', 'action': 'HEDGE', 'reason': 'High downside volatility risk'})
            
            # Heston leverage effect signal
            if self.heston.is_calibrated and self.heston.rho < -0.5:
                signals.append({'type': 'HESTON', 'action': 'CRASH_RISK', 'reason': f'Strong leverage effect (ρ={self.heston.rho:.2f})'})
            
            # Tier 1: Jump Detection
            if self.merton_jump.is_calibrated:
                latest_return = float(np.diff(np.log(prices[-2:]))[0]) if len(prices) > 1 else 0
                jump_info = self.merton_jump.detect_jump(latest_return)
                analysis['jump'] = jump_info
                analysis['jump_risk'] = self.merton_jump.get_jump_risk()
                
                if jump_info['is_jump']:
                    direction = jump_info['direction']
                    signals.append({'type': 'JUMP', 'action': f'JUMP_{direction}', 'reason': f'Black swan detected ({jump_info["probability"]:.0f}% prob)'})
            else:
                analysis['jump'] = {'is_jump': False, 'probability': 0}
                analysis['jump_risk'] = {'risk_level': 'UNKNOWN'}
            
            # Tier 1: Rough Volatility
            if self.rough_vol.is_calibrated:
                analysis['rough_vol'] = self.rough_vol.get_roughness_analysis()
            else:
                analysis['rough_vol'] = {'H': 0.5, 'interpretation': 'UNKNOWN', 'roughness_score': 50}
            
            # Tier 1: Hilbert-Huang Transform
            if self.hht.is_decomposed:
                analysis['hht'] = self.hht.get_current_frequencies()
            else:
                analysis['hht'] = {'dominant_freq': 0, 'dominant_period_minutes': 0}
            
            # Tier 2: Bates SVJ
            if self.bates_svj.is_calibrated:
                analysis['bates_svj'] = self.bates_svj.get_analysis()
            else:
                analysis['bates_svj'] = {'status': 'NOT_CALIBRATED'}
            
            # Tier 2: Recurrence Quantification
            analysis['rqa'] = self.rqa.get_analysis()
            
            # Tier 2: Multifractal Analysis
            analysis['multifractal'] = self.multifractal.get_analysis()
            
            # Tier 2: Topological Data Analysis
            analysis['tda'] = self.tda.get_analysis()
            
            # Tier 3: Almgren-Chriss Execution
            if self.almgren_chriss.is_calibrated:
                analysis['execution'] = self.almgren_chriss.get_analysis()
            else:
                analysis['execution'] = {'status': 'NOT_CALIBRATED'}
            
            # Tier 2: Multifractal regime signal
            mf = analysis['multifractal']
            if mf.get('interpretation') == 'STRONGLY_MULTIFRACTAL':
                signals.append({'type': 'MFDFA', 'action': 'COMPLEX_REGIME', 'reason': f'Strong multifractality (ΔH={mf.get("delta_h", 0):.3f})'})
            
            # Tier 2: RQA determinism signal
            rqa = analysis['rqa']
            if rqa.get('interpretation') == 'HIGHLY_DETERMINISTIC':
                signals.append({'type': 'RQA', 'action': 'PATTERN_LOCK', 'reason': f'High determinism (DET={rqa.get("DET", 0):.2f})'})
            
            analysis['signals'] = signals


            self.last_analysis = analysis
            
        except Exception as e:
            logging.error(f"QuantEngine analysis error: {e}")
        
        return analysis


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 500
    
    # Simulate price with regime changes
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    volumes = np.abs(np.random.randn(n) * 1000 + 5000)
    
    print("=== Testing QuantEngine ===\n")
    
    engine = QuantEngine()
    engine.initialize(prices, volumes)
    
    analysis = engine.analyze(prices, volumes)
    
    print(f"Regime: {analysis['regime']['regime']} ({analysis['regime']['confidence']:.1f}% confidence)")
    print(f"Volatility Forecast: {analysis['volatility']['forecast']:.4f}")
    print(f"Asymmetry (gamma): {analysis['volatility']['asymmetry']:.4f}")
    print(f"Order Flow: {analysis['order_flow']['signal']} ({analysis['order_flow']['strength']})")
    print(f"Cycle Strengths: {analysis['cycles']['strengths']}")
    print(f"\nSignals: {analysis['signals']}")
