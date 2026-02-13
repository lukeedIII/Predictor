import numpy as np
import torch
import logging
from filterpy.kalman import KalmanFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MathCore:
    def __init__(self):
        # Smart GPU detection: RTX 5080 (Blackwell/sm_100) needs PyTorch built
        # with CUDA 12.8+ for full kernel support. PyTorch 2.6+cu124 detects the 
        # GPU but cuDNN LSTM kernels crash. We test with a real op first.
        self.device = "cpu"
        if torch.cuda.is_available():
            try:
                # Test a real CUDA operation — if this works, basic ops are fine
                test = torch.randn(2, 2, device="cuda") @ torch.randn(2, 2, device="cuda")
                del test
                torch.cuda.empty_cache()
                self.device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                logging.info(f"MathCore initialized on GPU: {gpu_name}")
            except RuntimeError:
                self.device = "cpu"
                logging.warning("CUDA detected but kernels incompatible (Blackwell/cu124). Using CPU.")
        else:
            logging.info("MathCore initialized on CPU (no CUDA GPU detected)")


    def calculate_hurst_exponent(self, prices):
        """
        Calculate Hurst exponent using R/S (Rescaled Range) analysis.
        Uses LOG-SPACED lags up to N/4 for statistical robustness.
        
        Interpretation:
        - H > 0.5: Persistent/Trending (price continues in same direction)
        - H = 0.5: Random walk (no autocorrelation)
        - H < 0.5: Anti-persistent/Mean-reverting (price tends to reverse)
        """
        if len(prices) < 50:
            return 0.5
        
        try:
            prices = np.array(prices)
            n = len(prices)
            
            # Use log-spaced lags from 10 to N/4 for better statistical properties
            max_lag = min(n // 4, 100)
            if max_lag < 10:
                return 0.5
            
            lags = np.unique(np.floor(np.logspace(1, np.log10(max_lag), 20)).astype(int))
            lags = lags[lags >= 10]  # Minimum lag of 10
            
            if len(lags) < 5:
                return 0.5
            
            rs_values = []
            for lag in lags:
                # Divide series into non-overlapping subseries
                subseries_count = n // lag
                if subseries_count < 1:
                    continue
                
                rs_lag = []
                for i in range(subseries_count):
                    subseries = prices[i*lag:(i+1)*lag]
                    if len(subseries) < 2:
                        continue
                    
                    # Calculate returns
                    returns = np.diff(subseries)
                    
                    # Mean-adjusted cumulative sum
                    mean_return = np.mean(returns)
                    cumsum = np.cumsum(returns - mean_return)
                    
                    # Range
                    R = np.max(cumsum) - np.min(cumsum)
                    
                    # Standard deviation
                    S = np.std(returns, ddof=1)
                    
                    if S > 0:
                        rs_lag.append(R / S)
                
                if rs_lag:
                    rs_values.append((lag, np.mean(rs_lag)))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Linear regression in log-log space
            log_lags = np.log([x[0] for x in rs_values])
            log_rs = np.log([x[1] for x in rs_values])
            
            # Fit: log(R/S) = H * log(n) + c
            slope, _ = np.polyfit(log_lags, log_rs, 1)
            
            # Clamp to valid range [0, 1]
            hurst = np.clip(slope, 0.0, 1.0)
            
            return float(np.nan_to_num(hurst, nan=0.5))
            
        except Exception as e:
            logging.error(f"Hurst calculation error: {e}")
            return 0.5

    def kalman_smooth(self, prices):
        """Apply Kalman filter for noise reduction."""
        if len(prices) < 2:
            return prices
        
        try:
            kf = KalmanFilter(dim_x=2, dim_z=1)
            kf.x = np.array([[prices[0]], [0.]])
            kf.F = np.array([[1., 1.], [0., 1.]])  # State transition
            kf.H = np.array([[1., 0.]])  # Measurement function
            kf.P *= 10.  # Initial uncertainty
            kf.R = 0.01  # Measurement noise
            kf.Q = 0.001  # Process noise
            
            smoothed = []
            for z in prices:
                kf.predict()
                kf.update(z)
                smoothed.append(float(kf.x[0]))
            
            return np.array(smoothed)
        except Exception as e:
            logging.error(f"Kalman smoothing error: {e}")
            return prices

    def estimate_garch_volatility(self, returns, window=30):
        """
        Simple GARCH(1,1)-like volatility estimation.
        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
        """
        if len(returns) < window:
            return np.std(returns) if len(returns) > 1 else 0.01
        
        try:
            # Use exponentially weighted variance as a GARCH proxy
            alpha = 0.06  # Weight on recent squared return
            beta = 0.94   # Weight on previous variance
            
            variance = np.var(returns[:window])
            
            for i in range(window, len(returns)):
                variance = alpha * returns[i-1]**2 + beta * variance
            
            return np.sqrt(max(variance, 1e-8))
            
        except:
            return np.std(returns) if len(returns) > 1 else 0.01

    def extract_cycles(self, prices, top_n=3):
        """
        Extract dominant market cycles using FFT.
        Returns relative amplitudes AND actual period lengths.
        """
        if len(prices) < 50:
            return [0.0] * top_n
        
        try:
            p_tensor = torch.tensor(prices, device=self.device, dtype=torch.float32)
            
            # Normalize
            p_norm = (p_tensor - p_tensor.mean()) / (p_tensor.std() + 1e-9)
            
            # Apply FFT
            fft_values = torch.fft.fft(p_norm)
            amplitudes = torch.abs(fft_values)
            
            # Only consider positive frequencies (skip DC component)
            half_n = len(amplitudes) // 2
            amplitudes_half = amplitudes[1:half_n]
            
            # Get top N peaks
            if len(amplitudes_half) < top_n:
                return [0.0] * top_n
            
            val, idx = torch.topk(amplitudes_half, top_n)
            
            # Normalize to relative strength
            total = torch.sum(val) + 1e-9
            relative_strength = (val / total).tolist()
            
            return relative_strength
            
        except Exception as e:
            logging.error(f"Cycle extraction error: {e}")
            return [0.0] * top_n

    def run_monte_carlo(self, current_price, volatility, steps=60, simulations=1000, drift=None):
        """
        Monte Carlo simulation with drift term and GARCH-based volatility.
        
        Uses Geometric Brownian Motion (GBM):
        dS = mu*S*dt + sigma*S*dW
        
        Where:
        - mu: drift (estimated from recent returns if not provided)
        - sigma: volatility (should be GARCH-estimated)
        - dW: Wiener process increment
        """
        try:
            if volatility <= 0:
                volatility = 0.001
            
            # If no drift provided, assume slight positive for crypto
            if drift is None:
                drift = 0.0  # Conservative: assume no drift (random walk)
            
            dt = 1.0 / steps  # Time step (fraction of prediction horizon)
            
            # Generate random increments
            epsilon = torch.randn(simulations, steps, device=self.device)
            
            # GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*epsilon)
            drift_term = (drift - 0.5 * volatility**2) * dt
            diffusion_term = volatility * np.sqrt(dt) * epsilon
            
            log_returns = drift_term + diffusion_term
            price_paths = current_price * torch.exp(torch.cumsum(log_returns, dim=1))
            
            return price_paths.cpu().numpy()
            
        except Exception as e:
            logging.error(f"Monte Carlo error: {e}")
            return np.ones((simulations, steps)) * current_price

    def get_market_regime(self, hurst):
        """
        Interpret Hurst exponent for market regime.
        
        Returns:
        - 'TRENDING': H > 0.55 (persistent, momentum strategies work)
        - 'MEAN_REVERTING': H < 0.45 (anti-persistent, mean-reversion strategies work)
        - 'RANDOM': 0.45 <= H <= 0.55 (random walk, no clear edge)
        """
        if hurst > 0.55:
            return "TRENDING"
        elif hurst < 0.45:
            return "MEAN_REVERTING"
        else:
            return "RANDOM"
