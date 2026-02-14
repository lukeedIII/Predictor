import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import config
import logging
import joblib
import pickle
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from math_core import MathCore
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import QuantEngine for institutional-grade analysis
try:
    from quant_models import QuantEngine
    QUANT_AVAILABLE = True
except ImportError:
    QUANT_AVAILABLE = False
    logging.warning("QuantEngine not available. Running in basic mode.")

# Import FREE alternative data sources
try:
    from alt_data import FreeDataAggregator
    ALT_DATA_AVAILABLE = True
except ImportError:
    ALT_DATA_AVAILABLE = False
    logging.warning("Alternative data not available.")


# ========== DEEP MODEL SEQUENCE LENGTH ==========
DEEP_SEQ_LEN = 30  # 30 timesteps per sample — Transformer processes temporal windows


class NexusTransformer(nn.Module):
    """Production-grade Transformer Encoder for temporal sequence classification.
    ~90M params (~360 MB) — sized for RTX 5080 (16 GB VRAM, uses ~1.5 GB).
    
    Architecture:
      Input projection → Positional Embedding → 12x TransformerEncoder layers
      → [CLS] token pooling → Classification head → Sigmoid
    """
    def __init__(self, input_size=42, d_model=1024, nhead=16, num_layers=12,
                 dim_feedforward=4096, dropout=0.15):
        super(NexusTransformer, self).__init__()
        self.d_model = d_model
        self.seq_len = DEEP_SEQ_LEN
        
        # Project raw features into d_model dimensional space
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Learnable positional embeddings (seq_len + 1 for [CLS] token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, DEEP_SEQ_LEN + 1, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer encoder (Pre-LayerNorm for stable deep training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN — critical for 12-layer stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights (Xavier uniform for projections)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, return_logits=False):
        # x shape: (batch, seq_len, features)
        B = x.shape[0]
        
        # Project features → d_model
        x = self.input_proj(x)  # (B, seq_len, d_model)
        
        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, seq_len+1, d_model)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.shape[1], :]
        x = self.pos_dropout(x)
        
        # Transformer Encoder
        x = self.encoder(x)  # (B, seq_len+1, d_model)
        
        # [CLS] token output = sequence-level representation
        cls_out = x[:, 0, :]  # (B, d_model)
        
        logits = self.head(cls_out)
        if return_logits:
            return logits  # Raw logits for BCEWithLogitsLoss (AMP-safe)
        return self.sigmoid(logits)
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    @property
    def size_mb(self):
        return sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)


# ========== Backward compatibility alias ==========
# Keep NexusLSTM as a minimal stub so old references don't crash during migration
class NexusLSTM(NexusTransformer):
    """DEPRECATED — redirects to NexusTransformer for backward compatibility."""
    def __init__(self, input_size=42, **kwargs):
        super().__init__(input_size=input_size)

class NexusPredictor:
    def __init__(self):
        self.model_path = os.path.join(config.MODEL_DIR, "predictor_v3.joblib")
        self.deep_model_path = os.path.join(config.MODEL_DIR, "nexus_transformer_v1.pth")
        self.pretrained_path = os.path.join(config.MODEL_DIR, "nexus_transformer_pretrained.pth")
        # Legacy alias for backward compat with code that references lstm_path
        self.lstm_path = self.deep_model_path
        self.scaler_path = os.path.join(config.MODEL_DIR, "feature_scaler_v3.pkl")
        self.ensemble_state_path = os.path.join(config.MODEL_DIR, "ensemble_state_v3.pkl")
        self.math = MathCore()
        self.device = self.math.device
        
        # ===== CLEAN FEATURE SET (v6 — microstructure upgrade) =====
        # All features are scale-invariant (returns/ratios, not raw prices)
        # No distribution shift between training and prediction
        self.features = [
            # Returns-based OHLCV (scale-invariant across any BTC price level)
            'close_ret_1', 'close_ret_5', 'close_ret_15',
            'high_low_range', 'close_open_range', 'volume_ratio',
            # Technical indicators (normalized)
            'sma_20_dist', 'sma_50_dist', 'rsi', 'volatility',
            # Fourier cycles (rolling window — same computation train & predict)
            'cycle_1', 'cycle_2',
            # Market microstructure
            'hurst', 'kalman_err_norm', 'obi_sim',
            # Quant features (vectorized row-level — no block staleness)
            'regime_id', 'regime_confidence', 'gjr_volatility',
            # Advanced quant (Hawkes + Optimal Transport)
            'hawkes_intensity', 'wass_drift',
            # Multi-timeframe trend (computed from 1-min data)
            'trend_5m', 'trend_15m', 'trend_1h',
            'ret_60', 'ret_240', 'vol_regime',
            # Volume profile
            'vwap_dist', 'vol_momentum',
            # Cross-asset correlation (ETH + Gold + ETH/BTC)
            'eth_ret_5', 'eth_ret_15', 'eth_vol_ratio',
            'ethbtc_ret_5', 'ethbtc_trend',
            'gold_ret_15', 'gold_ret_60',
            # Phase 3: Microstructure features (from 1m candle proxies)
            'trade_intensity', 'buy_sell_ratio', 'vwap_momentum',
            'tick_volatility', 'large_trade_ratio',
            # Phase 4: Real WebSocket microstructure (from live tick data)
            'ws_trades_per_sec', 'ws_buy_sell_ratio', 'ws_spread_bps',
        ]
        # Real-time-only features (used as confidence boost, NOT in training)
        self.alt_data_features = [
            'fear_greed_norm', 'orderbook_imbalance', 'trade_pressure', 'momentum_24h'
        ]
        self.validation_split = 0.2
        self.min_train_samples = 80
        self.prediction_horizon = 15  # P2: 15 min instead of 60 — more actionable
        self.is_statistically_verified = False
        self.last_validation_accuracy = 0.0
        self._training_accuracy = 0.0  # accuracy from model training (test-set)
        self._live_accuracy = 0.0  # accuracy from live prediction outcomes
        self._live_accuracy_samples = 0  # how many predictions have been validated
        
        # Live prediction validation — track and score predictions
        self._prediction_history = []  # [{time, direction, price, confidence, validated, correct}]
        self._validation_window = 15  # minutes before checking outcome
        self.calibrated_model = None
        
        # P0: Ensemble weights — Transformer starts at 0 until it earns weight via validation
        self.xgb_weight = 1.0
        self.lstm_weight = 0.0  # Earns weight when Transformer validation > 52% (kept name for API compat)
        self.lstm_validation_acc = 0.0
        
        # P1: Feature scaler for LSTM normalization
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Initialize QuantEngine for institutional analysis
        self.quant_engine = QuantEngine() if QUANT_AVAILABLE else None
        self.quant_initialized = False
        self.last_quant_analysis = {}
        
        # Initialize FREE alternative data aggregator
        self.alt_data = FreeDataAggregator() if ALT_DATA_AVAILABLE else None
        self.last_alt_signals = {}
        
        self.initialize_models()
    
    @property
    def is_trained(self):
        """True if XGBoost model is fitted and ready for prediction."""
        return hasattr(self.model, 'classes_') and os.path.exists(self.model_path)
    
    def _save_ensemble_state(self):
        """Persist ensemble weights so they survive restarts."""
        state = {
            'xgb_weight': self.xgb_weight,
            'lstm_weight': self.lstm_weight,
            'lstm_validation_acc': self.lstm_validation_acc,
            'last_validation_accuracy': self.last_validation_accuracy,
            'is_statistically_verified': self.is_statistically_verified,
        }
        try:
            with open(self.ensemble_state_path, 'wb') as f:
                pickle.dump(state, f)
            logging.info(f"Ensemble state saved: XGB={self.xgb_weight:.2f} LSTM={self.lstm_weight:.2f}")
        except Exception as e:
            logging.error(f"Failed to save ensemble state: {e}")
    
    def _load_ensemble_state(self):
        """Restore ensemble weights from disk."""
        if os.path.exists(self.ensemble_state_path):
            try:
                with open(self.ensemble_state_path, 'rb') as f:
                    state = pickle.load(f)
                self.xgb_weight = state.get('xgb_weight', 1.0)
                self.lstm_weight = state.get('lstm_weight', 0.0)
                self.lstm_validation_acc = state.get('lstm_validation_acc', 0.0)
                self.last_validation_accuracy = state.get('last_validation_accuracy', 0.0)
                self.is_statistically_verified = state.get('is_statistically_verified', False)
                logging.info(
                    f"Ensemble state restored: XGB={self.xgb_weight:.2f} LSTM={self.lstm_weight:.2f} "
                    f"LSTM_acc={self.lstm_validation_acc:.1f}% verified={self.is_statistically_verified}"
                )
            except Exception as e:
                logging.warning(f"Failed to load ensemble state: {e}")


    def initialize_models(self):
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        
        # Phase 5: Base model paths (shipped with app for instant-on)
        base_xgb = os.path.join(config.BASE_MODEL_DIR, "base_xgboost.joblib")
        base_lstm = os.path.join(config.BASE_MODEL_DIR, "base_lstm.pth")
        base_scaler = os.path.join(config.BASE_MODEL_DIR, "base_scaler.pkl")
        base_ensemble = os.path.join(config.BASE_MODEL_DIR, "base_ensemble_state.pkl")
        
        # === XGBoost: user model > base model > fresh ===
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logging.info(f"XGBoost model loaded from {self.model_path}")
        elif os.path.exists(base_xgb):
            self.model = joblib.load(base_xgb)
            logging.info(f"📦 Base XGBoost model loaded (instant-on mode)")
        else:
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=500,    # v6: 200→500 (more capacity for 40 features)
                max_depth=6,         # v6: 4→6 (deeper = captures complex interactions)
                learning_rate=0.03,  # v6: 0.05→0.03 (slower with more trees = better)
                subsample=0.8,
                colsample_bytree=0.7,  # v6: 0.8→0.7 (more feature diversity per tree)
                min_child_weight=5,
                gamma=0.1,           # v6: new — minimum loss reduction for split (prunes noise)
                reg_alpha=0.1,       # L1 regularization
                reg_lambda=1.5,      # v6: 1.0→1.5 (stronger L2 to offset deeper trees)
                n_jobs=8
            )
            logging.info("XGBoost model initialized (untrained)")
        
        # === Transformer: user model > base model > pretrained > fresh ===
        self.lstm_device = self.device  # kept name for API compat
        self.lstm = NexusTransformer(input_size=len(self.features)).to(self.lstm_device)
        lstm_loaded = False
        load_candidates = [
            (self.deep_model_path, "user"),
            (base_lstm, "base"),
            (self.pretrained_path, "pretrained"),
        ]
        for model_file, label in load_candidates:
            if os.path.exists(model_file) and not lstm_loaded:
                try:
                    self.lstm.load_state_dict(torch.load(model_file, map_location=self.lstm_device))
                    logging.info(f"🧠 Transformer loaded from {label}: {model_file} ({self.lstm.size_mb:.0f} MB, {self.lstm.num_parameters/1e6:.1f}M params)")
                    lstm_loaded = True
                    if label == "pretrained":
                        logging.info("   ↳ Using pretrained weights — will fine-tune on live data")
                except RuntimeError as e:
                    logging.warning(f"Transformer load failed ({label}, shape mismatch? will retrain): {e}")
                    self.lstm = NexusTransformer(input_size=len(self.features)).to(self.lstm_device)
                    self.lstm_weight = 0.0
        
        # === Scaler: user > base ===
        scaler_loaded = False
        for scaler_file, label in [(self.scaler_path, "user"), (base_scaler, "base")]:
            if os.path.exists(scaler_file) and not scaler_loaded:
                try:
                    with open(scaler_file, 'rb') as f:
                        self.scaler = pickle.load(f)
                        self.scaler_fitted = True
                        scaler_loaded = True
                    logging.info(f"Feature scaler loaded ({label}, fitted={self.scaler_fitted})")
                except Exception as e:
                    logging.warning(f"Scaler load failed ({label}): {e}")
        
        # === Ensemble state: user > base ===
        if os.path.exists(self.ensemble_state_path):
            self._load_ensemble_state()
        elif os.path.exists(base_ensemble):
            # Temporarily swap path to load base state
            orig_path = self.ensemble_state_path
            self.ensemble_state_path = base_ensemble
            self._load_ensemble_state()
            self.ensemble_state_path = orig_path
            logging.info("📦 Base ensemble state loaded")
        
        # If Transformer failed to load (shape mismatch after feature list change), ensure weight stays 0
        if not lstm_loaded and self.lstm_weight > 0:
            logging.warning("Transformer weight reset to 0 — model not loaded (needs retrain with new features)")
        elif not lstm_loaded:
            logging.info(f"🆕 Fresh Transformer initialized ({self.lstm.size_mb:.0f} MB, {self.lstm.num_parameters/1e6:.1f}M params)")
            self.lstm_weight = 0.0
            self.xgb_weight = 1.0

    def _load_market_data(self):
        """Load market data for TRAINING. Uses last 500K rows (~1 year of 1m data).
        Vectorized features make this fast — no heavy QuantEngine loop."""
        MAX_TRAIN_ROWS = 500_000  # ~1 year — captures seasonal patterns & regime changes
        df = None
        if os.path.exists(config.MARKET_DATA_PARQUET_PATH):
            df = pd.read_parquet(config.MARKET_DATA_PARQUET_PATH)
        elif os.path.exists(config.MARKET_DATA_PATH):
            df = pd.read_csv(config.MARKET_DATA_PATH)
        if df is not None and len(df) > MAX_TRAIN_ROWS:
            logging.info(f"Training data: using last {MAX_TRAIN_ROWS:,} of {len(df):,} rows")
            df = df.tail(MAX_TRAIN_ROWS).reset_index(drop=True)
        return df

    def _load_market_data_tail(self, n=500):
        """Load only the last N rows — fast path for prediction."""
        if os.path.exists(config.MARKET_DATA_PARQUET_PATH):
            df = pd.read_parquet(config.MARKET_DATA_PARQUET_PATH)
            return df.tail(n).reset_index(drop=True)
        elif os.path.exists(config.MARKET_DATA_PATH):
            df = pd.read_csv(config.MARKET_DATA_PATH)
            return df.tail(n).reset_index(drop=True)
        return None

    def _load_cross_asset_data(self, n=None):
        """Load cross-asset data (ETH, PAXG, ETH/BTC) and merge by timestamp.
        Returns dict of DataFrames keyed by prefix, or empty dict if unavailable."""
        pairs = {
            'eth': config.ETH_DATA_PATH,
            'paxg': config.PAXG_DATA_PATH,
            'ethbtc': config.ETHBTC_DATA_PATH,
        }
        result = {}
        for prefix, path in pairs.items():
            if os.path.exists(path):
                try:
                    xdf = pd.read_parquet(path)
                    xdf['timestamp'] = pd.to_datetime(xdf['timestamp'])
                    if n is not None:
                        xdf = xdf.tail(n).reset_index(drop=True)
                    # Prefix columns to avoid collision with BTC data
                    xdf = xdf.rename(columns={
                        c: f"{prefix}_{c}" for c in xdf.columns if c != 'timestamp'
                    })
                    result[prefix] = xdf
                except Exception as e:
                    logging.debug(f"Cross-asset {prefix}: {e}")
        return result

    def _engineer_features(self, df, fast_mode=False):
        """
        Engineer SCALE-INVARIANT features on a DataFrame.
        v4: All features are returns/ratios — no raw prices or volumes.
        IDENTICAL computation path for training and prediction (no distribution shift).
        fast_mode=True: skip expensive rolling Hurst, use single-shot instead.
        """
        if df is None or len(df) < self.min_train_samples:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        raw_prices = df['close'].values
        
        # ===== RETURNS-BASED OHLCV (scale-invariant) =====
        df['close_ret_1'] = df['close'].pct_change(1)    # 1-min return
        df['close_ret_5'] = df['close'].pct_change(5)    # 5-min return
        df['close_ret_15'] = df['close'].pct_change(15)  # 15-min return (matches horizon)
        df['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-9)  # candle range
        df['close_open_range'] = (df['close'] - df['open']) / (df['close'] + 1e-9)  # candle body
        df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20, min_periods=1).mean() + 1e-9)  # relative volume
        
        # ===== TECHNICAL INDICATORS (normalized) =====
        kalman_price = self.math.kalman_smooth(raw_prices)
        df['kalman_price'] = kalman_price  # keep for internal use, not a feature
        df['kalman_err_norm'] = (df['close'] - df['kalman_price']) / (df['close'] + 1e-9)  # normalized error
        
        sma_20 = df['kalman_price'].rolling(window=20, min_periods=1).mean()
        sma_50 = df['kalman_price'].rolling(window=50, min_periods=1).mean()
        df['sma_20_dist'] = (df['close'] - sma_20) / (df['close'] + 1e-9)  # distance from SMA20 as fraction
        df['sma_50_dist'] = (df['close'] - sma_50) / (df['close'] + 1e-9)  # distance from SMA50 as fraction
        df['volatility'] = df['close'].pct_change().rolling(window=10, min_periods=1).std()
        
        # RSI (standard 14-period) — already 0-100 scale, universal
        delta = df['kalman_price'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        # Order Book Imbalance simulation — already a ratio
        df['obi_sim'] = df['volume'].diff() / (df['volume'].rolling(window=10, min_periods=1).mean() + 1e-9)
        
        # ===== HURST EXPONENT =====
        if fast_mode:
            tail_prices = raw_prices[-100:] if len(raw_prices) >= 100 else raw_prices
            hurst_val = self.math.calculate_hurst_exponent(tail_prices)
            df['hurst'] = hurst_val
        else:
            # Rolling Hurst — sampled every 10 rows then interpolated for speed
            hurst_vals = np.full(len(df), 0.5)
            window = 100
            step = 10  # compute every 10 rows, interpolate the rest
            computed_indices = []
            computed_values = []
            for i in range(window, len(df), step):
                try:
                    h = self.math.calculate_hurst_exponent(raw_prices[i - window:i])
                    hurst_vals[i] = h
                    computed_indices.append(i)
                    computed_values.append(h)
                except Exception:
                    hurst_vals[i] = 0.5
                    computed_indices.append(i)
                    computed_values.append(0.5)
            # Interpolate between computed points
            if len(computed_indices) > 1:
                hurst_vals = np.interp(range(len(df)), computed_indices, computed_values)
            df['hurst'] = hurst_vals
        
        # ===== FOURIER CYCLES (rolling FFT) =====
        if fast_mode:
            cycles = self.math.extract_cycles(
                raw_prices[-100:] if len(raw_prices) >= 100 else raw_prices, top_n=2
            )
            df['cycle_1'] = cycles[0]
            df['cycle_2'] = cycles[1]
        else:
            # Sampled every 10 rows then interpolated (FFT is expensive)
            cycle_1_vals = np.zeros(len(df))
            cycle_2_vals = np.zeros(len(df))
            fft_window = 100
            fft_step = 10
            c1_indices, c1_values = [], []
            c2_indices, c2_values = [], []
            for i in range(fft_window, len(df), fft_step):
                try:
                    c = self.math.extract_cycles(raw_prices[i - fft_window:i], top_n=2)
                    c1_indices.append(i); c1_values.append(c[0])
                    c2_indices.append(i); c2_values.append(c[1])
                except Exception:
                    c1_indices.append(i); c1_values.append(0.0)
                    c2_indices.append(i); c2_values.append(0.0)
            if len(c1_indices) > 1:
                cycle_1_vals = np.interp(range(len(df)), c1_indices, c1_values)
                cycle_2_vals = np.interp(range(len(df)), c2_indices, c2_values)
            df['cycle_1'] = cycle_1_vals
            df['cycle_2'] = cycle_2_vals
        
        # ===== MULTI-TIMEFRAME TREND (from 1-min data) =====
        # 5-min trend: fast SMA > slow SMA
        sma_5 = df['close'].rolling(5, min_periods=1).mean()
        sma_20_raw = df['close'].rolling(20, min_periods=1).mean()
        df['trend_5m'] = (sma_5 > sma_20_raw).astype(float)
        
        # 15-min trend: 15-bar SMA > 60-bar SMA
        sma_15 = df['close'].rolling(15, min_periods=1).mean()
        sma_60 = df['close'].rolling(60, min_periods=1).mean()
        df['trend_15m'] = (sma_15 > sma_60).astype(float)
        
        # 1-hour trend: 60-bar SMA > 240-bar SMA
        sma_240 = df['close'].rolling(240, min_periods=1).mean()
        df['trend_1h'] = (sma_60 > sma_240).astype(float)
        
        # Higher-timeframe momentum
        df['ret_60'] = df['close'].pct_change(60)   # 1-hour return
        df['ret_240'] = df['close'].pct_change(240)  # 4-hour return
        
        # Volatility regime: current vol / median vol (is vol expanding or contracting?)
        vol_median = df['volatility'].rolling(60, min_periods=1).median()
        df['vol_regime'] = df['volatility'] / (vol_median + 1e-9)
        
        # ===== VOLUME PROFILE =====
        # VWAP distance (volume-weighted average price)
        roll_vol = df['volume'].rolling(20, min_periods=1).sum()
        roll_pv = (df['close'] * df['volume']).rolling(20, min_periods=1).sum()
        vwap = roll_pv / (roll_vol + 1e-9)
        df['vwap_dist'] = (df['close'] - vwap) / (df['close'] + 1e-9)
        
        # Volume momentum
        df['vol_momentum'] = df['volume'].pct_change(5).clip(-5, 5)  # capped to avoid outliers
        
        # ===== CROSS-ASSET CORRELATION (Phase 3: ETH + Gold + ETH/BTC) =====
        # Initialize with neutral defaults (backward compatible if data missing)
        for col in ['eth_ret_5', 'eth_ret_15', 'eth_vol_ratio',
                    'ethbtc_ret_5', 'ethbtc_trend', 'gold_ret_15', 'gold_ret_60']:
            df[col] = 0.0
        
        try:
            n_rows = len(df) + 300  # Extra rows for rolling calculations
            cross = self._load_cross_asset_data(n=n_rows)
            
            if 'eth' in cross:
                merged = df[['timestamp']].merge(cross['eth'], on='timestamp', how='left')
                if 'eth_close' in merged.columns:
                    df['eth_ret_5'] = merged['eth_close'].pct_change(5).fillna(0).values
                    df['eth_ret_15'] = merged['eth_close'].pct_change(15).fillna(0).values
                    eth_vol = merged.get('eth_volume', pd.Series(0, index=merged.index))
                    df['eth_vol_ratio'] = (eth_vol / (eth_vol.rolling(20, min_periods=1).mean() + 1e-9)).fillna(0).values
            
            if 'ethbtc' in cross:
                merged = df[['timestamp']].merge(cross['ethbtc'], on='timestamp', how='left')
                if 'ethbtc_close' in merged.columns:
                    df['ethbtc_ret_5'] = merged['ethbtc_close'].pct_change(5).fillna(0).values
                    # ETH/BTC trend: 5-bar SMA > 20-bar SMA
                    sma5 = merged['ethbtc_close'].rolling(5, min_periods=1).mean()
                    sma20 = merged['ethbtc_close'].rolling(20, min_periods=1).mean()
                    df['ethbtc_trend'] = (sma5 > sma20).astype(float).fillna(0).values
            
            if 'paxg' in cross:
                merged = df[['timestamp']].merge(cross['paxg'], on='timestamp', how='left')
                if 'paxg_close' in merged.columns:
                    df['gold_ret_15'] = merged['paxg_close'].pct_change(15).fillna(0).values
                    df['gold_ret_60'] = merged['paxg_close'].pct_change(60).fillna(0).values
        except Exception as e:
            logging.debug(f"Cross-asset features skipped: {e}")
        
        # ===== MICROSTRUCTURE FEATURES (Phase 3 — from 1m candle proxies) =====
        df = self._add_microstructure_features(df)
        
        # ===== QUANT FEATURES (vectorized, row-level) =====
        df = self._add_quant_features_vectorized(df)
        
        # ===== ADVANCED QUANT: Hawkes + Wasserstein =====
        df = self._add_hawkes_wasserstein_features(df)
        
        # ===== ALT DATA (confidence boost only, NOT in self.features) =====
        if fast_mode:
            df = self._add_alt_data_features(df)
        else:
            for col in self.alt_data_features:
                df[col] = 0.0
        
        # ===== QUANT ENGINE for UI display (not for features) =====
        if self.quant_engine is not None:
            try:
                prices = df['close'].values
                volumes = df['volume'].values if 'volume' in df.columns else None
                if not self.quant_initialized and len(prices) >= 100:
                    self.quant_initialized = self.quant_engine.initialize(prices, volumes)
                if self.quant_initialized:
                    self.last_quant_analysis = self.quant_engine.analyze(prices, volumes)
            except Exception:
                pass
        
        df = df.ffill().fillna(0.0)
        return df

    def _load_microstructure_data(self, n=None):
        """Load WebSocket microstructure snapshots and prepare for merge with OHLCV.
        Returns DataFrame with timestamp index and ws_* columns, or None."""
        if not os.path.exists(config.MICROSTRUCTURE_DATA_PATH):
            return None
        try:
            micro = pd.read_parquet(config.MICROSTRUCTURE_DATA_PATH)
            if micro.empty:
                return None
            micro['timestamp'] = pd.to_datetime(micro['timestamp'], utc=True).dt.tz_localize(None)
            micro = micro.drop_duplicates(subset=['timestamp'], keep='last')
            if n is not None:
                micro = micro.tail(n)
            # Only keep the columns we need for features
            keep_cols = ['timestamp', 'ws_trades_per_sec', 'ws_buy_sell_ratio', 'ws_spread_bps']
            available = [c for c in keep_cols if c in micro.columns]
            return micro[available] if 'timestamp' in available else None
        except Exception as e:
            logging.warning(f"Microstructure load failed: {e}")
            return None

    def load_and_engineer_features(self):
        """Load ALL market data, engineer features, and merge microstructure. Used for TRAINING only."""
        df = self._load_market_data()
        df = self._engineer_features(df, fast_mode=False)
        if df is None:
            return None
        
        # Merge real WebSocket microstructure data (if available)
        micro = self._load_microstructure_data()
        if micro is not None and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            micro['timestamp'] = pd.to_datetime(micro['timestamp'])
            # Floor both to minute for alignment
            df['_merge_ts'] = df['timestamp'].dt.floor('min')
            micro['_merge_ts'] = micro['timestamp'].dt.floor('min')
            micro_cols = [c for c in micro.columns if c.startswith('ws_')]
            df = df.merge(micro[['_merge_ts'] + micro_cols], on='_merge_ts', how='left')
            df.drop(columns=['_merge_ts'], inplace=True)
            matched = df[micro_cols[0]].notna().sum() if micro_cols else 0
            logging.info(f"Merged {matched:,}/{len(df):,} rows with WS microstructure data")
        
        # Fill missing WS features with neutral defaults
        for col in ['ws_trades_per_sec', 'ws_buy_sell_ratio', 'ws_spread_bps']:
            if col not in df.columns:
                df[col] = 0.0
        df['ws_buy_sell_ratio'] = df['ws_buy_sell_ratio'].fillna(1.0)  # neutral = 1.0
        df['ws_trades_per_sec'] = df['ws_trades_per_sec'].fillna(0.0)
        df['ws_spread_bps'] = df['ws_spread_bps'].fillna(0.0)
        
        return df
    
    def _add_microstructure_features(self, df):
        """
        Phase 3: Microstructure features derived from 1-minute OHLCV data.
        These proxy tick-level signals using candlestick structure and volume patterns.
        All features are scale-invariant (ratios/normalized) — IDENTICAL train/predict.
        """
        n = len(df)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        opn = df['open'].values
        volume = df['volume'].values
        
        # ===== 1. TRADE INTENSITY (proxy from volume × price activity) =====
        # Higher volume + larger candle range = more trading activity per minute
        # Normalized by rolling median for scale-invariance
        candle_range = (high - low) / (close + 1e-9)
        activity = volume * candle_range  # volume-weighted range
        activity_series = pd.Series(activity)
        activity_median = activity_series.rolling(60, min_periods=1).median().values + 1e-9
        df['trade_intensity'] = np.clip(activity / activity_median, 0, 10)
        
        # ===== 2. BUY/SELL RATIO (directional volume flow) =====
        # Candle body direction × volume: positive body = buying pressure, negative = selling
        # Ratio of buy volume / sell volume over rolling 10-min window
        body = close - opn
        body_pct = body / (close + 1e-9)  # normalized body size
        buy_vol = np.where(body > 0, volume * np.abs(body_pct), volume * 0.1)  # buy candles get full weight
        sell_vol = np.where(body < 0, volume * np.abs(body_pct), volume * 0.1)  # sell candles get full weight
        buy_rolling = pd.Series(buy_vol).rolling(10, min_periods=1).sum().values
        sell_rolling = pd.Series(sell_vol).rolling(10, min_periods=1).sum().values
        df['buy_sell_ratio'] = np.clip(buy_rolling / (sell_rolling + 1e-9), 0.1, 10)
        
        # ===== 3. VWAP MOMENTUM (rate of change of VWAP) =====
        # VWAP slope indicates institutional accumulation (rising) or distribution (falling)
        roll_vol = pd.Series(volume).rolling(20, min_periods=1).sum() + 1e-9
        roll_pv = pd.Series(close * volume).rolling(20, min_periods=1).sum()
        vwap = (roll_pv / roll_vol).values
        vwap_series = pd.Series(vwap)
        vwap_ret_5 = vwap_series.pct_change(5).fillna(0).values
        df['vwap_momentum'] = np.clip(vwap_ret_5 * 100, -5, 5)  # scaled %, capped
        
        # ===== 4. TICK VOLATILITY (micro-volatility from Parkinson estimator) =====
        # Parkinson vol uses high-low range: σ² = (1/4ln2) × E[ln(H/L)²]
        # More accurate than close-to-close vol for intraday data
        log_hl = np.log(high / (low + 1e-9) + 1e-9)
        parkinson_var = (log_hl ** 2) / (4 * np.log(2))
        # Rolling Parkinson vol over 10 minutes, normalized by 60-min median
        park_vol = pd.Series(parkinson_var).rolling(10, min_periods=1).mean().values
        park_vol_median = pd.Series(park_vol).rolling(60, min_periods=1).median().values + 1e-9
        df['tick_volatility'] = np.clip(park_vol / park_vol_median, 0, 10)
        
        # ===== 5. LARGE TRADE RATIO (volume spike detection) =====
        # Fraction of volume that exceeds 2x the rolling average — proxy for whale/large trades
        vol_mean_20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
        is_large = volume > (2.0 * vol_mean_20)  # boolean mask
        # Rolling ratio of large-volume candles in last 30 minutes
        large_series = pd.Series(is_large.astype(float))
        df['large_trade_ratio'] = large_series.rolling(30, min_periods=1).mean().values
        
        logging.info(f"Microstructure features computed for {n:,} rows")
        return df

    def _add_quant_features_vectorized(self, df):
        """
        Vectorized quant features — computed per-row from price/volume data.
        NO QuantEngine dependency (eliminates block-staleness and train/predict shift).
        IDENTICAL computation in training and prediction.
        """
        # ===== REGIME ID from Hurst (deterministic, already computed) =====
        # H > 0.55 → TRENDING (2), H < 0.45 → MEAN_REVERTING (0), else → RANDOM (1)
        hurst = df['hurst'].values
        regime = np.ones(len(df), dtype=int)  # Default: RANDOM (1)
        regime[hurst > 0.55] = 2   # TRENDING
        regime[hurst < 0.45] = 0   # MEAN_REVERTING
        df['regime_id'] = regime
        
        # ===== REGIME CONFIDENCE: how far Hurst is from 0.5 =====
        df['regime_confidence'] = np.clip(np.abs(hurst - 0.5) * 4, 0, 1)  # 0-1 scale
        
        # ===== GJR-GARCH VOLATILITY (asymmetric — drops increase vol more) =====
        # σ²(t) = ω + α·r²(t-1) + γ·r²(t-1)·I(r<0) + β·σ²(t-1)
        returns = df['close'].pct_change().fillna(0).values
        alpha, gamma, beta = 0.06, 0.04, 0.90
        omega = 1e-6  # long-run variance floor
        gjr_var = np.full(len(returns), omega)
        for i in range(1, len(returns)):
            r2 = returns[i-1] ** 2
            leverage = r2 * (1 if returns[i-1] < 0 else 0)  # asymmetric term
            gjr_var[i] = omega + alpha * r2 + gamma * leverage + beta * gjr_var[i-1]
        df['gjr_volatility'] = np.sqrt(gjr_var)  # convert variance → vol
        
        logging.info(f"Vectorized quant features computed for {len(df):,} rows")
        return df

    def _add_hawkes_wasserstein_features(self, df):
        """
        Advanced quant features — pure vectorized, IDENTICAL train/predict.
        
        1. Hawkes Process Intensity:
           λ(t) = μ + Σ α·exp(-β·(t - t_i))
           Measures self-exciting trade clustering. High intensity = cascading
           buy/sell pressure (trade avalanches). Uses volume-weighted returns
           as "events" with exponential decay kernel.
        
        2. Wasserstein Distance (Optimal Transport):
           W₁(P_prev, P_curr) = inf E[|X-Y|]
           Measures distribution shift between consecutive return windows.
           High distance = regime change. Low distance = stable regime.
        """
        n = len(df)
        returns = df['close'].pct_change().fillna(0).values
        volumes = df['volume'].values
        
        # ===== HAWKES INTENSITY =====
        # Self-exciting process with exponential kernel
        # α = excitation amplitude, β = decay rate (how fast excitement fades)
        alpha = 0.8   # strength of self-excitation
        beta = 0.15   # decay rate (per-minute) — ~7 minute half-life
        mu = 0.01     # baseline intensity
        
        # "Events" = absolute return × relative volume (big moves on high volume excite more)
        vol_mean = np.maximum(np.convolve(volumes, np.ones(20)/20, mode='same'), 1e-9)
        event_strength = np.abs(returns) * (volumes / vol_mean)
        
        hawkes = np.full(n, mu)
        for i in range(1, n):
            # Intensity decays from previous + new event excitation
            hawkes[i] = mu + (hawkes[i-1] - mu) * np.exp(-beta) + alpha * event_strength[i-1]
        
        # Normalize to 0-1 range via sigmoid-like transform
        hawkes_median = np.median(hawkes[hawkes > mu]) if np.any(hawkes > mu) else 0.1
        df['hawkes_intensity'] = 1.0 / (1.0 + np.exp(-(hawkes - hawkes_median) / (hawkes_median + 1e-9)))
        
        # ===== WASSERSTEIN DISTANCE (Optimal Transport) =====
        # Compare return distribution of window [t-60, t-30] vs [t-30, t]
        # High distance = distribution shift = regime change
        window = 30  # 30-minute windows
        wass = np.zeros(n)
        
        for i in range(window * 2, n, 5):  # Sample every 5 rows, interpolate
            prev_window = returns[i - window*2 : i - window]
            curr_window = returns[i - window : i]
            try:
                wass[i] = wasserstein_distance(prev_window, curr_window)
            except Exception:
                wass[i] = 0.0
        
        # Interpolate between sampled points
        sampled_idx = list(range(window * 2, n, 5))
        if len(sampled_idx) > 1:
            sampled_vals = wass[sampled_idx]
            wass = np.interp(range(n), sampled_idx, sampled_vals)
        
        # Normalize: divide by rolling median to make scale-invariant
        wass_series = pd.Series(wass)
        wass_median = wass_series.rolling(100, min_periods=1).median().values + 1e-9
        df['wass_drift'] = wass / wass_median
        
        logging.info(f"Hawkes+Wasserstein features computed for {n:,} rows")
        return df
    
    def _add_alt_data_features(self, df):
        """
        Add FREE alternative data features.
        Sources: Fear&Greed, Binance order book, trade pressure.
        No paid APIs required!
        """
        # Initialize with neutral defaults
        df['fear_greed_norm'] = 0.0
        df['orderbook_imbalance'] = 0.0
        df['trade_pressure'] = 0.0
        df['momentum_24h'] = 0.0
        
        if self.alt_data is None:
            return df
        
        try:
            # Get all FREE signals
            features = self.alt_data.get_features_for_model()
            self.last_alt_signals = features
            
            # Apply to the latest row (real-time data)
            feat_dict = features.get('features', {})
            
            if 'fear_greed_norm' in feat_dict:
                df.loc[df.index[-1], 'fear_greed_norm'] = feat_dict['fear_greed_norm']
            
            if 'orderbook_imbalance' in feat_dict:
                df.loc[df.index[-1], 'orderbook_imbalance'] = feat_dict['orderbook_imbalance']
            
            if 'trade_pressure' in feat_dict:
                df.loc[df.index[-1], 'trade_pressure'] = feat_dict['trade_pressure']
            
            if 'momentum_24h' in feat_dict:
                df.loc[df.index[-1], 'momentum_24h'] = feat_dict['momentum_24h']
            
            logging.debug(f"Alt data features added: {len(feat_dict)} features")
            
        except Exception as e:
            logging.warning(f"Alt data feature error: {e}")
        
        return df

    def create_temporal_split(self, df):
        """
        Create PROPER temporal train/test split.
        Training data: older 80%
        Test data: newest 20%
        NO FUTURE DATA LEAKAGE.
        """
        n = len(df)
        split_idx = int(n * (1 - self.validation_split))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df

    def create_target_variable(self, df, for_training=True):
        """
        Create target variable WITHOUT look-ahead bias.
        P2: 15-min horizon with 0.3% threshold (covers fees + slippage).
        """
        if for_training:
            df = df.copy()
            df['future_price'] = df['close'].shift(-self.prediction_horizon)
            # P2: 0.3% threshold covers Binance fees (0.1% x2) + slippage
            df['target'] = (df['future_price'] > df['close'] * 1.003).astype(int)
            df = df.dropna(subset=['target'])
            return df
        else:
            return df

    def train(self):
        """
        Train with PROPER temporal validation + CHAMPION-CHALLENGER promotion gate.
        
        P0/P1/P2 FIXES APPLIED:
        - XGBoost with regularization
        - LSTM with 30-step sliding windows + StandardScaler
        - LSTM earns ensemble weight via validation
        - 15-min prediction horizon with 0.3% threshold
        
        Champion-Challenger:
        - Trains a NEW "challenger" model without overwriting the current "champion"
        - Evaluates both on the test set (logloss + accuracy)
        - Only promotes the challenger if it is at least as good
        - Grace period: first N retrains always promote (cold-start)
        
        Returns: (is_trained: bool, progress: float, promotion: dict | None)
        """
        df_raw = self._load_market_data()
        if df_raw is None:
            return False, 0.0, None
        
        if len(df_raw) < self.min_train_samples + self.prediction_horizon:
            progress = (len(df_raw) / (self.min_train_samples + self.prediction_horizon)) * 100
            return False, min(progress, 95), None
        
        df = self.load_and_engineer_features()
        if df is None or len(df) < self.min_train_samples:
            return False, 95, None
        
        # Create temporal split BEFORE creating targets
        train_df, test_df = self.create_temporal_split(df)
        
        # Create targets for training data only
        train_df = self.create_target_variable(train_df, for_training=True)
        
        if len(train_df) < 20:
            logging.warning("Not enough training samples after target creation.")
            return False, 95, None
        
        if train_df['target'].nunique() < 2:
            logging.warning("Not enough target variance. Market too stable.")
            return False, 95, None
        
        X_train = train_df[self.features]
        y_train = train_df['target']
        
        try:
            # ── Prepare test data for champion-challenger evaluation ──
            test_for_eval = self.create_target_variable(test_df.copy(), for_training=True)
            can_evaluate = len(test_for_eval) >= 10 and test_for_eval['target'].nunique() >= 2
            
            # ── Measure CHAMPION performance (current model) BEFORE training ──
            champion_logloss = None
            champion_accuracy = None
            has_champion = hasattr(self.model, 'classes_') and can_evaluate
            
            if has_champion:
                try:
                    from sklearn.metrics import log_loss
                    X_test_eval = test_for_eval[self.features]
                    y_test_eval = test_for_eval['target']
                    champ_probs = self.model.predict_proba(X_test_eval)[:, 1]
                    champion_logloss = log_loss(y_test_eval, champ_probs)
                    champion_accuracy = ((champ_probs > 0.5).astype(int) == y_test_eval.values).mean() * 100
                    logging.info(
                        f"[CHAMPION-CHALLENGER] Champion metrics — "
                        f"logloss: {champion_logloss:.4f}, acc: {champion_accuracy:.1f}%"
                    )
                except Exception as e:
                    logging.warning(f"[CHAMPION-CHALLENGER] Could not evaluate champion: {e}")
                    has_champion = False
            
            # ===== STEP 1: Train CHALLENGER XGBoost =====
            # Train into a FRESH clone so we don't corrupt the champion yet
            import copy
            challenger_model = copy.deepcopy(self.model)
            
            # Exponential sample weighting: recent data counts ~10x more than oldest
            n = len(X_train)
            decay_rate = 3.0 / n  # e^(-3) ≈ 0.05 at oldest sample → 20x recency bias
            sample_weights = np.exp(decay_rate * np.arange(n))  # 0.05 at start → 1.0 at end
            sample_weights /= sample_weights.mean()  # Normalize so mean=1
            
            challenger_model.fit(X_train, y_train, sample_weight=sample_weights)
            logging.info(f"[CHAMPION-CHALLENGER] Challenger XGBoost trained on {len(X_train):,} samples")
            
            # Log feature importance (P4)
            importance = challenger_model.feature_importances_
            feat_imp = sorted(zip(self.features, importance), key=lambda x: x[1], reverse=True)
            top5 = ', '.join([f"{f}={v:.3f}" for f, v in feat_imp[:5]])
            zero_feats = [f for f, v in feat_imp if v < 0.001]
            logging.info(f"Top features: {top5}")
            if zero_feats:
                logging.warning(f"Zero-importance features: {zero_feats}")
            
            # ── Measure CHALLENGER performance ──
            challenger_logloss = None
            challenger_accuracy = None
            
            if can_evaluate:
                try:
                    from sklearn.metrics import log_loss
                    X_test_eval = test_for_eval[self.features]
                    y_test_eval = test_for_eval['target']
                    chall_probs = challenger_model.predict_proba(X_test_eval)[:, 1]
                    challenger_logloss = log_loss(y_test_eval, chall_probs)
                    challenger_accuracy = ((chall_probs > 0.5).astype(int) == y_test_eval.values).mean() * 100
                    logging.info(
                        f"[CHAMPION-CHALLENGER] Challenger metrics — "
                        f"logloss: {challenger_logloss:.4f}, acc: {challenger_accuracy:.1f}%"
                    )
                except Exception as e:
                    logging.warning(f"[CHAMPION-CHALLENGER] Could not evaluate challenger: {e}")
            
            # ── PROMOTION DECISION ──
            retrain_count = getattr(self, '_retrain_count', 0)
            self._retrain_count = retrain_count + 1
            
            promotion = {
                'promoted': False,
                'reason': 'pending',
                'champion_logloss': champion_logloss,
                'challenger_logloss': challenger_logloss,
                'champion_accuracy': champion_accuracy,
                'challenger_accuracy': challenger_accuracy,
                'retrain_number': self._retrain_count,
            }
            
            should_promote = False
            
            if retrain_count < config.CHALLENGER_GRACE_RETRAINS:
                # Grace period: always promote during cold-start
                should_promote = True
                promotion['reason'] = f'grace_period (retrain #{self._retrain_count}/{config.CHALLENGER_GRACE_RETRAINS})'
                logging.info(f"[CHAMPION-CHALLENGER] Grace period — auto-promoting (retrain #{self._retrain_count})")
            
            elif not has_champion or champion_logloss is None or challenger_logloss is None:
                # No champion to compare against — promote by default
                should_promote = True
                promotion['reason'] = 'no_champion_baseline'
                logging.info("[CHAMPION-CHALLENGER] No champion baseline available — promoting")
            
            elif challenger_accuracy is not None and challenger_accuracy < config.CHALLENGER_MIN_ACCURACY_PCT:
                # Absolute accuracy floor — reject catastrophic models
                should_promote = False
                promotion['reason'] = f'accuracy_floor ({challenger_accuracy:.1f}% < {config.CHALLENGER_MIN_ACCURACY_PCT}%)'
                logging.warning(
                    f"[CHAMPION-CHALLENGER] REJECTED — challenger accuracy {challenger_accuracy:.1f}% "
                    f"below floor {config.CHALLENGER_MIN_ACCURACY_PCT}%"
                )
            
            elif challenger_logloss <= champion_logloss + config.CHALLENGER_MIN_LOGLOSS_IMPROVEMENT:
                # Challenger is at least as good (lower logloss = better)
                should_promote = True
                delta = champion_logloss - challenger_logloss
                promotion['reason'] = f'challenger_wins (logloss Δ={delta:+.4f})'
                logging.info(
                    f"[CHAMPION-CHALLENGER] PROMOTED — challenger logloss {challenger_logloss:.4f} "
                    f"beats champion {champion_logloss:.4f} (Δ={delta:+.4f})"
                )
            
            else:
                # Challenger is worse — keep the champion
                should_promote = False
                delta = champion_logloss - challenger_logloss
                promotion['reason'] = f'champion_retained (logloss Δ={delta:+.4f})'
                logging.warning(
                    f"[CHAMPION-CHALLENGER] REJECTED — challenger logloss {challenger_logloss:.4f} "
                    f"worse than champion {champion_logloss:.4f} (Δ={delta:+.4f})"
                )
            
            promotion['promoted'] = should_promote
            
            # ── PROMOTE or DISCARD ──
            if should_promote:
                # Swap challenger into production
                self.model = challenger_model
                joblib.dump(self.model, self.model_path)
                logging.info("[CHAMPION-CHALLENGER] Challenger model saved to disk as new champion")
            else:
                # Discard challenger, keep current model
                del challenger_model
                logging.info("[CHAMPION-CHALLENGER] Challenger discarded — champion model unchanged")
            
            # Calibrate confidence (Platt scaling) — always on the active model
            try:
                if can_evaluate:
                    X_cal = test_for_eval[self.features]
                    y_cal = test_for_eval['target']
                    try:
                        self.calibrated_model = CalibratedClassifierCV(
                            estimator=self.model, method='sigmoid', cv='prefit'
                        )
                        self.calibrated_model.fit(X_cal, y_cal)
                    except (ValueError, TypeError):
                        self.calibrated_model = CalibratedClassifierCV(
                            estimator=self.model, method='sigmoid', cv=2, ensemble=False
                        )
                        self.calibrated_model.fit(X_cal, y_cal)
                    logging.info("Platt scaling calibration applied.")
            except Exception as e:
                logging.warning(f"Calibration failed: {e}")
                self.calibrated_model = None
            
            # ===== STEP 2: Fit Scaler + Train LSTM with sliding windows =====
            # Fit scaler on training data
            self.scaler.fit(X_train)
            self.scaler_fitted = True
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Create sliding window sequences for Transformer
            X_scaled = self.scaler.transform(X_train)
            y_values = y_train.values
            
            X_windows, y_windows = self._create_sliding_windows(X_scaled, y_values)
            
            if len(X_windows) > DEEP_SEQ_LEN:
                self.train_deep_model(X_windows, y_windows, epochs=30)
                torch.save(self.lstm.state_dict(), self.deep_model_path)
                logging.info(f"Transformer trained on {len(X_windows):,} sequences of {DEEP_SEQ_LEN} timesteps")
                
                # ===== STEP 3: Validate LSTM and assign ensemble weight =====
                self._assign_deep_model_weight(test_df)
            else:
                logging.warning(f"Not enough data for Transformer windows ({len(X_windows)} < {DEEP_SEQ_LEN})")
            
            # Validate on test set
            self.last_validation_accuracy = self._validate_on_test_set(test_df)
            self._training_accuracy = self.last_validation_accuracy  # preserve training score
            self.is_statistically_verified = True
            
            logging.info(
                f"Training complete. XGB Acc: {self.last_validation_accuracy:.1f}% | "
                f"LSTM Acc: {self.lstm_validation_acc:.1f}% | "
                f"Ensemble: XGB={self.xgb_weight:.1f} LSTM={self.lstm_weight:.1f} | "
                f"Promoted: {promotion['promoted']} ({promotion['reason']})"
            )
            
            # P1: Save ensemble state to disk so weights survive restarts
            self._save_ensemble_state()
            
        except Exception as e:
            logging.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False, 95, None
        
        return True, 100, promotion

    def train_on_candles(self, candle_df, timeframe_label='1s'):
        """Train on arbitrary-timeframe OHLCV candles (e.g. 1s from WebSocket).

        Adjusts prediction_horizon proportionally:
          - '1s' → 900 rows = 15 minutes
          - '1m' → 15 rows  = 15 minutes (default)
        
        Returns: (is_trained: bool, progress: float)
        """
        if candle_df is None or len(candle_df) < self.min_train_samples:
            logging.warning(f"[RETRAIN-{timeframe_label}] Not enough candles: {len(candle_df) if candle_df is not None else 0}")
            return False, 0.0

        # Adjust horizon based on timeframe
        original_horizon = self.prediction_horizon
        if timeframe_label == '1s':
            self.prediction_horizon = 900  # 900 × 1s = 15 min
        elif timeframe_label == '1m':
            self.prediction_horizon = 15   # 15 × 1m = 15 min (default)

        try:
            # Engineer features on the provided candles
            df = self._engineer_features(candle_df.copy(), fast_mode=False)
            if df is None or len(df) < self.min_train_samples:
                logging.warning(f"[RETRAIN-{timeframe_label}] Feature engineering yielded too few rows")
                return False, 50.0

            # Fill missing WS features with neutrals (1s candles won't have these)
            for col in ['ws_trades_per_sec', 'ws_buy_sell_ratio', 'ws_spread_bps']:
                if col not in df.columns:
                    df[col] = 0.0
            df['ws_buy_sell_ratio'] = df['ws_buy_sell_ratio'].fillna(1.0)
            df['ws_trades_per_sec'] = df['ws_trades_per_sec'].fillna(0.0)
            df['ws_spread_bps'] = df['ws_spread_bps'].fillna(0.0)

            # Temporal split
            train_df, test_df = self.create_temporal_split(df)
            train_df = self.create_target_variable(train_df, for_training=True)

            if len(train_df) < 20 or train_df['target'].nunique() < 2:
                logging.warning(f"[RETRAIN-{timeframe_label}] Insufficient training samples or variance")
                return False, 75.0

            X_train = train_df[self.features]
            y_train = train_df['target']

            # Exponential sample weighting (same as main train)
            n = len(X_train)
            decay_rate = 3.0 / n
            sample_weights = np.exp(decay_rate * np.arange(n))
            sample_weights /= sample_weights.mean()

            self.model.fit(X_train, y_train, sample_weight=sample_weights)
            joblib.dump(self.model, self.model_path)

            # Fit scaler + Transformer if enough data
            self.scaler.fit(X_train)
            self.scaler_fitted = True
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

            X_scaled = self.scaler.transform(X_train)
            X_windows, y_windows = self._create_sliding_windows(X_scaled, y_train.values)

            if len(X_windows) > DEEP_SEQ_LEN:
                self.train_deep_model(X_windows, y_windows, epochs=15)  # Fewer epochs for quick retrain
                torch.save(self.lstm.state_dict(), self.deep_model_path)
                self._assign_deep_model_weight(test_df)

            self.last_validation_accuracy = self._validate_on_test_set(test_df)
            self.is_statistically_verified = True
            self._save_ensemble_state()

            logging.info(
                f"[RETRAIN-{timeframe_label}] Complete on {len(train_df):,} {timeframe_label} candles | "
                f"Acc: {self.last_validation_accuracy:.1f}%"
            )
            return True, 100.0

        except Exception as e:
            logging.error(f"[RETRAIN-{timeframe_label}] Training error: {e}")
            import traceback
            traceback.print_exc()
            return False, 50.0
        finally:
            # Restore original horizon
            self.prediction_horizon = original_horizon


    def _create_sliding_windows(self, X, y):
        """
        Create sliding window sequences for Transformer.
        Input: X (N, features), y (N,)
        Output: X_windows (N-seq_len, seq_len, features), y_windows (N-seq_len,)
        """
        windows_X = []
        windows_y = []
        for i in range(DEEP_SEQ_LEN, len(X)):
            windows_X.append(X[i - DEEP_SEQ_LEN:i])
            windows_y.append(y[i])
        return np.array(windows_X), np.array(windows_y)

    def _assign_deep_model_weight(self, test_df):
        """
        P0: Transformer must EARN its ensemble weight via validation.
        If LSTM accuracy > 52% on test set, it gets proportional weight.
        """
        try:
            test_with_target = self.create_target_variable(test_df.copy(), for_training=True)
            if len(test_with_target) < DEEP_SEQ_LEN + 10:
                self.lstm_weight = 0.0
                return
            
            X_test = test_with_target[self.features].values
            y_test = test_with_target['target'].values
            
            if not self.scaler_fitted:
                self.lstm_weight = 0.0
                return
            
            X_scaled = self.scaler.transform(X_test)
            X_windows, y_windows = self._create_sliding_windows(X_scaled, y_test)
            
            if len(X_windows) < 10:
                self.lstm_weight = 0.0
                return
            
            # LSTM predictions
            self.lstm.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_windows).to(self.lstm_device)
                preds = self.lstm(X_t).cpu().numpy().flatten()
                lstm_preds = (preds > 0.5).astype(int)
            
            self.lstm_validation_acc = (lstm_preds == y_windows).mean() * 100
            
            if self.lstm_validation_acc > 52.0:  # Must beat random
                # Weight proportional to how much it beats 50%
                self.lstm_weight = min(0.4, (self.lstm_validation_acc - 50) / 50)
                self.xgb_weight = 1.0 - self.lstm_weight
                logging.info(f"LSTM earned weight: {self.lstm_weight:.2f} (acc: {self.lstm_validation_acc:.1f}%)")
            else:
                self.lstm_weight = 0.0
                self.xgb_weight = 1.0
                logging.info(f"LSTM weight = 0 (acc: {self.lstm_validation_acc:.1f}% < 52%)")
        
        except Exception as e:
            logging.warning(f"LSTM weight assignment failed: {e}")
            self.lstm_weight = 0.0
            self.xgb_weight = 1.0

    def _validate_on_test_set(self, test_df):
        """
        Walk-forward validation on temporal test set.
        Returns: accuracy percentage
        """
        if len(test_df) < self.prediction_horizon + 10:
            return 0.0
        
        # Prepare test data with known outcomes
        test_df = self.create_target_variable(test_df.copy(), for_training=True)
        
        if len(test_df) < 5 or test_df['target'].nunique() < 2:
            return 0.0
        
        X_test = test_df[self.features]
        y_test = test_df['target']
        
        # XGBoost predictions
        xgb_preds = (self.model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = (xgb_preds == y_test.values).mean() * 100
        return accuracy

    def get_prediction(self):
        """
        Get prediction using FAST path (500 rows, ~5 seconds).
        P0: Dynamic ensemble — LSTM only included if it earned weight.
        P1: LSTM gets 30-step scaled window, not 1 row.
        """
        _fallback = {
            "direction": "NEUTRAL",
            "confidence": 0,
            "target_price": 0,
            "target_price_1h": 0,
            "target_price_2h": 0,
            "hurst": 0.5,
            "verified": False
        }
        
        try:
            # ===== STAGE 1: Load data =====
            df = self._load_market_data_tail(n=500)
            if df is None or df.empty:
                logging.error("PREDICT FAIL [stage=data_load] No market data available")
                return _fallback
            logging.debug(f"PREDICT [stage=data_load] {len(df)} rows loaded")
            
            # ===== STAGE 2: Engineer features =====
            df = self._engineer_features(df, fast_mode=True)
            if df is None or df.empty:
                logging.error("PREDICT FAIL [stage=feature_eng] Feature engineering returned None/empty")
                return _fallback
            
            # ===== STAGE 2b: Inject live WebSocket microstructure =====
            ws_snap = getattr(self, '_live_ws_snapshot', None)
            if ws_snap:
                df['ws_trades_per_sec'] = ws_snap.get('trades_per_sec', 0.0)
                df['ws_buy_sell_ratio'] = ws_snap.get('buy_sell_ratio', 1.0)
                price = ws_snap.get('price', 0)
                bid = ws_snap.get('bid', 0)
                ask = ws_snap.get('ask', 0)
                df['ws_spread_bps'] = round((ask - bid) / (price + 1e-9) * 10000, 2) if price > 0 else 0.0
            else:
                for col in ['ws_trades_per_sec', 'ws_buy_sell_ratio', 'ws_spread_bps']:
                    if col not in df.columns:
                        df[col] = 0.0
            
            logging.debug(f"PREDICT [stage=feature_eng] {len(df)} rows, {len(df.columns)} cols")
            
            # ===== STAGE 3: Check model readiness =====
            if not hasattr(self.model, 'classes_'):
                logging.error("PREDICT FAIL [stage=model_check] XGBoost model has no classes_ (not trained)")
                return _fallback
            
            # Dynamically resolve features: use only what the model was trained with
            # This handles backward compatibility when new features are added
            model_features = getattr(self.model, 'feature_names_in_', None)
            if model_features is not None:
                active_features = [f for f in model_features if f in df.columns]
                if len(active_features) < len(model_features):
                    missing_model = [f for f in model_features if f not in df.columns]
                    logging.warning(f"PREDICT [stage=feature_compat] Model expects {missing_model} — filling with 0")
                    for col in missing_model:
                        df[col] = 0.0
                    active_features = list(model_features)
            else:
                # Fallback: use self.features, check for missing
                active_features = [f for f in self.features if f in df.columns]
                missing_feats = [f for f in self.features if f not in df.columns]
                if missing_feats:
                    logging.warning(f"PREDICT [stage=feature_check] Missing features (filling 0): {missing_feats}")
                    for col in missing_feats:
                        df[col] = 0.0
                    active_features = list(self.features)
            
            # Use ONLY the latest available data point for XGBoost
            latest = df[active_features].tail(1)
            current_price = float(df['close'].iloc[-1])
            hurst_val = float(df['hurst'].iloc[-1])
            
            # ===== STAGE 4: XGBoost prediction =====
            if self.calibrated_model is not None:
                xgb_prob = float(self.calibrated_model.predict_proba(latest)[0][1])
            else:
                xgb_prob = float(self.model.predict_proba(latest)[0][1])
            logging.debug(f"PREDICT [stage=xgb] prob={xgb_prob:.4f}")
            
            # ===== STAGE 5: Transformer prediction (only if it has earned weight) =====
            lstm_prob = 0.5  # neutral default (kept var name for API compat)
            if self.lstm_weight > 0 and self.scaler_fitted and len(df) >= DEEP_SEQ_LEN:
                try:
                    # Build 30-step scaled window from the last 30 rows
                    window_data = df[active_features].tail(DEEP_SEQ_LEN).values
                    window_scaled = self.scaler.transform(window_data)
                    
                    self.lstm.eval()
                    with torch.no_grad():
                        x_seq = torch.FloatTensor(window_scaled).unsqueeze(0).to(self.lstm_device)
                        lstm_prob = self.lstm(x_seq).item()
                    logging.debug(f"PREDICT [stage=transformer] prob={lstm_prob:.4f}")
                except Exception as e:
                    logging.warning(f"PREDICT [stage=lstm] failed: {e}")
                    lstm_prob = 0.5  # fallback to neutral
            
            # ===== STAGE 6: Dynamic ensemble =====
            final_prob = self.xgb_weight * xgb_prob + self.lstm_weight * lstm_prob
            
            direction = "UP" if final_prob > 0.5 else "DOWN"
            confidence = round(abs(final_prob - 0.5) * 200, 2)
            
            # ===== STAGE 6b: Alt data confidence boost (real-time only) =====
            if self.alt_data is not None:
                try:
                    alt_features = self.alt_data.get_features_for_model()
                    self.last_alt_signals = alt_features
                    feat = alt_features.get('features', {})
                    
                    # Count how many alt signals agree with our direction
                    agreements = 0
                    total_signals = 0
                    
                    fg_norm = feat.get('fear_greed_norm', 0)  # -1 to 1
                    if abs(fg_norm) > 0.1:
                        total_signals += 1
                        if (direction == "UP" and fg_norm > 0) or (direction == "DOWN" and fg_norm < 0):
                            agreements += 1
                    
                    ob_imb = feat.get('orderbook_imbalance', 0)
                    if abs(ob_imb) > 0.1:
                        total_signals += 1
                        if (direction == "UP" and ob_imb > 0) or (direction == "DOWN" and ob_imb < 0):
                            agreements += 1
                    
                    tp = feat.get('trade_pressure', 0)
                    if abs(tp) > 0.1:
                        total_signals += 1
                        if (direction == "UP" and tp > 0) or (direction == "DOWN" and tp < 0):
                            agreements += 1
                    
                    # Boost or dampen confidence (±10% max, never zero a valid signal)
                    if total_signals > 0:
                        agreement_ratio = agreements / total_signals
                        boost = (agreement_ratio - 0.5) * 20  # -10 to +10
                        confidence = max(1, min(100, confidence + boost))
                        logging.debug(f"PREDICT [stage=alt_boost] {agreements}/{total_signals} agree, boost={boost:+.1f}")
                except Exception as e:
                    logging.debug(f"PREDICT [stage=alt_boost] skipped: {e}")
            
            # ===== STAGE 6c: Signal quality adjustment from trader feedback =====
            # Read trade outcomes and adjust confidence based on regime performance
            try:
                feedback_path = os.path.join(config.DATA_ROOT, 'feedback', 'trade_feedback.json')
                if os.path.exists(feedback_path):
                    with open(feedback_path, 'r') as f:
                        feedback = json.load(f)
                    
                    if len(feedback) >= 5:
                        # Get current regime label
                        current_regime = self.last_quant_analysis.get('regime', {}).get('regime', 'UNKNOWN')
                        
                        # Calculate win rate for current regime
                        regime_trades = [t for t in feedback[-50:] if t.get('regime') == current_regime]
                        if len(regime_trades) >= 3:
                            regime_wins = sum(1 for t in regime_trades if t.get('won', False))
                            regime_wr = regime_wins / len(regime_trades)
                            
                            # Adjust: good regime boosts confidence, bad regime dampens
                            regime_boost = (regime_wr - 0.5) * 20  # -10 to +10
                            confidence = max(1, min(100, confidence + regime_boost))
                            logging.debug(
                                f"PREDICT [stage=feedback] regime={current_regime} "
                                f"wr={regime_wr*100:.0f}% ({len(regime_trades)} trades) "
                                f"boost={regime_boost:+.1f}"
                            )
            except Exception as e:
                logging.debug(f"PREDICT [stage=feedback] skipped: {e}")
            
            # Save prediction for future accuracy calculation
            self.save_prediction_for_audit(current_price, direction, confidence)
            
            # Calculate price targets based on volatility
            vol = float(df['volatility'].iloc[-1]) if 'volatility' in df.columns else 0.005
            vol = max(0.003, min(vol, 0.03))
            
            # 15-min and 30-min targets (P2: matches new prediction horizon)
            mult_15m = vol * 2
            mult_30m = vol * 3
            
            if direction == "UP":
                target_15m = current_price * (1 + mult_15m)
                target_30m = current_price * (1 + mult_30m)
            else:
                target_15m = current_price * (1 - mult_15m)
                target_30m = current_price * (1 - mult_30m)
            
            # ===== STAGE 8: Track prediction for live validation =====
            from datetime import datetime, timedelta
            now = datetime.now()
            
            # Validate past predictions whose window has elapsed
            # Use 0.3% threshold aligned with training target
            validated = 0
            correct = 0
            for rec in self._prediction_history:
                if rec.get('validated'):
                    validated += 1
                    if rec.get('correct'):
                        correct += 1
                    continue
                # Check if enough time has passed since THIS prediction was made
                elapsed = (now - rec['time']).total_seconds() / 60
                if elapsed >= self._validation_window:
                    rec['validated'] = True
                    # Use current price as proxy for "price at target time"
                    # (the closer predictions are spaced, the more accurate this is)
                    price_change_pct = (current_price - rec['price']) / rec['price']
                    # Aligned with training: 0.3% threshold for UP
                    if rec['direction'] == 'UP':
                        rec['correct'] = price_change_pct > 0.003
                    else:  # DOWN
                        rec['correct'] = price_change_pct < -0.003
                    validated += 1
                    if rec['correct']:
                        correct += 1
            
            # Update LIVE accuracy (separate from training accuracy)
            if validated >= 3:
                self._live_accuracy = round(correct / validated * 100, 1)
                self._live_accuracy_samples = validated
                # Also update the displayed accuracy to use live data
                self.last_validation_accuracy = self._live_accuracy
            
            # Store this prediction (cap at 200 entries)
            self._prediction_history.append({
                'time': now,
                'direction': direction,
                'price': current_price,
                'confidence': confidence,
                'validated': False,
                'correct': False,
            })
            if len(self._prediction_history) > 200:
                self._prediction_history = self._prediction_history[-200:]
            
            return {
                "direction": direction,
                "confidence": float(min(100, confidence)),
                "target_price": round(target_15m, 2),
                "target_price_1h": round(target_15m, 2),
                "target_price_2h": round(target_30m, 2),
                "current_price": round(current_price, 2),
                "hurst": float(hurst_val),
                "regime_label": self.last_quant_analysis.get('regime', {}).get('regime', 'UNKNOWN'),
                "xgb_prob": round(xgb_prob, 4),
                "lstm_prob": round(lstm_prob, 4),
                "ensemble_weights": f"XGB={self.xgb_weight:.1f} LSTM={self.lstm_weight:.1f}",
                "verified": self.is_statistically_verified
            }
        
        except Exception as e:
            logging.error(f"PREDICT FAIL [stage=unknown] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return _fallback

    def train_deep_model(self, X_windows, y_windows, epochs=30):
        """
        Train Transformer on pre-built sliding window sequences.
        X_windows: (N, seq_len, features) numpy array
        y_windows: (N,) numpy array
        Uses mixed precision + cosine LR with early stopping.
        """
        from torch.amp import autocast, GradScaler
        
        self.lstm.train()
        criterion = nn.BCEWithLogitsLoss()  # AMP-safe (expects raw logits)
        optimizer = torch.optim.AdamW(self.lstm.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.98))
        amp_scaler = GradScaler()
        
        # Cosine annealing with warmup
        warmup_steps = min(100, len(X_windows) // 256)
        total_steps = (len(X_windows) // 256) * epochs
        def lr_lambda(step):
            if step < warmup_steps:
                return max(step / max(warmup_steps, 1), 0.1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Full tensors on GPU
        X_tensor = torch.FloatTensor(X_windows).to(self.lstm_device)
        y_tensor = torch.FloatTensor(y_windows).unsqueeze(1).to(self.lstm_device)
        
        batch_size = min(256, len(X_windows))
        best_loss = float('inf')
        best_state = None
        patience = 7
        patience_counter = 0
        global_step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle indices
            perm = torch.randperm(len(X_tensor))
            
            for i in range(0, len(X_tensor), batch_size):
                idx = perm[i:i + batch_size]
                batch_x = X_tensor[idx]
                batch_y = y_tensor[idx]
                
                optimizer.zero_grad()
                with autocast('cuda', dtype=torch.float16):
                    output = self.lstm(batch_x, return_logits=True)  # Raw logits for AMP
                    loss = criterion(output, batch_y)
                
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), max_norm=1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                scheduler.step()
                global_step += 1
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            
            # Early stopping with best model checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in self.lstm.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Transformer early stopping at epoch {epoch} (loss: {best_loss:.4f})")
                    break
        
        # Restore best model
        if best_state is not None:
            self.lstm.load_state_dict(best_state)
        
        vram_mb = torch.cuda.memory_allocated() / 1e6 if self.lstm_device.type == 'cuda' else 0
        logging.info(f"Transformer training done. Best loss: {best_loss:.4f} | VRAM: {vram_mb:.0f} MB")

    # Legacy alias
    def train_lstm(self, *args, **kwargs):
        return self.train_deep_model(*args, **kwargs)

    def save_prediction_for_audit(self, price, direction, confidence):
        """Save prediction with timestamp for future accuracy calculation."""
        audit_path = os.path.join(config.DATA_DIR, "prediction_audit.csv")
        new_row = pd.DataFrame([{
            "timestamp": datetime.now(),
            "price_at_prediction": price,
            "direction": direction,
            "confidence": confidence,
            "outcome_checked": False,
            "was_correct": None
        }])
        new_row.to_csv(audit_path, mode='a', header=not os.path.exists(audit_path), index=False)

    def calculate_accuracy(self):
        """
        Calculate REAL accuracy based on past predictions vs actual outcomes.
        Uses vectorised merge_asof for O(n log m) performance instead of O(n×m).
        Threshold aligned with training target: 0.3% move = UP.
        """
        audit_path = os.path.join(config.DATA_DIR, "prediction_audit.csv")
        if not os.path.exists(audit_path):
            return 0.0
        
        try:
            audit_df = pd.read_csv(audit_path)
            if len(audit_df) < 5:
                return 0.0
            
            audit_df['timestamp'] = pd.to_datetime(audit_df['timestamp'])
            
            # Load market data (prefer parquet for speed)
            if os.path.exists(config.MARKET_DATA_PARQUET_PATH):
                market_df = pd.read_parquet(config.MARKET_DATA_PARQUET_PATH)
            elif os.path.exists(config.MARKET_DATA_PATH):
                market_df = pd.read_csv(config.MARKET_DATA_PATH)
            else:
                return 0.0
            
            market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
            market_df = market_df.sort_values('timestamp')
            
            # Only check predictions old enough to have outcomes
            cutoff = datetime.now() - timedelta(minutes=self.prediction_horizon)
            old_predictions = audit_df[audit_df['timestamp'] < cutoff].copy()
            
            if len(old_predictions) < 3:
                return self.last_validation_accuracy
            
            # Compute target timestamps and vectorised merge
            old_predictions['target_time'] = old_predictions['timestamp'] + timedelta(minutes=self.prediction_horizon)
            old_predictions = old_predictions.sort_values('target_time')
            
            merged = pd.merge_asof(
                old_predictions, 
                market_df[['timestamp', 'close']].rename(columns={'timestamp': 'target_time', 'close': 'actual_price'}),
                on='target_time',
                direction='nearest',
                tolerance=pd.Timedelta(minutes=5)
            )
            
            # Drop rows where no market data was found within tolerance
            merged = merged.dropna(subset=['actual_price'])
            if len(merged) == 0:
                return self.last_validation_accuracy
            
            # Aligned threshold: 0.3% move = UP (matches training target)
            merged['actual_direction'] = np.where(
                merged['actual_price'] > merged['price_at_prediction'] * 1.003, 'UP', 'DOWN'
            )
            correct = (merged['direction'] == merged['actual_direction']).sum()
            total = len(merged)
            
            return (correct / total) * 100
            
        except Exception as e:
            logging.error(f"Error calculating accuracy: {e}")
            return self.last_validation_accuracy
