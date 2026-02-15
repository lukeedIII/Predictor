"""
derivatives_feed.py — Binance USDⓈ-M Futures Derivatives Data Pipeline
========================================================================
Collects funding rate, open interest, basis, and mark/index price data
from Binance Futures API endpoints. Stores to parquet with staleness tracking.

Endpoints used:
  A. GET /fapi/v1/premiumIndex     — mark price, index price, funding rate, next funding time
  B. GET /fapi/v1/fundingRate      — funding rate history
  C. GET /fapi/v1/openInterest     — current open interest
  D. GET /futures/data/openInterestHist — OI history (~30 days)
  E. GET /futures/data/basis       — spot-futures basis

Usage:
  feed = DerivativesFeed()
  await feed.collect_snapshot()        # Single collection cycle
  feed.get_features()                  # Returns dict of engineered features
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import config

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════

DERIVS_DIR = os.path.join(config.DATA_DIR, "derivatives")
os.makedirs(DERIVS_DIR, exist_ok=True)

FUNDING_HISTORY_PATH = os.path.join(DERIVS_DIR, "funding_history.parquet")
OI_HISTORY_PATH = os.path.join(DERIVS_DIR, "oi_history.parquet")
BASIS_HISTORY_PATH = os.path.join(DERIVS_DIR, "basis_history.parquet")
PREMIUM_INDEX_PATH = os.path.join(DERIVS_DIR, "premium_index.parquet")
SNAPSHOT_PATH = os.path.join(DERIVS_DIR, "latest_snapshot.parquet")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

FAPI_BASE = "https://fapi.binance.com"
DATA_BASE = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"  # Futures symbol (no slash)
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_HISTORY_ROWS = 50_000  # Cap stored history


class DerivativesFeed:
    """
    Collects and engineers derivatives features from Binance Futures API.
    
    Data cadence:
      - Premium index / funding: every 60s (real-time)
      - OI current: every 60s
      - OI history: backfilled on startup, then appended
      - Basis: every 60s from /futures/data/basis
      - Funding history: backfilled on startup, then appended every 8h
    """

    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "NexusSQ/7.0",
            "Accept": "application/json",
        })

        # Latest snapshot (updated each collect cycle)
        self._latest: Dict[str, Any] = {}
        self._last_collect_time: float = 0
        self._initialized = False

        log.info(f"[DERIVS] DerivativesFeed initialized for {symbol}")

    # ─── HTTP helpers ─────────────────────────────────────────────────────

    def _get(self, url: str, params: dict = None) -> Optional[Any]:
        """GET with retry logic. Returns parsed JSON or None on failure."""
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    log.warning(f"[DERIVS] GET {url} failed after {MAX_RETRIES} retries: {e}")
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # ENDPOINT A: Premium Index (mark price, index price, funding, next funding)
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_premium_index(self) -> Optional[Dict]:
        """
        GET /fapi/v1/premiumIndex
        Returns: markPrice, indexPrice, lastFundingRate, nextFundingTime, etc.
        """
        data = self._get(f"{FAPI_BASE}/fapi/v1/premiumIndex", {"symbol": self.symbol})
        if data is None:
            return None

        result = {
            "timestamp": pd.Timestamp.now(tz="UTC"),
            "mark_price": float(data.get("markPrice", 0)),
            "index_price": float(data.get("indexPrice", 0)),
            "last_funding_rate": float(data.get("lastFundingRate", 0)),
            "next_funding_time": int(data.get("nextFundingTime", 0)),
            "interest_rate": float(data.get("interestRate", 0)),
        }

        # Derived: mark-index spread in basis points
        if result["index_price"] > 0:
            result["mark_index_spread_bps"] = (
                (result["mark_price"] - result["index_price"])
                / result["index_price"]
                * 10000
            )
        else:
            result["mark_index_spread_bps"] = 0.0

        # Derived: time to next funding in minutes
        if result["next_funding_time"] > 0:
            now_ms = int(time.time() * 1000)
            result["time_to_funding_min"] = max(
                0, (result["next_funding_time"] - now_ms) / 60000
            )
        else:
            result["time_to_funding_min"] = 480  # default 8h

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # ENDPOINT B: Funding Rate History
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_funding_history(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        GET /fapi/v1/fundingRate
        Returns last N funding rate entries (8h intervals).
        """
        data = self._get(
            f"{FAPI_BASE}/fapi/v1/fundingRate",
            {"symbol": self.symbol, "limit": limit},
        )
        if data is None or len(data) == 0:
            return None

        df = pd.DataFrame(data)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        df = df.rename(columns={
            "fundingTime": "timestamp",
            "fundingRate": "funding_rate",
        })
        df = df[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)
        return df

    # ═══════════════════════════════════════════════════════════════════════
    # ENDPOINT C: Current Open Interest
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_open_interest(self) -> Optional[Dict]:
        """
        GET /fapi/v1/openInterest
        Returns current OI in contracts.
        """
        data = self._get(
            f"{FAPI_BASE}/fapi/v1/openInterest",
            {"symbol": self.symbol},
        )
        if data is None:
            return None

        return {
            "timestamp": pd.Timestamp.now(tz="UTC"),
            "open_interest": float(data.get("openInterest", 0)),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ENDPOINT D: Open Interest History (~30 days)
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_oi_history(self, period: str = "5m", limit: int = 500) -> Optional[pd.DataFrame]:
        """
        GET /futures/data/openInterestHist
        Returns OI history with sumOpenInterest and sumOpenInterestValue.
        period: '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
        """
        data = self._get(
            f"{DATA_BASE}/futures/data/openInterestHist",
            {"symbol": self.symbol, "period": period, "limit": limit},
        )
        if data is None or len(data) == 0:
            return None

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
        df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
        df = df.rename(columns={
            "sumOpenInterest": "oi_contracts",
            "sumOpenInterestValue": "oi_value_usd",
        })
        df = df[["timestamp", "oi_contracts", "oi_value_usd"]].sort_values("timestamp").reset_index(drop=True)
        return df

    # ═══════════════════════════════════════════════════════════════════════
    # ENDPOINT E: Basis (spot-futures spread)
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_basis(self, period: str = "5m", limit: int = 500) -> Optional[pd.DataFrame]:
        """
        GET /futures/data/basis
        Returns basis and basisRate between spot and futures.
        """
        data = self._get(
            f"{DATA_BASE}/futures/data/basis",
            {"pair": "BTCUSDT", "contractType": "PERPETUAL", "period": period, "limit": limit},
        )
        if data is None or len(data) == 0:
            return None

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["indexPrice"] = df["indexPrice"].astype(float)
        df["futuresPrice"] = df["futuresPrice"].astype(float)
        df["basis"] = df["basis"].astype(float)
        df["basisRate"] = df["basisRate"].astype(float)
        df = df.rename(columns={"basisRate": "basis_rate"})
        df = df[["timestamp", "indexPrice", "futuresPrice", "basis", "basis_rate"]]
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ═══════════════════════════════════════════════════════════════════════
    # LONG/SHORT RATIO (Top Trader Positions)
    # ═══════════════════════════════════════════════════════════════════════

    def fetch_long_short_ratio(self, period: str = "5m", limit: int = 500) -> Optional[pd.DataFrame]:
        """
        GET /futures/data/topLongShortPositionRatio
        Top trader long/short ratio.
        """
        data = self._get(
            f"{DATA_BASE}/futures/data/topLongShortPositionRatio",
            {"symbol": self.symbol, "period": period, "limit": limit},
        )
        if data is None or len(data) == 0:
            return None

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["longShortRatio"] = df["longShortRatio"].astype(float)
        df["longAccount"] = df["longAccount"].astype(float)
        df["shortAccount"] = df["shortAccount"].astype(float)
        df = df.rename(columns={
            "longShortRatio": "long_short_ratio",
            "longAccount": "long_pct",
            "shortAccount": "short_pct",
        })
        df = df[["timestamp", "long_short_ratio", "long_pct", "short_pct"]]
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ═══════════════════════════════════════════════════════════════════════
    # COLLECTION: Full snapshot + history backfill
    # ═══════════════════════════════════════════════════════════════════════

    def backfill_history(self):
        """
        One-time backfill of funding, OI, and basis history.
        Called on startup to seed the feature engine with enough data for z-scores.
        """
        log.info("[DERIVS] Backfilling derivatives history...")

        # Funding history (last 1000 entries ≈ 333 days at 8h intervals)
        funding = self.fetch_funding_history(limit=1000)
        if funding is not None and len(funding) > 0:
            self._save_append(funding, FUNDING_HISTORY_PATH)
            log.info(f"[DERIVS]   Funding history: {len(funding)} entries")

        # OI history (500 × 5m ≈ ~42 hours, but we want longer)
        # Fetch 1h intervals for more coverage (~20 days)
        oi = self.fetch_oi_history(period="1h", limit=500)
        if oi is not None and len(oi) > 0:
            self._save_append(oi, OI_HISTORY_PATH)
            log.info(f"[DERIVS]   OI history: {len(oi)} entries")

        # Basis history
        basis = self.fetch_basis(period="1h", limit=500)
        if basis is not None and len(basis) > 0:
            self._save_append(basis, BASIS_HISTORY_PATH)
            log.info(f"[DERIVS]   Basis history: {len(basis)} entries")

        self._initialized = True
        log.info("[DERIVS] Backfill complete")

    def collect_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Collect a single real-time snapshot from all endpoints.
        Called every 60s in the background loop.
        Returns dict of raw values + staleness metadata.
        """
        now = time.time()
        snapshot = {"collect_time": pd.Timestamp.now(tz="UTC")}

        # A. Premium Index
        premium = self.fetch_premium_index()
        if premium:
            snapshot.update(premium)
            # Append to premium index history
            row = pd.DataFrame([{
                "timestamp": premium["timestamp"],
                "mark_price": premium["mark_price"],
                "index_price": premium["index_price"],
                "last_funding_rate": premium["last_funding_rate"],
                "mark_index_spread_bps": premium["mark_index_spread_bps"],
                "time_to_funding_min": premium["time_to_funding_min"],
            }])
            self._save_append(row, PREMIUM_INDEX_PATH, max_rows=MAX_HISTORY_ROWS)

        # C. Current OI
        oi = self.fetch_open_interest()
        if oi:
            snapshot["open_interest"] = oi["open_interest"]

        # Staleness tracking
        snapshot["premium_stale"] = premium is None
        snapshot["oi_stale"] = oi is None
        snapshot["derivs_quality"] = sum([
            0.0 if premium is None else 0.4,
            0.0 if oi is None else 0.3,
            0.3 if self._initialized else 0.0,  # history backfilled?
        ])

        self._latest = snapshot
        self._last_collect_time = now
        return snapshot

    def collect_periodic_history(self):
        """
        Append recent OI/basis/long-short history (5m granularity).
        Called less frequently (every 5-10 minutes) to update rolling features.
        """
        # OI history (5m, last 30 entries = 2.5h)
        oi = self.fetch_oi_history(period="5m", limit=30)
        if oi is not None and len(oi) > 0:
            self._save_append(oi, OI_HISTORY_PATH)

        # Basis (5m, last 30 entries)
        basis = self.fetch_basis(period="5m", limit=30)
        if basis is not None and len(basis) > 0:
            self._save_append(basis, BASIS_HISTORY_PATH)

        # Long/short ratio (5m, last 30)
        # Note: stored inline with OI history for simplicity

    # ═══════════════════════════════════════════════════════════════════════
    # FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════════════

    def get_features(self) -> Dict[str, float]:
        """
        Engineer scale-invariant derivatives features for the predictor.
        Returns dict of feature_name → float value.
        All features follow the pattern: (level, delta, z-score).
        """
        features = {}

        # ── Funding Rate Features ──────────────────────────────────────
        funding_df = self._load_history(FUNDING_HISTORY_PATH)
        if funding_df is not None and len(funding_df) >= 3:
            fr = funding_df["funding_rate"]

            # Raw current
            features["funding_rate"] = float(fr.iloc[-1])

            # EMA (1-day = 3 funding periods, 7-day = 21 periods)
            features["funding_rate_ema_1d"] = float(fr.ewm(span=3).mean().iloc[-1])
            features["funding_rate_ema_7d"] = float(fr.ewm(span=21).mean().iloc[-1])

            # Z-score (7-day rolling)
            if len(fr) >= 21:
                mu = fr.rolling(21).mean().iloc[-1]
                sigma = fr.rolling(21).std().iloc[-1]
                features["funding_rate_z_7d"] = float(
                    (fr.iloc[-1] - mu) / max(sigma, 1e-10)
                )
            else:
                features["funding_rate_z_7d"] = 0.0

            # Regime bucket: -2 (very negative), -1, 0 (neutral), 1, 2 (very positive)
            q = fr.quantile([0.1, 0.3, 0.7, 0.9])
            current = fr.iloc[-1]
            if current <= q.iloc[0]:
                features["funding_regime"] = -2.0
            elif current <= q.iloc[1]:
                features["funding_regime"] = -1.0
            elif current >= q.iloc[3]:
                features["funding_regime"] = 2.0
            elif current >= q.iloc[2]:
                features["funding_regime"] = 1.0
            else:
                features["funding_regime"] = 0.0
        else:
            features.update({
                "funding_rate": 0.0,
                "funding_rate_ema_1d": 0.0,
                "funding_rate_ema_7d": 0.0,
                "funding_rate_z_7d": 0.0,
                "funding_regime": 0.0,
            })

        # ── Premium Index / Mark-Index Features ────────────────────────
        premium_df = self._load_history(PREMIUM_INDEX_PATH)
        if premium_df is not None and len(premium_df) >= 5:
            spread = premium_df["mark_index_spread_bps"]

            features["mark_index_spread_bps"] = float(spread.iloc[-1])

            # Z-scores (1-day: 1440 1-min snapshots, but we collect every ~60s)
            window_1d = min(len(spread), 1440)
            window_7d = min(len(spread), 10080)
            mu_1d = spread.tail(window_1d).mean()
            std_1d = spread.tail(window_1d).std()
            features["spread_z_1d"] = float(
                (spread.iloc[-1] - mu_1d) / max(std_1d, 1e-10)
            )
            if len(spread) >= 1440:
                mu_7d = spread.tail(window_7d).mean()
                std_7d = spread.tail(window_7d).std()
                features["spread_z_7d"] = float(
                    (spread.iloc[-1] - mu_7d) / max(std_7d, 1e-10)
                )
            else:
                features["spread_z_7d"] = features["spread_z_1d"]

            # Time to funding
            features["time_to_funding_min"] = float(
                premium_df["time_to_funding_min"].iloc[-1]
            )
        else:
            features.update({
                "mark_index_spread_bps": 0.0,
                "spread_z_1d": 0.0,
                "spread_z_7d": 0.0,
                "time_to_funding_min": 480.0,
            })

        # ── Open Interest Features ─────────────────────────────────────
        oi_df = self._load_history(OI_HISTORY_PATH)
        if oi_df is not None and len(oi_df) >= 5:
            # Use oi_value_usd if available, else oi_contracts
            oi_col = "oi_value_usd" if "oi_value_usd" in oi_df.columns else "oi_contracts"
            oi = oi_df[oi_col]

            features["oi_value"] = float(oi.iloc[-1])

            # Changes at multiple horizons
            for label, periods in [("5m", 1), ("15m", 3), ("1h", 12), ("4h", 48)]:
                if len(oi) > periods:
                    pct_chg = (oi.iloc[-1] - oi.iloc[-1 - periods]) / max(oi.iloc[-1 - periods], 1e-10)
                    features[f"oi_chg_{label}"] = float(pct_chg)
                else:
                    features[f"oi_chg_{label}"] = 0.0

            # Z-score (7-day)
            window = min(len(oi), 2016)  # 7d × 24h × 12 (5-min intervals)
            mu = oi.tail(window).mean()
            sigma = oi.tail(window).std()
            features["oi_z_7d"] = float((oi.iloc[-1] - mu) / max(sigma, 1e-10))
        else:
            features.update({
                "oi_value": 0.0,
                "oi_chg_5m": 0.0,
                "oi_chg_15m": 0.0,
                "oi_chg_1h": 0.0,
                "oi_chg_4h": 0.0,
                "oi_z_7d": 0.0,
            })

        # ── OI-Price Divergence ────────────────────────────────────────
        # Signal: OI rising but price flat/falling = potential trap/unwind
        if oi_df is not None and premium_df is not None and len(oi_df) > 12 and len(premium_df) > 12:
            oi_col = "oi_value_usd" if "oi_value_usd" in oi_df.columns else "oi_contracts"
            oi_chg = oi_df[oi_col].pct_change(12).iloc[-1]  # 1h change
            price_chg = premium_df["mark_price"].pct_change(60).iloc[-1] if len(premium_df) > 60 else 0
            features["oi_price_divergence"] = float(
                1.0 if (oi_chg > 0.01 and price_chg < -0.001) or
                       (oi_chg < -0.01 and price_chg > 0.001) else 0.0
            )
        else:
            features["oi_price_divergence"] = 0.0

        # ── Basis Features ─────────────────────────────────────────────
        basis_df = self._load_history(BASIS_HISTORY_PATH)
        if basis_df is not None and len(basis_df) >= 5:
            br = basis_df["basis_rate"]

            features["basis_rate"] = float(br.iloc[-1])

            # Change (1h)
            if len(br) > 12:
                features["basis_chg_1h"] = float(br.iloc[-1] - br.iloc[-13])
            else:
                features["basis_chg_1h"] = 0.0

            # Z-score (7-day)
            window = min(len(br), 2016)
            mu = br.tail(window).mean()
            sigma = br.tail(window).std()
            features["basis_z_7d"] = float((br.iloc[-1] - mu) / max(sigma, 1e-10))
        else:
            features.update({
                "basis_rate": 0.0,
                "basis_chg_1h": 0.0,
                "basis_z_7d": 0.0,
            })

        # ── Staleness & Quality Flags ──────────────────────────────────
        features["derivs_quality"] = self._latest.get("derivs_quality", 0.0)
        features["oi_is_stale"] = float(self._latest.get("oi_stale", True))
        features["premium_is_stale"] = float(self._latest.get("premium_stale", True))

        # Clip extreme z-scores to [-5, 5]
        for key in features:
            if key.endswith("_z_7d") or key.endswith("_z_1d"):
                features[key] = float(np.clip(features[key], -5, 5))

        return features

    def get_feature_names(self) -> list:
        """Return ordered list of all derivatives feature names."""
        return [
            # Funding (5)
            "funding_rate", "funding_rate_ema_1d", "funding_rate_ema_7d",
            "funding_rate_z_7d", "funding_regime",
            # Mark/Index spread (4)
            "mark_index_spread_bps", "spread_z_1d", "spread_z_7d",
            "time_to_funding_min",
            # Open Interest (7)
            "oi_value", "oi_chg_5m", "oi_chg_15m", "oi_chg_1h",
            "oi_chg_4h", "oi_z_7d", "oi_price_divergence",
            # Basis (3)
            "basis_rate", "basis_chg_1h", "basis_z_7d",
            # Quality (3)
            "derivs_quality", "oi_is_stale", "premium_is_stale",
        ]

    def get_snapshot_dict(self) -> Dict[str, Any]:
        """Return the latest raw snapshot for API/dashboard display."""
        return {
            "funding_rate": self._latest.get("last_funding_rate", 0),
            "mark_price": self._latest.get("mark_price", 0),
            "index_price": self._latest.get("index_price", 0),
            "open_interest": self._latest.get("open_interest", 0),
            "mark_index_spread_bps": self._latest.get("mark_index_spread_bps", 0),
            "time_to_funding_min": self._latest.get("time_to_funding_min", 0),
            "quality": self._latest.get("derivs_quality", 0),
            "last_update": str(self._latest.get("collect_time", "")),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # PERSISTENCE HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _save_append(
        self, df: pd.DataFrame, path: str, max_rows: int = MAX_HISTORY_ROWS
    ):
        """Append DataFrame to existing parquet, dedup by timestamp, cap rows."""
        try:
            if os.path.exists(path):
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, df]).drop_duplicates(
                    subset=["timestamp"], keep="last"
                )
                combined = combined.sort_values("timestamp").tail(max_rows)
                combined.to_parquet(path, index=False)
            else:
                df.sort_values("timestamp").to_parquet(path, index=False)
        except Exception as e:
            log.warning(f"[DERIVS] Save error ({path}): {e}")

    def _load_history(self, path: str) -> Optional[pd.DataFrame]:
        """Load a history parquet file, return None if missing/empty."""
        try:
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if len(df) > 0:
                    return df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            log.debug(f"[DERIVS] Load error ({path}): {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    feed = DerivativesFeed()

    print("\n=== Backfilling history ===")
    feed.backfill_history()

    print("\n=== Collecting snapshot ===")
    snap = feed.collect_snapshot()
    for k, v in (snap or {}).items():
        print(f"  {k}: {v}")

    print("\n=== Engineered Features ===")
    features = feed.get_features()
    for k, v in features.items():
        print(f"  {k:30s} = {v:+.6f}")

    print(f"\nTotal features: {len(features)}")
    print(f"Feature names: {feed.get_feature_names()}")
