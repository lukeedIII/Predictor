"""
FREE Alternative Data Sources for BTC Prediction
=================================================
100% free APIs only - no subscriptions required.

Sources:
- Fear & Greed Index (Alternative.me)
- Bitcoin network stats (blockchain.info)
- Google Trends (pytrends)
- Binance public data (already have)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FreeDataProvider:
    """Base class for free data sources."""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def get_cached(self, key: str) -> Optional[dict]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return data
        return None
    
    def set_cache(self, key: str, data: dict):
        self.cache[key] = (data, datetime.now())


class FearGreedProvider(FreeDataProvider):
    """
    Crypto Fear & Greed Index
    https://alternative.me/crypto/fear-and-greed-index/
    
    100% FREE, no API key needed.
    """
    
    def get_current(self) -> Dict:
        """Get current Fear & Greed value."""
        cache_key = "fear_greed"
        cached = self.get_cached(cache_key)
        if cached:
            return cached
        
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()['data'][0]
                
                value = int(data['value'])
                classification = data['value_classification']
                
                # Contrarian signal interpretation
                if value <= 20:
                    signal = 'EXTREME_FEAR'
                    bias = 1.0  # Strong buy signal
                elif value <= 35:
                    signal = 'FEAR'
                    bias = 0.5
                elif value >= 80:
                    signal = 'EXTREME_GREED'
                    bias = -1.0  # Strong sell signal
                elif value >= 65:
                    signal = 'GREED'
                    bias = -0.5
                else:
                    signal = 'NEUTRAL'
                    bias = 0.0
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'value': value,
                    'classification': classification,
                    'signal': signal,
                    'bias': bias,  # -1 to 1, positive = bullish
                    'available': True
                }
                
                self.set_cache(cache_key, result)
                logger.info(f"Fear & Greed: {value} ({classification})")
                return result
                
        except Exception as e:
            logger.warning(f"Fear & Greed API failed: {e}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'value': 50,
            'signal': 'UNAVAILABLE',
            'bias': 0.0,
            'available': False
        }
    
    def get_historical(self, days: int = 30) -> pd.DataFrame:
        """Get historical Fear & Greed data."""
        try:
            url = f"https://api.alternative.me/fng/?limit={days}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()['data']
                
                records = []
                for item in data:
                    records.append({
                        'timestamp': datetime.fromtimestamp(int(item['timestamp'])),
                        'fear_greed': int(item['value']),
                        'classification': item['value_classification']
                    })
                
                return pd.DataFrame(records).sort_values('timestamp')
                
        except Exception as e:
            logger.warning(f"Historical Fear & Greed failed: {e}")
        
        return pd.DataFrame()


class BlockchainInfoProvider(FreeDataProvider):
    """
    Bitcoin blockchain data from blockchain.info
    
    100% FREE, no API key needed.
    Rate limit: ~100 requests per 5 minutes
    """
    
    BASE_URL = "https://blockchain.info"
    
    def get_mempool_stats(self) -> Dict:
        """Get mempool statistics."""
        cache_key = "mempool"
        cached = self.get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Get unconfirmed transaction count
            unconfirmed_url = f"{self.BASE_URL}/q/unconfirmedcount"
            response = requests.get(unconfirmed_url, timeout=10)
            unconfirmed = int(response.text) if response.status_code == 200 else 0
            
            # Interpret congestion
            if unconfirmed > 100000:
                congestion = 'EXTREME'
                congestion_score = 1.0
            elif unconfirmed > 50000:
                congestion = 'HIGH'
                congestion_score = 0.7
            elif unconfirmed > 20000:
                congestion = 'MEDIUM'
                congestion_score = 0.4
            else:
                congestion = 'LOW'
                congestion_score = 0.1
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'unconfirmed_txs': unconfirmed,
                'congestion': congestion,
                'congestion_score': congestion_score,
                'available': True
            }
            
            self.set_cache(cache_key, result)
            logger.info(f"Mempool: {unconfirmed} txs ({congestion})")
            return result
            
        except Exception as e:
            logger.warning(f"Mempool stats failed: {e}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'unconfirmed_txs': 0,
            'congestion': 'UNKNOWN',
            'congestion_score': 0.0,
            'available': False
        }
    
    def get_difficulty(self) -> Dict:
        """Get current mining difficulty."""
        cache_key = "difficulty"
        cached = self.get_cached(cache_key)
        if cached:
            return cached
        
        try:
            url = f"{self.BASE_URL}/q/getdifficulty"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                difficulty = float(response.text)
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'difficulty': difficulty,
                    'difficulty_t': difficulty / 1e12,  # In trillions
                    'available': True
                }
                
                self.set_cache(cache_key, result)
                return result
                
        except Exception as e:
            logger.warning(f"Difficulty fetch failed: {e}")
        
        return {'available': False}
    
    def get_24h_stats(self) -> Dict:
        """Get 24-hour blockchain statistics."""
        cache_key = "stats_24h"
        cached = self.get_cached(cache_key)
        if cached:
            return cached
        
        try:
            url = f"{self.BASE_URL}/stats?format=json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'hash_rate': data.get('hash_rate', 0),
                    'n_tx': data.get('n_tx', 0),
                    'n_blocks_mined': data.get('n_blocks_mined', 0),
                    'total_btc_sent': data.get('total_btc_sent', 0) / 1e8,  # Satoshi to BTC
                    'market_price_usd': data.get('market_price_usd', 0),
                    'available': True
                }
                
                self.set_cache(cache_key, result)
                return result
                
        except Exception as e:
            logger.warning(f"24h stats failed: {e}")
        
        return {'available': False}


class GoogleTrendsProvider(FreeDataProvider):
    """
    Google Trends for "Bitcoin" search interest.
    
    100% FREE but rate-limited.
    High search = retail FOMO = often a top signal
    """
    
    def __init__(self):
        super().__init__()
        self.pytrends = None
        self._init_pytrends()
    
    def _init_pytrends(self):
        try:
            from pytrends.request import TrendReq
            self.pytrends = TrendReq(hl='en-US', tz=360)
        except ImportError:
            logger.warning("pytrends not installed. Run: pip install pytrends")
    
    def get_bitcoin_interest(self, timeframe: str = 'now 7-d') -> Dict:
        """
        Get Bitcoin search interest.
        
        Timeframes:
        - 'now 1-H': Past hour
        - 'now 4-H': Past 4 hours
        - 'now 1-d': Past day
        - 'now 7-d': Past week
        """
        cache_key = f"trends_{timeframe}"
        cached = self.get_cached(cache_key)
        if cached:
            return cached
        
        if not self.pytrends:
            return {
                'available': False,
                'error': 'pytrends not installed'
            }
        
        try:
            self.pytrends.build_payload(['Bitcoin'], timeframe=timeframe)
            data = self.pytrends.interest_over_time()
            
            if not data.empty:
                current_value = int(data['Bitcoin'].iloc[-1])
                avg_value = float(data['Bitcoin'].mean())
                max_value = int(data['Bitcoin'].max())
                
                # Relative interest (vs average)
                if avg_value > 0:
                    relative_interest = current_value / avg_value
                else:
                    relative_interest = 1.0
                
                # Signal interpretation
                if relative_interest > 1.5:
                    signal = 'HIGH_INTEREST'
                    fomo_score = 0.8
                elif relative_interest > 1.2:
                    signal = 'RISING_INTEREST'
                    fomo_score = 0.4
                elif relative_interest < 0.7:
                    signal = 'LOW_INTEREST'
                    fomo_score = -0.4
                else:
                    signal = 'NORMAL'
                    fomo_score = 0.0
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'current_interest': current_value,
                    'avg_interest': avg_value,
                    'max_interest': max_value,
                    'relative_interest': relative_interest,
                    'signal': signal,
                    'fomo_score': fomo_score,  # High = retail FOMO (contrarian sell)
                    'available': True
                }
                
                self.set_cache(cache_key, result)
                logger.info(f"Google Trends: {current_value} ({signal})")
                return result
                
        except Exception as e:
            logger.warning(f"Google Trends failed: {e}")
        
        return {
            'available': False,
            'fomo_score': 0.0
        }


class BinancePublicProvider(FreeDataProvider):
    """
    Binance public data (already used, but adding more).
    
    100% FREE, no API key for public endpoints.
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def get_24h_ticker(self, symbol: str = "BTCUSDT") -> Dict:
        """Get 24h ticker statistics."""
        cache_key = f"ticker_{symbol}"
        cached = self.get_cached(cache_key)
        if cached:
            return cached
        
        try:
            url = f"{self.BASE_URL}/ticker/24hr?symbol={symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                price_change_pct = float(data.get('priceChangePercent', 0))
                volume = float(data.get('volume', 0))
                quote_volume = float(data.get('quoteVolume', 0))
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'price': float(data.get('lastPrice', 0)),
                    'price_change_24h': price_change_pct,
                    'high_24h': float(data.get('highPrice', 0)),
                    'low_24h': float(data.get('lowPrice', 0)),
                    'volume_btc': volume,
                    'volume_usdt': quote_volume,
                    'trades_24h': int(data.get('count', 0)),
                    'available': True
                }
                
                self.set_cache(cache_key, result)
                return result
                
        except Exception as e:
            logger.warning(f"Binance ticker failed: {e}")
        
        return {'available': False}
    
    def get_order_book_imbalance(self, symbol: str = "BTCUSDT", depth: int = 20) -> Dict:
        """
        Calculate order book imbalance.
        Positive = more bids (bullish)
        Negative = more asks (bearish)
        """
        cache_key = f"orderbook_{symbol}"
        cached = self.get_cached(cache_key)
        if cached:
            return cached
        
        try:
            url = f"{self.BASE_URL}/depth?symbol={symbol}&limit={depth}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Sum bid and ask volumes
                bid_volume = sum(float(b[1]) for b in data.get('bids', []))
                ask_volume = sum(float(a[1]) for a in data.get('asks', []))
                
                total = bid_volume + ask_volume
                if total > 0:
                    imbalance = (bid_volume - ask_volume) / total  # -1 to 1
                else:
                    imbalance = 0.0
                
                # Signal interpretation
                if imbalance > 0.2:
                    signal = 'BID_HEAVY'
                elif imbalance < -0.2:
                    signal = 'ASK_HEAVY'
                else:
                    signal = 'BALANCED'
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'imbalance': imbalance,
                    'imbalance_signal': signal,
                    'available': True
                }
                
                self.set_cache(cache_key, result)
                return result
                
        except Exception as e:
            logger.warning(f"Order book failed: {e}")
        
        return {'available': False, 'imbalance': 0.0}
    
    def get_recent_trades_summary(self, symbol: str = "BTCUSDT") -> Dict:
        """Analyze recent trades for buy/sell pressure."""
        cache_key = f"trades_{symbol}"
        cached = self.get_cached(cache_key)
        if cached:
            return cached
        
        try:
            url = f"{self.BASE_URL}/trades?symbol={symbol}&limit=500"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                trades = response.json()
                
                buy_volume = 0
                sell_volume = 0
                
                for trade in trades:
                    qty = float(trade['qty'])
                    if trade['isBuyerMaker']:
                        sell_volume += qty  # Taker sold
                    else:
                        buy_volume += qty  # Taker bought
                
                total = buy_volume + sell_volume
                if total > 0:
                    buy_ratio = buy_volume / total
                else:
                    buy_ratio = 0.5
                
                # Signal interpretation
                if buy_ratio > 0.6:
                    signal = 'STRONG_BUYING'
                    pressure = 0.7
                elif buy_ratio > 0.55:
                    signal = 'BUYING'
                    pressure = 0.3
                elif buy_ratio < 0.4:
                    signal = 'STRONG_SELLING'
                    pressure = -0.7
                elif buy_ratio < 0.45:
                    signal = 'SELLING'
                    pressure = -0.3
                else:
                    signal = 'BALANCED'
                    pressure = 0.0
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'buy_ratio': buy_ratio,
                    'signal': signal,
                    'pressure': pressure,  # -1 to 1
                    'available': True
                }
                
                self.set_cache(cache_key, result)
                return result
                
        except Exception as e:
            logger.warning(f"Recent trades failed: {e}")
        
        return {'available': False, 'pressure': 0.0}


class FreeDataAggregator:
    """
    Aggregates all FREE data sources for prediction.
    No paid APIs required!
    """
    
    def __init__(self):
        self.fear_greed = FearGreedProvider()
        self.blockchain = BlockchainInfoProvider()
        self.trends = GoogleTrendsProvider()
        self.binance = BinancePublicProvider()
    
    def get_all_signals(self) -> Dict:
        """Get all available free signals."""
        return {
            'fear_greed': self.fear_greed.get_current(),
            'mempool': self.blockchain.get_mempool_stats(),
            'blockchain_24h': self.blockchain.get_24h_stats(),
            'google_trends': self.trends.get_bitcoin_interest(),
            'binance_ticker': self.binance.get_24h_ticker(),
            'order_book': self.binance.get_order_book_imbalance(),
            'trade_pressure': self.binance.get_recent_trades_summary(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_composite_signal(self) -> Dict:
        """
        Calculate a composite bullish/bearish signal from all sources.
        Returns score from -1 (bearish) to +1 (bullish).
        """
        signals = self.get_all_signals()
        
        scores = []
        weights = []
        
        # Fear & Greed (contrarian) - weight: 0.25
        fg = signals['fear_greed']
        if fg.get('available'):
            scores.append(fg.get('bias', 0))
            weights.append(0.25)
        
        # Google Trends (contrarian) - weight: 0.15
        trends = signals['google_trends']
        if trends.get('available'):
            # Negative because high FOMO = sell signal
            scores.append(-trends.get('fomo_score', 0))
            weights.append(0.15)
        
        # Order book imbalance - weight: 0.20
        ob = signals['order_book']
        if ob.get('available'):
            scores.append(ob.get('imbalance', 0))
            weights.append(0.20)
        
        # Trade pressure - weight: 0.25
        tp = signals['trade_pressure']
        if tp.get('available'):
            scores.append(tp.get('pressure', 0))
            weights.append(0.25)
        
        # Price momentum (24h change) - weight: 0.15
        ticker = signals['binance_ticker']
        if ticker.get('available'):
            change = ticker.get('price_change_24h', 0)
            # Normalize: 10% = max signal
            momentum = np.clip(change / 10, -1, 1)
            scores.append(momentum)
            weights.append(0.15)
        
        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            composite = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            composite = 0.0
        
        # Interpret composite
        if composite > 0.3:
            signal = 'BULLISH'
        elif composite > 0.1:
            signal = 'SLIGHTLY_BULLISH'
        elif composite < -0.3:
            signal = 'BEARISH'
        elif composite < -0.1:
            signal = 'SLIGHTLY_BEARISH'
        else:
            signal = 'NEUTRAL'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'composite_score': round(composite, 4),
            'signal': signal,
            'confidence': abs(composite),
            'sources_used': len(scores),
            'raw_signals': signals
        }
    
    def get_features_for_model(self) -> Dict:
        """
        Get normalized features for ML model integration.
        All features are -1 to 1 scale.
        """
        signals = self.get_all_signals()
        
        features = {}
        
        # Fear & Greed normalized
        fg = signals['fear_greed']
        if fg.get('available'):
            features['fear_greed_norm'] = (fg['value'] - 50) / 50  # 0-100 -> -1 to 1
        
        # Mempool congestion
        mempool = signals['mempool']
        if mempool.get('available'):
            features['mempool_congestion'] = mempool.get('congestion_score', 0)
        
        # Order book imbalance
        ob = signals['order_book']
        if ob.get('available'):
            features['orderbook_imbalance'] = ob.get('imbalance', 0)
        
        # Trade pressure
        tp = signals['trade_pressure']
        if tp.get('available'):
            features['trade_pressure'] = tp.get('pressure', 0)
        
        # Google Trends
        trends = signals['google_trends']
        if trends.get('available'):
            features['google_trends_fomo'] = trends.get('fomo_score', 0)
        
        # 24h momentum
        ticker = signals['binance_ticker']
        if ticker.get('available'):
            change = ticker.get('price_change_24h', 0)
            features['momentum_24h'] = np.clip(change / 10, -1, 1)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'n_features': len(features)
        }


# Test
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FREE ALTERNATIVE DATA SOURCES - TEST")
    print("  100% free, no API keys required!")
    print("=" * 60 + "\n")
    
    agg = FreeDataAggregator()
    
    # Test composite signal
    print("📊 COMPOSITE SIGNAL:")
    print("-" * 40)
    composite = agg.get_composite_signal()
    print(f"  Score: {composite['composite_score']:.4f}")
    print(f"  Signal: {composite['signal']}")
    print(f"  Confidence: {composite['confidence']:.2%}")
    print(f"  Sources: {composite['sources_used']}")
    
    # Test individual sources
    print("\n📈 INDIVIDUAL SIGNALS:")
    print("-" * 40)
    
    raw = composite['raw_signals']
    
    # Fear & Greed
    fg = raw['fear_greed']
    if fg.get('available'):
        print(f"  Fear & Greed: {fg['value']} ({fg['classification']})")
    
    # Mempool
    mem = raw['mempool']
    if mem.get('available'):
        print(f"  Mempool: {mem['unconfirmed_txs']:,} txs ({mem['congestion']})")
    
    # Order Book
    ob = raw['order_book']
    if ob.get('available'):
        print(f"  Order Book: {ob['imbalance']:.2%} imbalance ({ob['imbalance_signal']})")
    
    # Trade Pressure
    tp = raw['trade_pressure']
    if tp.get('available'):
        print(f"  Trade Pressure: {tp['buy_ratio']:.1%} buys ({tp['signal']})")
    
    # Ticker
    ticker = raw['binance_ticker']
    if ticker.get('available'):
        print(f"  24h Change: {ticker['price_change_24h']:.2f}%")
    
    # Model features
    print("\n🤖 FEATURES FOR MODEL:")
    print("-" * 40)
    features = agg.get_features_for_model()
    for name, value in features['features'].items():
        print(f"  {name}: {value:.4f}")
