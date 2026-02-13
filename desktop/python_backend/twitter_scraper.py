"""
Crypto News Scraper for Nexus Prediction Engine
Uses CryptoPanic (free, no API key for basic access) instead of dead Nitter instances.
Also supports direct RSS feeds as fallback.
"""

import requests
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CryptoNewsScraper:
    """
    Replaces the dead Nitter-based TwitterScraper.
    CryptoPanic provides pre-categorized crypto news with sentiment labels.
    Free tier: no API key needed for basic public access.
    """
    
    def __init__(self):
        self.base_url = "https://cryptopanic.com/api/free/v1/posts/"
        self.cache = []
        self.cache_time = None
        self.cache_ttl = 300  # Cache for 5 minutes
        
        # Influential accounts to watch (used for filtering, not scraping)
        self.key_influencers = [
            "elonmusk", "saborbtc", "caborbtc", "VitalikButerin",
            "CZ_Binance", "APompliano", "woonomic"
        ]
    
    def fetch_news(self, currency: str = "BTC", limit: int = 20) -> List[Dict]:
        """
        Fetch latest crypto news from CryptoPanic.
        Returns list of {title, source, sentiment, published_at, url}
        """
        # Return cache if fresh
        if self.cache and self.cache_time and (datetime.now() - self.cache_time).seconds < self.cache_ttl:
            return self.cache
        
        try:
            params = {
                "currencies": currency,
                "kind": "news",
                "public": "true"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code != 200:
                logging.warning(f"CryptoPanic API returned {response.status_code}")
                return self._fallback_rss()
            
            data = response.json()
            results = data.get("results", [])
            
            news_items = []
            for item in results[:limit]:
                votes = item.get("votes", {})
                # CryptoPanic provides community votes as sentiment signals
                positive = votes.get("positive", 0) + votes.get("liked", 0)
                negative = votes.get("negative", 0) + votes.get("disliked", 0)
                total_votes = positive + negative
                
                if total_votes > 0:
                    sentiment = "BULLISH" if positive > negative else "BEARISH"
                    sentiment_score = (positive - negative) / total_votes
                else:
                    sentiment = "NEUTRAL"
                    sentiment_score = 0.0
                
                news_items.append({
                    "title": item.get("title", ""),
                    "source": item.get("source", {}).get("title", "Unknown"),
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score,
                    "published_at": item.get("published_at", ""),
                    "url": item.get("url", ""),
                    "domain": item.get("domain", "")
                })
            
            self.cache = news_items
            self.cache_time = datetime.now()
            
            logging.info(f"CryptoPanic: fetched {len(news_items)} news items for {currency}")
            return news_items
            
        except Exception as e:
            logging.error(f"CryptoPanic fetch error: {e}")
            return self._fallback_rss()
    
    def _fallback_rss(self) -> List[Dict]:
        """Fallback to basic RSS if CryptoPanic is down."""
        try:
            import feedparser
            rss_urls = [
                "https://cointelegraph.com/rss",
                "https://www.coindesk.com/arc/outboundfeeds/rss/",
            ]
            items = []
            for url in rss_urls:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    title = entry.get("title", "")
                    sentiment, score = self._analyze_headline(title)
                    items.append({
                        "title": title,
                        "source": feed.feed.get("title", "RSS"),
                        "sentiment": sentiment,
                        "sentiment_score": score,
                        "published_at": entry.get("published", ""),
                        "url": entry.get("link", ""),
                        "domain": "rss"
                    })
            return items
        except Exception as e:
            logging.error(f"RSS fallback error: {e}")
            return []
    
    @staticmethod
    def _analyze_headline(title: str) -> tuple:
        """Keyword-based sentiment analysis for news headlines."""
        t = title.lower()
        bullish = ['surge', 'soar', 'rally', 'breakout', 'bullish', 'record high',
                    'all-time high', 'ath', 'moon', 'adoption', 'approval', 'buy',
                    'accumulate', 'inflow', 'upgrade', 'etf approved', 'partnership',
                    'institutional', 'support', 'recover', 'bounce', 'gain', 'rising']
        bearish = ['crash', 'plunge', 'dump', 'bearish', 'sell-off', 'selloff',
                    'hack', 'exploit', 'fraud', 'ban', 'crackdown', 'regulation',
                    'lawsuit', 'sec', 'fine', 'collapse', 'bankruptcy', 'liquidat',
                    'outflow', 'decline', 'drop', 'fall', 'warn', 'risk', 'fear']
        
        bull_hits = sum(1 for kw in bullish if kw in t)
        bear_hits = sum(1 for kw in bearish if kw in t)
        
        if bull_hits > bear_hits:
            return "BULLISH", min(0.3 + bull_hits * 0.15, 0.9)
        elif bear_hits > bull_hits:
            return "BEARISH", max(-0.3 - bear_hits * 0.15, -0.9)
        return "NEUTRAL", 0.0
    
    def get_aggregate_sentiment(self, currency: str = "BTC") -> float:
        """
        Get aggregate sentiment score from recent news.
        Returns: float in [-1.0, +1.0] (bearish to bullish)
        """
        news = self.fetch_news(currency)
        if not news:
            return 0.0
        
        scores = [item["sentiment_score"] for item in news if item["sentiment_score"] != 0]
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def check_important_news(self, currency: str = "BTC") -> Optional[Dict]:
        """
        Check for high-impact news (many votes = high impact).
        Returns the most impactful recent item or None.
        """
        news = self.fetch_news(currency)
        if not news:
            return None
        
        # Find the item with strongest sentiment (most voted)
        strongest = max(news, key=lambda x: abs(x["sentiment_score"]), default=None)
        if strongest and abs(strongest["sentiment_score"]) > 0.5:
            return strongest
        
        return None


# Backwards compatibility: alias for code that imports TwitterScraper
TwitterScraper = CryptoNewsScraper


if __name__ == "__main__":
    scraper = CryptoNewsScraper()
    
    print("=== CryptoPanic News Feed ===\n")
    news = scraper.fetch_news("BTC")
    
    for item in news[:10]:
        emoji = "🟢" if item["sentiment"] == "BULLISH" else ("🔴" if item["sentiment"] == "BEARISH" else "⚪")
        print(f"{emoji} [{item['source']}] {item['title'][:80]}")
        print(f"   Sentiment: {item['sentiment_score']:+.2f}")
        print()
    
    agg = scraper.get_aggregate_sentiment()
    print(f"\nAggregate BTC Sentiment: {agg:+.2f}")
