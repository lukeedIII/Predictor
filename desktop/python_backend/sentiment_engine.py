from transformers import pipeline
import feedparser
import logging
import os
import config
import requests
from alt_data import FearGreedProvider
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SentimentEngine:
    def __init__(self):
        logging.info("Initializing Sentiment Engine with FinBERT...")
        model_name = "ProsusAI/finbert"
        cache_path = os.path.join(config.MODEL_DIR, "finbert")
        os.makedirs(cache_path, exist_ok=True)
        
        import torch
        device = -1
        if "cuda" in config.DEFAULT_DEVICE:
            try:
                torch.zeros(1).to("cuda")
                device = 0
            except:
                logging.warning("RTX device incompatible. Falling back to CPU for Sentiment.")
                device = -1

        try:
            self.analyzer = pipeline(
                "sentiment-analysis", 
                model=model_name, 
                device=device,
                model_kwargs={"cache_dir": cache_path}
            )
            logging.info(f"FinBERT loaded (Device: {device})")
        except Exception as e:
            logging.error(f"Failed to load FinBERT: {e}. Falling back to default.")
            self.analyzer = pipeline("sentiment-analysis", device=-1)
        
        # RSS Feed sources for crypto news
        self.rss_feeds = [
            "https://cointelegraph.com/rss",
            "https://cryptonews.com/news/feed/",
            "https://decrypt.co/feed"
        ]
        
        # Fear & Greed Index (delegated to alt_data.FearGreedProvider)
        self._fear_greed_provider = FearGreedProvider()
        
        # Twitter scraper (lazy load)
        self._twitter_scraper = None
        self._last_twitter_fetch = None
        self._twitter_cache = []

        
    @property
    def twitter_scraper(self):
        """Lazy load Twitter scraper to avoid import errors."""
        if self._twitter_scraper is None:
            try:
                from twitter_scraper import TwitterScraper
                self._twitter_scraper = TwitterScraper()
                logging.info("Twitter/X scraper initialized")
            except Exception as e:
                logging.warning(f"Twitter scraper not available: {e}")
        return self._twitter_scraper

    def fetch_crypto_news(self, max_items: int = 5) -> List[str]:
        """Fetch real crypto news from RSS feeds."""
        headlines = []
        
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:max_items]:
                    title = entry.get('title', '')
                    if title and ('bitcoin' in title.lower() or 'btc' in title.lower() or 'crypto' in title.lower()):
                        headlines.append(title)
            except Exception as e:
                logging.debug(f"RSS fetch error for {feed_url}: {e}")
                continue
        
        unique_headlines = list(set(headlines))[:max_items]
        
        if unique_headlines:
            logging.info(f"Fetched {len(unique_headlines)} crypto RSS headlines")
        
        return unique_headlines

    def fetch_twitter_sentiment(self, max_accounts: int = 20) -> List[str]:
        """
        Fetch weighted tweet texts from top crypto influencers.
        Uses Nitter scraping (no API required).
        """
        if self.twitter_scraper is None:
            return []
        
        try:
            tweets = self.twitter_scraper.get_weighted_sentiment_texts(max_accounts)
            logging.info(f"Fetched {len(tweets)} weighted Twitter texts for sentiment")
            return tweets
        except Exception as e:
            logging.error(f"Twitter sentiment fetch error: {e}")
            return []

    def check_elon_alert(self) -> dict:
        """
        Check for Elon Musk BTC tweets.
        Returns alert dict if found, None otherwise.
        """
        if self.twitter_scraper is None:
            return None
        
        try:
            return self.twitter_scraper.get_elon_alert()
        except:
            return None

    def fetch_fear_greed_index(self) -> dict:
        """
        Fetch Fear & Greed Index (delegates to alt_data.FearGreedProvider).
        Returns dict with value (0-100), classification, and normalized score.
        """
        result = self._fear_greed_provider.get_current()
        
        value = result.get('value', 50)
        classification = result.get('classification', result.get('signal', 'Neutral'))
        # Normalize to -1 to +1 scale (0=Extreme Fear=-1, 50=Neutral=0, 100=Extreme Greed=+1)
        normalized = (value - 50) / 50
        
        return {
            'value': value,
            'classification': classification,
            'normalized': normalized,
            'timestamp': result.get('timestamp')
        }

    def get_combined_sentiment(self, max_twitter_accounts: int = 15) -> float:
        """
        Get combined sentiment from RSS + Twitter + Fear & Greed Index.
        
        Weights:
        - Twitter: 40% (most real-time)
        - RSS: 25% (news)  
        - Fear & Greed: 35% (market-wide sentiment)
        
        Returns: float between -1 (very negative) and +1 (very positive)
        """
        # Fetch from all sources
        rss_texts = self.fetch_crypto_news(max_items=10)
        twitter_texts = self.fetch_twitter_sentiment(max_accounts=max_twitter_accounts)
        fear_greed = self.fetch_fear_greed_index()

        
        # Analyze separately
        rss_sentiment = self.analyze_text(rss_texts) if rss_texts else 0.0
        twitter_sentiment = self.analyze_text(twitter_texts) if twitter_texts else 0.0
        fg_sentiment = fear_greed.get('normalized', 0.0)
        
        # Calculate weighted combined sentiment
        # Weights: Twitter 40%, RSS 25%, Fear & Greed 35%
        sources_available = 0
        combined = 0.0
        
        if twitter_texts:
            combined += 0.40 * twitter_sentiment
            sources_available += 1
        if rss_texts:
            combined += 0.25 * rss_sentiment
            sources_available += 1
        
        # Fear & Greed is always available (has fallback)
        combined += 0.35 * fg_sentiment
        sources_available += 1
        
        # Normalize if not all sources available
        if sources_available < 3:
            # Redistribute missing weights proportionally
            actual_weight = 0.35  # Base F&G weight
            if twitter_texts:
                actual_weight += 0.40
            if rss_texts:
                actual_weight += 0.25
            if actual_weight > 0:
                combined = combined / actual_weight
        
        logging.info(f"Combined Sentiment: RSS={rss_sentiment:+.2f}, Twitter={twitter_sentiment:+.2f}, F&G={fg_sentiment:+.2f} ({fear_greed.get('classification', 'N/A')}), Final={combined:+.2f}")
        
        return max(-1.0, min(1.0, combined))


    def analyze_text(self, texts: List[str]) -> float:
        """
        Analyzes a list of texts and returns sentiment score.
        
        Returns: float between -1 (very negative) and +1 (very positive)
                 0.0 = neutral or no data
        """
        if not texts:
            return 0.0
            
        try:
            # Limit batch size to avoid memory issues
            texts = texts[:50]
            
            results = self.analyzer(texts, truncation=True)
            scores = []
            
            for res in results:
                label = res['label'].lower()
                confidence = res['score']
                
                if label == 'positive':
                    scores.append(confidence)
                elif label == 'negative':
                    scores.append(-confidence)
                else:  # neutral
                    scores.append(0.0)
            
            if not scores:
                return 0.0
            
            avg_score = sum(scores) / len(scores)
            
            return max(-1.0, min(1.0, avg_score))
            
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")
            return 0.0


if __name__ == "__main__":
    engine = SentimentEngine()
    
    # Test combined sentiment
    print("=== Testing Combined Sentiment ===")
    combined = engine.get_combined_sentiment(max_twitter_accounts=5)
    print(f"Combined Sentiment: {combined:+.2f}")
    
    # Test Elon alert
    elon = engine.check_elon_alert()
    if elon:
        print(f"\n🚨 ELON ALERT: {elon}")
