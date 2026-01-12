"""
FinBERT Sentiment Analysis for Forex Trading
=============================================
Source: ProsusAI/finBERT, ahmedrachid/FinancialBERT-Sentiment-Analysis

Pre-trained BERT fine-tuned on financial text.
Classifies text as: Bullish, Neutral, Bearish

Usage for HFT:
1. Monitor real-time news feeds
2. Run FinBERT on headlines 50ms before price impact
3. Pre-position before market reaction
4. Mean reversion after overreaction

Studies show: 24.06 bps average return per news trade (18-year OOS)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

# Try importing transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("Transformers not available, using keyword-based fallback")


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    text: str
    sentiment: str  # 'bullish', 'neutral', 'bearish'
    score: float  # Confidence (0-1)
    bullish_prob: float
    neutral_prob: float
    bearish_prob: float


class FinBERTSentiment:
    """
    FinBERT-based sentiment analysis for financial text.

    Models available:
    - ProsusAI/finbert (original)
    - ahmedrachid/FinancialBERT-Sentiment-Analysis
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

        if HAS_TRANSFORMERS:
            self._load_model()

    def _load_model(self):
        """Load FinBERT model."""
        try:
            logger.info(f"Loading FinBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self.is_loaded = True
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}")
            self.is_loaded = False

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Financial text (headline, news, tweet)

        Returns:
            SentimentResult with probabilities
        """
        if not self.is_loaded:
            return self._keyword_fallback(text)

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0].numpy()

            # FinBERT labels: [negative, neutral, positive]
            bearish_prob = probs[0]
            neutral_prob = probs[1]
            bullish_prob = probs[2]

            # Determine sentiment
            max_idx = np.argmax(probs)
            sentiments = ['bearish', 'neutral', 'bullish']
            sentiment = sentiments[max_idx]
            score = probs[max_idx]

            return SentimentResult(
                text=text,
                sentiment=sentiment,
                score=score,
                bullish_prob=bullish_prob,
                neutral_prob=neutral_prob,
                bearish_prob=bearish_prob
            )

        except Exception as e:
            logger.warning(f"FinBERT analysis failed: {e}")
            return self._keyword_fallback(text)

    def _keyword_fallback(self, text: str) -> SentimentResult:
        """Keyword-based sentiment when FinBERT unavailable."""
        text_lower = text.lower()

        # Bullish keywords
        bullish_words = ['surge', 'rally', 'gain', 'rise', 'jump', 'soar', 'bullish',
                        'strong', 'beat', 'exceed', 'upgrade', 'buy', 'positive',
                        'growth', 'expand', 'optimistic', 'confident']

        # Bearish keywords
        bearish_words = ['fall', 'drop', 'decline', 'plunge', 'crash', 'bearish',
                        'weak', 'miss', 'cut', 'downgrade', 'sell', 'negative',
                        'contraction', 'pessimistic', 'concern', 'fear', 'risk']

        bullish_count = sum(1 for w in bullish_words if w in text_lower)
        bearish_count = sum(1 for w in bearish_words if w in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return SentimentResult(text, 'neutral', 0.6, 0.2, 0.6, 0.2)

        bullish_prob = bullish_count / (total + 1)
        bearish_prob = bearish_count / (total + 1)
        neutral_prob = 1 - bullish_prob - bearish_prob

        if bullish_prob > bearish_prob and bullish_prob > neutral_prob:
            sentiment = 'bullish'
            score = bullish_prob
        elif bearish_prob > bullish_prob and bearish_prob > neutral_prob:
            sentiment = 'bearish'
            score = bearish_prob
        else:
            sentiment = 'neutral'
            score = neutral_prob

        return SentimentResult(text, sentiment, score, bullish_prob, neutral_prob, bearish_prob)

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts efficiently."""
        return [self.analyze(text) for text in texts]


class ForexNewsSentiment:
    """
    Forex-specific news sentiment analyzer.

    Currency-specific keyword mappings:
    - "Fed hawkish" -> USD bullish
    - "ECB dovish" -> EUR bearish
    - "Risk-on" -> AUD, NZD bullish; JPY, CHF bearish
    """

    CURRENCY_KEYWORDS = {
        'USD': {
            'bullish': ['fed hawkish', 'rate hike', 'strong jobs', 'inflation hot',
                       'dollar strength', 'safe haven', 'treasury yields rise'],
            'bearish': ['fed dovish', 'rate cut', 'weak jobs', 'inflation cool',
                       'dollar weakness', 'risk on', 'treasury yields fall']
        },
        'EUR': {
            'bullish': ['ecb hawkish', 'eurozone growth', 'eu recovery',
                       'euro strength', 'german data strong'],
            'bearish': ['ecb dovish', 'eurozone recession', 'eu crisis',
                       'euro weakness', 'german data weak']
        },
        'GBP': {
            'bullish': ['boe hawkish', 'uk growth', 'brexit progress',
                       'pound strength', 'uk inflation hot'],
            'bearish': ['boe dovish', 'uk recession', 'brexit risk',
                       'pound weakness', 'uk inflation cool']
        },
        'JPY': {
            'bullish': ['boj hawkish', 'yen strength', 'japan intervention',
                       'risk off', 'safe haven'],
            'bearish': ['boj dovish', 'yen weakness', 'yield curve control',
                       'risk on', 'carry trade']
        }
    }

    def __init__(self):
        self.finbert = FinBERTSentiment()

    def analyze_for_pair(self, text: str, pair: str) -> Dict[str, float]:
        """
        Analyze news sentiment for specific currency pair.

        Args:
            text: News text
            pair: Currency pair (e.g., 'EURUSD')

        Returns:
            Dict with signal strength (-1 to 1) and confidence
        """
        # Get base currencies
        base = pair[:3]  # EUR in EURUSD
        quote = pair[3:]  # USD in EURUSD

        # General sentiment
        general = self.finbert.analyze(text)

        # Currency-specific adjustments
        base_sentiment = self._currency_sentiment(text, base)
        quote_sentiment = self._currency_sentiment(text, quote)

        # Net signal: base bullish or quote bearish -> buy pair
        net_signal = base_sentiment - quote_sentiment

        # Adjust with general sentiment
        if 'bullish' in general.sentiment:
            net_signal += 0.2 * general.score
        elif 'bearish' in general.sentiment:
            net_signal -= 0.2 * general.score

        # Normalize to -1 to 1
        signal = np.clip(net_signal, -1, 1)

        # Confidence based on keyword hits
        confidence = min(1.0, (abs(base_sentiment) + abs(quote_sentiment)) / 2 + general.score * 0.3)

        return {
            'signal': signal,
            'confidence': confidence,
            'base_sentiment': base_sentiment,
            'quote_sentiment': quote_sentiment,
            'general_sentiment': general.sentiment,
            'general_score': general.score
        }

    def _currency_sentiment(self, text: str, currency: str) -> float:
        """Get sentiment score for specific currency."""
        if currency not in self.CURRENCY_KEYWORDS:
            return 0.0

        text_lower = text.lower()
        keywords = self.CURRENCY_KEYWORDS[currency]

        bullish_hits = sum(1 for kw in keywords['bullish'] if kw in text_lower)
        bearish_hits = sum(1 for kw in keywords['bearish'] if kw in text_lower)

        total = bullish_hits + bearish_hits
        if total == 0:
            return 0.0

        return (bullish_hits - bearish_hits) / total


class NewsTradingSignal:
    """
    Generate trading signals from news events.

    Strategy:
    1. Monitor news feeds (Forex Factory, Bloomberg, Reuters)
    2. Analyze sentiment in real-time
    3. Generate entry signal if |sentiment| > threshold
    4. Exit on mean reversion or time-based stop
    """

    def __init__(self, sentiment_threshold: float = 0.6,
                 hold_period_seconds: int = 300):
        self.sentiment_threshold = sentiment_threshold
        self.hold_period = timedelta(seconds=hold_period_seconds)
        self.sentiment_analyzer = ForexNewsSentiment()

        # Active positions from news
        self.active_signals = {}

    def process_news(self, timestamp: datetime, headline: str,
                    pair: str) -> Optional[Dict]:
        """
        Process incoming news and generate signal.

        Args:
            timestamp: News timestamp
            headline: News headline
            pair: Currency pair to trade

        Returns:
            Trading signal dict or None
        """
        analysis = self.sentiment_analyzer.analyze_for_pair(headline, pair)

        if abs(analysis['signal']) < self.sentiment_threshold:
            return None

        signal = {
            'timestamp': timestamp,
            'pair': pair,
            'headline': headline,
            'direction': 1 if analysis['signal'] > 0 else -1,
            'strength': abs(analysis['signal']),
            'confidence': analysis['confidence'],
            'exit_time': timestamp + self.hold_period
        }

        self.active_signals[f"{pair}_{timestamp}"] = signal
        return signal

    def get_active_signals(self, current_time: datetime) -> List[Dict]:
        """Get all active news signals (not yet expired)."""
        active = []
        expired = []

        for key, signal in self.active_signals.items():
            if current_time < signal['exit_time']:
                active.append(signal)
            else:
                expired.append(key)

        # Clean up expired
        for key in expired:
            del self.active_signals[key]

        return active


def compute_sentiment_features(headlines: List[Tuple[datetime, str]],
                              pair: str = 'EURUSD') -> pd.DataFrame:
    """
    Compute sentiment features from news headlines.

    Args:
        headlines: List of (timestamp, headline) tuples
        pair: Currency pair

    Returns:
        DataFrame with sentiment features
    """
    analyzer = ForexNewsSentiment()

    features = []
    for timestamp, headline in headlines:
        analysis = analyzer.analyze_for_pair(headline, pair)
        features.append({
            'timestamp': timestamp,
            'sentiment_signal': analysis['signal'],
            'sentiment_confidence': analysis['confidence'],
            'base_sentiment': analysis['base_sentiment'],
            'quote_sentiment': analysis['quote_sentiment']
        })

    return pd.DataFrame(features).set_index('timestamp')
