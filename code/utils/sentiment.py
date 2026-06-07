"""
Sentiment scoring for Polymarket comment text using VADER.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is well-suited to
short, informal social-media text — slang, capitalization, punctuation, and
emoji-adjacent symbols are all handled.

Outputs a compound score in [-1, +1]:
    > +0.05  → positive  (bullish)
    < -0.05  → negative  (bearish)
    otherwise → neutral

We also compute a binary direction flag relative to the YES outcome:
  sentiment_direction = +1 if compound > 0.05
                       = -1 if compound < -0.05
                       =  0 otherwise

This lets us test: does a bullish whale comment predict a YES price increase,
and does a bearish whale comment predict a YES price decrease?
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Polymarket-specific terms to add to the VADER lexicon
# Positive = bullish on YES outcome; negative = bearish on YES outcome
_POLYMARKET_LEXICON = {
    # bullish signals
    "yes":      1.5,
    "gonna":    0.5,
    "moon":     2.0,
    "pump":     1.5,
    "win":      1.5,
    "winning":  1.5,
    "sure":     1.2,
    "certain":  1.2,
    "obvious":  1.0,
    "easy":     1.0,
    "lock":     1.5,
    "locks":    1.5,
    "bullish":  2.5,
    # bearish signals
    "no":      -1.5,
    "nope":    -1.5,
    "never":   -1.5,
    "dump":    -1.5,
    "lose":    -1.5,
    "losing":  -1.5,
    "doubt":   -1.2,
    "unlikely":-1.5,
    "bearish": -2.5,
    "scam":    -2.0,
    "rigged":  -1.8,
}
_analyzer.lexicon.update(_POLYMARKET_LEXICON)


def score(text: str) -> dict:
    """
    Score a single comment string.

    Returns a dict with:
        compound          float in [-1, 1]
        positive          float in [0, 1]
        negative          float in [0, 1]
        neutral           float in [0, 1]
        sentiment_label   'positive' | 'negative' | 'neutral'
        sentiment_direction  +1 | 0 | -1
    """
    if not text or not isinstance(text, str):
        return {
            "compound": 0.0, "positive": 0.0,
            "negative": 0.0, "neutral": 1.0,
            "sentiment_label": "neutral", "sentiment_direction": 0,
        }

    cleaned = _clean(text)
    scores = _analyzer.polarity_scores(cleaned)

    compound = scores["compound"]
    if compound >= 0.05:
        label, direction = "positive", 1
    elif compound <= -0.05:
        label, direction = "negative", -1
    else:
        label, direction = "neutral", 0

    return {
        "compound":            compound,
        "positive":            scores["pos"],
        "negative":            scores["neg"],
        "neutral":             scores["neu"],
        "sentiment_label":     label,
        "sentiment_direction": direction,
    }


def score_series(texts) -> list[dict]:
    """Score an iterable of text strings. Returns a list of score dicts."""
    return [score(t) for t in texts]


def _clean(text: str) -> str:
    """Light pre-processing: strip URLs, keep punctuation (VADER uses it)."""
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)             # remove @mentions
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    # Quick smoke test
    examples = [
        "Trump is definitely going to win this, easy money bullish 🚀",
        "No way this resolves yes, complete scam rigged market",
        "Interesting market, not sure which way this goes",
        "MOON 🌙🌙 100% YES",
        "Selling all my YES shares, this is done",
    ]
    for t in examples:
        s = score(t)
        print(f"[{s['sentiment_label']:8s} {s['compound']:+.3f}]  {t[:60]}")
