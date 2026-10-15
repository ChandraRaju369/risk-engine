"""Single entry point that turns one piece of text into structured risk signals.

Used by both the batch pipeline and the API's /analyze endpoint so the two
can never disagree.
"""
from .entities import link_tickers
from .events import classify
from .filters import is_spam
from .impact import impact_score
from .sentiment import score


def analyze(text: str, source: str = "news", ticker_hint: str | None = None) -> list[dict]:
    """Return one signal dict per tracked ticker mentioned in `text` (may be empty)."""
    if is_spam(text):
        return []
    tickers = link_tickers(text) or ([ticker_hint] if ticker_hint else [])
    if not tickers:
        return []
    sentiment, confidence = score(text, source)
    event, event_conf, method = classify(text)
    impact = impact_score(event, sentiment, confidence, source)
    return [{
        "ticker": t,
        "sentiment": sentiment,
        "confidence": confidence,
        "event_type": event,
        "event_confidence": event_conf,
        "event_method": method,
        "impact": impact,
    } for t in tickers]
