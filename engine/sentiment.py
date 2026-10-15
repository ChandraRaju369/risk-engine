"""Sentiment scoring. News -> FinBERT, social -> Twitter RoBERTa.

Score = P(positive) - P(negative), a value in [-1, 1].
Confidence = max class probability.
Models load lazily on first use so imports stay fast.
"""
from functools import lru_cache

MODELS = {
    "news": "ProsusAI/finbert",
    "social": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}


@lru_cache(maxsize=None)
def _pipe(source: str):
    from transformers import pipeline
    return pipeline("text-classification", model=MODELS[source], top_k=None,
                    truncation=True, max_length=256)


def score(text: str, source: str = "news") -> tuple[float, float]:
    """Return (sentiment in [-1, 1], confidence in [0, 1])."""
    probs = {r["label"].lower(): r["score"] for r in _pipe(source)(text)[0]}
    sentiment = probs.get("positive", 0.0) - probs.get("negative", 0.0)
    return round(sentiment, 4), round(max(probs.values()), 4)


def score_batch(texts: list[str], source: str = "news") -> list[tuple[float, float]]:
    out = []
    for res in _pipe(source)(texts, batch_size=16):
        probs = {r["label"].lower(): r["score"] for r in res}
        out.append((round(probs.get("positive", 0.0) - probs.get("negative", 0.0), 4),
                    round(max(probs.values()), 4)))
    return out
