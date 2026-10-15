"""Ingestion. Each fetcher yields dicts: {source, ts, text, ticker_hint}.

Sources:
  - news:   Yahoo Finance RSS per ticker
  - social: StockTwits public stream per ticker (no key). If it is blocked or
            offline the fetcher prints a warning and moves on, so demos never crash.
  - replay: a JSONL file, so the demo runs without live feeds
"""
import json
import time
from datetime import datetime, timezone

import feedparser
import requests

from .config import NEWS_FEEDS, UNIVERSE

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"


def _clean(text: str) -> str:
    return " ".join(text.split())


def fetch_news(limit_per_ticker: int = 10):
    for ticker in UNIVERSE:
        for template in NEWS_FEEDS:
            feed = feedparser.parse(template.format(ticker=ticker))
            for entry in feed.entries[:limit_per_ticker]:
                parsed = entry.get("published_parsed")
                ts = (datetime.fromtimestamp(time.mktime(parsed), tz=timezone.utc)
                      if parsed else datetime.now(timezone.utc))
                text = _clean(entry.get("title", "") + ". " + entry.get("summary", ""))
                yield {"source": "news", "ts": ts.isoformat(), "text": text,
                       "ticker_hint": ticker}


def fetch_social(limit_per_ticker: int = 20):
    """Live social posts from StockTwits. Quietly skips tickers it cannot reach."""
    for ticker in UNIVERSE:
        try:
            resp = requests.get(STOCKTWITS_URL.format(ticker=ticker), timeout=10,
                                headers={"User-Agent": "risk-engine-hackathon/0.1"})
            if resp.status_code != 200:
                print(f"[social] {ticker}: HTTP {resp.status_code}, skipping "
                      f"(use the replay file demo_social.jsonl instead)")
                continue
            for msg in resp.json().get("messages", [])[:limit_per_ticker]:
                yield {"source": "social", "ts": msg["created_at"],
                       "text": _clean(msg["body"]), "ticker_hint": ticker}
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"[social] {ticker}: {exc.__class__.__name__}, skipping")


def load_replay(path: str):
    """Each line: {"source": "news"|"social", "ts": ISO8601, "text": "..."}"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["text"] = _clean(row["text"])
                row.setdefault("ticker_hint", None)
                yield row
