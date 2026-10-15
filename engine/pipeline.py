"""Pipeline: ingest -> analyze (tickers, sentiment, event, impact) -> SQLite.

Run:  python -m engine.pipeline                    live news only (Yahoo RSS)
      python -m engine.pipeline social             live social only (StockTwits)
      python -m engine.pipeline live               live news + social
      python -m engine.pipeline sample_replay.jsonl   replay a file from data/
Set RISK_ZERO_SHOT=0 to skip the zero-shot model (keyword rules only).
"""
import itertools
import sys

from . import db, ingest
from .analyze import analyze
from .config import DATA_DIR
from .filters import is_spam


def run(items):
    con = db.connect()
    count = spam = 0
    for item in items:
        if is_spam(item["text"]):
            spam += 1
            print(f"[skipped spam] {item['text'][:60]}")
            continue
        for sig in analyze(item["text"], item["source"], item.get("ticker_hint")):
            db.save(con, item["ts"], item["source"], item["text"], sig)
            count += 1
            print(f"{item['ts'][:16]} {item['source']:<6} {sig['ticker']:<5} "
                  f"{sig['sentiment']:+.2f}  {sig['event_type']:<18} impact {sig['impact']:>4}  "
                  f"{item['text'][:55]}")
    con.commit()
    con.close()
    print(f"\n{count} signals written to the database ({spam} spam posts skipped)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "news"
    if arg == "news":
        run(ingest.fetch_news())
    elif arg == "social":
        run(ingest.fetch_social())
    elif arg == "live":
        run(itertools.chain(ingest.fetch_news(), ingest.fetch_social()))
    else:
        run(ingest.load_replay(str(DATA_DIR / arg)))
