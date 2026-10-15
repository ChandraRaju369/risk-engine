"""Module A, step 1: turn raw signals into a sentiment score per stock at any moment.

score(t) = sum(w_i * decay_i * s_i) / (sum(w_i * decay_i) + PRIOR)

  s_i      sentiment of signal i (-1..1)
  w_i      how much the signal counts = source credibility x model confidence x (1 + impact/10)
  decay_i  0.5 ** (age / half_life)  -> old news fades out
  PRIOR    pulls thin evidence towards 0, so one lone tweet cannot swing a weight
"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from engine import db
from engine.config import SOURCE_CREDIBILITY


@dataclass(frozen=True)
class Signal:
    ts: datetime
    source: str
    ticker: str
    text: str
    sentiment: float
    confidence: float
    event_type: str
    impact: float

    @property
    def weight(self) -> float:
        cred = SOURCE_CREDIBILITY.get(self.source, 0.5)
        return cred * self.confidence * (1 + self.impact / 10)


def parse_ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_signals() -> list[Signal]:
    con = db.connect(rows_as_dicts=True)
    rows = con.execute(
        "SELECT ts, source, ticker, text, sentiment, confidence, event_type, impact "
        "FROM signals").fetchall()
    con.close()
    signals = [Signal(parse_ts(r["ts"]), r["source"], r["ticker"], r["text"],
                      r["sentiment"] or 0.0, r["confidence"] or 0.5,
                      r["event_type"] or "Other", r["impact"] or 1.0) for r in rows]
    return sorted(signals, key=lambda s: s.ts)


def score_at(signals: list[Signal], t: datetime, half_life_h: float, prior: float,
             top_n: int = 3):
    """Return (score, drivers) for one stock at time t using only signals up to t."""
    parts = []
    num = den = 0.0
    for s in signals:
        age_h = (t - s.ts).total_seconds() / 3600
        if age_h < 0:
            continue
        w = s.weight * math.pow(0.5, age_h / half_life_h)
        num += w * s.sentiment
        den += w
        parts.append((w, s, age_h))
    if not parts:
        return 0.0, []
    score = num / (den + prior)
    ranked = sorted(((w * s.sentiment / (den + prior), s, age) for w, s, age in parts),
                    key=lambda x: -abs(x[0]))
    drivers = [{"text": s.text, "source": s.source, "sentiment": round(s.sentiment, 2),
                "event_type": s.event_type, "impact": s.impact,
                "contribution": round(c, 3), "age_h": round(age, 1)}
               for c, s, age in ranked[:top_n] if abs(c) >= 0.01]
    return round(score, 4), drivers
