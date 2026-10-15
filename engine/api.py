"""Risk Engine API.

Start:  python -m uvicorn engine.api:app --reload
Docs:   http://127.0.0.1:8000/docs   (interactive, good for the live demo)

GET  /health                 service check
GET  /signals                latest signals; filters: ticker, source, event_type,
                             min_impact, since (ISO time), limit
GET  /summary                per-ticker count, average sentiment, max impact
POST /analyze                analyze any text on the fly (optionally save it)
"""
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

from . import db
from .analyze import analyze

app = FastAPI(title="AI/NLP Risk Engine", version="0.3")


def _query(sql: str, params: tuple = ()) -> list[dict]:
    con = db.connect(rows_as_dicts=True)
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


@app.get("/health")
def health():
    total = _query("SELECT COUNT(*) AS n FROM signals")[0]["n"]
    return {"status": "ok", "signals_stored": total}


@app.get("/signals")
def signals(
    ticker: Optional[str] = None,
    source: Optional[Literal["news", "social"]] = None,
    event_type: Optional[str] = None,
    min_impact: float = Query(0, ge=0, le=10),
    since: Optional[str] = Query(None, description="ISO time, e.g. 2026-10-12T09:00"),
    limit: int = Query(100, ge=1, le=1000),
):
    where, params = ["impact >= ?"], [min_impact]
    for column, value in (("ticker", ticker and ticker.upper()), ("source", source),
                          ("event_type", event_type)):
        if value:
            where.append(f"{column} = ?")
            params.append(value)
    if since:
        where.append("ts >= ?")
        params.append(since)
    sql = (f"SELECT ts, source, ticker, text, sentiment, confidence, event_type, impact "
           f"FROM signals WHERE {' AND '.join(where)} ORDER BY ts DESC LIMIT ?")
    return _query(sql, tuple(params + [limit]))


@app.get("/summary")
def summary():
    return _query(
        "SELECT ticker, COUNT(*) AS signals, ROUND(AVG(sentiment), 3) AS avg_sentiment, "
        "MAX(impact) AS max_impact FROM signals GROUP BY ticker ORDER BY avg_sentiment DESC")


class AnalyzeIn(BaseModel):
    text: str
    source: Literal["news", "social"] = "news"
    ticker: Optional[str] = None   # use when the text does not name the company
    save: bool = False


@app.post("/analyze")
def analyze_text(body: AnalyzeIn):
    results = analyze(body.text, body.source, body.ticker and body.ticker.upper())
    if body.save and results:
        con = db.connect()
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for sig in results:
            db.save(con, ts, body.source, body.text, sig)
        con.commit()
        con.close()
    return {"text": body.text, "source": body.source, "signals": results}
