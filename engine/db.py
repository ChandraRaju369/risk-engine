"""SQLite storage for signals."""
import hashlib
import sqlite3

from .config import DATA_DIR, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    ts TEXT, source TEXT, ticker TEXT, text TEXT,
    sentiment REAL, confidence REAL,
    event_type TEXT, impact REAL
)"""


def connect(rows_as_dicts: bool = False) -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    if rows_as_dicts:
        con.row_factory = sqlite3.Row
    con.execute(SCHEMA)
    return con


def save(con: sqlite3.Connection, ts: str, source: str, text: str, sig: dict) -> None:
    """Insert or refresh one signal. Same ticker + text = same row (dedupe)."""
    sig_id = hashlib.md5(f"{sig['ticker']}|{text}".encode()).hexdigest()
    con.execute(
        "INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?,?)",
        (sig_id, ts, source, sig["ticker"], text, sig["sentiment"],
         sig["confidence"], sig["event_type"], sig["impact"]))
