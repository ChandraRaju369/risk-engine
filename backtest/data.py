"""Data for the backtest: the Kaggle news dataset and daily prices.

The news loader does not assume exact file or column names. It accepts a .zip, a folder or a
.csv, reads every CSV that has a headline column, a date column and a ticker column, and
keeps only our 15 stocks inside the chosen date window.
"""
import zipfile
from pathlib import Path

import pandas as pd

TEXT_COLS = ("headline", "title", "text")
DATE_COLS = ("date", "published", "datetime", "timestamp", "time")
TICK_COLS = ("stock", "ticker", "symbol")
NY = "America/New_York"


def _pick(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return None


def _iter_chunks(path: Path, member: str | None, chunksize: int = 200_000):
    """Yield (file_name, DataFrame chunk) for every candidate CSV."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if member:
                names = [n for n in names if member.lower() in n.lower()]
            print(f"CSV files inside the zip: {names}")
            for name in names:
                with zf.open(name) as fh:
                    for chunk in pd.read_csv(fh, chunksize=chunksize, low_memory=False):
                        yield name, chunk
    else:
        files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
        if member:
            files = [f for f in files if member.lower() in f.name.lower()]
        print(f"CSV files: {[f.name for f in files]}")
        for f in files:
            for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False):
                yield f.name, chunk


def _parse_dates(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(series, utc=True, errors="coerce")


def load_news(path, tickers, start, end, member=None, max_per_ticker=None, seed=0) -> pd.DataFrame:
    """Return DataFrame[ts (UTC), ticker, text] for our tickers within [start, end]."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    wanted = {t.upper() for t in tickers}
    frames, seen_unusable = [], set()
    for name, chunk in _iter_chunks(Path(path), member):
        tcol = _pick(chunk.columns, TEXT_COLS)
        dcol = _pick(chunk.columns, DATE_COLS)
        kcol = _pick(chunk.columns, TICK_COLS)
        if not (tcol and dcol and kcol):
            if name not in seen_unusable:
                print(f"  skipping {name}: columns are {list(chunk.columns)}")
                seen_unusable.add(name)
            continue
        chunk = chunk[[tcol, dcol, kcol]].rename(columns={tcol: "text", dcol: "date", kcol: "ticker"})
        chunk["ticker"] = chunk["ticker"].astype(str).str.upper().str.strip()
        chunk = chunk[chunk["ticker"].isin(wanted)]
        if chunk.empty:
            continue
        chunk["ts"] = _parse_dates(chunk["date"])
        chunk = chunk.dropna(subset=["ts", "text"])
        chunk = chunk[(chunk["ts"] >= start_ts - pd.Timedelta(days=1)) & (chunk["ts"] <= end_ts)]
        frames.append(chunk[["ts", "ticker", "text"]])
    if not frames:
        return pd.DataFrame(columns=["ts", "ticker", "text"])

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).str.split().str.join(" ")
    df = df[df["text"].str.len() >= 10]

    # A timestamp at exactly midnight New York time means "date only". Treat it as arriving after
    # the close (20:00), so it can only influence the NEXT trading day - never lookahead.
    local = df["ts"].dt.tz_convert(NY)
    midnight = (local.dt.hour == 0) & (local.dt.minute == 0) & (local.dt.second == 0)
    local = local.where(~midnight, local + pd.Timedelta(hours=20))
    df["ts"] = local.dt.tz_convert("UTC")

    df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)]
    df = df.drop_duplicates(subset=["ticker", "text", "ts"])
    df["day"] = df["ts"].dt.date
    df = df.drop_duplicates(subset=["ticker", "text", "day"]).drop(columns="day")   # same story repeated

    if max_per_ticker:   # random (but repeatable) sample per ticker keeps the time coverage even
        df = df.sample(frac=1, random_state=seed).groupby("ticker").head(max_per_ticker)
    return df.sort_values("ts").reset_index(drop=True)


def load_prices(tickers, start, end, cache: Path, csv: str | None = None) -> pd.DataFrame:
    """Daily adjusted close prices, one column per ticker. Cached on disk after the first download."""
    source = Path(csv) if csv else cache
    if source.exists():
        df = pd.read_csv(source, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        covers = (df.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=5)
                  and df.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=5)
                  and all(t in df.columns for t in tickers))
        if covers or csv:
            return df[[t for t in tickers if t in df.columns]]
    import yfinance as yf
    raw = yf.download(list(tickers), start=start, end=str(pd.Timestamp(end) + pd.Timedelta(days=7))[:10],
                      auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw else raw
    df = close.dropna(how="all")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    cache.parent.mkdir(exist_ok=True)
    df.to_csv(cache)
    return df[[t for t in tickers if t in df.columns]]
