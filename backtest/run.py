"""Backtest: does tilting an index by news sentiment beat plain equal weight?

Run (after downloading the Kaggle dataset):
  python -m backtest.run --news "C:\\Users\\you\\Downloads\\archive.zip"
  python -m backtest.run --news archive.zip --start 2018-01-01 --end 2018-12-31
Then open output/backtest.html.

How the test avoids cheating
  * Decisions are made at each day's 16:00 New York close using only news published before that.
  * Those weights earn the NEXT day's close-to-close return.
  * Headlines stamped only with a date are treated as arriving after the close.
  * Trading costs are charged on everything traded, including drift back to target weights.
  * Parameters are fixed up front (below), not tuned on the results; a sensitivity table shows the rest.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from engine.config import DATA_DIR, ROOT, UNIVERSE
from engine.events import classify
from engine.impact import impact_score
from engine.sentiment import score_batch
from module_a.aggregator import Signal
from module_a.rebalancer import Params, build_timeline

from .data import NY, load_news, load_prices

OUTPUT_DIR = ROOT / "output"
CACHE = DATA_DIR / "backtest_scored.csv"
TEMPLATE = __file__.replace("run.py", "template.html")
TRADING_DAYS = 252


# ---------------------------------------------------------------- scoring (cached) ----------
def _key(ticker: str, text: str) -> str:
    return hashlib.md5(f"{ticker}|{text}".encode()).hexdigest()


def score_news(news: pd.DataFrame, batch: int = 64) -> pd.DataFrame:
    """Run the engine on every headline. Results are cached so re-runs are instant."""
    news = news.copy()
    news["key"] = [_key(t, x) for t, x in zip(news["ticker"], news["text"])]
    cache = pd.read_csv(CACHE).drop_duplicates("key").set_index("key") if CACHE.exists() else pd.DataFrame()
    todo = news[~news["key"].isin(cache.index)] if len(cache) else news
    if len(todo):
        print(f"Scoring {len(todo)} new headlines with FinBERT ({len(news) - len(todo)} already cached)...")
        rows = []
        texts = todo["text"].tolist()
        for i in range(0, len(texts), batch):
            part = texts[i:i + batch]
            for text, (sent, conf) in zip(part, score_batch(part, "news")):
                event = classify(text, use_model=False)[0]      # rules only: fast, and the backtest needs sentiment
                rows.append((sent, conf, event, impact_score(event, sent, conf, "news")))
            if (i // batch) % 10 == 0:
                print(f"  {min(i + batch, len(texts))}/{len(texts)}")
        scored = todo[["key"]].copy()
        scored[["sentiment", "confidence", "event_type", "impact"]] = rows
        cache = pd.concat([cache.reset_index(), scored]) if len(cache) else scored
        DATA_DIR.mkdir(exist_ok=True)
        cache.drop_duplicates("key").to_csv(CACHE, index=False)
        cache = cache.drop_duplicates("key").set_index("key")
    return news.join(cache[["sentiment", "confidence", "event_type", "impact"]], on="key").dropna()


# ---------------------------------------------------------------- simulation -----------------
def simulate(weights: np.ndarray, rets: np.ndarray, cost_bps: float):
    """weights[k] is decided at the close of day k and earns rets[k+1].  Returns (net, turnover)."""
    K = len(weights)
    net, turnover = np.zeros(K - 1), np.zeros(K - 1)
    for k in range(K - 1):
        if k == 0:
            drift = weights[0]
        else:
            grown = weights[k - 1] * (1 + rets[k])
            drift = grown / grown.sum()
        traded = np.abs(weights[k] - drift).sum()
        net[k] = weights[k] @ rets[k + 1] - cost_bps / 1e4 * traded
        turnover[k] = 0.5 * traded
    return net, turnover


def metrics(net: np.ndarray, turnover: np.ndarray) -> dict:
    equity = np.cumprod(1 + net)
    n = len(net)
    peak = np.maximum.accumulate(equity)
    sd = net.std(ddof=1)
    return {
        "total_return": float(equity[-1] - 1),
        "annual_return": float(equity[-1] ** (TRADING_DAYS / n) - 1),
        "annual_vol": float(sd * math.sqrt(TRADING_DAYS)),
        "sharpe": float(net.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        "max_drawdown": float(((equity - peak) / peak).min()),
        "avg_turnover": float(turnover.mean()),
    }


def t_stat(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


def information_coefficient(scores: np.ndarray, rets: np.ndarray) -> np.ndarray:
    """Per day: rank correlation between today's sentiment scores and tomorrow's stock returns."""
    out = []
    for k in range(len(scores) - 1):
        s, r = scores[k], rets[k + 1]
        if np.std(s) < 1e-9:
            continue
        out.append(pd.Series(s).rank().corr(pd.Series(r).rank()))
    return np.array(out, dtype=float)


def run_strategy(signals, tickers, grid, rets, params: Params, cost_bps: float):
    tl = build_timeline(signals, tickers, grid, params)
    weights = np.array([[tl["weights"][t][k] for t in tickers] for k in range(len(grid))])
    scores = np.array([[tl["scores"][t][k] for t in tickers] for k in range(len(grid))])
    net, turn = simulate(weights, rets, cost_bps)
    return net, turn, scores, weights


# ---------------------------------------------------------------- main ------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--news", required=True, help="Kaggle zip, folder or CSV")
    ap.add_argument("--member", help="only read CSV files whose name contains this text")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2019-12-31")
    ap.add_argument("--max-per-ticker", type=int, default=800, help="cap headlines per stock (speed); 0 = all")
    ap.add_argument("--prices", help="optional CSV of daily prices (date index, one column per ticker)")
    ap.add_argument("--k", type=float, default=1.5)
    ap.add_argument("--half-life-days", type=float, default=2.0)
    ap.add_argument("--max-turnover", type=float, default=0.05, help="max share of the index traded per day")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="cost per unit traded, in basis points")
    ap.add_argument("--no-sensitivity", action="store_true")
    args = ap.parse_args()

    tickers = list(UNIVERSE)
    news = load_news(args.news, tickers, args.start, args.end, args.member, args.max_per_ticker or None)
    if news.empty:
        raise SystemExit("No headlines found for the 15 stocks in that window. Check --news, --start and --end, "
                         "and see the column list printed above.")
    print(f"\n{len(news)} headlines for {news['ticker'].nunique()} stocks, "
          f"{news['ts'].min():%Y-%m-%d} to {news['ts'].max():%Y-%m-%d}")
    counts = news["ticker"].value_counts()
    print("Headlines per stock: " + ", ".join(f"{t} {counts.get(t, 0)}" for t in tickers))
    thin = [t for t in tickers if counts.get(t, 0) < 30]
    if thin:
        print(f"  note: very little news for {thin}; those stocks will stay close to equal weight.")
    scored = score_news(news)

    prices = load_prices(tickers, args.start, args.end, DATA_DIR / "prices.csv", args.prices)
    tickers = [t for t in tickers if t in prices.columns]
    prices = prices.loc[args.start:args.end, tickers]
    if len(prices) < 30:
        raise SystemExit("Fewer than 30 trading days of prices; check the dates or the prices file.")
    rets = prices.pct_change().fillna(0.0).to_numpy()
    days = list(prices.index)
    grid = [(pd.Timestamp(d).tz_localize(NY) + pd.Timedelta(hours=16)).tz_convert("UTC").to_pydatetime()
            for d in days]

    signals = [Signal(r.ts.to_pydatetime(), "news", r.ticker, r.text, float(r.sentiment), float(r.confidence),
                      r.event_type, float(r.impact)) for r in scored.itertuples() if r.ticker in tickers]

    base = Params(k=args.k, half_life_h=args.half_life_days * 24, max_turnover=args.max_turnover, step_h=24)
    net_t, turn_t, scores, w_t = run_strategy(signals, tickers, grid, rets, base, args.cost_bps)
    n = len(tickers)
    equal_w = np.full((len(grid), n), 1 / n)
    net_e, turn_e = simulate(equal_w, rets, args.cost_bps)

    m_t, m_e = metrics(net_t, turn_t), metrics(net_e, turn_e)
    excess = net_t - net_e
    gross_t, _ = simulate(w_t, rets, 0.0)            # same weights, no trading costs
    gross_e, _ = simulate(equal_w, rets, 0.0)
    ic = information_coefficient(scores, rets)
    stats = {
        "excess_annual": float(excess.mean() * TRADING_DAYS),
        "excess_gross_annual": float((gross_t - gross_e).mean() * TRADING_DAYS),
        "tracking_error": float(excess.std(ddof=1) * math.sqrt(TRADING_DAYS)),
        "info_ratio": float(excess.mean() / excess.std(ddof=1) * math.sqrt(TRADING_DAYS)) if excess.std() > 0 else 0.0,
        "excess_t": t_stat(excess),
        "win_rate": float((excess > 0).mean()),
        "ic_mean": float(np.nanmean(ic)) if len(ic) else 0.0,
        "ic_t": t_stat(ic),
        "ic_days": int(len(ic)),
    }

    sens = []
    if not args.no_sensitivity:
        print("Sensitivity to tilt strength...")
        for k in (0.5, 1.0, 1.5, 2.0, 3.0):
            nt, tt, _, _ = run_strategy(signals, tickers, grid, rets, Params(
                k=k, half_life_h=base.half_life_h, max_turnover=base.max_turnover, step_h=24), args.cost_bps)
            mm = metrics(nt, tt)
            sens.append({"k": k, "total_return": mm["total_return"], "sharpe": mm["sharpe"],
                         "excess_annual": float((nt - net_e).mean() * TRADING_DAYS)})

    report = {
        "meta": {"start": args.start, "end": args.end, "days": len(net_t), "headlines": len(signals),
                 "tickers": tickers, "cost_bps": args.cost_bps, "k": base.k,
                 "half_life_days": args.half_life_days, "max_turnover": base.max_turnover,
                 "generated": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "dates": [d.strftime("%Y-%m-%d") for d in days[1:]],
        "tilted": [round(float(x), 5) for x in np.cumprod(1 + net_t)],
        "equal": [round(float(x), 5) for x in np.cumprod(1 + net_e)],
        "metrics": {"tilted": m_t, "equal": m_e}, "stats": stats, "sensitivity": sens,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "backtest.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    pd.DataFrame({"date": report["dates"], "tilted_growth": report["tilted"], "equal_growth": report["equal"],
                  "tilted_net_return": net_t, "equal_net_return": net_e}).to_csv(
        OUTPUT_DIR / "backtest_daily.csv", index=False)
    html = open(TEMPLATE, encoding="utf-8").read().replace("__DATA__", json.dumps(report).replace("</", "<\\/"))
    (OUTPUT_DIR / "backtest.html").write_text(html, encoding="utf-8")

    def pct(x): return f"{x:+.1%}"
    print(f"\n{'':<22}{'Sentiment-tilted':>18}{'Equal weight':>15}")
    for label, key, fmt in [("Total return", "total_return", pct), ("Annualised return", "annual_return", pct),
                            ("Annualised volatility", "annual_vol", lambda x: f"{x:.1%}"),
                            ("Sharpe ratio", "sharpe", lambda x: f"{x:.2f}"),
                            ("Max drawdown", "max_drawdown", lambda x: f"{x:.1%}"),
                            ("Avg daily turnover", "avg_turnover", lambda x: f"{x:.2%}")]:
        print(f"{label:<22}{fmt(m_t[key]):>18}{fmt(m_e[key]):>15}")
    print(f"\nExcess return per year {pct(stats['excess_annual'])} after costs, "
          f"{pct(stats['excess_gross_annual'])} before costs (t = {stats['excess_t']:.2f}); "
          f"mean daily IC {stats['ic_mean']:+.3f} (t = {stats['ic_t']:.2f}, {stats['ic_days']} days)")
    print(f"\nReport: {OUTPUT_DIR / 'backtest.html'}")


if __name__ == "__main__":
    main()
