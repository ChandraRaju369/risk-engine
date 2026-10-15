# AI/NLP Financial Risk Engine + Tactical Index Rebalancer

Turns unstructured news and social text into structured risk signals
(sentiment, event type, impact), then rebalances a 15-stock mock index (Module A).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m engine.pipeline sample_replay.jsonl         # news replay
python -m engine.pipeline demo_social.jsonl           # social replay
python -m engine.pipeline live                        # live news (Yahoo RSS) + social (StockTwits)
python -m uvicorn engine.api:app --reload             # API, docs at http://127.0.0.1:8000/docs
```
Set `RISK_ZERO_SHOT=0` to skip the 1.6 GB zero-shot model (keyword rules only).

> `sample_replay.jsonl` and `demo_social.jsonl` are small **synthetic** samples written for
> testing and demos. Live mode uses real feeds.

## Engine
| Stage | What it does | Model / method |
|---|---|---|
| Ingest | news RSS, StockTwits stream, JSONL replay | feedparser, requests |
| Guardrails | drops spam / pump posts | rules (`filters.py`) |
| Entity linking | text -> tickers in the 15-stock universe | alias dictionary + cashtags |
| Sentiment | -1..1 score + confidence | FinBERT (news), Twitter RoBERTa (social) |
| Event type | 8 categories + Other | keyword rules, zero-shot BART fallback |
| Impact | 1-10 severity | transparent formula (`impact.py`) |

## Signal schema
`ts, source, ticker, text, sentiment (-1..1), confidence, event_type, impact (1-10)`

## API
- `GET /signals?ticker=AAPL&source=news&event_type=Credit Event&min_impact=7&since=...&limit=100`
- `GET /summary` per-ticker count, average sentiment, max impact
- `POST /analyze` `{"text": "...", "source": "news", "save": false}`
- `GET /health`

## Module A: tactical index rebalancing
```bash
python -m module_a.run        # builds output/dashboard.html + output/weights.csv from the signals database
python -m tests.test_rebalancer   # checks: weights sum to 100%, stay within 2-15%, turnover cap holds
```
Open `output/dashboard.html` in a browser (no server needed).

1. **Aggregate** (`module_a/aggregator.py`): per stock, a time-decayed average of sentiment, weighted by
   source credibility x model confidence x (1 + impact/10), shrunk towards neutral when evidence is thin.
2. **Rebalance** (`module_a/rebalancer.py`): `weight = base x (1 + k x score)`, kept within 2-15% and 100% total,
   then limited to 10% turnover per step so one noisy post cannot churn the index.
3. **Dashboard** (`module_a/template.html`): replay slider and play button, per-stock weight panels,
   weights table, sentiment heatmap, "why did this move?" panel with the headlines responsible, and a
   News / Social filter.

Tunable: `python -m module_a.run --k 2 --half-life 6 --max-turnover 0.15`

## Backtest: does the sentiment tilt beat equal weight?
Uses the Kaggle "Massive Stock News Analysis DB for NLP Backtests" (Benzinga headlines) and yfinance prices.
```bash
python -m backtest.run --news "path/to/archive.zip" --max-per-ticker 100     # quick smoke run (~minutes)
python -m backtest.run --news "path/to/archive.zip"                          # default: 2019, up to 800 headlines/stock
python -m backtest.run --news "path/to/archive.zip" --start 2018-01-01 --end 2018-12-31
python -m tests.test_backtest
```
Writes `output/backtest.html` (open in a browser), `backtest.json`, `backtest_daily.csv`.

Method: weights are decided at each 4 pm New York close using only earlier news and earn the *next* day's
return (no lookahead; date-only headlines count as after-close). 5 bps cost on everything traded. Reports
return, Sharpe, drawdown, turnover, the t-statistic of the extra return, and the daily rank correlation (IC)
between sentiment and next-day returns, plus a sensitivity table. Parameters are fixed up front, not tuned.

## Accuracy check (human-labelled sample)
```bash
python -m validation.make_sample --news "path/to/archive.zip"   # writes output/label_me.xlsx (engine answers hidden)
# open the xlsx, fill the two dropdown columns, save
python -m validation.score                                     # writes output/validation.md
```

## Demo
`python -m scripts.demo_prep` then follow `docs/DEMO_SCRIPT.md`.

## Results

All numbers come from real runs of this code on the Kaggle Benzinga headline set (see the Backtest section).

### Backtest: sentiment-tilted index vs equal weight
Weights are set at each 4 pm New York close from earlier news only and earn the next day's return; 5 bps cost on everything traded; tilt k = 1.5, half-life 2 days, parameters fixed before looking at results.

| Year | Stocks with news | Tilted return | Equal-weight return | Extra per year (after costs) | t-stat | Daily IC |
|---|---|---|---|---|---|---|
| 2018 | 8 of 15 | -1.4% | -0.3% | -1.2% | -1.08 | +0.010 |
| 2019 | 9 of 15 | +28.5% | +31.4% | -2.3% | -1.63 | -0.010 |

**Conclusion: no measurable edge.** Neither year is statistically different from zero (|t| < 2) and the rank correlation between sentiment and next-day returns is about zero. The tilt trades about 4% of the index a day against 0.4% for equal weight, which costs roughly 0.8 to 0.9 points a year. A weaker tilt loses less (2019, k = 0.5: -0.8% a year). Most Kaggle headlines are analyst-rating notes, and 6 to 7 of the 15 stocks have no headlines in the file, so the test is weak evidence about news sentiment in general.

### NLP accuracy on 100 labelled headlines
Sample drawn from the 2019 headlines; the engine's answers were hidden while labelling. Labels were drafted with an LLM and then reviewed. With 100 items the 95% interval is about 9 points either way.

| Measure | Result |
|---|---|
| Direction right on clearly good or bad news (43 items) | 86% (73-93%) |
| Three-way sentiment agreement | 63% (53-72%), macro-F1 0.61; guessing "neutral" scores 57% |
| Event type agreement | 67% (57-75%); keyword rules right 52% on 31 items, "Other" default right 74% on 69 |

Main weakness: market-wide commentary (for example "Dow Rises Over 200 Points") is scored as company sentiment.

Reproduce: `python -m backtest.run --news "path/to/archive.zip"` and `python -m validation.make_sample --news "path/to/archive.zip"` then `python -m validation.score`.

## Status
- [x] Ingestion (RSS news, StockTwits, replay), entity linking, spam filter, SQLite
- [x] Sentiment, event classifier, impact score
- [x] FastAPI service
- [x] Module A: aggregator, rebalancer, dashboard
- [x] Backtest (code done; results to be added after the real run)
- [x] Validation tool, demo prep and demo script
- [x] Real results (backtest, accuracy) and slides
- [ ] Recorded demo
