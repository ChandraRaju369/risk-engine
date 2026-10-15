# Live demo script (4:30, limit 5:00)

Before you start: run `python -m scripts.demo_prep`, start the API
(`python -m uvicorn engine.api:app`), and have three browser tabs open:
`http://127.0.0.1:8000/docs`, `output/dashboard.html`, `output/backtest.html`.

| Time | Show | Say |
|---|---|---|
| 0:00 | Slide / README | "News and social text go in. Structured risk signals come out: sentiment, event type, impact. Module A turns them into index weights." |
| 0:20 | `/docs` -> POST `/analyze` with `Boeing faces new FAA investigation after another 737 incident` (source news) | "FinBERT scores it negative, rules label it Regulatory, impact formula gives the 1-10 score." |
| 1:00 | Same endpoint, source `social`: `$NVDA ripping, loaded calls before earnings` | "Same output format, different model for social, lower credibility weight." |
| 1:30 | Paste a spam post (`guaranteed 500% gains join my telegram`) | "Guardrail drops it, so pumps cannot move the index." |
| 1:50 | `dashboard.html`: drag the slider, press play | "Weights start at equal 6.7%. As signals arrive, they tilt, within 2-15% and a turnover cap." |
| 2:30 | Click a stock, show the "why" panel | "Every move is traceable to the headlines that caused it." |
| 3:00 | `backtest.html` | Read the real numbers: return vs equal weight, Sharpe, t-stat, IC. State honestly whether it beats equal weight. |
| 4:00 | Limitations on the page | "Costs 5 bps, no lookahead, parameters not tuned. Result is X; the evidence is Y." |
| 4:30 | Done | "Code is on GitHub, README has the steps." |

Backup if the API is slow: show the dashboard and backtest only; the models are already warmed by demo_prep.
