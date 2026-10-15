"""Build the Module A dashboard from the signals database.

Run:  python -m module_a.run
      python -m module_a.run --half-life 6 --k 2 --max-turnover 0.15
Then open output/dashboard.html in your browser (double-click it).
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from engine.config import ROOT, UNIVERSE

from .aggregator import load_signals
from .rebalancer import Params, build_timeline, make_grid, params_dict

OUTPUT_DIR = ROOT / "output"
TEMPLATE = Path(__file__).with_name("template.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=float, default=Params.k, help="tilt strength")
    ap.add_argument("--half-life", type=float, default=Params.half_life_h, help="sentiment half-life, hours")
    ap.add_argument("--max-turnover", type=float, default=Params.max_turnover, help="max turnover per step")
    args = ap.parse_args()
    params = Params(k=args.k, half_life_h=args.half_life, max_turnover=args.max_turnover)

    signals = load_signals()
    if not signals:
        raise SystemExit("No signals in the database yet. Run the pipeline first, e.g.\n"
                         "  python -m engine.pipeline sample_replay.jsonl\n"
                         "  python -m engine.pipeline demo_social.jsonl")

    tickers = list(UNIVERSE)
    grid = make_grid(signals, params.step_h)
    scopes = {
        "all": signals,
        "news": [s for s in signals if s.source == "news"],
        "social": [s for s in signals if s.source == "social"],
    }
    timelines = {name: build_timeline(sigs, tickers, grid, params)
                 for name, sigs in scopes.items() if sigs or name == "all"}
    for name in ("news", "social"):          # an empty scope still needs a flat, valid timeline
        if name not in timelines:
            timelines[name] = build_timeline([], tickers, grid, params)

    last_signal = max(s.ts for s in signals)
    start_index = next(i for i, t in enumerate(grid) if t >= last_signal)
    data = {
        "meta": {
            "tickers": tickers, "base": 1 / len(tickers), "params": params_dict(params),
            "start_index": start_index, "n_signals": len(signals),
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "times": [t.isoformat() for t in grid],
        "epochs": [int(t.timestamp()) for t in grid],
        "scopes": timelines,
        "feed": [{"epoch": int(s.ts.timestamp()), "ts": s.ts.isoformat(), "source": s.source,
                  "ticker": s.ticker, "text": s.text, "sentiment": round(s.sentiment, 2),
                  "event_type": s.event_type, "impact": s.impact} for s in signals],
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html_path = OUTPUT_DIR / "dashboard.html"
    html_path.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload),
                         encoding="utf-8")

    csv_path = OUTPUT_DIR / "weights.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "ticker", "score", "weight", "target_weight"])
        for i, t in enumerate(grid):
            for tk in tickers:
                tl = timelines["all"]
                w.writerow([t.isoformat(), tk, tl["scores"][tk][i], tl["weights"][tk][i],
                            tl["targets"][tk][i]])

    final = timelines["all"]
    print(f"{len(signals)} signals, {len(grid)} rebalance steps "
          f"({grid[0]:%Y-%m-%d %H:%M} to {grid[-1]:%H:%M} UTC)\n")
    print(f"{'Stock':<7}{'Score':>8}{'Weight':>9}{'vs equal':>10}")
    base = 1 / len(tickers)
    for tk in sorted(tickers, key=lambda x: -final["weights"][x][start_index]):
        wt = final["weights"][tk][start_index]
        print(f"{tk:<7}{final['scores'][tk][start_index]:>+8.2f}{wt:>9.1%}{(wt - base) * 100:>+9.1f}pp")
    print(f"\nDashboard: {html_path}\nWeights:   {csv_path}")


if __name__ == "__main__":
    main()
