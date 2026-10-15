"""Module A, step 2: turn sentiment scores into index weights.

Each step:
  1. tilt      raw_i = base * max(floor, 1 + K * score_i)      (positive -> bigger, negative -> smaller)
  2. bound     every weight stays in [W_MIN, W_MAX] and the weights sum to 100%
  3. throttle  total turnover per rebalance is capped, so noise cannot churn the index
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from .aggregator import Signal, score_at


@dataclass
class Params:
    k: float = 1.5               # tilt strength: weight multiplier = 1 + k * score
    w_min: float = 0.02          # lowest allowed weight per stock
    w_max: float = 0.15          # highest allowed weight per stock
    max_turnover: float = 0.10   # max share of the index that may change per rebalance
    half_life_h: float = 4.0     # sentiment half-life (use ~6 for live, shorter for compressed replays)
    prior: float = 0.5           # shrinkage towards neutral for thin evidence
    step_h: float = 1.0          # rebalance frequency in hours


def cap_normalize(raw: list[float], lo: float, hi: float) -> list[float]:
    """Scale raw numbers so that, after clipping each to [lo, hi], they sum to exactly 1.

    Finds the scale factor by bisection: sum(clip(lam * raw)) rises smoothly with lam.
    """
    low, high = 0.0, 1e6
    for _ in range(200):
        lam = (low + high) / 2
        total = sum(min(hi, max(lo, lam * r)) for r in raw)
        if total < 1:
            low = lam
        else:
            high = lam
    lam = (low + high) / 2
    weights = [min(hi, max(lo, lam * r)) for r in raw]
    scale = sum(weights)
    return [w / scale for w in weights]


def rebalance(prev: list[float], scores: list[float], p: Params):
    """Return (new_weights, target_weights, realised_turnover)."""
    n = len(prev)
    base = 1.0 / n
    raw = [base * max(0.05, 1 + p.k * s) for s in scores]
    target = cap_normalize(raw, p.w_min, p.w_max)
    turnover = 0.5 * sum(abs(t - w) for t, w in zip(target, prev))
    alpha = 1.0 if turnover <= p.max_turnover else p.max_turnover / turnover
    new = [w + alpha * (t - w) for w, t in zip(prev, target)]
    return new, target, turnover * alpha


def make_grid(signals: list[Signal], step_h: float, tail_h: int = 2) -> list[datetime]:
    first = min(s.ts for s in signals).replace(minute=0, second=0, microsecond=0)
    last = max(s.ts for s in signals).replace(minute=0, second=0, microsecond=0)
    start = first - timedelta(hours=step_h)           # one step of "before anything happened"
    end = last + timedelta(hours=1 + tail_h)           # a few steps after, to show sentiment fading
    grid, t = [], start
    while t <= end:
        grid.append(t)
        t += timedelta(hours=step_h)
    return grid


def build_timeline(signals: list[Signal], tickers: list[str], grid: list[datetime],
                   p: Params) -> dict:
    """Run the aggregator + rebalancer across the whole time grid for one set of signals."""
    by_ticker = {t: [s for s in signals if s.ticker == t] for t in tickers}
    n = len(tickers)
    weights = [1.0 / n] * n
    out = {"weights": {t: [] for t in tickers}, "targets": {t: [] for t in tickers},
           "scores": {t: [] for t in tickers}, "drivers": {t: [] for t in tickers},
           "turnover": [], "counts": []}
    for now in grid:
        scored = [score_at(by_ticker[t], now, p.half_life_h, p.prior) for t in tickers]
        scores = [sc for sc, _ in scored]
        weights, target, moved = rebalance(weights, scores, p)
        for idx, t in enumerate(tickers):
            out["weights"][t].append(round(weights[idx], 5))
            out["targets"][t].append(round(target[idx], 5))
            out["scores"][t].append(scores[idx])
            out["drivers"][t].append(scored[idx][1])
        out["turnover"].append(round(moved, 5))
        out["counts"].append(sum(1 for s in signals if s.ts <= now))
    return out


def params_dict(p: Params) -> dict:
    return asdict(p)
