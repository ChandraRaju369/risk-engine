"""Checks the guarantees the rebalancer must always keep. Run:  python -m tests.test_rebalancer"""
import random

from engine.events import classify
from engine.filters import is_spam
from engine.impact import impact_score
from module_a.rebalancer import Params, cap_normalize, rebalance

P = Params()
N = 15


def test_weights_always_valid():
    rnd = random.Random(1)
    prev = [1 / N] * N
    for _ in range(500):
        scores = [rnd.uniform(-1, 1) for _ in range(N)]
        prev, target, moved = rebalance(prev, scores, P)
        assert abs(sum(prev) - 1) < 1e-9, "weights must sum to 100%"
        assert all(P.w_min - 1e-9 <= w <= P.w_max + 1e-9 for w in prev), "weights must stay within bounds"
        assert moved <= P.max_turnover + 1e-9, "turnover cap must hold"


def test_direction():
    prev = [1 / N] * N
    scores = [0.0] * N
    scores[0], scores[1] = 0.9, -0.9
    new, _, _ = rebalance(prev, scores, P)
    assert new[0] > 1 / N > new[1], "positive sentiment raises a weight, negative lowers it"


def test_neutral_stays_equal():
    new, _, moved = rebalance([1 / N] * N, [0.0] * N, P)
    assert moved < 1e-9 and all(abs(w - 1 / N) < 1e-9 for w in new)


def test_cap_normalize_extremes():
    w = cap_normalize([100] + [0.001] * (N - 1), P.w_min, P.w_max)
    assert abs(sum(w) - 1) < 1e-9 and max(w) <= P.w_max + 1e-9 and min(w) >= P.w_min - 1e-9


def test_engine_rules():
    assert classify("Moody's downgrades Bank of America on rising credit losses", use_model=False)[0] == "Credit Event"
    assert classify("Exxon slides on Middle East ceasefire", use_model=False)[0] == "Geopolitical"
    assert is_spam("BUY NOW $NVDA $AAPL $TSLA guaranteed 500% gains join my telegram")
    assert not is_spam("Apple beats earnings expectations")
    assert impact_score("Credit Event", -0.9, 0.9, "news") > impact_score("Other", -0.9, 0.9, "news")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("\nall checks passed")
