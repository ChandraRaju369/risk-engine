"""Checks the backtest arithmetic. Run:  python -m tests.test_backtest"""
import numpy as np

from backtest.run import metrics, simulate


def test_weights_earn_the_next_days_return():
    # stock 0 gains 10% on day 2 only. Weights chosen at the close of day 1 should capture it;
    # weights chosen at the close of day 2 must NOT (that gain has already happened).
    rets = np.zeros((4, 2)); rets[2, 0] = 0.10
    w = np.array([[0.5, 0.5], [1.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    net, _ = simulate(w, rets, cost_bps=0)
    assert abs(net[1] - 0.10) < 1e-12, "weights set on day 1 earn day 2's return"
    assert abs(net[2]) < 1e-12, "weights set on day 2 do not earn day 2's return"


def test_costs_charged_on_trades():
    rets = np.zeros((3, 2))                                  # flat prices, so only costs move the result
    hold = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    net, turn = simulate(hold, rets, cost_bps=10)
    assert np.allclose(net, 0) and np.allclose(turn, 0), "no trades, no cost"
    trade = np.array([[0.5, 0.5], [1.0, 0.0], [1.0, 0.0]])   # day 1: move 50% of the index (gross change 1.0)
    net, turn = simulate(trade, rets, cost_bps=10)
    assert abs(net[1] - (-0.0010)) < 1e-12, "10 bps charged on the 1.0 gross change"
    assert abs(turn[1] - 0.5) < 1e-12, "turnover is half the gross change"
    assert abs(net[0]) < 1e-12, "no cost on a day with no trade"


def test_metrics_basic():
    m = metrics(np.array([0.01, -0.02, 0.03]), np.array([0.0, 0.0, 0.0]))
    assert abs(m["total_return"] - (1.01 * 0.98 * 1.03 - 1)) < 1e-12
    assert m["max_drawdown"] < 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("\nall checks passed")
