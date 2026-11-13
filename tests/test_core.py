"""Smallest checks that fail if the statistics break. Run: .venv/Scripts/python -m pytest -q"""
import inspect

import numpy as np

from driftwatch.core import AlarmPolicy, EProcess, attribute

ALPHA = 0.05


def test_signature_frozen():
    assert list(inspect.signature(AlarmPolicy.update).parameters) == ["self", "stream", "stratum", "outcome", "t"]


def test_anytime_validity_all_strategies():
    """Under the null, P(ever crossing 1/alpha over a long horizon, checked at every step) <= alpha."""
    rng = np.random.default_rng(0)
    for strat in ("agrapa", "ons", "fixed"):
        hits = 0
        for _ in range(400):
            ep, d = EProcess(0.0, -1.0, strat), rng.choice([-1, 0, 1], 3000, p=[.1, .8, .1])
            hits += any(ep.update(x) >= np.log(1 / ALPHA) for x in d)
        assert hits / 400 <= ALPHA + 0.03, (strat, hits / 400)  # 3 MC sigma


def test_power_on_shift():
    rng = np.random.default_rng(1)
    ep, t = EProcess(0.0, -1.0), None
    for i, x in enumerate(rng.choice([-1, 0, 1], 5000, p=[.05, .75, .20])):
        if ep.update(x) >= np.log(1 / ALPHA):
            t = i
            break
    assert t is not None and t < 1000


def test_attribution_and_precedence():
    assert attribute({("M", 1): 100}, 50).label == "model"
    assert attribute({("H", 0): 100}, 50).label == "context"
    assert attribute({("M", 1): 100, ("P", 2): 120}, 50).label == "multiple"
    assert attribute({("M", 1): 100, ("J", 0): 120}, 50).label == "measurement"  # J contaminates the rest
    assert attribute({("M", 1): 100, ("P", 2): 900}, 50).label == "model"  # outside the guard window
    assert attribute({}, 50) is None


def test_policy_abstains_on_low_support_and_stays_valid_when_null():
    pol = AlarmPolicy(3, p0_gold=[.1, .1, .1], support=[100, 5, 100])
    assert pol.abstained == [1] and ("J", 1) not in pol.eps
    rng = np.random.default_rng(2)
    for t in range(2000):
        pol.update("J", t % 3, float(rng.random() < .1), t)
    assert not pol.crossings
