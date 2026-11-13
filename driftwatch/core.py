"""DRIFTWATCH core (paper §6): betting e-processes + the frozen alarm-policy interface.

Interface frozen:  AlarmPolicy.update(stream, stratum, outcome, t=None) -> Verdict | None
Streams: M model, C context, P population (paired differences in [-1,1], null mean 0),
         J measurement (gold-anchor judge error, null mean = exact p0), H context-hash (null: never drops).
Validity: every (stream, stratum) is a non-negative test martingale under its null; budget alpha/K each,
union bound + Ville => P(any alarm ever) <= alpha over an unbounded horizon.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

LMAX = 0.5  # betting fraction cap in normalised space (z >= -1 so capital stays positive)
LABEL = {"M": "model", "C": "context", "P": "population", "J": "measurement", "H": "context"}


class EProcess:
    """One-sided betting e-process for H0: E[y]=mu (y in [lo,hi]); bets on an increase. log-capital in .logE."""

    __slots__ = ("mu", "D", "strategy", "lam", "logE", "n", "sz", "szz", "A", "fixed")

    def __init__(self, mu: float = 0.0, lo: float = 0.0, strategy: str = "agrapa", fixed: float = 0.25):
        self.mu = mu
        self.D = (mu - lo) if mu > lo else 1.0  # z=(y-mu)/D >= -1
        self.strategy, self.fixed = strategy, fixed
        self.lam = fixed if strategy == "fixed" else 0.0  # predictable: set from the past only
        self.logE, self.n, self.sz, self.szz, self.A = 0.0, 0, 0.0, 0.0, 1.0

    def update(self, y: float) -> float:
        z = (y - self.mu) / self.D
        self.logE += math.log1p(self.lam * z)
        self.n += 1
        if self.strategy == "agrapa":  # plug-in lambda = m / E[z^2], with a unit prior
            self.sz += z
            self.szz += z * z
            self.lam = min(LMAX, max(0.0, (self.sz / (self.n + 1)) / ((1.0 + self.szz) / (self.n + 1))))
        elif self.strategy == "ons":  # online Newton step on -log(1+lam z)
            g = -z / (1.0 + self.lam * z)
            self.A += g * g
            self.lam = min(LMAX, max(0.0, self.lam - (2.0 / (2.0 - math.log(3.0))) * g / self.A))
        return self.logE


@dataclass
class Verdict:
    label: str  # none | model | context | population | measurement | multiple
    first: float
    streams: tuple


def attribute(crossings: dict, window: float) -> Optional[Verdict]:
    """Algorithm 1: J has precedence (a drifting judge contaminates the rest); never force one label."""
    if not crossings:
        return None
    first = min(crossings.values())
    X = tuple(sorted({("C" if k == "H" else k) for (k, _s), t in crossings.items() if t <= first + window}))
    if "J" in X:
        return Verdict("measurement", first, X)
    if len(X) == 1:
        return Verdict(LABEL[X[0]], first, X)
    return Verdict("multiple", first, X)


class AlarmPolicy:
    def __init__(self, n_strata, alpha=0.05, window=1500, p0_gold=None, support=None, min_support=30,
                 strategy="agrapa", active=("M", "C", "P", "J", "H")):
        self.window, self.crossings, self._verdict, self._n = window, {}, None, 0
        # abstain (never report "no drift") on strata whose gold support is below the gate
        self.abstained = [] if support is None else [s for s in range(n_strata) if support[s] < min_support]
        self.eps = {}
        for k in active:
            if k == "H":
                self.eps[("H", 0)] = EProcess(0.0, 0.0, "fixed", 1.0)  # any dropped constraint is a violation
                continue
            for s in range(n_strata):
                if k == "J":
                    if s in self.abstained:
                        continue
                    self.eps[(k, s)] = EProcess(p0_gold[s], 0.0, strategy)
                else:
                    self.eps[(k, s)] = EProcess(0.0, -1.0, strategy)
        self.K = len(self.eps)
        self.thr = math.log(self.K / alpha)  # equal split of the global budget

    def update(self, stream, stratum, outcome, t=None):
        key = (stream, 0 if stream == "H" else stratum)
        ep = self.eps.get(key)
        if t is None:
            t, self._n = self._n, self._n + 1
        if ep is None:
            return None
        if ep.update(outcome) >= self.thr and key not in self.crossings:
            self.crossings[key] = t
        if self._verdict is None and self.crossings and t >= min(self.crossings.values()) + self.window:
            self._verdict = attribute(self.crossings, self.window)
            return self._verdict
        return None

    def verdict(self, window=None):
        return attribute(self.crossings, self.window if window is None else window)
