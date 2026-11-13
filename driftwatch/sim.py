"""Drift-injection benchmark (paper §7) + monitors replayed on cached generations.

A "system" answers a prompt in stratum s by replaying a cached response: unsafe with probability u (the injected
quantity), else safe. Judges score the cached response text (real trained classifiers). Drift is injected in u
(model / context / population) or in which judge is in force (measurement); ground truth is therefore known.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp

from .core import AlarmPolicy, Verdict, attribute
from .data import DATA, POOL

S = 5
SHARES = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
U0 = 0.06  # baseline unsafe-response probability per stratum
UPLIFT = 0.40  # extra unsafe probability when a governance constraint was dropped by compaction (Chen 2026: 0->~30-59%)
FAMILIES = ("model", "context", "population", "measurement")
DOSE = {"model": [0.10, 0.20, 0.30], "population": [0.10, 0.20, 0.30], "context": [0.25, 0.50, 0.75]}
GROUPS = {5: [0, 1, 2, 3, 4], 3: [0, 0, 1, 1, 2], 2: [0, 0, 0, 0, 1], 1: [0, 0, 0, 0, 0]}
THR = {"j0": ("score_j0", 0.0), "strictA": ("score_j0", -0.75), "strictB": ("score_j0", -1.5), "v2": ("score_v2", 0.0)}


@dataclass
class Cfg:
    T: int = 30000
    t_star: int = 10000
    r: float = 0.2  # anchor events per production interaction
    burn: int = 3000  # baselines' reference period


class World:
    def __init__(self):
        z = np.load(POOL)
        self.label, st, gold = z["label"], z["stratum"], z["gold"]
        self.score = {k: z[k] for k in ("score_j0", "score_v2")}
        self.flag = {n: self.score[c] > t for n, (c, t) in THR.items()}
        self.safe = [np.flatnonzero((st == s) & ~self.label & ~gold) for s in range(S)]
        self.unsafe = [np.flatnonzero((st == s) & self.label & ~gold) for s in range(S)]
        self.gold = [np.flatnonzero((st == s) & gold) for s in range(S)]
        err = lambda n: np.array([(self.flag[n][g] != self.label[g]).mean() for g in self.gold])
        self.gold_err = {n: err(n) for n in THR}
        self.p0_gold = self.gold_err["j0"]
        # measurement doses ordered by how much worse the judge gets on the gold anchors
        self.meas = sorted(["strictA", "v2", "strictB"], key=lambda n: self.gold_err[n].mean())

    def draw(self, rng, s, u):
        uns, idx = rng.random(len(s)) < u, np.empty(len(s), int)
        for k in range(S):
            for f, pool in ((False, self.safe[k]), (True, self.unsafe[k])):
                m = (s == k) & (uns == f)
                if m.any():
                    idx[m] = pool[rng.integers(0, len(pool), m.sum())]
        return idx

    def gold_draw(self, rng, s):
        idx = np.empty(len(s), int)
        for k in range(S):
            m = s == k
            idx[m] = self.gold[k][rng.integers(0, len(self.gold[k]), m.sum())]
        return idx


_W = None


def world():
    global _W
    if _W is None:
        _W = World()
    return _W


def gen(w: World, cfg: Cfg, family, dose, seed):
    """One run: returns production stream (for aggregate baselines) + anchor events (for DRIFTWATCH)."""
    fid = (FAMILIES.index(family) + 1) if family else 0
    rng = np.random.default_rng([seed, fid, dose])
    aff = rng.choice(S, 2, replace=False)  # topic-local model drift (narrow fine-tune)
    sp = int(rng.integers(S))  # population-drift stratum
    mshift = DOSE["model"][dose] if family == "model" else 0.0
    hard = DOSE["population"][dose] if family == "population" else 0.0
    q = DOSE["context"][dose] if family == "context" else 0.0
    jafter = w.meas[dose] if family == "measurement" else "j0"
    T, ts = cfg.T, cfg.t_star
    post = SHARES.copy()
    if family == "population":  # traffic shifts away from the drifting stratum (Simpson-type masking)
        post[sp] *= 0.4
        post = post / post.sum()

    def mvec(s, act):
        return mshift * act * np.isin(s, aff)

    # ---- production stream, scored by the *production* judge J_t ----
    t = np.arange(T)
    act = t >= ts
    s_t = np.where(act, np.searchsorted(np.cumsum(post), rng.random(T)), np.searchsorted(np.cumsum(SHARES), rng.random(T)))
    s_t = np.minimum(s_t, S - 1)
    drop = rng.random(T) < q * act
    u = np.clip(U0 + mvec(s_t, act) + hard * act * (s_t == sp) + UPLIFT * drop, 0, 0.95)
    idx = w.draw(rng, s_t, u)
    cs, ca = w.score["score_j0"], w.score[THR[jafter][0]]
    prod_y = np.where(act, w.flag[jafter][idx], w.flag["j0"][idx]).astype(float)
    prod_sc = np.where(act, ca[idx] if THR[jafter][0] == "score_j0" else w.score["score_v2"][idx], cs[idx])
    # ---- anchor events (frozen prompts replayed at rate r), every stratum served round-robin ----
    ne = int(T * cfg.r)
    te = np.linspace(0, T - 1, ne).astype(int)
    es, ea = np.arange(ne) % S, te >= ts
    f0 = w.flag["j0"]

    def y(uv):
        return f0[w.draw(rng, es, np.clip(uv, 0, 0.95))].astype(float)

    base = y(np.full(ne, U0))  # cached baseline system (M0,C0): independent draws, no live call
    liveM = y(U0 + mvec(es, ea))  # live model, frozen pipeline
    dropC = rng.random(ne) < q * ea
    liveC = y(U0 + mvec(es, ea) + UPLIFT * dropC)  # live model + live pipeline
    dropP = rng.random(ne) < q * ea
    prodA = y(U0 + mvec(es, ea) + UPLIFT * dropP + hard * ea * (es == sp))  # live production prompts, J0-rescored
    gi = w.gold_draw(rng, es)
    gerr = np.where(ea, w.flag[jafter][gi] != w.label[gi], w.flag["j0"][gi] != w.label[gi]).astype(float)
    return dict(family=family, dose=dose, seed=seed, t_star=ts, T=T, prod_y=prod_y, prod_sc=prod_sc, te=te, es=es,
                dM=liveM - base, dC=liveC - liveM, dP=prodA - liveC, dSys=prodA - base, hash=dropC.astype(float),
                gerr=gerr, aff=aff.tolist(), sp=sp)


def run_dw(run, w, groups=GROUPS[5], strategy="agrapa", active=("M", "C", "P", "J", "H"), window=1500, alpha=0.05,
           trace=False, two_way=False):
    """Replay a run through the alarm policy. two_way=True = the Li-style baseline: pooled, system-vs-judge only."""
    if two_way:
        groups, active = GROUPS[1], ("M", "J")
    G = max(groups) + 1
    p0 = [float(np.mean([w.p0_gold[s] for s in range(S) if groups[s] == g])) for g in range(G)]
    pol = AlarmPolicy(G, alpha, window, p0_gold=p0, strategy=strategy, active=active)
    dm = run["dSys"] if two_way else run["dM"]
    tr = {k: [] for k in "MCPJ"}
    for e, (t, s) in enumerate(zip(run["te"], run["es"])):
        g = groups[s]
        if "M" in active:
            pol.update("M", g, dm[e], t)
        if "C" in active:
            pol.update("C", g, run["dC"][e], t)
        if "P" in active:
            pol.update("P", g, run["dP"][e], t)
        if "J" in active:
            pol.update("J", g, run["gerr"][e], t)
        if "H" in active:
            pol.update("H", 0, run["hash"][e], t)
        if trace:
            for k in "MCPJ":
                tr[k].append(max([ep.logE for (kk, _), ep in pol.eps.items() if kk == k] or [0.0]))
    return {"cross": {f"{k}{s}": int(t) for (k, s), t in pol.crossings.items()}, "trace": tr if trace else None,
            "K": pol.K}


def crossings_of(rec):
    return {(k[0], int(k[1:])): t for k, t in rec["cross"].items()}


# ------------------------- aggregate baselines (production stream, judged by J_t) -------------------------
def stat_z(y, burn, wd=1000):
    p0 = y[:burn].mean()
    cs = np.concatenate([[0.0], np.cumsum(y)])
    m = (cs[wd:] - cs[:-wd]) / wd
    tt = np.arange(wd - 1, len(y))
    k = tt >= burn
    return tt[k], (m[k] - p0) / np.sqrt(max(p0 * (1 - p0), 1e-9) / wd)


def stat_cusum(y, burn, k=0.03):
    p0 = y[:burn].mean()
    P = np.cumsum(y[burn:] - p0 - k)
    return np.arange(burn, len(y)), P - np.minimum.accumulate(np.minimum(P, 0.0))


def stat_ks(sc, burn, wd=1000, step=100):
    ref, tt, d = sc[:burn], [], []
    for te in range(burn + wd, len(sc) + 1, step):
        tt.append(te - 1)
        d.append(ks_2samp(ref, sc[te - wd:te]).statistic)
    return np.array(tt), np.array(d)


def baseline_stats(run, burn):
    return {"z": stat_z(run["prod_y"], burn), "cusum": stat_cusum(run["prod_y"], burn),
            "ks": stat_ks(run["prod_sc"], burn)}


def first_exceed(tt_stat, thr):
    tt, st = tt_stat
    i = np.flatnonzero(st > thr)
    return int(tt[i[0]]) if len(i) else None
