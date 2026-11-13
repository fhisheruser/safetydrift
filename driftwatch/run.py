"""Experiments + analysis.  python -m driftwatch.run [prep|calibrate|experiments|analyze|all]"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

from .data import POOL, RES, prep
from .sim import (DOSE, FAMILIES, GROUPS, Cfg, baseline_stats, crossings_of, first_exceed, gen, run_dw, world)

ALPHA = 0.05
CFG = Cfg()
N_SEEDS, N_NULL, N_CAL = 10, 40, 200
CONFIGS = {  # name -> run_dw kwargs  (all replayed on the SAME cached run)
    "dw5": {}, "dw3": {"groups": GROUPS[3]}, "dw2": {"groups": GROUPS[2]}, "dw1": {"groups": GROUPS[1]},
    "ons": {"strategy": "ons"}, "fixed": {"strategy": "fixed"}, "nohash": {"active": ("M", "C", "P", "J")},
    "li2": {"two_way": True},
}


def _work(a):
    fam, dose, seed, thr, cfg = a
    w = world()
    run = gen(w, cfg, fam, dose, seed)
    st = baseline_stats(run, cfg.burn)
    rec = {"family": fam, "dose": dose, "seed": seed, "aff": run["aff"], "sp": run["sp"]}
    if thr is None:  # calibration pass: keep max statistics
        rec["max"] = {k: float(v[1].max()) for k, v in st.items()}
    else:
        rec["bl"] = {k: first_exceed(st[k], thr[k]) for k in st}
    cfgs = CONFIGS if thr else {n: CONFIGS[n] for n in ("dw5", "dw1", "li2", "ons", "fixed")}
    rec["dw"] = {n: run_dw(run, w, **kw) for n, kw in cfgs.items()}
    return rec


def _pool(tasks):
    n = max(1, min(cpu_count() - 1, 8))
    with Pool(n) as p:
        return p.map(_work, tasks, chunksize=2)


def calibrate():
    t0 = time.time()
    recs = _pool([(None, 0, 1000 + i, None, CFG) for i in range(N_CAL)])
    thr = {k: float(np.quantile([r["max"][k] for r in recs], 1 - ALPHA, method="higher")) for k in ("z", "ks", "cusum")}
    fa = {n: float(np.mean([bool(r["dw"][n]["cross"]) for r in recs])) for n in recs[0]["dw"]}
    out = {"thr": thr, "n_null": N_CAL, "alpha": ALPHA, "dw_false_alarm_rate": fa,
           "rolling_z_uncalibrated_fa(z>1.96)": float(np.mean([r["max"]["z"] > 1.96 for r in recs]))}
    (RES / "calibration.json").write_text(json.dumps(out, indent=1))
    print("calibrated", out, f"{time.time()-t0:.0f}s")
    return out


def experiments():
    t0 = time.time()
    cal = json.loads((RES / "calibration.json").read_text())
    tasks = [(f, d, s, cal["thr"], CFG) for f in FAMILIES for d in range(3) for s in range(N_SEEDS)]
    tasks += [(None, 0, s, cal["thr"], CFG) for s in range(N_NULL)]
    recs = _pool(tasks)
    (RES / "runs.json").write_text(json.dumps(recs))
    print(f"{len(recs)} runs {time.time()-t0:.0f}s")
    # cost sweep: anchor interleave rate r vs delay (RQ5)
    cost = []
    for r in (0.02, 0.05, 0.1, 0.2, 0.4):
        cfg = Cfg(r=r)
        for f, d in (("model", 1), ("context", 0), ("population", 1)):
            for rec in _pool([(f, d, s, cal["thr"], cfg) for s in range(6)]):
                rec["r"] = r
                rec["dw"] = {"dw5": rec["dw"]["dw5"], "nohash": rec["dw"]["nohash"]}
                cost.append(rec)
    (RES / "cost.json").write_text(json.dumps(cost))
    # e-process traces for Figure 3
    w = world()
    run = gen(w, CFG, "context", 1, 123)
    (RES / "trace.json").write_text(json.dumps({"te": run["te"].tolist(), "t_star": CFG.t_star,
                                               "trace": run_dw(run, w, trace=True)["trace"], "thr": float(np.log(21 / ALPHA))}))
    print("sweeps done", f"{time.time()-t0:.0f}s")


def all_():
    RES.mkdir(exist_ok=True)
    prep() if not POOL.exists() else None
    calibrate()
    experiments()
    from .analyze import analyze
    analyze()


if __name__ == "__main__":
    {"prep": prep, "calibrate": calibrate, "experiments": experiments, "all": all_,
     "analyze": lambda: __import__("driftwatch.analyze", fromlist=["x"]).analyze()}[sys.argv[1] if len(sys.argv) > 1 else "all"]()
