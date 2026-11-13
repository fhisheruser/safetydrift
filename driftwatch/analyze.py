"""Tables, figures, pre-registered prediction checks. Every number in results/summary.md comes from results/*.json."""
from __future__ import annotations

import json

import numpy as np

from .core import attribute
from .data import RES
from .run import ALPHA, CFG
from .sim import FAMILIES, DOSE, crossings_of, world

BASE = ("z", "ks", "cusum")
NAME = {"z": "Rolling z-test", "ks": "Sliding-window KS", "cusum": "CUSUM (aggregate)", "dw1": "E-process, 1 stratum",
        "li2": "Two-way e-process (Li-style)", "dw5": "DRIFTWATCH (5 strata)"}
PAL = {"z": "#999999", "ks": "#BBBBBB", "cusum": "#666666", "li2": "#0072B2", "dw1": "#56B4E9", "dw5": "#D55E00"}


def alarm_time(rec, m):
    if m in BASE:
        return rec["bl"][m]
    c = rec["dw"][m]["cross"]
    return min(c.values()) if c else None


def outcome(rec, m):
    t = alarm_time(rec, m)
    if t is None:
        return "miss", CFG.T - CFG.t_star  # right-censored at the horizon
    return ("fa", None) if t < CFG.t_star else ("det", t - CFG.t_star)


def med_delay(recs, m):
    d = [outcome(r, m)[1] for r in recs if outcome(r, m)[0] != "fa"]
    return float(np.median(d)) if d else float("nan")


def det_rate(recs, m):
    return float(np.mean([outcome(r, m)[0] == "det" for r in recs])) if recs else float("nan")


def verdict_of(rec, m="dw5", W=1500):
    v = attribute(crossings_of(rec["dw"][m]), W)
    if v is None:
        return "none"
    return "system" if (m == "li2" and v.label == "model") else v.label


def analyze():
    runs = json.loads((RES / "runs.json").read_text())
    cal = json.loads((RES / "calibration.json").read_text())
    cost = json.loads((RES / "cost.json").read_text())
    trace = json.loads((RES / "trace.json").read_text())
    jm = json.loads((RES / "judge_metrics.json").read_text())
    w = world()
    by = {f: [r for r in runs if r["family"] == f] for f in FAMILIES}
    null = [r for r in runs if r["family"] is None]
    drift = [r for r in runs if r["family"]]
    L = []
    P = L.append

    P("# DRIFTWATCH results (auto-generated; every number below is computed from results/*.json)\n")
    P(f"Horizon T={CFG.T} production interactions, drift injected at t*={CFG.t_star}, anchor rate r={CFG.r}, alpha={ALPHA}. "
      f"{len(runs)} runs = 4 families x 3 doses x {len(by['model'])//3} seeds + {len(null)} null runs.\n")
    P("## Judges (trained on PKU-SafeRLHF response text; replay pool = held-out prompts)\n")
    P(f"- J0 ({jm.get('judge_j0', 'word TF-IDF+LR')}): acc {jm['acc_j0']:.3f}, AUROC {jm['auroc_j0']:.3f}; v2 ({jm.get('judge_v2', 'char TF-IDF+LR')}): acc {jm['acc_v2']:.3f}, AUROC {jm['auroc_v2']:.3f}")
    P(f"- Strata: {jm['strata']}  (replay pool sizes {jm['pool_per_stratum']})")
    P("- Gold-anchor error of each judge in force after a measurement drift: " +
      ", ".join(f"{n}={w.gold_err[n].mean():.3f}" for n in ["j0"] + w.meas) + f"  (dose order: {w.meas})\n")

    P("## Validity: false alarms on drift-free streams (nominal alpha=0.05 over the whole horizon)\n")
    P("| method | false-alarm rate | n null runs |\n|---|---|---|")
    P(f"| Rolling z-test, uncalibrated (z>1.96) | {cal['rolling_z_uncalibrated_fa(z>1.96)']:.3f} | {cal['n_null']} |")
    for m, lab in (("z", "Rolling z, calibrated"), ("ks", "KS, calibrated"), ("cusum", "CUSUM, calibrated")):
        P(f"| {lab} | {ALPHA:.3f} (by construction) | {cal['n_null']} |")
    for n, v in cal["dw_false_alarm_rate"].items():
        P(f"| DRIFTWATCH variant `{n}` | {v:.3f} | {cal['n_null']} |")
    P(f"| DRIFTWATCH `dw5` on the {len(null)} experiment null runs | {np.mean([outcome(r,'dw5')[0]!='miss' for r in null]):.3f} | {len(null)} |\n")

    P("## Table 1: median detection delay in interactions (misses right-censored at the horizon); detection rate in brackets\n")
    P("Baselines are calibrated to the same horizon false-alarm probability (alpha).\n")
    ms = ["z", "ks", "cusum", "dw1", "li2", "dw5"]
    P("| method | " + " | ".join(FAMILIES) + " |\n|---|" + "---|" * 4)
    T1 = {}
    for m in ms:
        row = []
        for f in FAMILIES:
            T1[(m, f)] = med_delay(by[f], m)
            row.append(f"{T1[(m,f)]:.0f} ({det_rate(by[f], m):.0%})")
        P(f"| {NAME[m]} | " + " | ".join(row) + " |")
    P("")

    P("## Table 2: attribution (rows = injected truth, columns = verdict, window W=1500)\n")
    cols = ["model", "context", "population", "measurement", "multiple", "none"]
    P("| injected \\ verdict | " + " | ".join(cols) + " |\n|---|" + "---|" * 6)
    ok = tot = 0
    for f in FAMILIES + (None,):
        rs = by[f] if f else null
        cnt = {c: sum(verdict_of(r) == c for r in rs) for c in cols}
        P(f"| {f or 'none (null)'} | " + " | ".join(str(cnt[c]) for c in cols) + " |")
        if f:
            ok += cnt[f]
            tot += len(rs)
    acc4 = ok / tot
    P(f"\n4-way attribution accuracy (DRIFTWATCH): **{acc4:.1%}** over {tot} drift runs (chance 25%).")
    ok2 = sum((verdict_of(r, 'li2') == ("measurement" if r["family"] == "measurement" else "system")) for r in drift)
    P(f"Two-way accuracy of the Li-style baseline (system vs judge only): {ok2/len(drift):.1%}. "
      f"DRIFTWATCH collapsed to the same 2-way question: "
      f"{np.mean([(verdict_of(r)=='measurement')==(r['family']=='measurement') and verdict_of(r) not in ('none',) for r in drift]):.1%}.\n")
    for f in FAMILIES:
        P(f"- {f}: DRIFTWATCH correct on {np.mean([verdict_of(r)==f for r in by[f]]):.0%}, li2 correct (2-way) on "
          f"{np.mean([verdict_of(r,'li2')==('measurement' if f=='measurement' else 'system') for r in by[f]]):.0%}")
    P("")

    P("## Masking (RQ3): runs where a stratum alarm fires after t* but the aggregate monitor never alarms\n")
    agg_never = lambda r: all(alarm_time(r, m) is None for m in ("z", "cusum", "ks", "dw1"))
    mk = lambda rs, m="dw5": float(np.mean([outcome(r, m)[0] == "det" and agg_never(r) for r in rs]))
    mkz = lambda rs: float(np.mean([outcome(r, "dw5")[0] == "det" and alarm_time(r, "z") is None for r in rs]))
    P("| family | masked vs (z, KS, CUSUM and 1-stratum e-process) | masked vs rolling z only |\n|---|---|---|")
    for f in FAMILIES:
        P(f"| {f} | {mk(by[f]):.1%} | {mkz(by[f]):.1%} |")
    P("")
    P("### Stratification granularity sweep (Figure 1)\n")
    P("| strata | masking rate (vs 1-stratum e-process, all drift runs) | median delay (all drift runs) |\n|---|---|---|")
    gran = {}
    for g, n in ((1, "dw1"), (2, "dw2"), (3, "dw3"), (5, "dw5")):
        mr = float(np.mean([outcome(r, n)[0] == "det" and alarm_time(r, "dw1") is None for r in drift]))
        gran[g] = (mr, med_delay(drift, n))
        P(f"| {g} | {mr:.1%} | {gran[g][1]:.0f} |")
    P("")

    P("## Ablations\n")
    P("Betting strategy (median delay, model family; null false-alarm rate): " + ", ".join(
        f"{n}: {med_delay(by['model'], n):.0f} / FA {cal['dw_false_alarm_rate'][n]:.3f}" for n in ("dw5", "ons", "fixed")))
    P(f"\nLayers on context drift (hash layer = Layer 3 context-integrity signal): behavioural only {med_delay(by['context'],'nohash'):.0f} "
      f"({det_rate(by['context'],'nohash'):.0%} detected) vs with hash {med_delay(by['context'],'dw5'):.0f} ({det_rate(by['context'],'dw5'):.0%}).")
    Ws = [0, 100, 300, 600, 1000, 1500, 3000, 6000]
    wacc = [float(np.mean([(lambda v: v is not None and (v.label == r["family"]))(attribute(crossings_of(r["dw"]["dw5"]), W)) for r in drift])) for W in Ws]
    P("\nGuard window W vs attribution accuracy: " + ", ".join(f"W={W}: {a:.0%}" for W, a in zip(Ws, wacc)) + "\n")

    P("## Cost (RQ5): anchor rate r vs delay\n")
    P("| r | live system calls per 1k interactions | judge calls per 1k | median delay model | context | population |\n|---|---|---|---|---|---|")
    rs_ = sorted({c["r"] for c in cost})
    cost_pts = {}
    for r in rs_:
        cr = [c for c in cost if c["r"] == r]
        d = {f: float(np.median([(lambda t: (CFG.T - CFG.t_star) if t is None else max(t - CFG.t_star, 0))(alarm_time(c, "dw5")) for c in cr if c["family"] == f])) for f in ("model", "context", "population")}
        cost_pts[r] = d
        P(f"| {r} | {2*r*1000:.0f} | {5*r*1000:.0f} | {d['model']:.0f} | {d['context']:.0f} | {d['population']:.0f} |")
    P("")

    P("## Pre-registered predictions (preregistration.md)\n")
    p1 = T1[("dw5", "model")] < min(T1[(m, "model")] for m in ("z", "ks", "cusum", "li2"))
    P(f"- P1 DRIFTWATCH beats all aggregate baselines on model drift: **{'SUPPORTED' if p1 else 'FALSIFIED'}** "
      f"(DW {T1[('dw5','model')]:.0f} vs best baseline {min(T1[(m,'model')] for m in ('z','ks','cusum','li2')):.0f}).")
    p2 = "SUPPORTED" if acc4 > 0.8 else ("FALSIFIED" if acc4 < 0.6 else "INCONCLUSIVE (60-80%)")
    P(f"- P2 attribution accuracy > 80% on single-source injections: **{p2}** ({acc4:.1%}).")
    m3 = mk(by["population"])
    P(f"- P3 masking rate > 20% on population-drift runs: **{'SUPPORTED' if m3 > 0.2 else ('FALSIFIED' if m3 < 0.05 else 'INCONCLUSIVE (5-20%)')}** ({m3:.1%}).")
    p4 = med_delay(by["context"], "dw5") < 0.5 * med_delay(by["context"], "nohash")
    P(f"- P4 context drift detected almost entirely by the constraint-survival signal: **{'SUPPORTED' if p4 else 'FALSIFIED'}** "
      f"(behavioural-only {med_delay(by['context'],'nohash'):.0f} vs with hash {med_delay(by['context'],'dw5'):.0f}).")
    P("- P5 camouflage vs public anchors: **NOT EVALUATED** (camouflage partition requires an evaluation-aware LLM; out of scope here).\n")
    (RES / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    figures(by, runs, trace, gran, cost_pts, T1)


def figures(by, runs, trace, gran, cost_pts, T1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    out = RES / "figures"
    out.mkdir(exist_ok=True)
    drift = [r for r in runs if r["family"]]
    cap = CFG.T - CFG.t_star

    def save(fig, n):
        fig.savefig(out / f"{n}.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / f"{n}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Fig 1: masking vs granularity
    g = sorted(gran)
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    ax[0].plot(g, [100 * gran[k][0] for k in g], "o-", color=PAL["dw5"])
    ax[0].set(xlabel="strata monitored |S|", ylabel="% runs masked from aggregate", xticks=g)
    ax[1].plot(g, [gran[k][1] for k in g], "s-", color="#0072B2")
    ax[1].set(xlabel="strata monitored |S|", ylabel="median delay (interactions)", xticks=g)
    fig.suptitle("Stratification recovers drift that aggregation masks", fontsize=10)
    save(fig, "fig1_masking_vs_granularity")

    # Fig 2: delay vs dose per family
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, f in zip(axs, FAMILIES):
        for m in ("z", "ks", "cusum", "li2", "dw5"):
            y = [med_delay([r for r in by[f] if r["dose"] == d], m) for d in range(3)]
            ax.plot(range(3), y, "o-" if m == "dw5" else "o--", color=PAL[m], lw=2.2 if m == "dw5" else 1.2,
                    label=NAME[m], zorder=3 if m == "dw5" else 2)
        ax.axhline(cap, color="k", lw=0.5, ls=":")
        ax.set(title=f, xlabel="dose level", xticks=range(3), xticklabels=["low", "mid", "high"])
    axs[0].set_ylabel("median delay (interactions)")
    axs[0].text(0.02, cap * 0.93, "not detected", fontsize=7)
    axs[-1].legend(fontsize=7, frameon=False, loc="center right")
    save(fig, "fig2_delay_vs_dose")

    # Fig 3: e-process traces
    fig, ax = plt.subplots(figsize=(6, 3))
    cols = {"M": "#0072B2", "C": "#D55E00", "P": "#009E73", "J": "#CC79A7"}
    for k, lab in (("M", "model"), ("C", "context"), ("P", "population"), ("J", "measurement")):
        ax.plot(trace["te"], trace["trace"][k], color=cols[k], label=lab, lw=1.4)
    ax.axhline(trace["thr"], color="k", ls="--", lw=0.8)
    ax.axvline(trace["t_star"], color="k", ls=":", lw=0.8)
    ax.text(trace["t_star"], ax.get_ylim()[1] * 0.92, " injection", fontsize=8)
    ax.set(xlabel="production interactions", ylabel="max log-capital over strata", title="Context-drift run: only the context stream crosses")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "fig3_eprocess_traces")

    # Fig 4: cost-benefit
    fig, ax = plt.subplots(figsize=(5, 3.2))
    rs_ = sorted(cost_pts)
    ax.plot([200 * r / 10 for r in rs_], [np.mean(list(cost_pts[r].values())) for r in rs_], "o-", color=PAL["dw5"], label="DRIFTWATCH (vary r)")
    for r in rs_:
        ax.annotate(f"r={r}", (20 * r, np.mean(list(cost_pts[r].values()))), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for m in ("z", "ks", "cusum"):
        ax.axhline(np.mean([T1[(m, f)] for f in ("model", "context", "population")]), color=PAL[m], ls="--", lw=1, label=NAME[m])
    ax.set(xlabel="live-system overhead (% extra calls)", ylabel="mean delay, 3 families", title="Cost-benefit")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig4_cost_benefit")


if __name__ == "__main__":
    analyze()
