# DRIFTWATCH results (auto-generated; every number below is computed from results/*.json)

Horizon T=30000 production interactions, drift injected at t*=10000, anchor rate r=0.2, alpha=0.05. 160 runs = 4 families x 3 doses x 10 seeds + 40 null runs.

## Judges (trained on PKU-SafeRLHF response text; replay pool = held-out prompts)

- J0 (microsoft/MiniLM-L12-H384-uncased): acc 0.882, AUROC 0.946; v2 (word TF-IDF+LR): acc 0.862, AUROC 0.918
- Strata: ['Cybercrime', 'Economic Crime', 'Privacy Violation', 'Mental Manipulation', 'other_harm']  (replay pool sizes [11496, 5550, 2784, 5104, 9980])
- Gold-anchor error of each judge in force after a measurement drift: j0=0.162, strictA=0.176, v2=0.192, strictB=0.192  (dose order: ['strictA', 'v2', 'strictB'])

## Validity: false alarms on drift-free streams (nominal alpha=0.05 over the whole horizon)

| method | false-alarm rate | n null runs |
|---|---|---|
| Rolling z-test, uncalibrated (z>1.96) | 0.870 | 200 |
| Rolling z, calibrated | 0.050 (by construction) | 200 |
| KS, calibrated | 0.050 (by construction) | 200 |
| CUSUM, calibrated | 0.050 (by construction) | 200 |
| DRIFTWATCH variant `dw5` | 0.015 | 200 |
| DRIFTWATCH variant `dw1` | 0.020 | 200 |
| DRIFTWATCH variant `li2` | 0.030 | 200 |
| DRIFTWATCH variant `ons` | 0.020 | 200 |
| DRIFTWATCH variant `fixed` | 0.020 | 200 |
| DRIFTWATCH `dw5` on the 40 experiment null runs | 0.000 | 40 |

## Table 1: median detection delay in interactions (misses right-censored at the horizon); detection rate in brackets

Baselines are calibrated to the same horizon false-alarm probability (alpha).

| method | model | context | population | measurement |
|---|---|---|---|---|
| Rolling z-test | 904 (90%) | 334 (97%) | 20000 (43%) | 599 (97%) |
| Sliding-window KS | 2699 (77%) | 449 (100%) | 20000 (37%) | 20000 (37%) |
| CUSUM (aggregate) | 784 (87%) | 193 (97%) | 20000 (47%) | 386 (97%) |
| E-process, 1 stratum | 14070 (70%) | 61 (100%) | 20000 (27%) | 18526 (53%) |
| Two-way e-process (Li-style) | 11992 (60%) | 3476 (97%) | 20000 (30%) | 15736 (67%) |
| DRIFTWATCH (5 strata) | 10974 (70%) | 88 (93%) | 12135 (80%) | 17508 (63%) |

## Table 2: attribution (rows = injected truth, columns = verdict, window W=1500)

| injected \ verdict | model | context | population | measurement | multiple | none |
|---|---|---|---|---|---|---|
| model | 22 | 1 | 0 | 0 | 0 | 7 |
| context | 1 | 28 | 1 | 0 | 0 | 0 |
| population | 0 | 0 | 24 | 0 | 0 | 6 |
| measurement | 0 | 0 | 0 | 19 | 0 | 11 |
| none (null) | 0 | 0 | 0 | 0 | 0 | 40 |

4-way attribution accuracy (DRIFTWATCH): **77.5%** over 120 drift runs (chance 25%).
Two-way accuracy of the Li-style baseline (system vs judge only): 65.0%. DRIFTWATCH collapsed to the same 2-way question: 80.0%.

- model: DRIFTWATCH correct on 73%, li2 correct (2-way) on 60%
- context: DRIFTWATCH correct on 93%, li2 correct (2-way) on 100%
- population: DRIFTWATCH correct on 80%, li2 correct (2-way) on 33%
- measurement: DRIFTWATCH correct on 63%, li2 correct (2-way) on 67%

## Masking (RQ3): runs where a stratum alarm fires after t* but the aggregate monitor never alarms

| family | masked vs (z, KS, CUSUM and 1-stratum e-process) | masked vs rolling z only |
|---|---|---|
| model | 0.0% | 3.3% |
| context | 0.0% | 0.0% |
| population | 20.0% | 43.3% |
| measurement | 0.0% | 0.0% |

### Stratification granularity sweep (Figure 1)

| strata | masking rate (vs 1-stratum e-process, all drift runs) | median delay (all drift runs) |
|---|---|---|
| 1 | 0.0% | 15238 |
| 2 | 14.2% | 11872 |
| 3 | 15.0% | 11910 |
| 5 | 20.0% | 11102 |

## Ablations

Betting strategy (median delay, model family; null false-alarm rate): dw5: 10974 / FA 0.015, ons: 8377 / FA 0.020, fixed: 8757 / FA 0.020

Layers on context drift (hash layer = Layer 3 context-integrity signal): behavioural only 6567 (87% detected) vs with hash 88 (93%).

Guard window W vs attribution accuracy: W=0: 78%, W=100: 78%, W=300: 78%, W=600: 78%, W=1000: 78%, W=1500: 78%, W=3000: 78%, W=6000: 78%

## Cost (RQ5): anchor rate r vs delay

| r | live system calls per 1k interactions | judge calls per 1k | median delay model | context | population |
|---|---|---|---|---|---|
| 0.02 | 40 | 100 | 20000 | 1819 | 20000 |
| 0.05 | 100 | 250 | 20000 | 666 | 20000 |
| 0.1 | 200 | 500 | 16192 | 328 | 20000 |
| 0.2 | 400 | 1000 | 10974 | 171 | 11785 |
| 0.4 | 800 | 2000 | 6086 | 68 | 6004 |

## Pre-registered predictions (preregistration.md)

- P1 DRIFTWATCH beats all aggregate baselines on model drift: **FALSIFIED** (DW 10974 vs best baseline 784).
- P2 attribution accuracy > 80% on single-source injections: **INCONCLUSIVE (60-80%)** (77.5%).
- P3 masking rate > 20% on population-drift runs: **INCONCLUSIVE (5-20%)** (20.0%).
- P4 context drift detected almost entirely by the constraint-survival signal: **SUPPORTED** (behavioural-only 6567 vs with hash 88).
- P5 camouflage vs public anchors: **NOT EVALUATED** (camouflage partition requires an evaluation-aware LLM; out of scope here).
