# DRIFTWATCH results (auto-generated; every number below is computed from results/*.json)

Horizon T=30000 production interactions, drift injected at t*=10000, anchor rate r=0.2, alpha=0.05. 160 runs = 4 families x 3 doses x 10 seeds + 40 null runs.

## Judges (trained on PKU-SafeRLHF response text; replay pool = held-out prompts)

- J0 (word TF-IDF+LR): acc 0.862, AUROC 0.918; v2 (char TF-IDF+LR): acc 0.838, AUROC 0.895
- Strata: ['Cybercrime', 'Economic Crime', 'Privacy Violation', 'Mental Manipulation', 'other_harm']  (replay pool sizes [11496, 5550, 2784, 5104, 9980])
- Gold-anchor error of each judge in force after a measurement drift: j0=0.192, v2=0.218, strictA=0.228, strictB=0.294  (dose order: ['v2', 'strictA', 'strictB'])

## Validity: false alarms on drift-free streams (nominal alpha=0.05 over the whole horizon)

| method | false-alarm rate | n null runs |
|---|---|---|
| Rolling z-test, uncalibrated (z>1.96) | 0.855 | 200 |
| Rolling z, calibrated | 0.050 (by construction) | 200 |
| KS, calibrated | 0.050 (by construction) | 200 |
| CUSUM, calibrated | 0.050 (by construction) | 200 |
| DRIFTWATCH variant `dw5` | 0.030 | 200 |
| DRIFTWATCH variant `dw1` | 0.020 | 200 |
| DRIFTWATCH variant `li2` | 0.035 | 200 |
| DRIFTWATCH variant `ons` | 0.015 | 200 |
| DRIFTWATCH variant `fixed` | 0.040 | 200 |
| DRIFTWATCH `dw5` on the 40 experiment null runs | 0.000 | 40 |

## Table 1: median detection delay in interactions (misses right-censored at the horizon); detection rate in brackets

Baselines are calibrated to the same horizon false-alarm probability (alpha).

| method | model | context | population | measurement |
|---|---|---|---|---|
| Rolling z-test | 1850 (90%) | 437 (100%) | 20000 (27%) | 478 (77%) |
| Sliding-window KS | 2599 (80%) | 499 (97%) | 20000 (20%) | 20000 (37%) |
| CUSUM (aggregate) | 2124 (97%) | 274 (100%) | 20000 (33%) | 285 (77%) |
| E-process, 1 stratum | 15708 (63%) | 61 (97%) | 20000 (23%) | 11405 (80%) |
| Two-way e-process (Li-style) | 12802 (67%) | 4606 (97%) | 20000 (17%) | 8800 (80%) |
| DRIFTWATCH (5 strata) | 13376 (67%) | 88 (100%) | 17873 (57%) | 10870 (90%) |

## Table 2: attribution (rows = injected truth, columns = verdict, window W=1500)

| injected \ verdict | model | context | population | measurement | multiple | none |
|---|---|---|---|---|---|---|
| model | 20 | 0 | 0 | 0 | 0 | 10 |
| context | 0 | 30 | 0 | 0 | 0 | 0 |
| population | 0 | 0 | 17 | 1 | 0 | 12 |
| measurement | 0 | 0 | 0 | 27 | 0 | 3 |
| none (null) | 0 | 0 | 0 | 0 | 0 | 40 |

4-way attribution accuracy (DRIFTWATCH): **78.3%** over 120 drift runs (chance 25%).
Two-way accuracy of the Li-style baseline (system vs judge only): 65.8%. DRIFTWATCH collapsed to the same 2-way question: 78.3%.

- model: DRIFTWATCH correct on 67%, li2 correct (2-way) on 67%
- context: DRIFTWATCH correct on 100%, li2 correct (2-way) on 100%
- population: DRIFTWATCH correct on 57%, li2 correct (2-way) on 17%
- measurement: DRIFTWATCH correct on 90%, li2 correct (2-way) on 80%

## Masking (RQ3): runs where a stratum alarm fires after t* but the aggregate monitor never alarms

| family | masked vs (z, KS, CUSUM and 1-stratum e-process) | masked vs rolling z only |
|---|---|---|
| model | 0.0% | 3.3% |
| context | 0.0% | 0.0% |
| population | 20.0% | 30.0% |
| measurement | 0.0% | 20.0% |

### Stratification granularity sweep (Figure 1)

| strata | masking rate (vs 1-stratum e-process, all drift runs) | median delay (all drift runs) |
|---|---|---|
| 1 | 0.0% | 12883 |
| 2 | 6.7% | 9574 |
| 3 | 8.3% | 10492 |
| 5 | 13.3% | 10532 |

## Ablations

Betting strategy (median delay, model family; null false-alarm rate): dw5: 13376 / FA 0.030, ons: 10787 / FA 0.015, fixed: 11977 / FA 0.040

Layers on context drift (hash layer = Layer 3 context-integrity signal): behavioural only 9442 (87% detected) vs with hash 88 (100%).

Guard window W vs attribution accuracy: W=0: 78%, W=100: 78%, W=300: 78%, W=600: 78%, W=1000: 78%, W=1500: 78%, W=3000: 78%, W=6000: 78%

## Cost (RQ5): anchor rate r vs delay

| r | live system calls per 1k interactions | judge calls per 1k | median delay model | context | population |
|---|---|---|---|---|---|
| 0.02 | 40 | 100 | 20000 | 1819 | 20000 |
| 0.05 | 100 | 250 | 20000 | 666 | 20000 |
| 0.1 | 200 | 500 | 20000 | 328 | 20000 |
| 0.2 | 400 | 1000 | 11842 | 171 | 17360 |
| 0.4 | 800 | 2000 | 6940 | 68 | 8529 |

## Pre-registered predictions (preregistration.md)

- P1 DRIFTWATCH beats all aggregate baselines on model drift: **FALSIFIED** (DW 13376 vs best baseline 1850).
- P2 attribution accuracy > 80% on single-source injections: **INCONCLUSIVE (60-80%)** (78.3%).
- P3 masking rate > 20% on population-drift runs: **INCONCLUSIVE (5-20%)** (20.0%).
- P4 context drift detected almost entirely by the constraint-survival signal: **SUPPORTED** (behavioural-only 9442 vs with hash 88).
- P5 camouflage vs public anchors: **NOT EVALUATED** (camouflage partition requires an evaluation-aware LLM; out of scope here).
