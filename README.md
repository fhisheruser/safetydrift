# SafetyDrift

Continuous monitoring for deployed language models that says **which part of the system made it less safe**: the model, the chat context, the user population, or the safety judge itself.

- **Live demo:** https://fhisheruser.github.io/safetydrift/
- **Paper draft:** *Attributing Hidden Safety Drift in Deployed Language Models: A Continuous, Anytime-Valid Monitoring Framework* (framework name: DRIFTWATCH)

Most drift monitors watch one aggregate safety score and raise an alarm when it moves. That alarm says nothing about what to do, because four different processes can move the score and each needs a different fix: roll back the model, pin the safety rules in the context pipeline, recalibrate for a new population, or re-pin the judge. DRIFTWATCH runs one anytime-valid betting test per cause and per harm topic, keeps the overall false-alarm probability under a fixed budget however long it runs, and returns a verdict naming the cause.

---

## Contents

| path | what it is |
|---|---|
| `index.html` | the interactive demo (runs entirely in the browser, no server) |
| `driftwatch/core.py` | the monitor: betting e-processes, alarm budget, attribution logic |
| `driftwatch/data.py` | data preparation, strata, the two TF-IDF judges, cached scoring |
| `driftwatch/train_judge.py` | fine-tuning the MiniLM transformer judge and re-scoring the pool |
| `driftwatch/sim.py` | drift-injection benchmark and the baseline monitors |
| `driftwatch/run.py`, `driftwatch/analyze.py` | calibration, the 160-run experiment, tables and figures |
| `tests/` | validity, power and attribution checks (`pytest -q`) |
| `preregistration.md` | protocol and predictions written before the first analysed run, with dated amendments |
| `results/` | benchmark with the TF-IDF judge as the production judge |
| `results_real/` | benchmark with the fine-tuned MiniLM judge as the production judge, plus the judge evaluation |
| `data/pool.npz`, `data/pool_real.npz` | cached judge scores for the 34,914 held-out replies (what the benchmark replays) |

## Running it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt     # Windows; use .venv/bin/pip elsewhere

# dataset: PKU-SafeRLHF (73,907 prompts, two human-labelled replies each)
curl -L -o data/pku_train.parquet \
  https://huggingface.co/api/datasets/PKU-Alignment/PKU-SafeRLHF/parquet/default/train/0.parquet

python -m driftwatch.run all                 # TF-IDF judges -> calibrate -> 160 runs -> results/
python -m driftwatch.train_judge             # fine-tune MiniLM (about 70 min on a 2 GB laptop GPU) -> data/pool_real.npz
DW_POOL=pool_real.npz DW_RESULTS=results_real python -m driftwatch.run all
python -m pytest -q
```

The benchmark replays cached judge scores, so after the judges exist a full 160-run experiment takes about ten minutes on a laptop CPU. The trained judge weights (130 MB) are not committed; `train_judge.py` regenerates them.

---

## Data

PKU-SafeRLHF, flattened to one row per reply: 147,814 replies with human safe/unsafe labels and harm categories. Prompts are split by hash, 60 % for training the judges and 40 % held out. The held-out replies whose prompt carries at least one harm category form the **replay pool**: 34,914 replies, 87.5 % of them labelled unsafe.

Strata (the paper's "topics") are the four most frequent harm categories plus a remainder bucket:

| stratum | replies in pool |
|---|---|
| Cybercrime | 11,496 |
| Economic Crime | 5,550 |
| Privacy Violation | 2,784 |
| Mental Manipulation | 5,104 |
| Other harm | 9,980 |

50 safe and 50 unsafe replies per stratum are frozen as **gold anchors** with human labels; they are what identifies judge drift.

## Safety judges: accuracy

Three judges were trained, all on the same 88,692 training replies and all evaluated on the 34,914 held-out replies from prompts they never saw. Predictions are taken at the default threshold (score > 0 means unsafe).

| judge | accuracy | AUROC | precision (unsafe) | recall (unsafe) | specificity (safe) |
|---|---|---|---|---|---|
| **MiniLM-L12 fine-tuned** (33 M params, 1 epoch, max length 160) | **0.882** | **0.946** | 0.976 | 0.887 | 0.846 |
| Word TF-IDF (1–2-grams) + logistic regression | 0.862 | 0.918 | 0.966 | 0.874 | 0.783 |
| Char TF-IDF (3–4-grams) + logistic regression, 40 k-row subset | 0.838 | 0.895 | — | — | — |

Confusion matrices (rows: human label, columns: judge prediction):

**MiniLM-L12**

| | predicted safe | predicted unsafe |
|---|---|---|
| truly safe (4,368) | 3,696 | 672 |
| truly unsafe (30,546) | 3,457 | 27,089 |

**Word TF-IDF + LR**

| | predicted safe | predicted unsafe |
|---|---|---|
| truly safe (4,368) | 3,422 | 946 |
| truly unsafe (30,546) | 3,858 | 26,688 |

Per-stratum accuracy / AUROC:

| stratum | MiniLM-L12 | Word TF-IDF |
|---|---|---|
| Cybercrime | 0.957 / 0.977 | 0.942 / 0.954 |
| Economic Crime | 0.909 / 0.958 | 0.896 / 0.937 |
| Privacy Violation | 0.877 / 0.939 | 0.847 / 0.912 |
| Mental Manipulation | 0.911 / 0.949 | 0.888 / 0.910 |
| Other harm | 0.767 / 0.897 | 0.744 / 0.860 |

Both judges err mostly by missing unsafe replies (false negatives), and both are weakest on the broad "other harm" bucket. Error rate on the 100 gold anchors per stratum: MiniLM 0.11, 0.12, 0.14, 0.17, 0.27; word TF-IDF 0.18, 0.12, 0.16, 0.21, 0.29. The measurement-drift family raises this by swapping the judge or tightening its threshold, and the gold anchors are what detects it. Full numbers: `results_real/judge_eval.json`.

---

## Benchmark

Protocol (fixed in `preregistration.md` before the first analysed run): horizon T = 30,000 production interactions, drift injected at t* = 10,000, anchor rate r = 0.2, α = 0.05 split equally over K = 21 streams (5 strata × {model, context, population, measurement} + 1 context-integrity stream), guard window W = 1,500, aGRAPA betting. Four drift families × 3 doses × 10 seeds + 40 null runs = **160 runs**. Baselines (rolling z-test, sliding-window KS, CUSUM, pooled two-way e-process) are calibrated on 200 separate null streams to the same horizon false-alarm probability. Comparing delays at unmatched false-alarm rates is the most common way drift papers overclaim, so every comparison below is at matched α.

The benchmark was run twice with the identical protocol: once with the TF-IDF judge as the production judge (`results/`), once with the fine-tuned MiniLM judge (`results_real/`). In each run the "swapped judge" of the measurement-drift family is a different model (char TF-IDF, and word TF-IDF respectively).

### False alarms on drift-free streams (200 null streams, nominal α = 0.05 over the whole horizon)

| monitor | TF-IDF judge | MiniLM judge |
|---|---|---|
| Rolling z-test, textbook threshold (z > 1.96) | 0.855 | 0.870 |
| Rolling z / KS / CUSUM, calibrated | 0.050 by construction | 0.050 by construction |
| DRIFTWATCH, 5 strata | **0.030** | **0.015** |
| DRIFTWATCH, 1 stratum | 0.020 | 0.020 |
| Two-way e-process | 0.035 | 0.030 |

The uncalibrated rolling z-test, which is what most teams actually run, false-alarms on roughly 85 % of streams where nothing changed. DRIFTWATCH stays under its budget at every stopping time, which is the anytime-validity guarantee.

### Median detection delay in interactions, detection rate in brackets (MiniLM judge)

Misses are right-censored at the horizon (20,000).

| monitor | model | context | population | measurement |
|---|---|---|---|---|
| Rolling z-test | 904 (90 %) | 334 (97 %) | 20,000 (43 %) | 599 (97 %) |
| Sliding-window KS | 2,699 (77 %) | 449 (100 %) | 20,000 (37 %) | 20,000 (37 %) |
| CUSUM | 784 (87 %) | 193 (97 %) | 20,000 (47 %) | 386 (97 %) |
| E-process, 1 stratum | 14,070 (70 %) | 61 (100 %) | 20,000 (27 %) | 18,526 (53 %) |
| Two-way e-process | 11,992 (60 %) | 3,476 (97 %) | 20,000 (30 %) | 15,736 (67 %) |
| **DRIFTWATCH, 5 strata** | 10,974 (70 %) | 88 (93 %) | **12,135 (80 %)** | 17,508 (63 %) |

Same table with the TF-IDF judge: DRIFTWATCH 13,376 (67 %) / 88 (100 %) / 17,873 (57 %) / 10,870 (90 %); rolling z-test 1,850 (90 %) / 437 (100 %) / 20,000 (27 %) / 478 (77 %).

### Attribution: which cause did the monitor name? (MiniLM judge)

Rows are the injected truth, columns the verdict. 30 runs per family, 40 null runs.

| injected ↓ / verdict → | model | context | population | measurement | multiple | none |
|---|---|---|---|---|---|---|
| model | **22** | 1 | 0 | 0 | 0 | 7 |
| context | 1 | **28** | 1 | 0 | 0 | 0 |
| population | 0 | 0 | **24** | 0 | 0 | 6 |
| measurement | 0 | 0 | 0 | **19** | 0 | 11 |
| none (null) | 0 | 0 | 0 | 0 | 0 | **40** |

4-way attribution accuracy **77.5 %** (chance 25 %), 3 wrong labels, 24 misses. The two-way baseline that only separates "system" from "judge" reaches 65.0 % on its own easier question.

With the TF-IDF judge: 78.3 % accuracy, 0 wrong labels, 26 misses (model 20/30, context 30/30, population 17/30, measurement 27/30, null 40/40 quiet).

### Masking: drift the aggregate monitor never sees

Share of runs where a per-stratum alarm fired after the change but neither the rolling z-test, KS, CUSUM nor the pooled 1-stratum e-process ever alarmed:

| family | TF-IDF judge | MiniLM judge |
|---|---|---|
| population | 20.0 % | 20.0 % |
| model / context / measurement | 0 % | 0 % |

Against the rolling z-test alone, 43.3 % of population-drift runs were masked with the MiniLM judge. Sweeping the number of strata from 1 to 5 raises the masking rate recovered from 0 % to 20 % (MiniLM) while the median delay falls from 15,238 to 11,102.

### Ablations (MiniLM judge)

- **Context-integrity signal.** Hashing whether the safety constraints survived compaction catches context drift in a median of **88** interactions, against 6,567 for the behavioural layer alone. It costs no extra model calls.
- **Betting strategy.** aGRAPA 10,974 / ONS 8,377 / fixed λ 8,757 median delay on model drift, all with null false-alarm rate ≤ 0.02.
- **Guard window.** Attribution accuracy is flat at 78 % for W from 0 to 6,000, because single-source injections rarely produce a second crossing.
- **Cost.** Delay against anchor rate r: at r = 0.1 (200 extra model calls per 1,000 interactions) model drift is caught in 16,192; at r = 0.4 (800 per 1,000) in 6,086. The paper's target of under 5 % overhead is not met at any rate that detects model drift within the horizon.

### Pre-registered predictions

| | prediction | TF-IDF judge | MiniLM judge |
|---|---|---|---|
| P1 | DRIFTWATCH beats all aggregate baselines on model drift | **falsified** (13,376 vs 1,850) | **falsified** (10,974 vs 784) |
| P2 | 4-way attribution accuracy > 80 % | inconclusive (78.3 %) | inconclusive (77.5 %) |
| P3 | masking rate > 20 % on population drift | inconclusive (20.0 %) | inconclusive (20.0 %) |
| P4 | context drift caught almost entirely by the constraint-survival signal | **supported** (88 vs 9,442) | **supported** (88 vs 6,567) |
| P5 | camouflage vs public anchors differ | not testable here | not testable here |

P1 fails for a structural reason: the aggregate baselines read every production interaction for free, while DRIFTWATCH only sees the anchor probes (one per five interactions), pays a variance doubling for paired differences, and splits α over 21 streams. Its advantage is on drift that aggregation hides (population drift: 80 % detected vs 43–47 % for the baselines) and on saying *why*. The protocol was not changed after seeing results.

---

## What is real and what is simulated

| component | status |
|---|---|
| Dataset, strata, gold anchors | real (PKU-SafeRLHF, human labels) |
| Safety judges | real, trained here: MiniLM-L12 fine-tune and two TF-IDF classifiers |
| Replay of cached replies and judge scores | real: every pool reply is scored once by every judge |
| Monitor, baselines, calibration | real, as in the paper |
| Drift injection | **simulated at the response-mix level**: model/context/population drift change the probability that the system emits an unsafe cached reply; measurement drift swaps the judge or its threshold. No language model was fine-tuned to produce the drift, so ground truth is exact but the drift is a caricature of production drift. |
| Representational layer, trajectory layer, camouflage anchors, online FDR | not implemented |

## Limitations

- Drift is injected, not induced by fine-tuning a deployed model. Producing real model drift (LoRA on Llama-3.1-8B / Qwen2.5-7B as the paper specifies) needs a GPU with far more than the 2 GB used here.
- Harm is never observed directly; every claim inherits the judge's errors, and both judges miss about 11 % of unsafe replies.
- The five strata are fixed in advance; drift in a topic nobody defined is invisible by construction.
- Under simultaneous drift the monitor reports "multiple" rather than a cause.

## Licence and data

Code under MIT. PKU-SafeRLHF is distributed by its authors under CC BY-NC 4.0 and is not redistributed here; the download command above fetches it from the Hugging Face Hub.
