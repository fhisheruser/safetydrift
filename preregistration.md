# Pre-registration (written BEFORE the first analysed run; amendments below, dated, never edited in place)

Protocol (fixed): 4 drift families x 3 dose levels x 10 seeds + 40 null runs = 160 runs; horizon T=30000 production
interactions, injection at t*=10000, anchor rate r=0.2, alpha=0.05, equal alpha split over K=21 streams
(5 strata x {M,C,P,J} + 1 context-hash), guard window W=1500, betting strategy aGRAPA.
Doses: model +0.10/+0.20/+0.30 unsafe-probability in 2 random strata; population hardness +0.10/+0.20/+0.30 in 1 stratum with
its traffic share x0.4; context constraint-drop probability 0.25/0.50/0.75 (x0.40 uplift); measurement = three judges ordered by gold error.
Baselines (z-test, KS, CUSUM, Li-style two-way e-process) are calibrated on 200 separate null runs (seeds 1000+) to the same
horizon false-alarm probability alpha. Seeds 0-9 = experiment; calibration seeds are disjoint. No tuning on experiment seeds.

Predictions and what falsifies them (paper §8.1):
- P1 DRIFTWATCH detects model drift with shorter median delay than all aggregate baselines. Falsified if any baseline (incl. two-way e-process) matches or beats it.
- P2 4-way attribution accuracy > 80% on single-source injections. Falsified below 60%.
- P3 Masking rate > 20% on population-drift runs. Falsified below 5%.
- P4 Context drift is detected almost entirely by the constraint-survival signal. Falsified if the behavioural-only variant detects it equally fast.
- P5 Camouflage vs public anchors differ. NOT TESTABLE in this implementation (no evaluation-aware LLM in the loop).

## Amendments
(none)
- 2026-11-13: the identical protocol (same doses, seeds, alpha, K, W, calibration seeds) was re-run with a fine-tuned MiniLM-L12 safety judge as the production judge J0 and the TF-IDF word judge as the swapped judge for the measurement-drift family. Results are kept separately in `results_real/`; the original TF-IDF-judge results in `results/` are unchanged.
