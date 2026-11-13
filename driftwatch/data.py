"""Data + judges. Uses PKU-SafeRLHF (73.9k prompts x 2 responses, human safe/unsafe labels, 19 harm categories).

prep(): flatten -> strata (top-4 harm categories + other) -> split by prompt (60% judge-train / 40% replay pool)
        -> train judge J0 (word TF-IDF + LR) and a different judge v2 (char TF-IDF + LR)
        -> score EVERY pool response once and cache (the "cached-generation replay" of paper D1).
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / os.environ.get("DW_RESULTS", "results")
POOL = DATA / os.environ.get("DW_POOL", "pool.npz")
SEED = 0


def _cats(v):
    if isinstance(v, str):
        v = ast.literal_eval(v)
    return [k for k, b in dict(v).items() if b]


def _safe(v):
    return v == "True" if isinstance(v, str) else bool(v)


def load_rows():
    d = pd.read_parquet(DATA / "pku_train.parquet")
    rows = []
    for pid, r in enumerate(d.itertuples(index=False)):
        for i in (0, 1):
            rows.append((pid, getattr(r, f"response_{i}"), not _safe(getattr(r, f"is_response_{i}_safe")),
                         _cats(getattr(r, f"response_{i}_harm_category"))))
    return pd.DataFrame(rows, columns=["pid", "text", "unsafe", "cats"])


def prep(n_gold=50):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    RES.mkdir(exist_ok=True)
    df = load_rows()
    freq = pd.Series([c for cs in df.cats[df.unsafe] for c in cs]).value_counts()
    top = list(freq.index[:4])
    names = top + ["other_harm"]

    def prompt_stratum(cs_union):
        for i, c in enumerate(top):
            if c in cs_union:
                return i
        return 4 if cs_union else -1  # -1: both responses safe (benign prompt) -> not in the harm strata

    pu = df.groupby("pid").cats.apply(lambda s: set(c for cs in s for c in cs))
    pstrat = pu.map(prompt_stratum)
    df["stratum"] = df.pid.map(pstrat)
    rng = np.random.default_rng(SEED)
    is_train = (df.pid.map(lambda p: (p * 2654435761) % 10) < 6).values  # split by prompt: no leakage
    tr, pool = df[is_train], df[(~is_train) & (df.stratum >= 0)].reset_index(drop=True)

    # --- judge J0: word TF-IDF + logistic regression (the deployed certifier) ---
    v0 = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150_000, sublinear_tf=True)
    X = v0.fit_transform(tr.text)
    j0 = LogisticRegression(C=4.0, max_iter=300, solver="liblinear").fit(X, tr.unsafe)
    s0 = j0.decision_function(v0.transform(pool.text))
    # --- judge v2: a different model family (char n-grams, smaller train set) = the "silent version swap" ---
    sub = tr.sample(40_000, random_state=SEED)
    v2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=5, max_features=100_000, sublinear_tf=True)
    j2 = LogisticRegression(C=2.0, max_iter=300, solver="liblinear").fit(v2.fit_transform(sub.text), sub.unsafe)
    s2 = j2.decision_function(v2.transform(pool.text))

    # --- gold anchors: balanced, frozen human labels ---
    gold = np.zeros(len(pool), bool)
    for s in range(5):
        for lab in (False, True):
            idx = np.flatnonzero((pool.stratum == s).values & (pool.unsafe == lab).values)
            gold[rng.choice(idx, n_gold, replace=False)] = True

    y = pool.unsafe.values
    np.savez_compressed(DATA / "pool.npz", stratum=pool.stratum.values.astype(np.int8), label=y,
                        score_j0=s0.astype(np.float32), score_v2=s2.astype(np.float32), gold=gold)
    m = {"strata": names, "n_pool": int(len(pool)), "n_train": int(len(tr)),
         "pool_per_stratum": [int((pool.stratum == s).sum()) for s in range(5)],
         "unsafe_rate_pool": float(y.mean()),
         "auroc_j0": float(roc_auc_score(y, s0)), "acc_j0": float(((s0 > 0) == y).mean()),
         "auroc_v2": float(roc_auc_score(y, s2)), "acc_v2": float(((s2 > 0) == y).mean()),
         "acc_j0_per_stratum": [float(((s0 > 0) == y)[(pool.stratum == s).values].mean()) for s in range(5)]}
    (RES / "judge_metrics.json").write_text(json.dumps(m, indent=1))
    return m
