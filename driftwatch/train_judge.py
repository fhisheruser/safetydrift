"""Fine-tune a real transformer safety judge on PKU-SafeRLHF and re-score the replay pool.

Produces data/pool_real.npz (same layout as pool.npz, but score_j0 = transformer, score_v2 = the TF-IDF word judge),
results_real/judge_metrics.json and results_real/judge_eval.json (confusion matrices, per-stratum metrics).
Then `DW_POOL=pool_real.npz DW_RESULTS=results_real python -m driftwatch.run all` re-runs the whole benchmark on it.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import DATA, ROOT, load_rows

MODEL = os.environ.get("DW_JUDGE_MODEL", "microsoft/MiniLM-L12-H384-uncased")
MAXLEN, EPOCHS, LR, SEED = 160, 1, 5e-5, 0
OUT = ROOT / "results_real"
LOG = OUT / "train.log"


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf8") as f:
        f.write(s + "\n")


def batches(tok, texts, labels, bs, shuffle, rng):
    idx = np.arange(len(texts))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        enc = tok([texts[k] for k in j], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        yield enc, (torch.tensor(labels[j]) if labels is not None else None)


@torch.no_grad()
def score(model, tok, texts, dev, bs):
    model.eval()
    out = []
    for enc, _ in batches(tok, texts, None, bs, False, None):
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            lg = model(**enc).logits.float()
        out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())  # logit(unsafe) - logit(safe)
    return np.concatenate(out)


def main():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device", dev, "model", MODEL)

    old = np.load(DATA / "pool.npz")
    df = load_rows()
    is_train = (df.pid.map(lambda p: (p * 2654435761) % 10) < 6).values  # identical prompt-level split to data.prep
    tr = df[is_train].reset_index(drop=True)
    # rebuild the pool exactly as prep() did (held-out prompts with at least one harm category)
    freq = pd.Series([c for cs in df.cats[df.unsafe] for c in cs]).value_counts()
    top = list(freq.index[:4])
    pu = df.groupby("pid").cats.apply(lambda s: set(c for cs in s for c in cs))
    pstrat = pu.map(lambda u: next((i for i, c in enumerate(top) if c in u), 4 if u else -1))
    df["stratum"] = df.pid.map(pstrat)
    pool = df[(~is_train) & (df.stratum >= 0)].reset_index(drop=True)
    assert len(pool) == len(old["label"]) and (pool.unsafe.values == old["label"]).all(), "pool mismatch"
    log(f"train rows {len(tr)}  pool rows {len(pool)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
    bs = 32 if dev.type == "cuda" else 16
    steps = EPOCHS * ((len(tr) + bs - 1) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.06, anneal_strategy="linear")
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    texts, labels = tr.text.tolist(), tr.unsafe.values.astype(np.int64)
    t0, step, run_loss = time.time(), 0, 0.0
    for ep in range(EPOCHS):
        model.train()
        for enc, y in batches(tok, texts, labels, bs, True, rng):
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = model(**enc, labels=y.to(dev)).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss = 0.98 * run_loss + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 100 == 0 or step == steps:
                el = time.time() - t0
                log(f"step {step}/{steps} loss {run_loss:.4f} elapsed {el/60:.1f}m eta {(steps-step)*el/step/60:.1f}m")
    model.save_pretrained(DATA / "judge_minilm")
    tok.save_pretrained(DATA / "judge_minilm")

    log("scoring pool")
    s_tf = score(model, tok, pool.text.tolist(), dev, bs * 4)
    y = pool.unsafe.values
    st = pool.stratum.values
    names = top + ["other_harm"]
    s_word = old["score_j0"]

    def ev(s):
        p = s > 0
        cm = confusion_matrix(y, p, labels=[False, True]).tolist()
        return {"acc": float((p == y).mean()), "auroc": float(roc_auc_score(y, s)),
                "precision_unsafe": float((p & y).sum() / max(p.sum(), 1)), "recall_unsafe": float((p & y).sum() / y.sum()),
                "specificity_safe": float((~p & ~y).sum() / (~y).sum()),
                "confusion": {"rows": "true [safe, unsafe]", "cols": "predicted [safe, unsafe]", "matrix": cm},
                "per_stratum": [{"stratum": names[k], "n": int((st == k).sum()), "acc": float((p == y)[st == k].mean()),
                                 "auroc": float(roc_auc_score(y[st == k], s[st == k]))} for k in range(5)]}

    evald = {"transformer_judge": {"model": MODEL, "train_rows": int(len(tr)), "epochs": EPOCHS, "max_len": MAXLEN,
                                   "device": str(dev), "train_minutes": round((time.time() - t0) / 60, 1), **ev(s_tf)},
             "tfidf_word_judge": ev(s_word), "n_pool": int(len(pool))}
    (OUT / "judge_eval.json").write_text(json.dumps(evald, indent=1))
    np.savez_compressed(DATA / "pool_real.npz", stratum=old["stratum"], label=y, score_j0=s_tf.astype(np.float32),
                        score_v2=s_word.astype(np.float32), gold=old["gold"])
    m = {"strata": names, "n_pool": int(len(pool)), "n_train": int(len(tr)), "judge_j0": MODEL, "judge_v2": "word TF-IDF+LR",
         "pool_per_stratum": [int((st == k).sum()) for k in range(5)], "unsafe_rate_pool": float(y.mean()),
         "auroc_j0": evald["transformer_judge"]["auroc"], "acc_j0": evald["transformer_judge"]["acc"],
         "auroc_v2": evald["tfidf_word_judge"]["auroc"], "acc_v2": evald["tfidf_word_judge"]["acc"],
         "acc_j0_per_stratum": [r["acc"] for r in evald["transformer_judge"]["per_stratum"]]}
    (OUT / "judge_metrics.json").write_text(json.dumps(m, indent=1))
    log("done", json.dumps({k: evald["transformer_judge"][k] for k in ("acc", "auroc", "confusion")}))


if __name__ == "__main__":
    main()
