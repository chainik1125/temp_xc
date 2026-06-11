"""verify_theory.py — direct tests of the sprint's theory propositions.

A) Coincidence-line overlap statistics (P4): for velocity pairs (y, y'), the
   distribution of #shared symbols between windows, as a function of the ratio
   r = y'/y mod M. Pure combinatorics (exact, all phases).

B) Pair-task double contrast (P3 + P5): two-class velocity tasks under both
   embeddings. Random embedding: accuracy should depend only on ratio orbit
   {r, 1/r}. Circle embedding: accuracy should depend on |y - y'| (frequency
   distance / Rayleigh). Architecture: TXC (H=128) + bag-of-symbols readout
   (mean-pooled token-SAE codes), 3 seeds each.

Run on GPU pod:  python verify_theory.py --out theory_out
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from fb_core import (DEVICE, TaskData, TaskSpec, TokenSAE, build_model,
                     fit_probe, oracle_accuracy, train_model)

M, W = 101, 16

# pairs grouped by ratio class (ratio r = y'/y mod M, orbit {r, 1/r})
PAIRS = [
    (1, 2), (2, 4), (8, 16), (16, 32),   # ratio 2
    (1, 3), (3, 9), (16, 48),            # ratio 3
    (1, 50), (2, 100),                   # ratio 50 (orbit {50, 99...}): scaled pair
    (3, 7),                              # generic ratio 36
    (3, 98),                             # SIGN pair {y, -y}: identical symbol
                                         # sets -> bag-of-symbols must fail
]


def ratio_of(y1, y2):
    return (y2 * pow(y1, -1, M)) % M


def overlap_stats(y1, y2, W=W, M=M):
    """Exact distribution over relative phase of #shared symbol slots between
    a velocity-y1 window and a velocity-y2 window."""
    counts = []
    t = np.arange(W)
    for db in range(M):
        # slots (t, s): y1*t - y2*s = db  ->  for each s, t = (db + y2 s)/y1
        s = np.arange(W)
        tt = ((db + y2 * s) * pow(int(y1), -1, M)) % M
        counts.append(int(np.sum(tt < W)))
    c = np.array(counts)
    return {"mean": float(c.mean()), "max": int(c.max()),
            "frac_ge4": float((c >= 4).mean()), "hist": np.bincount(c).tolist()}


def make_pair_spec(y1, y2, embedding):
    d = 8 if embedding == "circle" else 0
    return TaskSpec(f"pair_{y1}_{y2}_{embedding}", M=M, W=W, n_classes=2,
                    sigma=0.25, d=d, velocities=(y1, y2), a_loc_star=0.5,
                    embedding=embedding)


def run_pair(y1, y2, embedding, seed, steps=4000):
    spec = make_pair_spec(y1, y2, embedding)
    data = TaskData(spec, seed=seed)
    torch.manual_seed(31_337 + seed)
    out = {"pair": [y1, y2], "embedding": embedding, "seed": seed,
           "ratio": ratio_of(y1, y2)}
    # oracle
    Xte, yte, _ = data.sample(5000)
    out["oracle"] = oracle_accuracy(spec, Xte, yte, data.U, data.R)

    # TXC, capacity-limited
    model = build_model("txc", spec.d, spec.W, 128, 32).to(DEVICE)
    tr = train_model(model, data, steps=steps)
    Xtr, ytr, _ = data.sample(20000)
    with torch.no_grad():
        Ftr = model.window_codes(Xtr)
        Fte = model.window_codes(Xte)
    out["txc_fvu"] = tr.fvu
    out["txc_linear"] = fit_probe(Ftr, ytr, Fte, yte, 2)["acc_test"]

    # bag-of-symbols readout: mean-pooled token-SAE codes + MLP
    # (an explicit "smoothing detector": all order information destroyed)
    tok = TokenSAE(spec.d, 128, 2).to(DEVICE)
    tr2 = train_model(tok, data, steps=steps)
    with torch.no_grad():
        Btr = tok.encode(Xtr).mean(1)
        Bte = tok.encode(Xte).mean(1)
    out["bag_fvu"] = tr2.fvu
    out["bag_mlp"] = fit_probe(Btr, ytr, Bte, yte, 2, hidden=256)["acc_test"]
    print(json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory_out")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # A) overlap combinatorics
    ov = {}
    for (y1, y2) in PAIRS:
        ov[f"{y1}_{y2}"] = {"ratio": ratio_of(y1, y2),
                            **overlap_stats(y1, y2)}
    with open(os.path.join(args.out, "overlaps.json"), "w") as f:
        json.dump(ov, f, indent=1)
    print("overlaps:", json.dumps(ov, indent=0), flush=True)

    # B) pair tasks
    results = []
    for emb in ["random", "circle"]:
        for (y1, y2) in PAIRS:
            for seed in args.seeds:
                tag = f"pair_{y1}_{y2}_{emb}_s{seed}"
                path = os.path.join(args.out, tag + ".json")
                if os.path.exists(path):
                    continue
                r = run_pair(y1, y2, emb, seed, steps=args.steps)
                with open(path, "w") as f:
                    json.dump(r, f, indent=1)
                results.append(r)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
