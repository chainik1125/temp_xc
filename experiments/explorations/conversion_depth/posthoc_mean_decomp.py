"""POST-HOC (not preregistered) — aggregation-vs-order decomposition of g(ℓ).

Motivation, recorded before running (see RECORD § 4): the phase-4 EM
sweep shows g(ℓ) > 0 at mid depth, while the paper's § 5.3 verdict was
shuffle_gap ≈ 0 (order-insensitive codes). Both can be true if the
window's advantage over the single token is ORDER-FREE aggregation
(pooling more evidence of an ambient property), not temporal structure.

Decomposition: add one probe per cell — **window-mean linear** (linear
probe on the T-window MEAN, a position-symmetric reader = the order-free
linear ceiling). Then
    g_agg(ℓ)   = AUC(window mean)   − AUC(per-token)      [order-free part]
    g_order(ℓ) = AUC(window flatten) − AUC(window mean)   [order part]
with g(ℓ) = g_agg + g_order. The frozen window-flatten and per-token
numbers come from the § 2/§ 4 runs and are NOT recomputed.

This is a diagnostic DECOMPOSITION of an already-measured quantity —
same rows, same frozen probe hyperparameters, no target/metric change —
labeled post-hoc in the record.

Run:  .venv/bin/python -m experiments.explorations.conversion_depth.posthoc_mean_decomp
Writes results/posthoc_mean_decomp.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.conversion_depth.phase4_em_depth import (
    EM_DIR, N_FOLDS, NULL_SEED, POS_STRIDE, PROBE_HS as EM_PROBE_HS, T as T4,
)

CACHE_ROOT = Path("/workspace/conv_depth_caches")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "posthoc_mean_decomp.json"
T = 16


def gather_mean(acts_t, rows):
    w = torch.from_numpy(rows[:, 0])
    p = torch.from_numpy(rows[:, 1])
    d = acts_t.shape[-1]
    acc = torch.zeros((len(rows), d), dtype=torch.float32)
    for j in range(T):
        acc += acts_t[w, p - (T - 1) + j].float()
    return (acc / T).to(torch.float16)


def phase3(done):
    z = np.load(RES / "probe_rows.npz", allow_pickle=False)
    targets = ["ant_kw", "ant_bts", "is_bt"]
    for tag in ["base", "distill"]:
        cache = CACHE_ROOT / tag
        if not (cache / "meta.json").exists():
            continue
        meta = json.loads((cache / "meta.json").read_text())
        for k in meta["hs_capture"]:
            fkey = f"p3/{tag}/hs{k}"
            if all(f"{fkey}/{t}" in done for t in targets):
                continue
            acts_t = torch.from_numpy(np.ascontiguousarray(
                np.load(cache / f"hs{k}.npy", mmap_mode="r")))
            for tname in targets:
                key = f"{fkey}/{tname}"
                if key in done:
                    continue
                rtr = z[f"{tname}_train_rows"]
                rte = z[f"{tname}_test_rows"]
                ytr = torch.from_numpy(z[f"{tname}_train_y"])
                yte = torch.from_numpy(z[f"{tname}_test_y"])
                Xtr = gather_mean(acts_t, rtr)
                Xte = gather_mean(acts_t, rte)
                r = fit_probe(Xtr, ytr, Xte, yte, 2, class_weight=True)
                done[key] = {"window_mean_linear_auc": r["auc"],
                             "balacc_opt": r.get("balacc_opt")}
                print(f"[posthoc {key}] mean_auc={r['auc']:.3f}", flush=True)
                OUT.write_text(json.dumps(done, indent=1))
            del acts_t


def phase4(done):
    if not (EM_DIR / "meta.json").exists():
        return
    lens = np.load(EM_DIR / "lens.npy")
    labels = np.load(EM_DIR / "labels.npy")
    qids = np.load(EM_DIR / "qids.npy")
    rows, row_lab, row_q = [], [], []
    for ri in range(len(lens)):
        for p in range(T4 - 1, int(lens[ri]), POS_STRIDE):
            rows.append((ri, p))
            row_lab.append(labels[ri])
            row_q.append(qids[ri])
    rows = np.array(rows, dtype=np.int64)
    row_lab = np.array(row_lab, dtype=np.int64)
    row_q = np.array(row_q, dtype=np.int64)
    folds = [(row_q % N_FOLDS) != f for f in range(N_FOLDS)]
    for k in EM_PROBE_HS:
        key = f"p4/em/hs{k}"
        if key in done:
            continue
        acts_t = torch.from_numpy(np.ascontiguousarray(
            np.load(EM_DIR / f"hs{k}.npy")))
        w = torch.from_numpy(rows[:, 0])
        p = torch.from_numpy(rows[:, 1])
        d = acts_t.shape[-1]
        acc = torch.zeros((len(rows), d), dtype=torch.float32)
        for j in range(T4):
            acc += acts_t[w, p - (T4 - 1) + j].float()
        Xm = (acc / T4).to(torch.float16)
        del acts_t
        aucs = []
        for f, tr_mask in enumerate(folds):
            te_mask = ~tr_mask
            r = fit_probe(Xm[torch.from_numpy(tr_mask)],
                          torch.from_numpy(row_lab[tr_mask]),
                          Xm[torch.from_numpy(te_mask)],
                          torch.from_numpy(row_lab[te_mask]),
                          2, class_weight=True)
            aucs.append(r["auc"])
        done[key] = {"window_mean_linear_auc": float(np.mean(aucs)),
                     "auc_folds": aucs}
        print(f"[posthoc {key}] mean_auc={np.mean(aucs):.3f}", flush=True)
        OUT.write_text(json.dumps(done, indent=1))


def main():
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    phase4(done)
    phase3(done)
    OUT.write_text(json.dumps(done, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
