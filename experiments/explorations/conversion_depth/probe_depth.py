"""Phase 3 (probes) — g(ℓ) curves for base vs generator on the Ward stream.

Implements RECORD.md § 2 EXACTLY (frozen prereg): for each model
(base, distill), each capture point (hs0 + resid_post 0,2,…,30), each
target (ant_kw PRIMARY, ant_bts SECONDARY, is_bt COMPANION), the four
frozen probes (per-token linear, window linear T=16 right-edge, both
MLP-512 presence checks) + permutation nulls on the linear pair.

Probe rows are built ONCE per target (positions shared across all
layers and BOTH models — identical rows, so every difference is
attributable to the representation).

Run:  .venv/bin/python -m experiments.explorations.conversion_depth.probe_depth [base|distill|both]
Writes results/depth_probe_<tag>.json (+ probe-set row dump for
reproducibility, results/probe_rows.npz).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import (
    ceilings_for_target, fit_probe,
)

STREAM_DIR = Path("/workspace/conv_depth_caches/ward_stream")
CACHE_ROOT = Path("/workspace/conv_depth_caches")
HERE = Path(__file__).resolve().parent

T = 16                      # frozen window (right-edge)
DPLUS_LO, DPLUS_HI = 8, 13  # Ward D+ convention
NEG_BUFFER = 25
NEG_CAND_PER_TRACE = 60
NEG_PER_POS = 5
SPLIT_SEED = 7
NEG_SEED_TR, NEG_SEED_TE = 13, 14
NULL_SEED = 99
TAGS = {"base": "base", "distill": "distill"}


def load_sidecars():
    d = {}
    for name in ["map_ok", "is_bt", "dist_next_kw", "dist_next_btsent",
                 "dist_prev_kw", "dist_prev_btsent", "in_think",
                 "trace_idx", "win_start"]:
        d[name] = np.load(STREAM_DIR / f"{name}.npy")
    return d


def build_rows(sc):
    """Probe rows per target: dict target -> dict split -> (rows, y).

    rows = (w, p) pairs, y ∈ {0,1}. Split by trace (80/20, rng(7)).
    Negative recipe per bt_freq (see prereg § 2).
    """
    N, L = sc["map_ok"].shape
    traces = np.unique(sc["trace_idx"])
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(traces))
    test_traces = set(traces[perm[:len(traces) // 5]].tolist())

    elig = sc["map_ok"] & sc["in_think"]
    elig[:, :T] = False                       # p >= 16

    def dist_ok(dn, dp):
        dn = np.where(dn < 0, 10 ** 9, dn)
        dp = np.where(dp < 0, 10 ** 9, dp)
        return (dn > NEG_BUFFER) & (dp > NEG_BUFFER)

    specs = {
        "ant_kw": {
            "pos": (sc["dist_next_kw"] >= DPLUS_LO)
                   & (sc["dist_next_kw"] <= DPLUS_HI),
            "negok": dist_ok(sc["dist_next_kw"], sc["dist_prev_kw"]),
        },
        "ant_bts": {
            "pos": (sc["dist_next_btsent"] >= DPLUS_LO)
                   & (sc["dist_next_btsent"] <= DPLUS_HI),
            "negok": dist_ok(sc["dist_next_btsent"], sc["dist_prev_btsent"]),
        },
        "is_bt": {
            "pos": sc["is_bt"] == 1,
            "negok": dist_ok(sc["dist_next_btsent"], sc["dist_prev_btsent"]),
        },
    }

    out = {}
    for tname, spec in specs.items():
        out[tname] = {}
        for split, is_train in [("train", True), ("test", False)]:
            rng = np.random.default_rng(NEG_SEED_TR if is_train
                                        else NEG_SEED_TE)
            pos_rows, neg_rows = [], []
            for ti in traces:
                if (int(ti) in test_traces) == is_train:
                    continue
                wmask = sc["trace_idx"] == ti
                widx = np.where(wmask)[0]
                pm = spec["pos"][widx] & elig[widx]
                nm = spec["negok"][widx] & elig[widx] & ~spec["pos"][widx]
                pw, pp = np.where(pm)
                pos_rows.extend(zip(widx[pw].tolist(), pp.tolist()))
                nw, np_ = np.where(nm)
                cand = list(zip(widx[nw].tolist(), np_.tolist()))
                if cand:
                    take = rng.choice(len(cand),
                                      size=min(len(cand), NEG_CAND_PER_TRACE),
                                      replace=False)
                    neg_rows.extend(cand[i] for i in take)
            rng2 = np.random.default_rng(17)
            neg_idx = rng2.permutation(len(neg_rows))
            neg_rows = [neg_rows[i]
                        for i in neg_idx[:NEG_PER_POS * max(1, len(pos_rows))]]
            rows = np.array(pos_rows + neg_rows, dtype=np.int64)
            y = np.concatenate([np.ones(len(pos_rows), dtype=np.int64),
                                np.zeros(len(neg_rows), dtype=np.int64)])
            out[tname][split] = (rows, y)
            print(f"[rows] {tname}/{split}: {len(pos_rows)} pos, "
                  f"{len(neg_rows)} neg", flush=True)
    return out, sorted(test_traces)


def gather(acts_t, rows):
    """acts_t: torch fp16 (N, 128, d) in RAM. rows: (n, 2) (w, p).
    Returns X_tok (n, d) and X_win (n, T*d), both fp16 CPU."""
    w = torch.from_numpy(rows[:, 0])
    p = torch.from_numpy(rows[:, 1])
    X_tok = acts_t[w, p]
    n, d = X_tok.shape
    X_win = torch.empty((n, T, d), dtype=acts_t.dtype)
    for j in range(T):
        X_win[:, j] = acts_t[w, p - (T - 1) + j]
    return X_tok, X_win.reshape(n, T * d)


def main(which: str):
    sc = load_sidecars()
    rows, test_traces = build_rows(sc)
    np.savez(HERE / "results" / "probe_rows.npz",
             **{f"{t}_{s}_rows": rows[t][s][0] for t in rows for s in rows[t]},
             **{f"{t}_{s}_y": rows[t][s][1] for t in rows for s in rows[t]},
             test_traces=np.array(test_traces))

    tags = ["base", "distill"] if which == "both" else [which]
    for tag in tags:
        cache = CACHE_ROOT / tag
        meta = json.loads((cache / "meta.json").read_text())
        out_path = HERE / "results" / f"depth_probe_{tag}.json"
        done = json.loads(out_path.read_text()) if out_path.exists() else {
            "meta": {"model_id": meta["model_id"], "T": T,
                     "prereg": "RECORD.md § 2"}, "cells": {}}
        for k in meta["hs_capture"]:
            t0 = time.time()
            acts = np.load(cache / f"hs{k}.npy", mmap_mode="r")
            acts_t = torch.from_numpy(np.ascontiguousarray(acts))
            for tname in rows:
                cell_key = f"hs{k}/{tname}"
                if cell_key in done["cells"]:
                    continue
                (rtr, ytr), (rte, yte) = rows[tname]["train"], rows[tname]["test"]
                Xtr_tok, Xtr_win = gather(acts_t, rtr)
                Xte_tok, Xte_win = gather(acts_t, rte)
                ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
                cell = ceilings_for_target(Xtr_tok, Xte_tok, Xtr_win, Xte_win,
                                           ytr_t, yte_t, 2,
                                           class_weight=True)
                # permutation nulls on the linear pair (frozen seed)
                g = torch.Generator().manual_seed(NULL_SEED)
                ytr_p = ytr_t[torch.randperm(len(ytr_t), generator=g)]
                yte_p = yte_t[torch.randperm(len(yte_t), generator=g)]
                cell["null_window_linear"] = fit_probe(
                    Xtr_win, ytr_p, Xte_win, yte_p, 2, class_weight=True)
                cell["null_per_token_linear"] = fit_probe(
                    Xtr_tok, ytr_p, Xte_tok, yte_p, 2, class_weight=True)
                cell["g_auc"] = (cell["window_linear"]["auc"]
                                 - cell["per_token_linear"]["auc"])
                done["cells"][cell_key] = cell
                print(f"[{tag} hs{k:>2} {tname:>7}] "
                      f"tok={cell['per_token_linear']['auc']:.3f} "
                      f"win={cell['window_linear']['auc']:.3f} "
                      f"g={cell['g_auc']:+.3f} "
                      f"mlp_win={cell['window_mlp']['auc']:.3f} "
                      f"null={cell['null_window_linear']['auc']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                out_path.write_text(json.dumps(done, indent=1))
            del acts_t, acts
        print(f"[{tag}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "both")
