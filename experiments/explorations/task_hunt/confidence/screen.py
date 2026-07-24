"""Stage-1 screen — confidence trend (executes ../confidence/CARD.md).

Substrate: the Ward stream rebuilt on this pod
(/workspace/conv_depth_caches/ward_stream) + the cache_depth.py reader
caches; screen layer resid_post L13 = hs14 for BOTH readers (base,
distill) — the measured g(ℓ) peak. Labels: ../labels/confidence.npz
(runpod-b, frozen; slope8_bin uses ITS committed tercile edges).

Row construction (the card's guard, frozen in CARD § screen cells):
  eligibility  valid & p ≥ 63 (uniform so every screened T ≤ 64 fits)
               & hedge(anchor) defined
  targets      slope8_bin (PRIMARY, 3-class), slope4_bin (terciles
               computed here over eligible rows, backup), state
               (hedge 3-class, regime-1 CONTROL)
  matching     slope targets: exact class-histogram matching over
               (anchor hedge state × position bucket) cells — kills the
               "slope readable from current state" ambient route;
               state control: position-bucket matching only
  split        by trace (labels' trace_split), caps 4000/1500 per class

Probes: frozen problib stack; T ∈ {16, 32, 64}; per T window linear /
window-mean linear / context-shuffled linear (anchor fixed); MLPs at
T ∈ {32, 64}; permutation nulls at T = 32; per-token pair once.

Run: .venv/bin/python -m experiments.explorations.task_hunt.confidence.screen [base|distill]
Writes results/screen_<reader>.json.
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.labels import lib
from experiments.explorations.task_hunt.replag.build_labels import (
    matched_sample,
)

CACHE_ROOT = Path("/workspace/conv_depth_caches")
HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels" / "confidence.npz"
RES = HERE / "results"
SCREEN_HS = 15                    # resid_post L14 — nearest captured
                                  # L13-equivalent (cache_depth stores
                                  # odd hs = resid_post of even layers)
T_GRID = [16, 32, 64]
MLP_T = [32, 64]
NULL_T = 32
NULL_SEED = 99
SHUF_SEED = 1234
MATCH_SEED = 1013
P_MIN = 63
POS_EDGES = [63, 80, 96, 112, 128]
CAP = {"train": 4000, "test": 1500}
MIN_ROWS = 300
READERS = {"base": "NousResearch/Meta-Llama-3.1-8B",
           "distill": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}


def build_rows():
    z = np.load(LABELS)
    valid = z["valid"].astype(bool)
    hedge = z["hedge"]
    slope8_bin = z["slope8_bin"]
    trace_idx, trace_split = z["trace_idx"], z["trace_split"]
    N, L = hedge.shape

    elig = valid.copy()
    elig[:, :P_MIN] = False
    elig &= hedge >= 0

    s4_edges, slope4_bin = lib.tercile_bins(
        np.where(elig, z["slope4"], np.nan))

    def hbucket(p):
        return int(np.searchsorted(POS_EDGES, p, side="right") - 1)

    tasks = {"slope8": (slope8_bin, True), "slope4": (slope4_bin, True),
             "state": (hedge, False)}
    manifests, stats = {}, {"slope4_edges": [float(e) for e in s4_edges]}
    for split_name, split_flag in [("train", 0), ("test", 1)]:
        in_split = trace_split[trace_idx] == split_flag
        for tname, (cls_grid, hedge_match) in tasks.items():
            pools: dict = {}
            for w in np.flatnonzero(in_split):
                for p in np.flatnonzero(elig[w]):
                    c = int(cls_grid[w, p])
                    if c < 0:
                        continue
                    cell_tok = (int(hedge[w, p]) if hedge_match else 0)
                    pools.setdefault(c, []).append(
                        (int(w), int(p), cell_tok, hbucket(p)))
            rng = np.random.default_rng(MATCH_SEED + zlib.crc32(
                f"conf/{tname}/{split_name}".encode()) % 2 ** 16)
            out, joint = matched_sample(pools, CAP[split_name], rng, 3)
            n_per = {int(c): len(v) for c, v in out.items()}
            stats[f"{tname}/{split_name}"] = {
                "rows_per_class": n_per, "joint_matched": bool(joint),
                "ok": bool(min(n_per.values(), default=0) >= MIN_ROWS)}
            rows = np.array([r[:2] for c in sorted(out) for r in out[c]],
                            dtype=np.int64)
            y = np.concatenate([np.full(len(out[c]), c, dtype=np.int64)
                                for c in sorted(out)]) if len(rows) else \
                np.zeros(0, dtype=np.int64)
            manifests[tname, split_name] = (rows, y)
    return manifests, stats


def gather_tok(acts, rows):
    return acts[torch.from_numpy(rows[:, 0]), torch.from_numpy(rows[:, 1])]


def gather_win(acts, rows, T):
    w = torch.from_numpy(rows[:, 0])
    p = torch.from_numpy(rows[:, 1])
    n, d = len(rows), acts.shape[-1]
    X = torch.empty((n, T, d), dtype=acts.dtype)
    for j in range(T):
        X[:, j] = acts[w, p - (T - 1) + j]
    return X


def shuffle_context(X_win, rng):
    n, T, d = X_win.shape
    if T <= 2:
        return X_win.clone()
    perms = rng.permuted(np.tile(np.arange(T - 1), (n, 1)), axis=1)
    out = X_win.clone()
    out[:, :T - 1] = X_win[torch.arange(n)[:, None],
                           torch.from_numpy(perms)]
    return out


def summarize(r):
    return {k: r[k] for k in ["acc_test", "per_class", "n_train", "n_test"]
            if k in r}


def screen(tag: str):
    RES.mkdir(exist_ok=True)
    out_path = RES / f"screen_{tag}.json"
    manifests, mstats = build_rows()
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"reader": READERS[tag], "screen_hs": SCREEN_HS,
                 "t_grid": T_GRID, "mlp_t": MLP_T, "null_t": NULL_T,
                 "p_min": P_MIN, "rows": mstats,
                 "card": "../confidence/CARD.md"},
        "cells": {}}
    cells = done["cells"]

    def run(cell_key, fn):
        if cell_key in cells:
            return
        t0 = time.time()
        cells[cell_key] = fn()
        cells[cell_key]["wall_s"] = round(time.time() - t0, 1)
        print(f"[{tag} {cell_key}] "
              + " ".join(f"{k}={v:.3f}" for k, v in cells[cell_key].items()
                         if isinstance(v, float) and k != "wall_s"),
              flush=True)
        out_path.write_text(json.dumps(done, indent=1))

    acts = torch.from_numpy(np.ascontiguousarray(
        np.load(CACHE_ROOT / tag / f"hs{SCREEN_HS}.npy", mmap_mode="r")))

    for tname in ["slope8", "slope4", "state"]:
        if not (mstats[f"{tname}/train"]["ok"]
                and mstats[f"{tname}/test"]["ok"]):
            print(f"[{tag} {tname}] SKIP (insufficient matched rows)")
            continue
        (rtr, ytr), (rte, yte) = (manifests[tname, "train"],
                                  manifests[tname, "test"])
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
        Xtr_tok, Xte_tok = gather_tok(acts, rtr), gather_tok(acts, rte)
        run(f"{tname}/tok_linear", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3)))
        run(f"{tname}/tok_mlp", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3, hidden=512)))
        for T in T_GRID:
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            flat_tr = Wtr.reshape(len(rtr), -1)
            flat_te = Wte.reshape(len(rte), -1)
            run(f"{tname}/T{T}/win_linear", lambda: summarize(
                fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3)))
            mtr = (Wtr.float().mean(1)).to(torch.float16)
            mte = (Wte.float().mean(1)).to(torch.float16)
            run(f"{tname}/T{T}/win_mean_linear", lambda: summarize(
                fit_probe(mtr, ytr_t, mte, yte_t, 3)))
            srng = np.random.default_rng(SHUF_SEED + zlib.crc32(
                f"{tname}/T{T}".encode()) % 2 ** 16)
            Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
            Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
            run(f"{tname}/T{T}/win_shuf_linear", lambda: summarize(
                fit_probe(Str, ytr_t, Ste, yte_t, 3)))
            if T in MLP_T:
                run(f"{tname}/T{T}/win_mlp", lambda: summarize(
                    fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3,
                              hidden=512)))
                run(f"{tname}/T{T}/win_shuf_mlp", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, 3, hidden=512)))
            if T == NULL_T:
                g = torch.Generator().manual_seed(NULL_SEED)
                ytr_p = ytr_t[torch.randperm(len(ytr_t), generator=g)]
                yte_p = yte_t[torch.randperm(len(yte_t), generator=g)]
                run(f"{tname}/T{T}/null_win_linear", lambda: summarize(
                    fit_probe(flat_tr, ytr_p, flat_te, yte_p, 3)))
                run(f"{tname}/null_tok_linear", lambda: summarize(
                    fit_probe(Xtr_tok, ytr_p, Xte_tok, yte_p, 3)))
            del Wtr, Wte, flat_tr, flat_te, Str, Ste
    del acts
    print(f"[{tag}] SCREEN DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    for tag in (sys.argv[1:] or ["distill", "base"]):
        screen(tag)
