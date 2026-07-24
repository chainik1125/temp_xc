"""Stage-1 screen — repetition-lag Δ (executes CARD.md exactly).

Per model (gpt2 / gemma2_2b / llama31_8b), on the frozen screen layer:
for each target (det4/det8/det16/det32 binary, lag4 4-class) —

  per-token linear + per-token MLP(512)          (T-independent, once)
  T ∈ {2,4,8,16,32}: window linear (flatten, right-edge anchor),
      window-MEAN linear, window SHUFFLED linear (context slots
      permuted per row, anchor slot fixed; seeded)
  window MLP + shuffled-window MLP at T ∈ {8, 32}
  permutation nulls (NULL_SEED 99) on the linear pair at T = 16

Probe stack: conversion_depth.problib (frozen — no retuning). Binary
metric: rank-AUC (class_weight=True); lag4: acc_test (balanced by
construction) + per_class. Incremental/resumable per cell.

Run: .venv/bin/python -m experiments.explorations.task_hunt.replag.screen [model ...]
Writes results/screen_<model>.json next to this file.
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
from experiments.explorations.task_hunt.replag.build_labels import (
    BUCKETS, CACHE_ROOT, LABELS_DIR, MODELS,
)
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS

T_GRID = [2, 4, 8, 16, 32]
MLP_T = [8, 32]
NULL_T = 16
NULL_SEED = 99          # probe_depth convention
SHUF_SEED = 1234
TARGETS = [f"det{b[1:]}" for b in BUCKETS] + ["lag4"]
HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def gather_tok(acts, rows):
    return acts[torch.from_numpy(rows[:, 0].astype(np.int64)),
                torch.from_numpy(rows[:, 1].astype(np.int64))]


def gather_win(acts, rows, T):
    w = torch.from_numpy(rows[:, 0].astype(np.int64))
    p = torch.from_numpy(rows[:, 1].astype(np.int64))
    n, d = len(rows), acts.shape[-1]
    X = torch.empty((n, T, d), dtype=acts.dtype)
    for j in range(T):
        X[:, j] = acts[w, p - (T - 1) + j]
    return X


def shuffle_context(X_win, rng):
    """Permute slots 0..T−2 per row; anchor slot T−1 fixed."""
    n, T, d = X_win.shape
    if T <= 2:
        return X_win.clone()
    perms = rng.permuted(np.tile(np.arange(T - 1), (n, 1)), axis=1)
    out = X_win.clone()
    out[:, :T - 1] = X_win[torch.arange(n)[:, None],
                           torch.from_numpy(perms)]
    return out


def win_mean(X_win):
    return (X_win.float().mean(1)).to(torch.float16)


def summarize(r, n_classes):
    keep = ["acc_test", "per_class", "n_train", "n_test"]
    out = {k: r[k] for k in keep if k in r}
    if n_classes == 2:
        out.update({k: r[k] for k in ["auc", "balacc", "balacc_opt"]
                    if k in r})
    return out


def screen(key: str):
    RES.mkdir(exist_ok=True)
    out_path = RES / f"screen_{key}.json"
    meta = json.loads((CACHE_ROOT / key / "acts_meta.json").read_text())
    stats = json.loads(
        (LABELS_DIR / f"replag_{key}_stats.json").read_text())
    hs = SCREEN_HS[key]
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"model_id": MODELS[key]["hf"], "screen_hs": hs,
                 "card": "CARD.md (frozen)", "t_grid": T_GRID,
                 "mlp_t": MLP_T, "null_t": NULL_T,
                 "coverage": stats["coverage"]},
        "cells": {}}
    cells = done["cells"]

    def save():
        out_path.write_text(json.dumps(done, indent=1))

    def run(cell_key, fn):
        if cell_key in cells:
            return
        t0 = time.time()
        cells[cell_key] = fn()
        cells[cell_key]["wall_s"] = round(time.time() - t0, 1)
        print(f"[{key} {cell_key}] "
              + " ".join(f"{k}={v:.3f}" for k, v in cells[cell_key].items()
                         if isinstance(v, float) and k != "wall_s"),
              flush=True)
        save()

    acts = torch.from_numpy(np.ascontiguousarray(
        np.load(CACHE_ROOT / key / f"hs{hs}.npy", mmap_mode="r")))
    z = np.load(LABELS_DIR / f"replag_{key}_manifests.npz")

    for target in TARGETS:
        if not (stats["tasks"][f"{target}/train"]["ok"]
                and stats["tasks"][f"{target}/test"]["ok"]):
            print(f"[{key} {target}] SKIP (insufficient matched rows)")
            continue
        n_classes = 4 if target == "lag4" else 2
        cw = n_classes == 2
        rtr, ytr = z[f"{target}_train_rows"], z[f"{target}_train_y"]
        rte, yte = z[f"{target}_test_rows"], z[f"{target}_test_y"]
        ytr_t = torch.from_numpy(ytr.astype(np.int64))
        yte_t = torch.from_numpy(yte.astype(np.int64))

        Xtr_tok, Xte_tok = gather_tok(acts, rtr), gather_tok(acts, rte)
        run(f"{target}/tok_linear", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, n_classes,
                      class_weight=cw), n_classes))
        run(f"{target}/tok_mlp", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, n_classes,
                      hidden=512, class_weight=cw), n_classes))

        for T in T_GRID:
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            flat_tr = Wtr.reshape(len(rtr), -1)
            flat_te = Wte.reshape(len(rte), -1)
            run(f"{target}/T{T}/win_linear", lambda: summarize(
                fit_probe(flat_tr, ytr_t, flat_te, yte_t, n_classes,
                          class_weight=cw), n_classes))
            run(f"{target}/T{T}/win_mean_linear", lambda: summarize(
                fit_probe(win_mean(Wtr), ytr_t, win_mean(Wte), yte_t,
                          n_classes, class_weight=cw), n_classes))
            srng = np.random.default_rng(
                SHUF_SEED + zlib.crc32(f"{target}/T{T}".encode()) % 2 ** 16)
            Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
            Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
            run(f"{target}/T{T}/win_shuf_linear", lambda: summarize(
                fit_probe(Str, ytr_t, Ste, yte_t, n_classes,
                          class_weight=cw), n_classes))
            if T in MLP_T:
                run(f"{target}/T{T}/win_mlp", lambda: summarize(
                    fit_probe(flat_tr, ytr_t, flat_te, yte_t, n_classes,
                              hidden=512, class_weight=cw), n_classes))
                run(f"{target}/T{T}/win_shuf_mlp", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, n_classes,
                              hidden=512, class_weight=cw), n_classes))
            if T == NULL_T:
                g = torch.Generator().manual_seed(NULL_SEED)
                ytr_p = ytr_t[torch.randperm(len(ytr_t), generator=g)]
                yte_p = yte_t[torch.randperm(len(yte_t), generator=g)]
                run(f"{target}/T{T}/null_win_linear", lambda: summarize(
                    fit_probe(flat_tr, ytr_p, flat_te, yte_p, n_classes,
                              class_weight=cw), n_classes))
                run(f"{target}/null_tok_linear", lambda: summarize(
                    fit_probe(Xtr_tok, ytr_p, Xte_tok, yte_p, n_classes,
                              class_weight=cw), n_classes))
            del Wtr, Wte, flat_tr, flat_te, Str, Ste
    del acts
    print(f"[{key}] SCREEN DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    for key in (sys.argv[1:] or list(MODELS)):
        screen(key)
