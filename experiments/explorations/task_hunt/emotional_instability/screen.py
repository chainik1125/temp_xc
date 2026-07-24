"""Stage-1 screen — emotional-instability onset (executes CARD.md).

Substrate: gemma-3-12b-it, screen layer resid_post L24 = hs25 over the
ragged conversation cache (flat token axis; manifests store flat
indices; eligibility keeps every T ≤ 64 window inside its conversation).

Targets (from build_labels.py manifests): ant4/ant8/ant16 (anticipation
D-ladder, binary), esc3 (escalation tercile, 3-class), det (post-onset
sanity anchor, binary — validates labels, not a KEEP input).

Frozen probe grid (CARD): per-token linear/MLP once; T ∈ {16, 32, 64}:
window linear / window-mean linear / context-shuffled linear (anchor
slot fixed); MLPs at T ∈ {32, 64}; permutation nulls at T = 32.

Run: .venv/bin/python -m experiments.explorations.task_hunt.emotional_instability.screen
Writes results/screen.json.
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe

ACTS = Path("/workspace/emo_caches/acts")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SCREEN_HS = 25
T_GRID = [16, 32, 64]
MLP_T = [32, 64]
NULL_T = 32
NULL_SEED = 99
SHUF_SEED = 1234
TARGETS = {"ant4": 2, "ant8": 2, "ant16": 2, "esc3": 3, "det": 2}


def gather_win(acts, rows, T):
    n, d = len(rows), acts.shape[-1]
    X = torch.empty((n, T, d), dtype=acts.dtype)
    r = torch.from_numpy(rows)
    for j in range(T):
        X[:, j] = acts[r - (T - 1) + j]
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


def summarize(r, n_classes):
    keep = ["acc_test", "per_class", "n_train", "n_test"]
    out = {k: r[k] for k in keep if k in r}
    if n_classes == 2:
        out.update({k: r[k] for k in ["auc", "balacc", "balacc_opt"]
                    if k in r})
    return out


def main():
    out_path = RES / "screen.json"
    stats = json.loads((RES / "label_stats.json").read_text())
    z = np.load(RES / "manifests.npz")
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"model_id": "google/gemma-3-12b-it",
                 "screen_hs": SCREEN_HS, "t_grid": T_GRID,
                 "mlp_t": MLP_T, "null_t": NULL_T,
                 "card": "CARD.md (frozen)"},
        "cells": {}}
    cells = done["cells"]

    def run(cell_key, n_classes, fn):
        if cell_key in cells:
            return
        t0 = time.time()
        cells[cell_key] = fn()
        cells[cell_key]["wall_s"] = round(time.time() - t0, 1)
        print(f"[emo {cell_key}] "
              + " ".join(f"{k}={v:.3f}" for k, v in cells[cell_key].items()
                         if isinstance(v, float) and k != "wall_s"),
              flush=True)
        out_path.write_text(json.dumps(done, indent=1))

    acts = torch.from_numpy(np.ascontiguousarray(
        np.load(ACTS / f"hs{SCREEN_HS}.npy", mmap_mode="r")))

    for target, n_classes in TARGETS.items():
        st_tr = stats.get(f"{target}/train", {})
        st_te = stats.get(f"{target}/test", {})
        if not (st_tr.get("ok") and st_te.get("ok")):
            print(f"[emo {target}] SKIP (insufficient matched rows: "
                  f"{st_tr.get('rows_per_class')} / "
                  f"{st_te.get('rows_per_class')})", flush=True)
            continue
        cw = n_classes == 2
        rtr = z[f"{target}_train_rows"]
        rte = z[f"{target}_test_rows"]
        ytr = torch.from_numpy(z[f"{target}_train_y"])
        yte = torch.from_numpy(z[f"{target}_test_y"])
        Xtr_tok = acts[torch.from_numpy(rtr)]
        Xte_tok = acts[torch.from_numpy(rte)]
        run(f"{target}/tok_linear", n_classes, lambda: summarize(
            fit_probe(Xtr_tok, ytr, Xte_tok, yte, n_classes,
                      class_weight=cw), n_classes))
        run(f"{target}/tok_mlp", n_classes, lambda: summarize(
            fit_probe(Xtr_tok, ytr, Xte_tok, yte, n_classes,
                      hidden=512, class_weight=cw), n_classes))
        for T in T_GRID:
            Wtr = gather_win(acts, rtr, T)
            Wte = gather_win(acts, rte, T)
            flat_tr = Wtr.reshape(len(rtr), -1)
            flat_te = Wte.reshape(len(rte), -1)
            run(f"{target}/T{T}/win_linear", n_classes, lambda: summarize(
                fit_probe(flat_tr, ytr, flat_te, yte, n_classes,
                          class_weight=cw), n_classes))
            mtr = (Wtr.float().mean(1)).to(torch.float16)
            mte = (Wte.float().mean(1)).to(torch.float16)
            run(f"{target}/T{T}/win_mean_linear", n_classes,
                lambda: summarize(fit_probe(mtr, ytr, mte, yte,
                                            n_classes, class_weight=cw),
                                  n_classes))
            srng = np.random.default_rng(SHUF_SEED + zlib.crc32(
                f"{target}/T{T}".encode()) % 2 ** 16)
            Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
            Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
            run(f"{target}/T{T}/win_shuf_linear", n_classes,
                lambda: summarize(fit_probe(Str, ytr, Ste, yte,
                                            n_classes, class_weight=cw),
                                  n_classes))
            if T in MLP_T:
                run(f"{target}/T{T}/win_mlp", n_classes,
                    lambda: summarize(fit_probe(flat_tr, ytr, flat_te,
                                                yte, n_classes,
                                                hidden=512,
                                                class_weight=cw),
                                      n_classes))
                run(f"{target}/T{T}/win_shuf_mlp", n_classes,
                    lambda: summarize(fit_probe(Str, ytr, Ste, yte,
                                                n_classes, hidden=512,
                                                class_weight=cw),
                                      n_classes))
            if T == NULL_T:
                g = torch.Generator().manual_seed(NULL_SEED)
                ytr_p = ytr[torch.randperm(len(ytr), generator=g)]
                yte_p = yte[torch.randperm(len(yte), generator=g)]
                run(f"{target}/T{T}/null_win_linear", n_classes,
                    lambda: summarize(fit_probe(flat_tr, ytr_p, flat_te,
                                                yte_p, n_classes,
                                                class_weight=cw),
                                      n_classes))
                run(f"{target}/null_tok_linear", n_classes,
                    lambda: summarize(fit_probe(Xtr_tok, ytr_p, Xte_tok,
                                                yte_p, n_classes,
                                                class_weight=cw),
                                      n_classes))
            del Wtr, Wte, flat_tr, flat_te, Str, Ste
    print(f"[emo] SCREEN DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
