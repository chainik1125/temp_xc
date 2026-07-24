"""CAPACITY control for the dialevel within-dialogue arm.

**Status: committed BEFORE it runs** (git order is the evidence). This
is a POST-HOC diagnostic added after the frozen screen's cells landed;
it does NOT alter `CARD.md`'s KEEP/KILL scoring (already determined),
it determines what the record is ALLOWED TO CLAIM about the mechanism —
the same discipline the Stage-2 probe-capacity diagnostic and the
punctint within-document control followed.

## Why it exists

The screen's three window arms are not capacity-matched to each other
or to per-token, and this agent's own Stage-2 finding is that probe
capacity can REVERSE a T-ordering on identical codes:

    per-token          d features
    window-MEAN        d features   (but the anchor is diluted to 1/T)
    anchor+context     T*d features (shuffled: order removed)
    window-flatten     T*d features
    window-MLP         T*d -> 512

Two specific readings are at risk. (i) `win_mlp` is the best cell on
every model (+0.05..+0.07 over `tok_mlp`) — but it is also the widest
probe in the grid. (ii) `win_linear` beats `win_shuf_linear` by
+0.025..+0.062 at T in {16,32} on all three models, which reads as the
hunt's first capacity-matched ORDER carriage — but only if a T*d-wide
probe cannot manufacture that much from the anchor alone.

## The three controls

1. **FOREIGN context** (`win_foreign_*`) — flatten with context slots
   0..T-2 taken from a DIFFERENT row (seeded row permutation), anchor
   slot kept true. Same dimensionality, same marginal statistics, ZERO
   true context. This is the capacity null for the flatten and MLP
   arms: whatever it scores is what width plus the anchor buy.
2. **anchor + context-mean** (`actxmean_*`) — concat(anchor,
   mean of slots 0..T-2), 2d features. This is the order-free window
   arm that does NOT dilute the anchor, i.e. the arm `win_mean` should
   have been: `win_mean` pools the anchor into the average at weight
   1/T, so its decline in T may be anchor dilution rather than evidence
   about window structure.
3. **MEAN + MLP** (`mean_mlp`) — MLP(512) on `win_mean`, exactly d
   input features, so it is capacity-matched to `tok_mlp` cell for
   cell.

Together they give a capacity-controlled ladder:
`tok` -> `actxmean` (does order-free context add?) -> `flatten` (does
order add?) -> `foreign` (is any of it real, or just width?).

Same rows, same caches, same frozen probe stack, same seeds as the
screen. Writes `results/capacity_check.json`.

Run: .venv/bin/python -m experiments.explorations.task_hunt.dialevel.capacity_check [model ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.dialevel.cache_acts import CACHE_ROOT
from experiments.explorations.task_hunt.dialevel.screen import (
    MODELS,
    build_rows,
)
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
from experiments.explorations.task_hunt.replag.screen import (
    gather_win,
    summarize,
    win_mean,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TS = (16, 32)
MEAN_TS = (16, 32, 64)
FOREIGN_SEED = 4242


def foreign_context(W, rng):
    """Context slots from a DIFFERENT row; anchor slot (T-1) kept true."""
    n, T, _ = W.shape
    perm = torch.from_numpy(rng.permutation(n))
    out = W.clone()
    out[:, :T - 1] = W[perm][:, :T - 1]
    return out


def anchor_ctxmean(W):
    """concat(anchor, mean over context slots) — order-free, anchor NOT
    diluted (2d features)."""
    return torch.cat([W[:, -1], W[:, :-1].float().mean(1).to(W.dtype)], 1)


def run_model(key: str, done: dict):
    hs = SCREEN_HS[key]
    man, mstats = build_rows(key)
    if not (mstats["wd/train"]["ok"] and mstats["wd/test"]["ok"]):
        print(f"[{key}] SKIP (rows under floor)")
        return
    rtr, ytr = man[("wd", "train")]
    rte, yte = man[("wd", "test")]
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
    acts = torch.from_numpy(np.ascontiguousarray(np.load(
        CACHE_ROOT / key / f"hs{hs}.npy", mmap_mode="r")))
    cells = done.setdefault("cells", {})
    done.setdefault("meta", {})[key] = {
        "screen_hs": hs, "n_per_class": mstats["wd/train"]["n_per_class"],
        "foreign_seed": FOREIGN_SEED}

    def run(k, fn):
        if k in cells:
            return
        t0 = time.time()
        cells[k] = fn()
        cells[k]["wall_s"] = round(time.time() - t0, 1)
        print(f"[{key} {k}] " + " ".join(f"{a}={b:.3f}" for a, b in
                                         cells[k].items()
                                         if isinstance(b, float)
                                         and a != "wall_s"), flush=True)
        (RES / "capacity_check.json").write_text(json.dumps(done, indent=1))

    for T in sorted(set(TS) | set(MEAN_TS)):
        Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
        if T in MEAN_TS:
            run(f"{key}/T{T}/mean_mlp", lambda: summarize(fit_probe(
                win_mean(Wtr), ytr_t, win_mean(Wte), yte_t, 2, hidden=512,
                class_weight=True), 2))
            atr, ate = anchor_ctxmean(Wtr), anchor_ctxmean(Wte)
            run(f"{key}/T{T}/actxmean_linear", lambda: summarize(fit_probe(
                atr, ytr_t, ate, yte_t, 2, class_weight=True), 2))
            run(f"{key}/T{T}/actxmean_mlp", lambda: summarize(fit_probe(
                atr, ytr_t, ate, yte_t, 2, hidden=512,
                class_weight=True), 2))
            del atr, ate
        if T in TS:
            rng = np.random.default_rng(FOREIGN_SEED + T)
            ftr = foreign_context(Wtr, rng).reshape(len(rtr), -1)
            fte = foreign_context(Wte, np.random.default_rng(
                FOREIGN_SEED + T + 1)).reshape(len(rte), -1)
            run(f"{key}/T{T}/win_foreign_linear", lambda: summarize(
                fit_probe(ftr, ytr_t, fte, yte_t, 2, class_weight=True), 2))
            run(f"{key}/T{T}/win_foreign_mlp", lambda: summarize(fit_probe(
                ftr, ytr_t, fte, yte_t, 2, hidden=512,
                class_weight=True), 2))
            del ftr, fte
        del Wtr, Wte
    del acts


def main():
    RES.mkdir(exist_ok=True)
    p = RES / "capacity_check.json"
    done = json.loads(p.read_text()) if p.exists() else {}
    for k in (sys.argv[1:] or list(MODELS)):
        run_model(k, done)
    p.write_text(json.dumps(done, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
