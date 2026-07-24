"""Width-matched NULL for the `anchor + context-mean` arm.

**Status: committed BEFORE it runs.** Second post-hoc diagnostic; like
`capacity_check.py` it changes nothing about the frozen KEEP/KILL
scoring, only what the record may claim.

`capacity_check.py` found that `anchor + context-mean` (2d features)
beats the frozen grid's `win_mean` (d features, anchor diluted to 1/T)
in **9 of 9** model x T comparisons, by +0.051 AUC on average, and that
it is the only window arm that clears the label-side `postst_floor` by
>= 0.05 on all three models. That is about to become a recommendation
to the whole hunt — every screen so far has used `win_mean` as its
order-free window arm — so it needs its own null, not an argument.

`win_foreign_*` already nulls the T*d flatten arms and showed that
width alone is expensive (foreign 0.58-0.62 against a per-token
0.65-0.74: 24k-131k noise features COST up to 0.15 AUC). The
`actxmean` arm is only 2d, so its width penalty should be negligible —
but "should be" is the word this script removes. Here the context-mean
half comes from a DIFFERENT row (seeded row permutation) while the
anchor half stays true: same 2d width, same marginal statistics, zero
true context. Whatever it scores is the arm's floor.

Writes `results/actxmean_null.json`.

Run: .venv/bin/python -m experiments.explorations.task_hunt.dialevel.actxmean_null [model ...]
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
from experiments.explorations.task_hunt.dialevel.capacity_check import (
    FOREIGN_SEED,
    MEAN_TS,
)
from experiments.explorations.task_hunt.dialevel.screen import (
    MODELS,
    build_rows,
)
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
from experiments.explorations.task_hunt.replag.screen import (
    gather_win,
    summarize,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def actxmean_foreign(W, rng):
    """concat(TRUE anchor, context-mean of a DIFFERENT row) — 2d wide,
    zero true context."""
    n, T, _ = W.shape
    perm = torch.from_numpy(rng.permutation(n))
    ctx = W[perm][:, :-1].float().mean(1).to(W.dtype)
    return torch.cat([W[:, -1], ctx], 1)


def run_model(key: str, done: dict):
    hs = SCREEN_HS[key]
    man, mstats = build_rows(key)
    rtr, ytr = man[("wd", "train")]
    rte, yte = man[("wd", "test")]
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
    acts = torch.from_numpy(np.ascontiguousarray(np.load(
        CACHE_ROOT / key / f"hs{hs}.npy", mmap_mode="r")))
    cells = done.setdefault("cells", {})
    done.setdefault("meta", {})[key] = {"screen_hs": hs,
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
        (RES / "actxmean_null.json").write_text(json.dumps(done, indent=1))

    for T in MEAN_TS:
        Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
        ftr = actxmean_foreign(Wtr, np.random.default_rng(FOREIGN_SEED + T))
        fte = actxmean_foreign(Wte, np.random.default_rng(
            FOREIGN_SEED + T + 1))
        run(f"{key}/T{T}/actxmean_foreign_linear", lambda: summarize(
            fit_probe(ftr, ytr_t, fte, yte_t, 2, class_weight=True), 2))
        run(f"{key}/T{T}/actxmean_foreign_mlp", lambda: summarize(fit_probe(
            ftr, ytr_t, fte, yte_t, 2, hidden=512, class_weight=True), 2))
        del Wtr, Wte, ftr, fte
    del acts


def main():
    RES.mkdir(exist_ok=True)
    p = RES / "actxmean_null.json"
    done = json.loads(p.read_text()) if p.exists() else {}
    for k in (sys.argv[1:] or list(MODELS)):
        run_model(k, done)
    p.write_text(json.dumps(done, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
