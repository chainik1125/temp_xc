"""RE-CHECK of the `novelty` NEGATIVE against the anchor-dilution bias.

**Status: committed BEFORE it runs.** Companion to
`interleave/anchor_arm_recheck.py`; same motivation, same
pre-registered outcome rule, applied to the second of my own published
verdicts that the instrument finding puts at risk.

`novelty` was recorded NEGATIVE because the window-MEAN gap over
per-token topped out at **+0.046 / +0.037 / ~+0.02** against a +0.05
bar — i.e. the verdict turned on a margin of four thousandths on the
best model. The `dialevel` capacity control then showed `win_mean`
dilutes the anchor to weight 1/T and loses 9 of 9 comparisons to
`anchor + context-mean`, by +0.051 AUC on average. A bias of that size
is larger than the margin this verdict turned on, so the NEGATIVE
cannot stand unexamined. The replag caches already exist: no forward
pass.

Face `nov_bin` (3-class terciles), same rows, eligibility, seeds,
screen layer and frozen probe stack as `novelty/screen.py`.
`actxmean_foreign_linear` is the width-matched null.

Outcome rule, stated before running: if the anchor-undiluted arm still
fails the card's +0.05 bar on ≥ 2 of 3 models, the NEGATIVE stands and
is now instrument-controlled. If it clears it, the verdict is withdrawn
to **INCONCLUSIVE-BY-INSTRUMENT** and `novelty` is re-screened on a
corrected grid — it does NOT silently flip to a KEEP, because the
card's growth clause is not tested by these three T values.

Writes `results/anchor_arm_recheck.json`.

Run: .venv/bin/python -m experiments.explorations.task_hunt.novelty.anchor_arm_recheck [model ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.dialevel.actxmean_null import (
    actxmean_foreign,
)
from experiments.explorations.task_hunt.dialevel.capacity_check import (
    FOREIGN_SEED,
    anchor_ctxmean,
)
from experiments.explorations.task_hunt.novelty.screen import (
    CACHE_ROOT,
    MODELS,
    build_rows,
)
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
from experiments.explorations.task_hunt.replag.screen import (
    gather_tok,
    gather_win,
    summarize,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TS = (16, 32, 64)
FACE = "nov_bin"
N_CLS = 3


def run_model(key: str, done: dict):
    hs = SCREEN_HS[key]
    man, mstats = build_rows(key)
    rtr, ytr = man[(FACE, "train")]
    rte, yte = man[(FACE, "test")]
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
    acts = torch.from_numpy(np.ascontiguousarray(np.load(
        CACHE_ROOT / key / f"hs{hs}.npy", mmap_mode="r")))
    cells = done.setdefault("cells", {})
    done.setdefault("meta", {})[key] = {
        "screen_hs": hs, "face": FACE, "foreign_seed": FOREIGN_SEED,
        "n_train": int(len(ytr)), "n_test": int(len(yte))}

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
        (RES / "anchor_arm_recheck.json").write_text(json.dumps(done, indent=1))

    Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
    run(f"{key}/tok_linear", lambda: summarize(
        fit_probe(Xtr, ytr_t, Xte, yte_t, N_CLS), N_CLS))
    run(f"{key}/tok_mlp", lambda: summarize(
        fit_probe(Xtr, ytr_t, Xte, yte_t, N_CLS, hidden=512), N_CLS))
    del Xtr, Xte
    for T in TS:
        Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
        atr, ate = anchor_ctxmean(Wtr), anchor_ctxmean(Wte)
        run(f"{key}/T{T}/actxmean_linear", lambda: summarize(
            fit_probe(atr, ytr_t, ate, yte_t, N_CLS), N_CLS))
        run(f"{key}/T{T}/actxmean_mlp", lambda: summarize(
            fit_probe(atr, ytr_t, ate, yte_t, N_CLS, hidden=512), N_CLS))
        ftr = actxmean_foreign(Wtr, np.random.default_rng(FOREIGN_SEED + T))
        fte = actxmean_foreign(Wte, np.random.default_rng(
            FOREIGN_SEED + T + 1))
        run(f"{key}/T{T}/actxmean_foreign_linear", lambda: summarize(
            fit_probe(ftr, ytr_t, fte, yte_t, N_CLS), N_CLS))
        run(f"{key}/T{T}/actxmean_foreign_mlp", lambda: summarize(
            fit_probe(ftr, ytr_t, fte, yte_t, N_CLS, hidden=512), N_CLS))
        del Wtr, Wte, atr, ate, ftr, fte
    del acts


def main():
    RES.mkdir(exist_ok=True)
    p = RES / "anchor_arm_recheck.json"
    done = json.loads(p.read_text()) if p.exists() else {}
    for k in (sys.argv[1:] or list(MODELS)):
        run_model(k, done)
    p.write_text(json.dumps(done, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
