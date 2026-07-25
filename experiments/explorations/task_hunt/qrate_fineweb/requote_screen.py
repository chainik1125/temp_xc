"""Stage-1 RE-QUOTE — punctint q margins on the corrected matched-class grid.

**Status: committed BEFORE it runs** (card § 10, frozen `b8f2f0bd`;
rewritten at the A40 restart — the original was lost unpushed,
APPENDIX A). The screen's q KEEP was scored on the MEAN arm, so its
recorded margins are LOWER BOUNDS (review of 2026-07-25); this states
them properly on the convention-of-record grid my own withdrawal wave
established: **fix the probe class, control width** — tok linear/MLP
vs `anchor ⊕ context-mean` linear/MLP at T ∈ {16, 32, 64}, with the
width-matched foreign-context null beside every window arm.

Same rows, eligibility, caps, seeds, screen layer and frozen probe
stack as `screen.py` (face `q_bin`, 3-class); the
`novelty/anchor_arm_recheck.py` recipe applied to the q face. No
leaderboard rows; no verdict rule — the unconditional KEEP is not at
stake, only the size of its properly-instrumented margins. Quote per
probe class: (actxmean − tok) linear and (actxmean − tok) MLP, each
with its foreign null printed beside (an arm that does not beat its
foreign null at matched width is width, not content).

Writes `results/requote_screen.json`.

Run: .venv/bin/python -m \
       experiments.explorations.task_hunt.qrate_fineweb.requote_screen [model ...]
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
from experiments.explorations.task_hunt.novelty.screen import MODELS
from experiments.explorations.task_hunt.qrate_fineweb.screen import (
    CACHE_ROOT,
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
FACE = "q"
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
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "rows": {k: v for k, v in mstats.items() if k.startswith("q/")}}

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
        (RES / "requote_screen.json").write_text(json.dumps(done, indent=1))

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
    p = RES / "requote_screen.json"
    done = json.loads(p.read_text()) if p.exists() else {}
    for k in (sys.argv[1:] or list(MODELS)):
        run_model(k, done)
    p.write_text(json.dumps(done, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
