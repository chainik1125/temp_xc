"""RE-CHECK of the `tss` KILL against the anchor-dilution instrument bias.

**Status: committed BEFORE it runs.** This does not re-open the frozen
`interleave/CARD.md` scoring; it tests whether that scoring rested on a
biased instrument, which is a different question and one this agent is
obliged to ask about its OWN published verdict.

## Why

The `dialevel` screen's capacity control (`dialevel/capacity_check.py`,
`dialevel/actxmean_null.py`) found the hunt's shared window instrument
biased AGAINST the window in two measurable ways:

- `win_mean` pools the ANCHOR into the average at weight 1/T. Where the
  anchor carries a strong per-position route, that dilutes the arm's
  best feature as T grows. Replacing it with `anchor + context-mean`
  (2d, order-free, anchor undiluted) won **9 of 9** model x T
  comparisons on dialevel, by +0.051 AUC on average.
- `win_flatten`'s width is expensive: a foreign-context null (true
  anchor, another row's context) scored BELOW per-token, so 24k-131k
  noise features cost up to 0.15 AUC.

`tss` was KILLED as converted on exactly the arms this affects — the
kill clause was "per-token >= 0.05 above its position floor while the
window adds < 0.05", and the window arms were `win_mean` and
`win_flatten`. Anchor dilution pushes `win_mean` DOWN, which makes that
clause MORE likely to fire. The verdict is therefore at risk of being
an instrument artifact, and the caches already exist, so the check
costs no forward pass.

Same rows, same eligibility, same frozen probe stack, same screen layer
as `interleave/screen.py`. `actxmean_foreign_*` is the width-matched
null (true anchor, context-mean from a different row).

Outcome rule, stated before running: if `actxmean` fails to lift the
window above per-token by the card's +0.05 at any T, the KILL stands
and is now instrument-controlled. If it does lift it, the KILL is
**withdrawn to INCONCLUSIVE-BY-INSTRUMENT** and `tss` is re-screened on
a corrected grid — the verdict does not silently flip to a KEEP.

Writes `results/anchor_arm_recheck.json`.

Run: .venv/bin/python -m experiments.explorations.task_hunt.interleave.anchor_arm_recheck [model ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.dialevel.capacity_check import (
    FOREIGN_SEED,
    anchor_ctxmean,
)
from experiments.explorations.task_hunt.dialevel.actxmean_null import (
    actxmean_foreign,
)
from experiments.explorations.task_hunt.interleave.cache_acts import CACHE_ROOT
from experiments.explorations.task_hunt.interleave.screen import (
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
FACE = "tss"
N_CLS = 3


def run_model(key: str, done: dict):
    hs = SCREEN_HS[key]
    man, mstats = build_rows(key, null=False)
    if not (mstats[f"{FACE}/train"]["ok"] and mstats[f"{FACE}/test"]["ok"]):
        print(f"[{key}] SKIP (rows under floor)")
        return
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

    # per-token re-run here so the comparison is inside one artifact
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
