"""End-to-end MPS smoke for the pf arm — briefing rlhf-mac-venue.md § 6.2.

Proves the LAST unproven link: the canonical `run_experiment` pathway on
Apple silicon. The hub's probes exercised the vendored arch and the
buffer directly; this drives the real runner, so it also covers device
selection, the cache-expect check, eval, and the leaderboard write.

Deliberately tiny (`--steps`, default 20) and tagged `smoke: True`, so
the row is excluded from every pf analysis — `render_writeup_fig`
drops smoke rows explicitly.

    .venv/bin/python -m experiments.explorations.actmix_rlhf.smoke_mps
"""
from __future__ import annotations

import argparse
import time

import torch

from experiments.explorations.actmix_rlhf.cells import pf
from temp_bench.core.runner import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1, help="cheapest cell by default")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    print(f"[smoke] torch {torch.__version__} "
          f"cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()}")
    from temp_bench.core.trainer import _select_device
    dev = _select_device()
    print(f"[smoke] _select_device() -> {dev}")
    assert dev == "mps", f"expected mps on this machine, got {dev}"

    cell = pf(args.T)
    tc = cell["training_cfg"].model_copy(update={"n_steps": args.steps})
    print(f"[smoke] cell {cell['cell_id']} T{args.T} seed {cell['seed']} "
          f"steps {args.steps} batch {tc.batch_size}")

    t0 = time.time()
    res = run_experiment(
        experiment="rlhf",
        arch_name=cell["arch"], seed=cell["seed"],
        datasource_name=cell["datasource"],
        training_cfg=tc,
        eval_cfg={**cell.get("eval_cfg", {}), "smoke": True},
        agent="mac-d",
        allow_dirty=True,
    )
    wall = time.time() - t0
    m = (res or {}).get("metrics", {}) if isinstance(res, dict) else {}
    print(f"[smoke] OK in {wall:.1f}s  ({wall/args.steps:.3f} s/step incl. setup)")
    print(f"[smoke] metrics: {m}")
    # HONEST SCOPE — eval_cfg.smoke SHORT-CIRCUITS the rlhf evaluator: it
    # returns {"smoke_ok": 1.0} and never scores. So this proves device
    # selection + cache resolution + the training loop + the leaderboard
    # write, and does NOT prove the eval path on this device. Do not read
    # a missing preference_auc_k20 as "noise at N steps"; it was skipped.
    print("[smoke] PROVEN: mps device, cache-expect, train loop, row write")
    print("[smoke] NOT PROVEN: the eval path (smoke short-circuits it) — "
          "the first real cell is what exercises it")


if __name__ == "__main__":
    main()
