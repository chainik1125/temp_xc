"""ITEM 6 — train EXACTLY the cells `frontier.py` loads. Nothing else.

Why this exists instead of `run_retrain 3 1`:

1. **`run_retrain` shard 1 spends 3 cells on `tsae_btkonly`**, which
   `frontier.py` never loads. At ~67 min/cell on this dataloader-bound
   pod that is ~67 min of wall-clock for an arm item 6 does not use.
   Measured, not assumed: shard-1 cells 4/5/6 ARE the tsae anchors
   (sort key `(arch, T, seed, n_steps)`, and `tsae_btkonly` sorts
   between `batchtopk_sae_btkonly` and `txc_batchtopk_post_btkonly`).

2. **`run_retrain` can never mint the SAE anchor checkpoints.** Their
   eval rows already exist, and `runner.py:141-150` returns
   `train_cached=True` off a leaderboard hit **without ever checking
   `checkpoint_exists`** — so those cells short-circuit forever, log
   `(cache t=True e=True)`, and write no weights. The sycgen SAE
   weights do not exist anywhere (pod: 0 `model.safetensors`; this mac:
   0 for these keys; the 07-25 HF mirror covers only the stage2
   panels). Without them `frontier.py` loses the pooled AND stacked
   arms — 2 of its 3.

   `train_key` hashes `training_cfg` + arch + `data_key` + section and
   **not** `eval_cfg`, so a fresh `retrain_tag` mints a new `eval_key`
   → cache miss → trains → **saves under the same `train_key`** that
   `frontier.py` looks up. Same weights, same key, new eval row.

3. **One tag for the whole item-6 run.** Every cell here trains on the
   REBUILT activation cache (`hs14.npy`), not pod-D's original, which
   was lost with the pod. Tagging all 15 `sycgen_keep_r1_rebuilt` keeps
   them from masquerading as the originals and makes the anchor
   comparison an explicit check: if the retrained SAE lands near the
   recorded ~0.4819, that is the FIRST genuine evidence the rebuilt
   cache is sound. The earlier "anchors reproduce at 0.487/0.470/0.489"
   was a leaderboard read — the eval never opened the file.

    .venv/bin/python -m experiments.explorations.task_hunt.sycgen.run_item6_cells [workers]
"""
from __future__ import annotations

import sys

from explorations.synthetic import grid

import experiments.explorations.task_hunt.sycgen.run_retrain as RR

TAG = "sycgen_keep_r1_rebuilt"

# Exactly the arches `frontier.py::main` calls `_load` on.
NEEDED = ("batchtopk_sae_btkonly", "txc_batchtopk_post_btkonly")


def cells():
    cs = [c for c in RR.cells()
          if c["arch"] in NEEDED and c["n_steps"] > 0]
    for c in cs:
        c["eval_extra"] = {"retrain_tag": TAG}
    # 3 SAE anchors (T=1 x 3 seeds) + 12 TXC (T{2,4,8,16} x 3 seeds).
    assert len(cs) == 15, f"item-6 needs 15 cells, built {len(cs)}"
    n_sae = sum(1 for c in cs if c["arch"] == "batchtopk_sae_btkonly")
    assert n_sae == 3, f"expected 3 SAE anchors, got {n_sae}"
    return cs


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    cs = cells()
    out = RR.HERE / "results" / "item6_cells.json"
    print(f"[item6] {len(cs)} cells, {workers} workers, tag={TAG}", flush=True)
    for c in cs:
        print(f"   {c['arch']:32s} T={c['T']:<3} s={c['seed']}", flush=True)
    grid.run_pool(cs, out, max_workers=workers, describe=RR._describe,
                  tag="item6")


if __name__ == "__main__":
    main()
