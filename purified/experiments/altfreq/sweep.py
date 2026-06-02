"""Alternate-frequency FreqBench sweep driver.

Validates the extended FreqBench (chirp / multitone / AM / relative-phase)
across four architectures:

    - topk_sae      — per-token baseline (T=1)
    - txcdr_t5      — txc_base slid with T=5 window
    - txc_base_TW   — txc_base joint T=W=16
    - tfa           — temporal-feature-attention baseline

Sweep axes:
    - 4 datasources (af_chirp_W16_s10, af_multitone_W16_s10, af_am_W16_s10,
                     af_relphase_W16_s10)
    - readout-optimal regime: k_pos=1, d_sae=1024 (established in
      docs/aniket/freq-bench/theory.md §2 — the sparsity slice that maximises AC signal)

All cells go through ``run_experiment(experiment='freq_bench')`` so rows land
in results/leaderboard.jsonl with full code-version stamps and are
idempotent on re-run.

Single-GPU usage (CUDA_VISIBLE_DEVICES=0 owned by this agent):
    CUDA_VISIBLE_DEVICES=0 TEMP_BENCH_ALLOW_DIRTY=1 \\
        .venv/bin/python experiments/altfreq/sweep.py --n-shards 1 --shard-id 0
"""

from __future__ import annotations

import argparse
import os
import time

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig

# Architecture configurations: (label, arch_name, T)
# txcdr_t5 = txc_base slid with T=5 (the established strong AC cell)
# txc_base_TW = txc_base joint with T=W=16
ARCH_CFGS = [
    ("topk_sae",      "topk_sae",  1),
    ("txcdr_t5",      "txc_base",  5),
    ("txc_base_TW",   "txc_base", 16),
    ("tfa",           "tfa",       5),
]

# W=16 production datasources (all use seq_len=W=16)
DATASOURCES = [
    "af_chirp_W16_s10",
    "af_multitone_W16_s10",
    "af_am_W16_s10",
    "af_relphase_W16_s10",
]

# Readout-optimal regime (docs/aniket/freq-bench/theory.md §2)
K_POS = 1
D_SAE = 1024
SEED = 0
N_STEPS = 5000   # sufficient for convergence on these tiny synthetic tasks


def build_cells() -> list[dict]:
    """All arch × datasource combinations at the readout-optimal slice."""
    cells = []
    for datasource in DATASOURCES:
        for label, arch, T in ARCH_CFGS:
            # topk_sae has T=1, which is always ≤ W=16
            # txc_base_TW has T=W=16, checked in run_cell
            if T > 16:
                continue
            cells.append({
                "label": label,
                "arch": arch,
                "T": T,
                "datasource": datasource,
                "k_pos": K_POS,
                "d_sae": D_SAE,
            })
    return cells


def run_cell(c: dict, n_steps: int) -> dict:
    override = {"d_sae": c["d_sae"], "k_pos": c["k_pos"], "T": c["T"]}
    training_cfg = TrainingConfig(
        n_steps=n_steps,
        batch_size=2048,
        buffer_tokens=1_500_000,
        warmup_steps=200,
        arch_hparams_override=override,
    )
    eval_cfg = {
        "smoke": False,
        "label": c["label"],
        "T": c["T"],
        "k_pos": c["k_pos"],
        "d_sae": c["d_sae"],
        "W": 16,
    }
    r = run_experiment(
        experiment="freq_bench",
        arch_name=c["arch"],
        seed=SEED,
        datasource_name=c["datasource"],
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,
        agent=os.environ.get("AGENT_NAME", "altfreq"),
        allow_dirty=True,
    )
    return r.row.metrics | {"cached": r.eval_cached}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    args = ap.parse_args()

    cells = build_cells()
    mine = [c for i, c in enumerate(cells) if i % args.n_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.n_shards}] {len(mine)}/{len(cells)} cells "
          f"on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    for i, c in enumerate(mine, 1):
        t0 = time.time()
        ds_short = c["datasource"].replace("af_", "")
        try:
            m = run_cell(c, args.n_steps)
            tag = "cache" if m.get("cached") else f"{time.time()-t0:.0f}s"
            print(
                f"[shard {args.shard_id}] [{i}/{len(mine)}] "
                f"{c['label']:14s} {ds_short:20s}: "
                f"NTPS={m['NTPS']:+.3f} "
                f"gap={m['order_gap']:+.3f} rev_drop={m['reverse_drop']:+.3f} "
                f"({tag})",
                flush=True,
            )
        except Exception as e:
            print(
                f"[shard {args.shard_id}] [{i}/{len(mine)}] {c['label']} "
                f"{ds_short}: FAILED {type(e).__name__}: {e}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
