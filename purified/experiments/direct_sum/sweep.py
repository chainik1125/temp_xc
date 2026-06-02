"""Direct-sum "which-process" sweep driver (arch × J), GPU-sharded.

Tests the core claim: all J processes share the same emission MARGINAL but
differ in temporal DYNAMICS (AC phase-walk vs DC constant-phase). A per-token
arch is pinned at A_loc=1/J (chance); order-sensitive temporal archs climb
toward A_oracle=1.

Results land in ``results/leaderboard.jsonl`` via the canonical
``run_experiment`` pathway (idempotent, flock-safe, code-version stamped).

Archs swept:
    topk_sae       — per-token baseline (expected NTPS ≈ 0)
    txcdr_t5       — txc_base slid T=5 (expected NTPS ≥ 0.50 @ J=4)
    txc_base_TW    — txc_base joint T=W=16 (moderate, per §3.6 theory)
    tfa            — attention-based TFA (expected NTPS ≥ 0.40)

Datasources:
    ds_which_J4_W16_s10   — J=4, W=16, σ=0.1
    ds_which_J8_W16_s10   — J=8, W=16, σ=0.1

Single GPU (GPU 1 owned by this agent):
    CUDA_VISIBLE_DEVICES=1 TEMP_BENCH_ALLOW_DIRTY=1 \\
      .venv/bin/python experiments/direct_sum/sweep.py --n-shards 1 --shard-id 0

Author: Aniket, 2026-06-01.
"""

from __future__ import annotations

import argparse
import os
import time

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig

# ── sweep configuration ──────────────────────────────────────────────────

# (label, arch_name, T)
# txcdr_t5 = txc_base slid with T=5 (the strong AC performer from FreqBench)
# txc_base_TW = txc_base with T=W (joint window, the degraded variant)
ARCH_CFGS = [
    ("topk_sae",      "topk_sae",  1),    # per-token baseline
    ("txcdr_t5",      "txc_base",  5),    # sliding T=5
    ("txc_base_TW",   "txc_base", 16),    # joint T=W=16 (W=16 for both ds)
    ("tfa",           "tfa",       5),    # attention-based temporal arch
]

# Both J datasources (W=16 for both → T=16 for txc_base_TW)
DATASOURCES = [
    ("ds_which_J4_W16_s10", 4),
    ("ds_which_J8_W16_s10", 8),
]

# Fixed hparams (readout-optimal per freq_bench_theory.md §2)
K_POS  = 1
D_SAE  = 1024
SEED   = 0


def build_cells() -> list[dict]:
    """Build the full (arch × datasource) cell list."""
    cells = []
    seen = set()
    for (label, arch, T), (ds_name, J) in [
        (a, d) for a in ARCH_CFGS for d in DATASOURCES
    ]:
        W = 16   # both datasources use W=16
        if T > W:
            continue   # skip if T > W (only matters if someone adds W<T combos)
        key = (label, ds_name)
        if key in seen:
            continue
        seen.add(key)
        cells.append({
            "label": label,
            "arch": arch,
            "T": T,
            "W": W,
            "J": J,
            "k_pos": K_POS,
            "d_sae": D_SAE,
            "datasource": ds_name,
        })
    return cells


def run_cell(c: dict, n_steps: int) -> dict:
    """Train one cell and return its metrics."""
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
        "W": c["W"],
        "J": c["J"],
    }
    r = run_experiment(
        experiment="freq_bench",
        arch_name=c["arch"],
        seed=SEED,
        datasource_name=c["datasource"],
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,
        agent=os.environ.get("AGENT_NAME", "direct_sum"),
        allow_dirty=True,
    )
    return r.row.metrics | {"cached": r.eval_cached}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Direct-sum which-process sweep (arch × J)."
    )
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=5000,
                    help="Training steps per cell (default: 5000).")
    args = ap.parse_args()

    cells = build_cells()
    mine = [c for i, c in enumerate(cells) if i % args.n_shards == args.shard_id]
    print(
        f"[shard {args.shard_id}/{args.n_shards}] {len(mine)}/{len(cells)} cells "
        f"on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
    )
    print(f"  cells: {[c['label']+'@'+c['datasource'] for c in mine]}")

    for i, c in enumerate(mine, 1):
        t0 = time.time()
        try:
            m = run_cell(c, args.n_steps)
            tag = "cache" if m.get("cached") else f"{time.time()-t0:.0f}s"
            ntps = m.get("NTPS", float("nan"))
            gap = m.get("order_gap", float("nan"))
            rev = m.get("reverse_drop", float("nan"))
            print(
                f"[shard {args.shard_id}] [{i}/{len(mine)}] "
                f"{c['label']:14s} J={c['J']:2d} ds={c['datasource']}: "
                f"NTPS={ntps:+.3f} gap={gap:+.3f} rev_drop={rev:+.3f} ({tag})",
                flush=True,
            )
        except Exception as e:
            print(
                f"[shard {args.shard_id}] [{i}/{len(mine)}] "
                f"{c['label']} J={c['J']} ds={c['datasource']}: "
                f"FAILED {type(e).__name__}: {e}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
