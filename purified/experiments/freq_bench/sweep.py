"""FreqBench AC sweep driver (raw_k × W × capacity × arch), GPU-sharded.

Validates the re-analysis correction with FRESH training in the v2 port:
the AC order-control signal (windowed archs encode signed direction; per-
token archs don't) and its dependence on sparsity (raw_k) and window (W).

Every cell goes through the canonical ``run_experiment`` pathway, so rows
land in ``results/leaderboard.jsonl`` with full code-version stamps and are
idempotent (cache-hit on re-run).

Parallelism: ``run_sweep`` is sequential, so this shards the cell list
round-robin across processes. Launch one process per GPU, each with its own
``CUDA_VISIBLE_DEVICES``; the flock-protected JSONL append makes concurrent
writes safe. Example (3 A40s):

    for g in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$g TEMP_BENCH_ALLOW_DIRTY=1 \
        .venv/bin/python experiments/freq_bench/sweep.py \
        --n-shards 3 --shard-id $g &
    done; wait
"""

from __future__ import annotations

import argparse
import itertools
import os
import time

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig

# (label, arch_name, T).  txcdr_t* = txc_base slid with a fixed T window.
ARCH_CFGS = [
    ("regular_sae", "topk_sae", 1),
    ("txcdr_t2", "txc_base", 2),
    ("txcdr_t5", "txc_base", 5),
]
W_DATASOURCES = {2: "fb_ac_W2_s10", 4: "fb_ac_W4_s10",
                 8: "fb_ac_W8_s10", 16: "fb_ac_W16_s10"}
RAW_KS = [1, 2, 5, 10, 20]
D_SAES = [40, 256]          # capacity axis (Dmitry used 40)
SEED = 0


def build_cells() -> list[dict]:
    """Full raw_k×W cross at d_sae=40 + a focused capacity slice.

    The full cross at Dmitry's capacity (40) answers the core question
    (raw_k facet + order controls). The capacity slice (W=16, raw_k=1, the
    strongest cell) tests whether a wider dictionary lifts the AC signal —
    without paying for a full second cross.
    """
    cells, seen = [], set()

    def add(label, arch, T, W, k, dsae):
        if T > W or k > dsae:
            return
        key = (label, W, k, dsae)
        if key in seen:
            return
        seen.add(key)
        cells.append({"label": label, "arch": arch, "T": T, "W": W,
                      "k_pos": k, "d_sae": dsae, "datasource": W_DATASOURCES[W]})

    # main cross @ d_sae=40
    for (label, arch, T), W, k in itertools.product(
            ARCH_CFGS, sorted(W_DATASOURCES), RAW_KS):
        add(label, arch, T, W, k, 40)
    # capacity slice @ strongest cell (W=16, raw_k=1)
    for (label, arch, T), dsae in itertools.product(ARCH_CFGS, [256, 1024]):
        add(label, arch, T, 16, 1, dsae)
    return cells


def run_cell(c: dict, n_steps: int) -> dict:
    override = {"d_sae": c["d_sae"], "k_pos": c["k_pos"], "T": c["T"]}
    training_cfg = TrainingConfig(
        n_steps=n_steps, batch_size=2048, buffer_tokens=1_500_000,
        warmup_steps=200, arch_hparams_override=override,
    )
    eval_cfg = {"smoke": False, "label": c["label"], "T": c["T"],
                "k_pos": c["k_pos"], "d_sae": c["d_sae"], "W": c["W"]}
    r = run_experiment(
        experiment="freq_bench", arch_name=c["arch"], seed=SEED,
        datasource_name=c["datasource"], training_cfg=training_cfg,
        eval_cfg=eval_cfg, agent=os.environ.get("AGENT_NAME"), allow_dirty=True,
    )
    return r.row.metrics | {"cached": r.eval_cached}


def _launch_gpus(n_gpus: int, n_steps: int) -> int:
    """Orchestrator: spawn one shard subprocess per GPU and wait for all.

    Run this under the harness background mechanism so the parent (and thus
    its GPU children) survives across turns.
    """
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))            # purified/
    procs = []
    for g in range(n_gpus):
        # Cap CPU threads per shard: 3 processes × 96 default threads would
        # thrash a 96-core box. 8 threads each is ample for these tiny ops.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                   TEMP_BENCH_ALLOW_DIRTY="1", AGENT_NAME="aniket",
                   OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                   OPENBLAS_NUM_THREADS="8")
        log = open(os.path.join(root, "logs", f"fb_sweep_shard{g}.log"), "w")
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(here, "sweep.py"),
             "--n-shards", str(n_gpus), "--shard-id", str(g),
             "--n-steps", str(n_steps)],
            stdout=log, stderr=subprocess.STDOUT, cwd=root, env=env))
        print(f"[orchestrator] spawned shard {g} pid {procs[-1].pid}", flush=True)
    rc = 0
    for p in procs:
        rc |= p.wait()
    print(f"[orchestrator] all shards done (rc={rc})", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=6_000)
    ap.add_argument("--launch-gpus", type=int, default=0,
                    help="Orchestrator: spawn this many per-GPU shards and wait.")
    args = ap.parse_args()

    if args.launch_gpus > 0:
        return _launch_gpus(args.launch_gpus, args.n_steps)

    cells = build_cells()
    mine = [c for i, c in enumerate(cells) if i % args.n_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.n_shards}] {len(mine)}/{len(cells)} cells "
          f"on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    for i, c in enumerate(mine, 1):
        t0 = time.time()
        try:
            m = run_cell(c, args.n_steps)
            tag = "cache" if m.get("cached") else f"{time.time()-t0:.0f}s"
            print(f"[shard {args.shard_id}] [{i}/{len(mine)}] "
                  f"{c['label']:11s} W={c['W']:2d} k={c['k_pos']:2d} "
                  f"dsae={c['d_sae']:3d}: NTPS={m['NTPS']:+.3f} "
                  f"gap={m['order_gap']:+.3f} rev_drop={m['reverse_drop']:+.3f} "
                  f"({tag})", flush=True)
        except Exception as e:
            print(f"[shard {args.shard_id}] [{i}/{len(mine)}] {c['label']} "
                  f"W={c['W']} k={c['k_pos']} dsae={c['d_sae']}: "
                  f"FAILED {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
