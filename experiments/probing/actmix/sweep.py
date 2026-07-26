"""ACTMIX § 5.1 sparse-probing grid: shuffle control + T-window sweep.

Drives the canonical pathway (``temp_bench.core.runner.run_experiment``)
over the rebuttal grid — every row lands in ``results/leaderboard.jsonl``
with the ACTMIX arm label in ``eval_cfg`` (mandatory per
``briefings/actmix-shared.md``):

- WINDOW archs (``--txc-archs``): retrained per T in ``--Ts`` via
  ``training_cfg.arch_hparams_override={"T": t}``.
- PER-TOKEN archs (``--token-archs``): T-invariant by construction —
  one train per seed; their table columns are flat bands (the hunt's
  fig4 convention). The shuffle control is exactly the identity for
  them (protocol 1.2.0 reports it as such).
- ``--untrained-only`` runs the SAME cell grid at n_steps=0 (the
  untrained twins) — invoked as a separate cheap pass so the twin
  policy (which seeds) is explicit in the launch script, not implicit.

Dispatch order (Aniket's fail-fast convention): untrained twins first
(cheapest pipeline gate), then per-token archs, then window archs with
seed 42 before seeds 1/2 and T endpoints (min, max) before interior
points — the T=1 controlled-limit anchor and the T_max claiming cell
are available before the curve fills in.

Two-GPU split: run one process per GPU with disjoint shards, e.g.::

    CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -m \
        experiments.probing.actmix.sweep --arm btk-only ... \
        --shard-index 0 --shard-count 2 > /workspace/logs/r1_s0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup ... --shard-index 1 --shard-count 2 ...

Sharding is round-robin over the ordered cell list, so both GPUs get a
mix of cheap and expensive cells. Leaderboard appends are flock-guarded
by the runner. ``--dry-run`` prints the exact cell queue and exits.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Any

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig

DATASOURCE = "gemma_2_2b_it_l13_fineweb_24k128"
ARMS = ("relu-mix", "btk-only", "paper-match")


@dataclass
class Cell:
    arch: str
    seed: int
    T: int | None            # None → per-token arch (no override)
    n_steps: int
    k_feats: tuple[int, ...]
    tags: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        t = f"/T={self.T}" if self.T is not None else ""
        u = "/UNTRAINED" if self.n_steps == 0 else ""
        return f"{self.arch}/seed={self.seed}{t}{u}"


def build_queue(args) -> list[Cell]:
    seeds = list(args.seeds)
    # seed 42 first (canonical gate), then the rest in given order.
    if 42 in seeds:
        seeds = [42] + [s for s in seeds if s != 42]
    Ts = list(args.Ts)
    # endpoints first: min, max, then interior in ascending order.
    if len(Ts) > 2:
        interior = [t for t in sorted(Ts)[1:-1]]
        Ts = [min(Ts), max(Ts)] + interior

    queue: list[Cell] = []
    kf = tuple(args.k_feats)
    n_steps = 0 if args.untrained_only else args.n_steps

    for arch in args.token_archs:
        for s in seeds:
            queue.append(Cell(arch, s, None, n_steps, kf))

    for arch in args.txc_archs:
        for s in seeds:
            for t in Ts:
                queue.append(Cell(arch, s, t, n_steps, kf))

    return queue


def run_cell(cell: Cell, args) -> None:
    override = {"T": cell.T} if cell.T is not None else None
    training_cfg = TrainingConfig(
        n_steps=cell.n_steps,
        batch_size=args.batch_size,
        arch_hparams_override=override,
    )
    for k_feat in cell.k_feats:
        eval_cfg = {
            "k_feat": int(k_feat),
            "S": int(args.S),
            "shuffle": "within_window",
            "shuffle_seed": int(args.shuffle_seed),
            "encode_batch_size": int(args.encode_batch_size),
            "arm": args.arm,
            "smoke": False,
        }
        t0 = time.time()
        result = run_experiment(
            experiment="probing",
            arch_name=cell.arch,
            seed=cell.seed,
            datasource_name=DATASOURCE,
            training_cfg=training_cfg,
            eval_cfg=eval_cfg,
            agent=os.environ.get("AGENT_NAME"),
        )
        m = result.row.metrics
        status = "CACHED" if result.eval_cached else f"ran {time.time()-t0:.0f}s"
        print(
            f"[{status}] {cell.label()}/k_feat={k_feat}  "
            f"mean_auc={m.get('mean_auc', float('nan')):.4f}  "
            f"auc_shuf={m.get('mean_auc_shuf', float('nan')):.4f}  "
            f"l0={m.get('realized_l0', float('nan')):.2f}  "
            f"eval_key={result.eval_key}",
            flush=True,
        )


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS,
                    help="ACTMIX arm label (mandatory on every row)")
    ap.add_argument("--txc-archs", nargs="*", default=[],
                    help="window archs, retrained per T")
    ap.add_argument("--token-archs", nargs="*", default=[],
                    help="per-token archs (T-invariant; single train per seed)")
    ap.add_argument("--Ts", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 42])
    ap.add_argument("--k-feats", type=int, nargs="*", default=[5, 20])
    ap.add_argument("--n-steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--S", type=int, default=32)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--encode-batch-size", type=int, default=64)
    ap.add_argument("--untrained-only", action="store_true",
                    help="run this grid as n_steps=0 untrained twins")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.txc_archs and not args.token_archs:
        ap.error("provide at least one of --txc-archs / --token-archs")

    # ── Preflight: refuse to run cells against a partial probe cache —
    # otherwise early cells eval on fewer tasks than late ones and the
    # suite is silently inconsistent across the grid.
    if not args.dry_run:
        from temp_bench.core.config import compute_data_key, data_cache_dir, load_datasource
        from temp_bench.data.probe_cache import list_probe_cache
        n_tasks = len(list_probe_cache(DATASOURCE))
        if n_tasks != 38:
            raise SystemExit(
                f"[sweep] preflight FAIL: probe cache has {n_tasks} complete tasks, "
                "expected 38 (SAEBench+CT). Finish the HF sync / prep_cache first."
            )
        acts = data_cache_dir(compute_data_key(load_datasource(DATASOURCE))) / "acts.npy"
        if not acts.exists():
            raise SystemExit(f"[sweep] preflight FAIL: training cache missing at {acts}")

    queue = build_queue(args)
    mine = [c for i, c in enumerate(queue) if i % args.shard_count == args.shard_index]

    print(f"[sweep] arm={args.arm} total_cells={len(queue)} "
          f"shard {args.shard_index}/{args.shard_count} -> {len(mine)} cells "
          f"(x {len(args.k_feats)} k_feats each)")
    for c in mine:
        print(f"  - {c.label()}")
    if args.dry_run:
        return

    failed: list[str] = []
    for i, c in enumerate(mine, 1):
        print(f"[sweep] ({i}/{len(mine)}) {c.label()}", flush=True)
        try:
            run_cell(c, args)
        except Exception as e:
            # One cell must never kill the pass (core run_sweep's
            # on_failure="continue" convention). Loud, then onward.
            failed.append(c.label())
            print(f"[sweep] FAILED {c.label()}: {type(e).__name__}: {e}",
                  flush=True)
    if failed:
        print(f"[sweep] PASS COMPLETE WITH {len(failed)} FAILED CELLS: "
              + "; ".join(failed), flush=True)
        raise SystemExit(3)   # chain sees a nonzero only after full sweep
    print("[sweep] PASS COMPLETE (all cells ok)", flush=True)


if __name__ == "__main__":
    cli()
