"""Unified eval entry point — chains sweep → probing (ordered + shuffled) → summary.

The harness wraps existing pieces rather than rewriting them:
  - `src.bench.sweep` for training (existing, unchanged)
  - `src.bench.saebench.probing_runner.run_probing` for probing
  - `src.bench.regressions.check_all` as a pre-flight gate

Usage:
    python -m src.bench.run_eval \\
        --architecture mlc \\
        --protocol A \\
        --t 5 \\
        --aggregation full_window \\
        --ckpt results/saebench/ckpts/mlc__gemma-2-2b__l10-11-12-13-14__k100__protA__seed42.pt \\
        --output results/saebench/results/mlc_protA_T5.jsonl

Ordered vs shuffled pair (team 4/18 ask):
    --shuffle-seed 42         # runs ONLY the shuffled pass
    --both-ordered-shuffled   # runs the pair at shuffle_seed=42
                              # and writes both to the same JSONL with
                              # shuffle_seed field distinguishing them

Bill / Han / new-arch workflow: register your arch in
`src/bench/architectures/__init__.py:REGISTRY`, then invoke this CLI
with `--architecture <your-key>`. See
`src/bench/architectures/README.md` for a worked example.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Run sparse-probing eval on a trained SAE/TempXC/MLC "
                    "checkpoint with ordered + shuffled controls."
    )
    ap.add_argument(
        "--architecture", "--arch", dest="arch", required=True,
        choices=["sae", "tempxc", "mlc"],
        help="Architecture family (matches src/bench/architectures REGISTRY).",
    )
    ap.add_argument(
        "--protocol", required=True, choices=["A", "B"],
        help="Sparsity matching protocol. See plan.md § 4 — A is per-token "
             "rate matched, B is total-window budget matched.",
    )
    ap.add_argument(
        "--t", type=int, default=5,
        help="Temporal window size (TempXC) or n_layers (MLC). Ignored for SAE.",
    )
    ap.add_argument(
        "--aggregation", required=True,
        choices=["last", "mean", "max", "full_window"],
        help="Aggregation strategy for TempXC. No-op for SAE / MLC but must "
             "be set for the JSONL schema.",
    )
    ap.add_argument("--ckpt", required=True, help="Path to trained checkpoint.")
    ap.add_argument(
        "--output", required=True,
        help="JSONL path to append probing records to.",
    )
    ap.add_argument(
        "--k-values", type=int, nargs="+", default=[1, 2, 5, 20],
        help="Top-k feature counts to sweep in probing.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Probe random seed.")
    ap.add_argument(
        "--device", default="cuda:0",
        help="Torch device. CPU works for probe fitting once Gemma acts are cached.",
    )
    # Shuffle-control knobs
    shuf = ap.add_mutually_exclusive_group()
    shuf.add_argument(
        "--shuffle-seed", type=int, default=None,
        help="If set, run the shuffled variant ONLY with this seed. "
             "Omit for ordered-only. Use --both-ordered-shuffled for the pair.",
    )
    shuf.add_argument(
        "--both-ordered-shuffled", action="store_true",
        help="Run ordered AND shuffled (seed=42) in one invocation. "
             "Both results go to the same JSONL — rows are distinguished "
             "by the 'shuffle_seed' field (None = ordered, int = shuffled).",
    )
    ap.add_argument(
        "--skip-regressions", action="store_true",
        help="Skip the startup regression gate (don't do this in CI).",
    )
    return ap.parse_args()


def main():
    args = _parse_args()

    # B1–B16 regression gate. Fails fast with a pointer to eval_infra_lessons.md
    # on any known-bug reintroduction. ~0.1 s; only skip via --skip-regressions.
    if not args.skip_regressions:
        from src.bench.regressions import check_all
        print(">> running regression checks (B1-B16 + item 8 hook)...")
        check_all(verbose=False)
        print("   all clean.")

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        sys.exit(f"FAIL: checkpoint not found: {ckpt}")

    from src.bench.saebench.probing_runner import run_probing

    # Decide which seeds to run.
    seeds_to_run: list[int | None]
    if args.both_ordered_shuffled:
        seeds_to_run = [None, 42]  # ordered then shuffled
    elif args.shuffle_seed is not None:
        seeds_to_run = [args.shuffle_seed]
    else:
        seeds_to_run = [None]  # ordered only (default)

    all_summaries = []
    t0 = time.time()
    for shuffle_seed in seeds_to_run:
        tag = "shuffled" if shuffle_seed is not None else "ordered"
        print(f"\n>> run_probing ({tag}): arch={args.arch} protocol={args.protocol} "
              f"T={args.t} agg={args.aggregation} seed={args.seed}")
        summary = run_probing(
            arch=args.arch,
            ckpt_path=str(ckpt),
            protocol=args.protocol,
            t=args.t,
            aggregation=args.aggregation,
            output_jsonl=args.output,
            k=0,  # placeholder; probing k is swept inside SAEBench via k_values
            k_values=tuple(args.k_values),
            device=args.device,
            random_seed=args.seed,
            shuffle_seed=shuffle_seed,
        )
        all_summaries.append({"tag": tag, **summary})

    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print(f" run_eval done in {elapsed:.1f}s across {len(seeds_to_run)} variant(s)")
    for s in all_summaries:
        skipped = " [SKIPPED]" if s.get("skipped") else ""
        print(f"   {s['tag']}: {s['n_records_written']} records written{skipped}")
    print("=" * 72)


if __name__ == "__main__":
    main()
