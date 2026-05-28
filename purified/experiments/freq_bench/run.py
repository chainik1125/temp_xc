"""Entry point for FreqBench — the §4 temporal-filtering stress test.

    python run.py freq_bench --arch <name> --seed <int> --datasource <key>
                             [--k-pos <int>] [--n-steps <int>] [--T <int>] [--smoke]

Each cell trains one SAE on a DC/AC/Mixed datasource, then probes its code
for the latent label with order-controls (see temp_bench/evals/freq_bench.py).
``--k-pos`` is the sparsity sweep axis (Dmitry's ``raw_k``); ``--T`` overrides
the arch's temporal window so one arch can stand in for txcdr_t2/t5/txc_base.
"""

from __future__ import annotations

import argparse
import os

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig


def _parse(extra: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run.py freq_bench")
    p.add_argument("--k-pos", type=int, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args(extra)


def run(args, extra: list[str]) -> int:
    sub = _parse(extra)

    if args.datasource is None:
        args.datasource = "fb_ac_smoke" if args.smoke else "fb_ac_W16_s10"
    if args.arch is None:
        args.arch = "txc_base"
        print(f"[freq_bench] --arch unset; defaulting to {args.arch}")

    if args.smoke:
        n_steps = sub.n_steps if sub.n_steps is not None else 30
        batch_size = sub.batch_size if sub.batch_size is not None else 64
        buffer_tokens = 8192
    else:
        n_steps = sub.n_steps if sub.n_steps is not None else 10_000
        batch_size = sub.batch_size if sub.batch_size is not None else 2048
        buffer_tokens = 2_000_000

    override: dict = {}
    if sub.k_pos is not None:
        override["k_pos"] = int(sub.k_pos)
    if sub.T is not None:
        override["T"] = int(sub.T)

    training_cfg = TrainingConfig(
        n_steps=n_steps,
        batch_size=batch_size,
        buffer_tokens=buffer_tokens,
        warmup_steps=0 if args.smoke else 200,
        arch_hparams_override=override or None,
    )

    eval_cfg = {"smoke": args.smoke}
    if sub.k_pos is not None:
        eval_cfg["k_pos"] = int(sub.k_pos)
    if sub.T is not None:
        eval_cfg["T"] = int(sub.T)

    print(f"[freq_bench] arch={args.arch} seed={args.seed} ds={args.datasource} "
          f"k_pos={override.get('k_pos')} T={override.get('T')} "
          f"n_steps={n_steps} smoke={args.smoke}")

    result = run_experiment(
        experiment="freq_bench",
        arch_name=args.arch,
        seed=args.seed,
        datasource_name=args.datasource,
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,
        agent=os.environ.get("AGENT_NAME"),
        allow_dirty=args.allow_dirty,
    )

    print("\n[freq_bench] DONE")
    print(f"  train_key: {result.train_key} (cached={result.train_cached})")
    print(f"  eval_key:  {result.eval_key} (cached={result.eval_cached})")
    for k, v in result.row.metrics.items():
        print(f"    {k:<14} = {v:.4f}")
    return 0
