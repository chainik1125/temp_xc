"""C4 qualitative-latents component runner.

Thin scaffold per ``experiments/_runner_template.py`` (PROTOCOL.md § 11
*Code reuse contract*). All training-loop / qualitative logic lives in
the shared modules:

- ``temp_bench.training.train_sae``         — canonical SAE trainer
- ``temp_bench.eval.qualitative.top_256_semantic`` — top-N SEMANTIC
- ``temp_bench.runner.run_cell``            — leaderboard append + caching

The C4 cells SHARE training checkpoints with C3 — both train on the
same FineWeb activation cache (``gemma_2_2b_it_l13_fineweb_24k128``)
with the same TrainingConfig. So if C3 has run for (arch, seed), C4's
my_train_fn just loads the cached checkpoint via runner.run_cell's
auto-skip.

Usage::

    # All three archs × 3 seeds (smoke first, then real):
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c4_qualitative.run \\
        --archs tsae_paper txc_base \\
        --seeds 1 2 42

    # Smoke (1 cell, n_features=8 to keep judge cost low):
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c4_qualitative.run --smoke

Note: the qualitative metric requires Anthropic API access. Set
``ANTHROPIC_API_KEY`` env var or place at ``/workspace/.tokens/anthropic_key``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from temp_bench import runner
from temp_bench.config import act_cache_dir, instantiate_arch, load_arch
from temp_bench.data import batch_iter_from_act_cache
from temp_bench.eval import qualitative
from temp_bench.schemas import TrainingConfig
from temp_bench.training import train_sae

# ── Per-component constants ─────────────────────────────────────────────

COMPONENT = "c4"
DATASOURCE = "gemma_2_2b_it_l13_fineweb_24k128"   # SHARED with C3
EVAL_PROTOCOL_VERSION = "1.0.0"

DEFAULT_ARCHS = ("tsae_paper", "txc_base")        # txc_pro pending port
DEFAULT_SEEDS = (1, 2, 42)
DEFAULT_N_FEATURES = 256

# Smoke: 1 cell, 8 features (cuts judge cost from $0.06 → $0.002)
SMOKE_TRAIN_STEPS = 200
SMOKE_BATCH = 64
SMOKE_N_FEATURES = 8


def _d_in_from_act_cache(act_cache_key: str) -> int:
    meta = json.loads((act_cache_dir(act_cache_key) / "meta.json").read_text())
    return int(meta["d_in"])


def _real_training_cfg() -> TrainingConfig:
    """Same config as C3's _real_training_cfg — checkpoints SHARE."""
    return TrainingConfig(
        n_steps=10_000,
        batch_size=256,
        learning_rate=3e-4,
        warmup_steps=500,
        precision="bf16",
    )


def my_train_fn(*, arch_name, arch_hparams, seed, training_cfg, act_cache_key, component):
    """Identical to C3's my_train_fn — runner.run_cell will auto-skip
    if the train_key is already cached (which it will be after C3 runs).
    """
    spec = load_arch(arch_name, component=component)
    d_in = _d_in_from_act_cache(act_cache_key)
    model = instantiate_arch(spec, d_in=d_in)

    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_iter = batch_iter_from_act_cache(act_cache_key, seed=seed)
    result = train_sae(model, batch_iter, training_cfg, device="cuda")
    return result["state_dict"]


def my_eval_fn(*, model, eval_cfg, component):
    """Compose qualitative.top_256_semantic — never inline judge logic."""
    del model  # we re-instantiate
    arch_name = eval_cfg["_arch_name"]
    state_dict = eval_cfg["_state_dict"]
    act_cache_key = eval_cfg["_act_cache_key"]
    eval_key_hint = eval_cfg["_eval_key_hint"]   # we set this in main()
    n_features = int(eval_cfg.get("n_features", DEFAULT_N_FEATURES))

    spec = load_arch(arch_name, component=component)
    d_in = _d_in_from_act_cache(act_cache_key)
    m = instantiate_arch(spec, d_in=d_in).cuda().eval()
    m.load_state_dict(state_dict)

    metrics = qualitative.top_256_semantic(
        m,
        eval_key=eval_key_hint,
        subject_model_name="google/gemma-2-2b-it",
        subject_layer=13,
        concat_corpora=("concat_A", "concat_B", "concat_random"),
        n_features=n_features,
    )
    return metrics, "top_N_semantic"


def main(*, archs, seeds, n_features, smoke, force_train=False, force_eval=False):
    if smoke:
        training_cfg = TrainingConfig(
            n_steps=SMOKE_TRAIN_STEPS,
            batch_size=SMOKE_BATCH,
            learning_rate=3e-4,
            warmup_steps=20,
            precision="bf16",
        )
    else:
        training_cfg = _real_training_cfg()

    from temp_bench.config import compute_act_cache_key, load_datasource, compute_train_key, compute_eval_key
    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))

    for arch in archs:
        for seed in seeds:
            # Pre-compute the eval_key so we can pass it to my_eval_fn for
            # judge_outputs.jsonl path. The runner re-computes the same
            # key from the same inputs, so this is safe.
            train_key = compute_train_key(
                arch=load_arch(arch, component=COMPONENT),
                seed=seed,
                training_cfg=training_cfg,
                act_cache_key=act_cache_key,
            )
            eval_cfg = {
                "n_features": n_features,
                "_act_cache_key": act_cache_key,
                "_datasource_name": DATASOURCE,
            }
            eval_key = compute_eval_key(
                train_key=train_key,
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                eval_cfg=eval_cfg,
            )
            eval_cfg["_eval_key_hint"] = eval_key

            result = runner.run_cell(
                component=COMPONENT,
                arch_name=arch,
                seed=seed,
                datasource_name=DATASOURCE,
                training_cfg=training_cfg,
                eval_cfg=eval_cfg,
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                train_fn=my_train_fn,
                eval_fn=my_eval_fn,
                force_train=force_train,
                force_eval=force_eval,
            )
            tag = "CACHED" if result.cached else "NEW"
            m = result.metrics or {}
            sem = m.get("top_N_semantic")
            agree = m.get("judge_agreement")
            njudged = m.get("n_features_judged")
            sem_s = f"{int(sem)}/{int(njudged)}" if sem is not None and njudged is not None else "-"
            agree_s = f"{agree:.3f}" if agree is not None else "-"
            print(f"[{tag}] {arch} seed={seed} n_features={n_features}  "
                  f"SEMANTIC={sem_s}  agreement={agree_s}  "
                  f"eval_key={result.eval_key}")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=list(DEFAULT_ARCHS))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    ap.add_argument("--n_features", type=int, default=DEFAULT_N_FEATURES)
    ap.add_argument("--smoke", action="store_true",
                    help="Run a 1-cell smoke test with n_features=8.")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        archs = args.archs[:1]
        seeds = args.seeds[:1]
        n_features = SMOKE_N_FEATURES
    else:
        archs, seeds, n_features = args.archs, args.seeds, args.n_features

    main(
        archs=archs, seeds=seeds, n_features=n_features,
        smoke=args.smoke,
        force_train=args.force_train, force_eval=args.force_eval,
    )


if __name__ == "__main__":
    cli()
