"""Component runner template — copy this into ``experiments/cN_*/run.py``.

A component's runner is ~40 lines. **It does not contain a training
loop, a probing routine, or a judge dispatcher.** Those live in
shared modules per PROTOCOL.md § 11 *Code reuse contract*:

- training: ``temp_bench.training.train_sae``
- evaluation: ``temp_bench.eval.<synthetic|probing|qualitative|steering|case_study>``
- caching + leaderboard append: ``temp_bench.runner.run_cell``
- model construction: ``temp_bench.config.instantiate_arch``

The runner's job is to: declare the per-component constants, define
two thin adapter functions (``my_train_fn`` + ``my_eval_fn``) that
delegate to the shared modules, and loop over (arch, seed, eval_cfg).

If you find yourself writing optimizer code, a `for step in range(...)`
training loop, or judge calls inline here — STOP. Push it into the
shared module. PROTOCOL.md § 11 is non-negotiable.
"""

from __future__ import annotations

import argparse

from temp_bench import runner
from temp_bench.config import instantiate_arch, load_arch
from temp_bench.eval import probing  # or synthetic / qualitative / steering / case_study
from temp_bench.schemas import TrainingConfig
from temp_bench.training import train_sae

# ── Per-component configuration ─────────────────────────────────────────

COMPONENT = "cN"                      # "c1", "c2", ..., "c7"
DATASOURCE = "<datasource_name>"      # from configs/datasources.yaml
EVAL_PROTOCOL_VERSION = "1.0.0"       # bump on metric/protocol change

DEFAULT_ARCHS = runner.list_archs()   # all from yaml; component may filter
DEFAULT_SEEDS = (1, 2, 42)
DEFAULT_K_FEATS = (5, 20)


# ── Component-specific train + eval adapters ───────────────────────────


def my_train_fn(*, arch_name, arch_hparams, seed, training_cfg, act_cache_key, component):
    """Thin adapter to the shared trainer.

    Builds the arch from the YAML spec via ``instantiate_arch``, then
    calls ``train_sae``. NEVER write a training loop here. If you need
    a new training pattern (e.g. RL fine-tuning, distillation), add it
    as a new function in ``temp_bench.training`` and call that.
    """
    spec = load_arch(arch_name, component=component)
    # `d_in` comes from the datasource — for real-LM components, look it
    # up via load_datasource(...).subject_model.config.hidden_size; for
    # toy components, it's `spec.hparams['d_in']` already.
    d_in = ...  # fill in
    model = instantiate_arch(spec, d_in=d_in)

    # Build the training-batch iterator from the activation cache.
    # The shared helper handles toy synthetic vs real-LM uniformly.
    from temp_bench.data import batch_iter_from_act_cache  # noqa
    batch_iter = batch_iter_from_act_cache(act_cache_key, seed=seed)

    result = train_sae(model, batch_iter, training_cfg)
    return result["state_dict"]


def my_eval_fn(*, model, eval_cfg, component):
    """Thin adapter to the shared eval module.

    Composes calls to ``temp_bench.eval.<module>``. NEVER write a
    probing routine, Pareto computation, or judge dispatcher inline
    here. Returns ``(metrics_dict, primary_metric_key)`` where
    ``primary_metric_key`` is the key in ``metrics_dict`` that goes
    into the headline column of the paper table.

    Example for C3 sparse probing::

        metrics = probing.s_tail_probe(
            model=model,
            sequences=eval_cfg["sequences"],
            labels=eval_cfg["labels"],
            S=eval_cfg["S"],
            k_feat=eval_cfg["k_feat"],
        )
        return metrics, "auc"
    """
    raise NotImplementedError(
        "Compose calls to temp_bench.eval.<module> here. "
        "Do not write probing / judge / metric logic inline."
    )


# ── Runner ──────────────────────────────────────────────────────────────


def main(*, archs=None, seeds=DEFAULT_SEEDS, k_feats=DEFAULT_K_FEATS,
         force_train=False, force_eval=False):
    archs = archs or DEFAULT_ARCHS
    for arch in archs:
        for seed in seeds:
            for k in k_feats:
                runner.run_cell(
                    component=COMPONENT,
                    arch_name=arch,
                    seed=seed,
                    datasource_name=DATASOURCE,
                    training_cfg=runner.default_training_cfg(arch),
                    eval_cfg={"k_feat": k, "S": 32},
                    eval_protocol_version=EVAL_PROTOCOL_VERSION,
                    train_fn=my_train_fn,
                    eval_fn=my_eval_fn,
                    force_train=force_train,
                    force_eval=force_eval,
                )


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--k_feats", type=int, nargs="*", default=DEFAULT_K_FEATS)
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()
    main(
        archs=args.archs,
        seeds=tuple(args.seeds),
        k_feats=tuple(args.k_feats),
        force_train=args.force_train,
        force_eval=args.force_eval,
    )


if __name__ == "__main__":
    cli()
