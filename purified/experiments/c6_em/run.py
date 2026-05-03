"""C6 entrypoint — drive the gap-close test through `runner.run_cell`.

Cells (in order):

1. ``sae_arditi`` × seed 42 (the apples-to-apples baseline)
2. ``txc_base`` × seed 42 with brickenauxk_a8 recipe
   (auxk_alpha=1/8, dead_threshold_tokens=128 000, bricken_enabled=True)

Both share the same activation cache (the C6 datasource ``finance_em_prompts``)
so the SAE and TXC see identical training distributions — apples-to-apples.

After both cells land, the headline number is

    gap = peak_align(sae_arditi) − peak_align(txc_base)

and the C6 decision tree (``docs/components/c6.md``) maps it to one of:

- gap ≤ 3 → Tied (TXC closed the gap with brickenauxk_a8)
- 3 < gap ≤ 9 → Mixed (step-efficiency win on Qwen-7B medical)
- gap > 9 → Honest negative (SAE arditi still wins on R1 30k)

Caveats baked into this run (vs Dmitry's published numbers in
``origin/em-nanda:docs/dmitry/results/em_features/em_nanda_results_paper.md``):

- **Judge swap**: Claude Haiku 4.5 instead of Gemini-3.1-flash-lite
  (Gemini key not provisioned).
- **Wang abbreviation**: stages 2+3 (causal screen + per-survivor
  coherence sweep) skipped; we go directly from Δz̄ ranking → stage-4
  6-α frontier on the top-3 features.
- **Corpus divergence**: training corpus is the C6 datasource's
  ``finance_em_prompts`` (cfierro/personality-qs-risky-financial-advice
  on HF), not Dmitry's pile/ultrachat 70-30 mix.
- **Hparam divergence**: TXC-base uses the locked yaml defaults
  (``d_sae=18432``, ``k_pos=20``, ``k_win=100``) rather than the
  larger d_sae=32768 + k=128 setup Dmitry used (locked yaml has no
  c6 per-component override; see agent_em briefing OQ #1).

These caveats mean absolute numbers won't match Dmitry's 95.16 / 91.25;
the **relative gap** between TXC-base+brickenauxk and SAE-arditi (both
trained on the same corpus, both evaluated by the same judge with the
same abbreviated procedure) is the headline.

    cd /workspace/temp_xc_em/purified
    source scripts/set_agent_env.sh agent_em
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em.run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from temp_bench import runner
from temp_bench.config import act_cache_dir, compute_act_cache_key, load_arch, load_datasource
from temp_bench.schemas import TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("c6.run")

COMPONENT = "c6"
DATASOURCE = "qwen_2_5_14b_instruct_finance_l24_resid_post"
EVAL_PROTOCOL_VERSION = "1.0.0"


# ── Per-arch training-config recipes ───────────────────────────────


def make_training_cfg(arch_name: str, *, n_steps: int = 30_000) -> TrainingConfig:
    """Return the training cfg for one C6 cell.

    - SAE-arditi: vanilla MSE, no Bricken, no AuxK (the locked recipe).
    - TXC-base + brickenauxk_a8: Bricken on, AuxK α=1/8,
      dead_threshold=128k tokens.
    """
    common = dict(
        n_steps=n_steps, batch_size=256, learning_rate=3e-4,
        optimizer="adam", warmup_steps=1_000,
        plateau_early_stop=False,  # train to completion regardless
        precision="bf16",
    )
    if arch_name == "sae_arditi":
        return TrainingConfig(**common,
                              bricken_enabled=False,
                              ema_auxk_alpha=0.0,             # unused
                              dead_threshold_tokens=10_000_000)  # unused
    if arch_name == "txc_base":
        # brickenauxk_a8 from origin/em-nanda:summary_brickenauxk_a8_frontier.md
        return TrainingConfig(
            **common,
            bricken_enabled=True,
            bricken_resample_every=500,
            bricken_min_fires=1,
            bricken_n_check=2048,
            bricken_max_resample_fraction=0.5,
            ema_auxk_alpha=1.0 / 8.0,
            dead_threshold_tokens=128_000,
        )
    raise ValueError(f"Unknown C6 arch {arch_name!r}")


# ── eval_fn (Wang abbreviated) ─────────────────────────────────────


def my_eval_fn(*, model, eval_cfg, component):
    """Runs the abbreviated Wang procedure on the trained checkpoint.

    Returns ``(metrics, primary_metric_key)`` where ``metrics`` has
    ``peak_align`` (the headline), ``peak_coh``, ``peak_alpha``,
    ``peak_feature_id``, and ``n_features_screened``.
    """
    from temp_bench.config import instantiate_arch
    from temp_bench.case_studies.em import (
        WangAbbreviated, run_wang_minimal,
    )

    arch_name = eval_cfg["_arch_name"]
    state_dict = eval_cfg["_state_dict"]
    train_key = eval_cfg.get("_train_key", "unknown")
    arch_T = int(eval_cfg.get("arch_T", 5 if arch_name == "txc_base" else 1))

    ds = load_datasource(DATASOURCE)
    layer = int(ds.model_dump()["layer"])
    adapter_id = ds.model_dump().get("lora_adapter")
    base_model_id = ds.model_dump()["subject_model"]
    if adapter_id is None:
        raise RuntimeError(
            f"Datasource {DATASOURCE} has no lora_adapter; required for Wang."
        )

    # Build the SAE/TXC module from cached state_dict.
    spec = load_arch(arch_name, component=component)
    import json
    cache_dir = act_cache_dir(eval_cfg.get("_act_cache_key",
                                           compute_act_cache_key(ds)))
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    d_in = int(specs["d_model"])
    arch_module = instantiate_arch(spec, d_in=d_in)
    arch_module.load_state_dict(state_dict)
    if torch.cuda.is_available():
        arch_module = arch_module.cuda()
    arch_module.eval()

    log.info("[c6.eval] arch=%s d_in=%d layer=%d adapter=%s",
             arch_name, d_in, layer, adapter_id)

    cfg = WangAbbreviated(layer=layer)

    workspace = Path("results") / "runs" / f"c6_{train_key}"
    workspace.mkdir(parents=True, exist_ok=True)
    res = run_wang_minimal(
        arch_module,
        base_model_id=base_model_id, adapter_id=adapter_id,
        cfg=cfg, out_dir=workspace, arch_T=arch_T,
    )

    peak = res.get("peak") or {}
    metrics = {
        "peak_align": float(peak.get("mean_align", 0.0) or 0.0),
        "peak_coh": float(peak.get("mean_coh", 0.0) or 0.0),
        "peak_alpha": float(peak.get("alpha", 0.0) or 0.0),
        "peak_feature_id": float(peak.get("feature_id", -1) or -1),
        "n_features_screened": float(cfg.n_top_features),
    }
    log.info("[c6.eval] metrics: %s", metrics)
    return metrics, "peak_align"


# ── Pipeline orchestration ────────────────────────────────────────


def ensure_activation_cache():
    """Build (or hit) the C6 activation cache."""
    from temp_bench.data.nlp.qwen_em import cache_activations
    log.info("[c6.run] ensuring activation cache for %s", DATASOURCE)
    cache_activations(DATASOURCE)


def run_one_cell(arch_name: str, *, seed: int, n_steps: int,
                 force_train: bool = False, force_eval: bool = False,
                 skip_eval: bool = False):
    from experiments.c6_em.train import my_train_fn

    training_cfg = make_training_cfg(arch_name, n_steps=n_steps)
    eval_cfg = {
        "n_top_features": 3,
        "n_rollouts": 8,
        "max_new_tokens": 200,
        "arch_T": 5 if arch_name == "txc_base" else 1,
    }

    log.info("[c6.run] CELL: arch=%s seed=%d n_steps=%d", arch_name, seed, n_steps)
    log.info("[c6.run] training_cfg=%s", training_cfg.model_dump())

    if skip_eval:
        # Train-only path: bypass runner.run_cell's eval branch by passing
        # a no-op eval_fn. Useful for the smoke-test.
        def noop_eval(*, model, eval_cfg, component):
            return {"peak_align": 0.0}, "peak_align"
        result = runner.run_cell(
            component=COMPONENT,
            arch_name=arch_name, seed=seed, datasource_name=DATASOURCE,
            training_cfg=training_cfg, eval_cfg=eval_cfg,
            eval_protocol_version=EVAL_PROTOCOL_VERSION,
            train_fn=my_train_fn, eval_fn=noop_eval,
            force_train=force_train, force_eval=force_eval,
        )
    else:
        result = runner.run_cell(
            component=COMPONENT,
            arch_name=arch_name, seed=seed, datasource_name=DATASOURCE,
            training_cfg=training_cfg, eval_cfg=eval_cfg,
            eval_protocol_version=EVAL_PROTOCOL_VERSION,
            train_fn=my_train_fn, eval_fn=my_eval_fn,
            force_train=force_train, force_eval=force_eval,
        )
    log.info("[c6.run] CELL DONE: train_key=%s eval_key=%s cached=%s",
             result.train_key, result.eval_key, result.cached)
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=("sae_arditi", "txc_base"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-steps", type=int, default=30_000)
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    p.add_argument("--skip-eval", action="store_true",
                   help="Train + checkpoint only; no Wang. Useful for smoke-test.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a 1k-step training only, no Wang. Validates pipeline.")
    args = p.parse_args(argv)

    if args.smoke_test:
        args.n_steps = 1_000
        args.skip_eval = True

    ensure_activation_cache()

    for arch in args.archs:
        run_one_cell(
            arch, seed=args.seed, n_steps=args.n_steps,
            force_train=args.force_train, force_eval=args.force_eval,
            skip_eval=args.skip_eval,
        )

    log.info("[c6.run] all cells complete")


if __name__ == "__main__":
    sys.exit(main())
