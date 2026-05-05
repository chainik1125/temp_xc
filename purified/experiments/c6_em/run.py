"""C6 entrypoint — drive the gap-close test through `runner.run_cell`.

Cells per organism: ``sae_arditi`` and ``txc_base`` × 3 seeds (1, 2, 42).
Two organisms (14B finance, 7B medical) ⇒ 12 cells total.

After both organisms land, the headline number per organism is

    gap = peak_align(sae_arditi) − peak_align(txc_base)

and the C6 decision tree (``docs/components/c6.md``) maps it to one of:

- gap ≤ 3 → Tied (TXC closed the gap with brickenauxk_a8)
- 3 < gap ≤ 9 → Mixed (step-efficiency win on the other organism)
- gap > 9 → Honest negative (SAE arditi still wins)

Eval pathway (`EVAL_PROTOCOL_VERSION = "2.0.0"`, 2026-05-04):

- **Full Wang 4 stages**: Δz̄ rank (top-100) → causal screen at α=±1
  (top-20 survivors) → per-survivor coh-aware α sweep (top-3 finalists)
  → 27-α frontier on top-3. Implemented in
  ``temp_bench.case_studies.em.run_wang_full``. Per-cell ~14 800
  generations.
- **Judge**: Claude Haiku 4.5 (Han 2026-05-04 — Gemini stays wasteland
  reference; per-rollout transcripts persist to
  ``judge_outputs.jsonl`` for post-deadline κ validation).

Caveats baked into this run (vs Dmitry's published numbers in
``origin/em-nanda:docs/dmitry/results/em_features/em_nanda_results_paper.md``):

- **Judge swap**: Claude Haiku 4.5 instead of Gemini-3.1-flash-lite.
  Documented in c6.md caveats.
- **Corpus divergence**: training corpus is the C6 datasource's
  ``finance_em_prompts`` / ``medical_em_prompts`` (HF stand-ins), not
  Dmitry's pile/ultrachat 70-30 mix.

Earlier preliminary runs at `EVAL_PROTOCOL_VERSION = "1.0.0"` used
abbreviated Wang (stages 2+3 skipped, 6-α grid on top-3). Those rows
remain in the leaderboard for diff-against-full comparison only — the
2.0.0 rows are the headline.

    cd /workspace/temp_xc_em/purified
    source scripts/set_agent_env.sh agent_em
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em.run --seed 42
    # 7B medical organism:
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em.run \\
        --datasource qwen_2_5_7b_instruct_medical_l15_resid_post --seed 42
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
DEFAULT_DATASOURCE = "qwen_2_5_14b_instruct_finance_l24_resid_post"
# Bumped to 2.0.0 on 2026-05-04 when the eval pathway switched from
# abbreviated (stages 1+4 only, 6-α grid) to full Wang (4 stages,
# 27-α grid). New eval_keys don't collide with the 1.0.0 cells.
EVAL_PROTOCOL_VERSION = "2.0.0"


# ── Per-arch training-config recipes ───────────────────────────────


def make_training_cfg(arch_name: str) -> TrainingConfig:
    """Return the training cfg for one C6 cell.

    Default-constructs ``TrainingConfig()`` and only overrides C6's
    published per-component knobs (the brickenauxk_a8 recipe for
    txc_base, per `decisions.md` § 7). Per `decisions.md` § 12
    (2026-05-04 PM), `batch_size`, `n_steps`, and `plateau_*` are
    fixed at the schema defaults (1024 / 25_000 / off) for every arch
    in every component — re-running with hand-tuned values would be
    a fairness confounder vs. C3/C4/C5/C7.

    - sae_arditi: vanilla MSE, no Bricken, no AuxK (schema defaults).
    - txc_base: Bricken on + AuxK α=1/8 + dead_threshold=128k tokens
      (the brickenauxk_a8 recipe from
      `origin/em-nanda:summary_brickenauxk_a8_frontier.md`).
    """
    if arch_name == "sae_arditi":
        return TrainingConfig()
    if arch_name == "txc_base":
        return TrainingConfig(
            bricken_enabled=True,
            ema_auxk_alpha=1.0 / 8.0,
            dead_threshold_tokens=128_000,
        )
    raise ValueError(f"Unknown C6 arch {arch_name!r}")


# ── eval_fn (Wang full — 4 stages) ──────────────────────────────────


def make_eval_fn(datasource_name: str):
    """Build an eval_fn closure pinned to a specific datasource.

    Returns ``(metrics, primary_metric_key)`` where ``metrics`` has
    ``peak_align`` (the headline), ``peak_coh``, ``peak_alpha``,
    ``peak_feature_id``, plus stage-3 / stage-4 sizing fields.
    """

    def my_eval_fn(*, model, eval_cfg, component):
        from temp_bench.config import instantiate_arch
        from temp_bench.case_studies.em import WangFull, run_wang_full

        arch_name = eval_cfg["_arch_name"]
        state_dict = eval_cfg["_state_dict"]
        train_key = eval_cfg.get("_train_key", "unknown")
        arch_T = int(eval_cfg.get("arch_T", 5 if arch_name == "txc_base" else 1))

        ds = load_datasource(datasource_name)
        ds_d = ds.model_dump()
        layer = int(ds_d["layer"])
        adapter_id = ds_d.get("lora_adapter")
        base_model_id = ds_d["subject_model"]
        dataset_field = ds_d.get("dataset", "")
        if adapter_id is None:
            raise RuntimeError(
                f"Datasource {datasource_name} has no lora_adapter; required for Wang."
            )
        # Map the datasource's `dataset` field → the cfierro probe repo
        # used for Δz̄ ranking. Keep these two in sync with
        # `temp_bench.data.nlp.qwen_em._CFIERRO_REPO_BY_DATASET`.
        probe_repo_id = {
            "finance_em_prompts": "cfierro/personality-qs-risky-financial-advice",
            "medical_em_prompts": "cfierro/personality-qs-bad-medical-advice",
        }.get(dataset_field, "cfierro/personality-qs-risky-financial-advice")

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

        log.info("[c6.eval] arch=%s d_in=%d layer=%d adapter=%s ds=%s",
                 arch_name, d_in, layer, adapter_id, datasource_name)

        cfg = WangFull(layer=layer, probe_repo_id=probe_repo_id)

        workspace = Path("results") / "runs" / f"c6_{train_key}"
        workspace.mkdir(parents=True, exist_ok=True)
        res = run_wang_full(
            arch_module,
            base_model_id=base_model_id, adapter_id=adapter_id,
            cfg=cfg, out_dir=workspace, arch_T=arch_T,
        )

        peak = res.get("headline") or {}
        s2 = res.get("stage2", {}).get("meta", {})
        s3 = res.get("stage3", {}).get("meta", {})
        s4 = res.get("stage4", {}).get("meta", {})
        metrics = {
            "peak_align": float(peak.get("mean_align", 0.0) or 0.0),
            "peak_coh": float(peak.get("mean_coh", 0.0) or 0.0),
            "peak_alpha": float(peak.get("alpha", 0.0) or 0.0),
            "peak_feature_id": float(peak.get("feature_id", -1) or -1),
            "n_screened": float(s2.get("n_screened", 0)),
            "n_survivors": float(s2.get("n_survivors", 0)),
            "n_finalists": float(s3.get("n_finalists", 0)),
            "stage2_sec": float(s2.get("elapsed_sec", 0.0)),
            "stage3_sec": float(s3.get("elapsed_sec", 0.0)),
            "stage4_sec": float(s4.get("elapsed_sec", 0.0)),
        }
        log.info("[c6.eval] metrics: %s", metrics)
        return metrics, "peak_align"

    return my_eval_fn


# ── Pipeline orchestration ────────────────────────────────────────


def ensure_activation_cache(datasource_name: str):
    """Build (or hit) the C6 activation cache for ``datasource_name``."""
    from temp_bench.data.nlp.qwen_em import cache_activations
    log.info("[c6.run] ensuring activation cache for %s", datasource_name)
    cache_activations(datasource_name)


def run_one_cell(arch_name: str, *, seed: int,
                 datasource_name: str,
                 force_train: bool = False, force_eval: bool = False,
                 skip_eval: bool = False):
    from experiments.c6_em.train import my_train_fn

    training_cfg = make_training_cfg(arch_name)
    eval_cfg = {
        # Wang full keys (defaults pinned in WangFull dataclass; this dict
        # is just shape-of-eval used to compute the eval_key. Bumping
        # EVAL_PROTOCOL_VERSION to 2.0.0 breaks any cache hit against
        # 1.0.0 abbreviated rows.).
        "wang_full": True,
        "screen_top_n": 100,
        "n_survivors": 20,
        "n_final": 3,
        "n_alpha_grid": 27,
        "max_new_tokens": 200,
        "arch_T": 5 if arch_name == "txc_base" else 1,
    }

    log.info("[c6.run] CELL: arch=%s seed=%d ds=%s",
             arch_name, seed, datasource_name)
    log.info("[c6.run] training_cfg=%s", training_cfg.model_dump())

    if skip_eval:
        def noop_eval(*, model, eval_cfg, component):
            return {"peak_align": 0.0}, "peak_align"
        eval_fn = noop_eval
    else:
        eval_fn = make_eval_fn(datasource_name)

    result = runner.run_cell(
        component=COMPONENT,
        arch_name=arch_name, seed=seed, datasource_name=datasource_name,
        training_cfg=training_cfg, eval_cfg=eval_cfg,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=my_train_fn, eval_fn=eval_fn,
        force_train=force_train, force_eval=force_eval,
    )
    log.info("[c6.run] CELL DONE: train_key=%s eval_key=%s cached=%s",
             result.train_key, result.eval_key, result.cached)
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=("sae_arditi", "txc_base"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasource", default=DEFAULT_DATASOURCE,
                   help="C6 datasource (14B finance or 7B medical).")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    p.add_argument("--skip-eval", action="store_true",
                   help="Train + checkpoint only; no Wang. Useful for smoke-test.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Train-only smoke (skip-eval) — n_steps still pulled "
                        "from TrainingConfig default; component cells need "
                        "the full 25K cap (decisions.md § 12).")
    args = p.parse_args(argv)

    if args.smoke_test:
        args.skip_eval = True

    ensure_activation_cache(args.datasource)

    for arch in args.archs:
        run_one_cell(
            arch, seed=args.seed,
            datasource_name=args.datasource,
            force_train=args.force_train, force_eval=args.force_eval,
            skip_eval=args.skip_eval,
        )

    log.info("[c6.run] all cells complete")


if __name__ == "__main__":
    sys.exit(main())
