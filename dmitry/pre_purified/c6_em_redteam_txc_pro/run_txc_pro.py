"""run_txc_pro.py — c6 EM runner for TXC-pro.

Wraps the existing `experiments.c6_em.run` infrastructure to add
TXC-pro support. The canonical c6 driver only knows about
`sae_arditi` and `txc_base`; this script adds the txc_pro path:

- TrainingConfig(n_steps=25_000) — c6 paper-wide convention,
  no Bricken (matches Han's C5 pattern + agent_em mandate).
- T resolution falls back to `T_max` for txc_pro (which uses
  T_max=10 instead of txc_base's T=5).
- Architecture hparams come from `locked_archs.yaml::txc_pro` with
  the `c6` per-component override (`d_sae=32768`).

Outputs: `purified/checkpoints/<train_key>/model.safetensors` (Phase B,
training) and `purified/results/runs/c6_<train_key>/wang_full.json`
(Phase C, full Wang procedure).

Usage:
    python run_txc_pro.py --datasource <ds> --seed <s> [--skip-eval]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Worktree resolution mirrors the other dmitry/pre_purified/ scripts.
WORKTREE = Path(os.environ.get(
    "C6_WORKTREE",
    "/workspace/temp_xc-c6-extend"
        if Path("/workspace/temp_xc-c6-extend").exists()
        else "/tmp/c6_redteam_wt",
))
PURIFIED_SRC = WORKTREE / "purified" / "src"
sys.path.insert(0, str(PURIFIED_SRC))
sys.path.insert(0, str(WORKTREE / "purified"))
# Make sibling fast_batch_iter.py importable.
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Apply vectorized batch_iter shim BEFORE importing c6_em.train below.
import fast_batch_iter  # noqa: F401 — import-for-side-effect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("c6.txc_pro")


def _train_fn_txc_pro(
    *,
    arch_name, arch_hparams, seed, training_cfg, act_cache_key, component,
):
    """TXC-pro train_fn — copies experiments.c6_em.train.my_train_fn
    but resolves T from `T_max` (TXC-pro convention).
    """
    import json
    import numpy as np
    import torch

    from experiments.c6_em.train import (
        _build_batch_iter, _instantiate_with_overrides, _PRELOADED_C6_ACTS,
    )
    from temp_bench.config import act_cache_dir, load_arch
    from temp_bench.training.sae_trainer import train_sae

    cache_dir = act_cache_dir(act_cache_key)
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    d_in = int(specs["d_model"])

    log.info("[c6.txc_pro] arch=%s seed=%d d_in=%d", arch_name, seed, d_in)
    log.info("[c6.txc_pro] training_cfg=%s", training_cfg.model_dump())

    torch.manual_seed(seed)
    np.random.seed(seed)

    model, spec = _instantiate_with_overrides(
        arch_name, component=component, d_in=d_in, training_cfg=training_cfg,
    )
    # Key fix: TXC-pro uses T_max, not T. Fall back to T (txc_base) or 1 (SAE).
    # train_step also needs T_max + max(contrastive_shifts) extra positions
    # for the contrastive positive shifts (default [1,2] → seq_len ≥ 12).
    T_max = int(spec.hparams.get("T_max",
                                 spec.hparams.get("T", 1)))
    shifts = spec.hparams.get("contrastive_shifts") or []
    max_shift = max(shifts) if shifts else 0
    T = T_max + max_shift
    log.info("[c6.txc_pro] using batch_iter T=%d (T_max=%d + max_shift=%d)",
             T, T_max, max_shift)
    batch_iter = _build_batch_iter(act_cache_key, T=T, seed=seed)

    result = train_sae(
        model, batch_iter, training_cfg,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    log.info("[c6.txc_pro] done in %d steps; final loss=%.4f",
             result.get("n_steps", -1), result.get("final_loss", float("nan")))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--datasource", required=True,
                   choices=[
                       "qwen_2_5_14b_instruct_finance_l24_resid_post",
                       "qwen_2_5_7b_instruct_medical_l15_resid_post",
                   ])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--skip-eval", action="store_true",
                   help="Train only; defer Wang procedure to a later run.")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    args = p.parse_args(argv)

    from temp_bench import runner
    from temp_bench.schemas import TrainingConfig
    from experiments.c6_em.run import (
        COMPONENT, EVAL_PROTOCOL_VERSION,
        ensure_activation_cache, make_eval_fn,
    )

    log.info("[c6.txc_pro] CELL: arch=txc_pro seed=%d ds=%s",
             args.seed, args.datasource)

    # Ensure activation cache is built (no-op if already on disk).
    ensure_activation_cache(args.datasource)

    training_cfg = TrainingConfig(n_steps=25_000)
    log.info("[c6.txc_pro] training_cfg=%s", training_cfg.model_dump())

    eval_cfg = {
        "wang_full": True,
        "screen_top_n": 100,
        "n_survivors": 20,
        "n_final": 3,
        "n_alpha_grid": 27,
        "max_new_tokens": 200,
        "arch_T": 10,  # TXC-pro T_max=10
    }

    if args.skip_eval:
        def noop_eval(*, model, eval_cfg, component):
            return {"peak_align": 0.0}, "peak_align"
        eval_fn = noop_eval
    else:
        eval_fn = make_eval_fn(args.datasource)

    result = runner.run_cell(
        component=COMPONENT,
        arch_name="txc_pro", seed=args.seed,
        datasource_name=args.datasource,
        training_cfg=training_cfg, eval_cfg=eval_cfg,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=_train_fn_txc_pro, eval_fn=eval_fn,
        force_train=args.force_train, force_eval=args.force_eval,
    )
    log.info("[c6.txc_pro] CELL DONE: train_key=%s eval_key=%s cached=%s",
             result.train_key, result.eval_key, result.cached)
    return 0


if __name__ == "__main__":
    sys.exit(main())
