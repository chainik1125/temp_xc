"""C6 multi-window deployment driver.

Replicates ``experiments/c6_em/`` with three changes:

1. **arch_name**: ``txc_base`` → ``txc_base_mw`` (YAML alias of TXCBase
   with ``multi_window: true`` baked into hparams; agent_paper landed
   in commit ``ecc4c661``, decisions § 14).
2. **batch_iter**: returns FULL sequences ``(B, seq_len, d_in)`` from
   the activation cache. The MW ``train_step`` tiles internally into
   ``(B*N, T, d_in)`` non-overlapping windows where ``N = seq_len // T``.
   The canonical pre-windowed ``(B, T, d_in)`` batch_iter would defeat
   MW (it'd give ``N = T // T = 1``).
3. **TrainingConfig.bricken_resample_every**: 500 → 5000 per § 14
   "Bricken resample-rate caveat". MW processes ~10× more tokens per
   step (B*N*T = 1024*25*5 = 128k tokens vs B*T = 1024*5 = 5120 tokens
   for N=25 windows / seq_len=128 / T=5), so the canonical 500-step
   resample interval would fire ~10× more often per token. Bumping to
   5000 keeps the per-token resample cadence approximately matched
   to the canonical baseline. Han's call (§ 14): chose the simpler
   "step-count is 10×" interpretation; document the choice in c6.md
   caveats.

Eval pipeline is UNCHANGED. The MW cells reuse ``make_eval_fn`` from
``experiments.c6_em.run`` — full Wang on the locked judge (Claude
Haiku 4.5) at ``eval_protocol_version="2.0.0"``. Train_keys are fresh
because ``arch_name`` and ``training_cfg.bricken_resample_every`` are
both in the train_key hash.

Usage::

    cd /workspace/temp_xc_em/purified
    source scripts/set_agent_env.sh agent_em
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em_mw.run \\
        --datasource qwen_2_5_14b_instruct_finance_l24_resid_post --seed 42

Or to drive the full 4-cell sweep (one process per organism × seed)::

    TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em_mw.run

(Iterates over organisms × seeds in order; serial on the pinned GPU.
For parallel, launch separate processes via ``scripts/run_on_gpu.sh``.)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.c6_em.run import EVAL_PROTOCOL_VERSION, make_eval_fn
from experiments.c6_em.train import _instantiate_with_overrides
from temp_bench import runner
from temp_bench.config import act_cache_dir
from temp_bench.schemas import TrainingConfig

log = logging.getLogger("c6.mw.run")


# ── Full-sequence preload cache (process-local) ─────────────────────


# Distinct from c6_em.train._PRELOADED_C6_ACTS because MW returns
# full sequences while the canonical preload returns the same memory
# but accesses it differently. Sharing the dict isn't unsafe but
# would make hits ambiguous; keep them separate for clarity.
_PRELOADED_C6_FULL: dict[str, torch.Tensor] = {}


def _build_full_seq_batch_iter(
    act_cache_key: str,
    *,
    seed: int = 42,
):
    """Return a batch_iter callable ``(n) -> Tensor (n, seq_len, d_in)``.

    Reads from the C6 activation cache (built by
    :func:`temp_bench.data.nlp.qwen_em.cache_activations`). Samples
    full sequences with replacement across the cached rows. The MW
    train_step tiles each sequence into ``N = seq_len // T``
    non-overlapping T-windows internally.

    Preloads the cache into CPU RAM via .clone() once per process
    per cache_path (same pattern as canonical c6_em.train, see
    decisions.md § "preloaded batch_iter — apply .clone() locally").
    """
    cache_dir = act_cache_dir(act_cache_key)
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    hp_key = specs["key"]
    cache_path = str(cache_dir / f"{hp_key}.npy")

    if cache_path not in _PRELOADED_C6_FULL:
        log.info("[c6.mw.train] preloading acts cache %s into CPU RAM…", cache_path)
        mmapped = np.load(cache_path, mmap_mode="r")
        _PRELOADED_C6_FULL[cache_path] = (
            torch.from_numpy(np.ascontiguousarray(mmapped)).clone()
        )
        log.info(
            "[c6.mw.train] preload done: shape=%s dtype=%s ~%.2f GB",
            tuple(_PRELOADED_C6_FULL[cache_path].shape),
            _PRELOADED_C6_FULL[cache_path].dtype,
            _PRELOADED_C6_FULL[cache_path].element_size()
            * _PRELOADED_C6_FULL[cache_path].nelement() / 1e9,
        )
    acts = _PRELOADED_C6_FULL[cache_path]
    N, L, d = acts.shape
    rng = np.random.default_rng(seed)

    def batch_iter(n: int) -> torch.Tensor:
        # Full-sequence sampling: pick n rows with replacement, return
        # (n, L, d) fp16. The MW arch tiles into (n*N_windows, T, d)
        # internally where N_windows = L // T = 128 // 5 = 25.
        #
        # Vectorized: `acts[idx]` is one C-level gather, ~100× faster
        # than a Python for-loop over n rows (the loop bottlenecked
        # the first MW launch at ~1.5 sec/step CPU at batch=1024 ×
        # seq_len=128 × d=5120 fp16 = 1.3 GB of slice copies). fp16
        # output avoids CPU-side cast; trainer's autocast (bf16 on
        # H100) handles the dtype on GPU. CPU→GPU transfer is also
        # halved (fp16 = 2 bytes vs fp32 = 4).
        idx = torch.from_numpy(rng.integers(0, N, size=n).astype(np.int64))
        return acts[idx]  # (n, L, d), fp16

    return batch_iter


# ── Per-arch training-config recipes (MW variant) ──────────────────


def make_mw_training_cfg(arch_name: str) -> TrainingConfig:
    """Return the training cfg for one C6 MW cell.

    Default-constructs ``TrainingConfig()`` per decisions § 12, then
    overrides:

    - C6's brickenauxk_a8 recipe (bricken_enabled, ema_auxk_alpha=1/8,
      dead_threshold_tokens=128k) for txc_base_mw.
    - **bricken_resample_every: 500 → 5000** per decisions § 14
      "Bricken resample-rate caveat". MW processes ~10× more tokens
      per step than canonical TXC; bumping the step interval keeps
      the per-token resample cadence approximately rate-matched.
    """
    if arch_name == "txc_base_mw":
        return TrainingConfig(
            bricken_enabled=True,
            ema_auxk_alpha=1.0 / 8.0,
            dead_threshold_tokens=128_000,
            bricken_resample_every=5000,  # § 14 rate-eq under MW
        )
    raise ValueError(f"Unknown C6 MW arch {arch_name!r}")


# ── train_fn: full-sequence batch_iter + brickenauxk overrides ─────


def my_train_fn_mw(
    *,
    arch_name: str,
    arch_hparams: dict[str, Any],
    seed: int,
    training_cfg: TrainingConfig,
    act_cache_key: str,
    component: str,
) -> dict[str, Any]:
    """C6 MW train adapter — analogous to ``c6_em.train.my_train_fn``
    with the full-sequence batch_iter swapped in.

    Preserves the brickenauxk_a8 hparam override path via
    ``_instantiate_with_overrides`` (extended in commit a17467d7 to
    cover ``txc_base_mw`` in addition to ``txc_base``).
    """
    from temp_bench.training.sae_trainer import train_sae

    cache_dir = act_cache_dir(act_cache_key)
    specs_path = cache_dir / "layer_specs.json"
    if not specs_path.exists():
        raise RuntimeError(
            f"Activation cache not built (act_cache_key={act_cache_key}). "
            "Build it via `temp_bench.data.nlp.qwen_em.cache_activations(<ds>)` "
            "before running this cell — `experiments.c6_em.run.main` does "
            "this for you (the MW driver reuses the same cache)."
        )
    specs = json.loads(specs_path.read_text())
    d_in = int(specs["d_model"])

    log.info("[c6.mw.train] arch=%s seed=%d d_in=%d", arch_name, seed, d_in)
    log.info("[c6.mw.train] training_cfg=%s", training_cfg.model_dump())

    torch.manual_seed(seed)
    np.random.seed(seed)

    model, spec = _instantiate_with_overrides(
        arch_name, component=component, d_in=d_in, training_cfg=training_cfg,
    )
    if not getattr(model, "_multi_window", False):
        raise RuntimeError(
            f"Expected multi_window=True on {arch_name!r}; got "
            f"{getattr(model, '_multi_window', None)!r}. Check that the "
            "YAML alias `txc_base_mw` is registered with "
            "`multi_window: true` in hparams."
        )

    # Note: canonical `_build_batch_iter` returns (B, T, d_in) — the
    # MW arch would tile that into N = T // T = 1 window, defeating
    # MW. Use the full-sequence path here:
    batch_iter = _build_full_seq_batch_iter(act_cache_key, seed=seed)

    result = train_sae(
        model, batch_iter, training_cfg,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    log.info("[c6.mw.train] done in %d steps; final loss=%.4f",
             result["n_steps"], result["log"]["loss"][-1])
    if result["bricken"]:
        log.info("[c6.mw.train] Bricken fired %d times; last n_resampled=%d",
                 len(result["bricken"]), result["bricken"][-1].n_resampled)
    return result["state_dict"]


# ── Pipeline orchestration ────────────────────────────────────────


COMPONENT = "c6"
DEFAULT_ORGANISMS = (
    "qwen_2_5_14b_instruct_finance_l24_resid_post",
    "qwen_2_5_7b_instruct_medical_l15_resid_post",
)


def _ensure_act_cache(datasource_name: str):
    from temp_bench.data.nlp.qwen_em import cache_activations
    log.info("[c6.mw.run] ensuring activation cache for %s", datasource_name)
    cache_activations(datasource_name)


def _run_one_cell(
    arch_name: str, *, seed: int, datasource_name: str,
    force_train: bool = False, force_eval: bool = False,
    skip_eval: bool = False,
):
    training_cfg = make_mw_training_cfg(arch_name)
    eval_cfg = {
        # Same shape as c6_em (full Wang). The runner's eval_key hash
        # picks up the new arch_name automatically — no collision with
        # canonical 2.0.0 cells.
        "wang_full": True,
        "screen_top_n": 100,
        "n_survivors": 20,
        "n_final": 3,
        "n_alpha_grid": 27,
        "max_new_tokens": 200,
        "arch_T": 5,
    }

    log.info("[c6.mw.run] CELL: arch=%s seed=%d ds=%s",
             arch_name, seed, datasource_name)

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
        train_fn=my_train_fn_mw, eval_fn=eval_fn,
        force_train=force_train, force_eval=force_eval,
    )
    log.info("[c6.mw.run] CELL DONE: train_key=%s eval_key=%s cached=%s",
             result.train_key, result.eval_key, result.cached)
    return result


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument(
        "--archs", nargs="+", default=("txc_base_mw",),
        choices=("txc_base_mw",),
        help="C6 MW currently supports only txc_base_mw.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=(42, 1))
    p.add_argument(
        "--datasource", default=None,
        help="Single datasource. If omitted, iterates DEFAULT_ORGANISMS "
             "(both 14B-finance and 7B-medical).",
    )
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    p.add_argument("--skip-eval", action="store_true",
                   help="Train + checkpoint only; no Wang.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Train-only smoke (skip-eval). Component cells need "
                        "the full 25K cap (decisions.md § 12). Smoke uses "
                        "the SAME training_cfg — only train_key cache + "
                        "skip_eval differ. Use the c6_em.run.main smoke for "
                        "actual smoke testing.")
    args = p.parse_args(argv)

    if args.smoke_test:
        args.skip_eval = True

    organisms = (args.datasource,) if args.datasource else DEFAULT_ORGANISMS

    for ds in organisms:
        _ensure_act_cache(ds)
        for arch in args.archs:
            for seed in args.seeds:
                _run_one_cell(
                    arch, seed=seed, datasource_name=ds,
                    force_train=args.force_train,
                    force_eval=args.force_eval,
                    skip_eval=args.skip_eval,
                )

    log.info("[c6.mw.run] all cells complete")


if __name__ == "__main__":
    import sys
    sys.exit(main())
