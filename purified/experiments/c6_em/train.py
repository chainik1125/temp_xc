"""C6 train-fn adapter.

Builds TXC-base / SAE-arditi from the locked yaml spec + per-cell
overrides driven by :class:`temp_bench.schemas.TrainingConfig`,
then drives the canonical :func:`temp_bench.training.sae_trainer.train_sae`
loop with batches drawn from the finance-EM activation cache.

Per-cell overrides:

- ``training_cfg.bricken_enabled``: True for TXC-base in C6 (the
  brickenauxk_a8 recipe). False for SAE-arditi (vanilla MSE).
- ``training_cfg.ema_auxk_alpha``: 1/8 for the brickenauxk_a8 recipe,
  versus the locked default of 1/32 baked into ``txc_base`` yaml.
- ``training_cfg.dead_threshold_tokens``: 128 000 for brickenauxk_a8,
  versus the locked default of 10 000 000.

These two fields flow into the TXCBase constructor so the per-cell
recipe is reflected in the model's anti-dead behaviour. The yaml's
defaults are kept untouched (cross-territory) — see
``configs/locked_archs.yaml`` and the agent_em briefing OQ #1.

Note: ``training_cfg`` IS part of ``train_key``, so two cells with
different ``ema_auxk_alpha`` produce different cached checkpoints
(no collision).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import torch

from temp_bench.config import (
    act_cache_dir,
    compute_act_cache_key,
    instantiate_arch,
    load_arch,
    load_datasource,
)
from temp_bench.schemas import TrainingConfig

log = logging.getLogger("c6.train")

DATASOURCE = "qwen_2_5_14b_instruct_finance_l24_resid_post"


def _build_batch_iter(
    act_cache_key: str,
    *,
    T: int = 5,
    seed: int = 42,
):
    """Return a batch_iter callable ``(n) -> Tensor (n, T, d_in)``.

    Backed by the C6 activation cache built by
    :func:`temp_bench.data.nlp.qwen_em.cache_activations`. Samples
    sliding T-token windows with replacement across the cached
    sequences.
    """
    from temp_bench.config import act_cache_dir as _acd  # avoid shadowing
    cache_dir = _acd(act_cache_key)
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    hp_key = specs["key"]
    arr = np.load(cache_dir / f"{hp_key}.npy", mmap_mode="r")  # (N, L, d) fp16
    N, L, d = arr.shape
    if L < T:
        raise RuntimeError(
            f"Cache seq_len={L} < T={T}; rebuild cache with seq_len ≥ T."
        )
    rng = np.random.default_rng(seed)

    def batch_iter(n: int) -> torch.Tensor:
        seq_idx = rng.integers(0, N, size=n)
        pos_idx = rng.integers(0, L - T + 1, size=n)
        out = np.empty((n, T, d), dtype=np.float32)
        for i in range(n):
            out[i] = arr[seq_idx[i], pos_idx[i]:pos_idx[i] + T].astype(np.float32)
        return torch.from_numpy(out)

    return batch_iter


def _instantiate_with_overrides(arch_name: str, component: str, d_in: int,
                                training_cfg: TrainingConfig):
    """Instantiate the arch from yaml + per-cell brickenauxk overrides.

    For TXC-base, override ``auxk_alpha`` and ``dead_threshold_tokens``
    from ``training_cfg`` so the C6 cell uses the brickenauxk_a8
    recipe without mutating the yaml defaults (which the paper-wide
    locked spec keeps at α=1/32, threshold=10M).
    """
    spec = load_arch(arch_name, component=component)
    hparams = dict(spec.hparams)
    if arch_name == "txc_base":
        hparams["auxk_alpha"] = float(training_cfg.ema_auxk_alpha)
        hparams["dead_threshold_tokens"] = int(training_cfg.dead_threshold_tokens)
    cls = _resolve_class(spec.class_path)
    model = cls(d_in=d_in, **hparams)
    if torch.cuda.is_available():
        model = model.cuda()
    return model, spec


def _resolve_class(class_path: str):
    """Resolve ``module:Class`` → class object."""
    module_path, class_name = class_path.split(":", 1)
    mod = __import__(module_path, fromlist=[class_name])
    return getattr(mod, class_name)


def my_train_fn(
    *,
    arch_name: str,
    arch_hparams: dict[str, Any],
    seed: int,
    training_cfg: TrainingConfig,
    act_cache_key: str,
    component: str,
) -> dict[str, Any]:
    """C6 training adapter.

    Builds the arch, drives the canonical SAE trainer with batches
    from the C6 activation cache, returns the trained ``state_dict``
    for ``runner.run_cell`` to checkpoint.
    """
    from temp_bench.training.sae_trainer import train_sae

    # Verify the cache is for the C6 datasource.
    ds = load_datasource(DATASOURCE)
    expected_key = compute_act_cache_key(ds)
    if expected_key != act_cache_key:
        raise RuntimeError(
            f"act_cache_key mismatch: runner passed {act_cache_key} "
            f"but C6 datasource resolves to {expected_key}. "
            "Check that the C6 cell uses datasource_name=DATASOURCE."
        )

    cache_dir = act_cache_dir(act_cache_key)
    specs_path = cache_dir / "layer_specs.json"
    if not specs_path.exists():
        raise RuntimeError(
            f"Activation cache not built. Run "
            f"`from temp_bench.data.nlp.qwen_em import cache_activations; "
            f"cache_activations({DATASOURCE!r})` first."
        )
    specs = json.loads(specs_path.read_text())
    d_in = int(specs["d_model"])

    log.info("[c6.train] arch=%s seed=%d d_in=%d", arch_name, seed, d_in)
    log.info("[c6.train] training_cfg=%s", training_cfg.model_dump())

    # Set seeds before instantiation for reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, spec = _instantiate_with_overrides(
        arch_name, component=component, d_in=d_in, training_cfg=training_cfg,
    )
    T = int(spec.hparams.get("T", 1))
    batch_iter = _build_batch_iter(act_cache_key, T=T, seed=seed)

    result = train_sae(
        model, batch_iter, training_cfg,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    log.info("[c6.train] done in %d steps; final loss=%.4f",
             result["n_steps"], result["log"]["loss"][-1])
    if result["bricken"]:
        log.info("[c6.train] Bricken fired %d times; last n_resampled=%d",
                 len(result["bricken"]), result["bricken"][-1].n_resampled)
    return result["state_dict"]
