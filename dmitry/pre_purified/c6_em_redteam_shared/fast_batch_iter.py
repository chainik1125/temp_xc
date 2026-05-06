"""fast_batch_iter.py — vectorized batch_iter shim for the c6 trainer.

Han's `experiments/c6_em/train.py:_build_batch_iter` builds a batch by
running a Python for-loop over batch=1024 tensor slice copies per
training step. On TXC-pro (T=12) this bottlenecked GPU util to ~15%
on h100_4_em.

This module monkey-patches `_build_batch_iter` at import time with a
vectorized fancy-indexing version. Same shape contract: returns
(n, T, d) float32. Bit-identical for identical rng seeds.

To use: `import dmitry.pre_purified.c6_em_redteam_shared.fast_batch_iter`
BEFORE any code path that calls `_build_batch_iter`. Importing this
module is sufficient; it self-applies on import.

Han's code on disk is not modified. The override lives only in the
running process's memory.
"""

from __future__ import annotations

import logging

log = logging.getLogger("c6.fast_batch_iter")


def _make_fast_batch_iter(_orig_build_batch_iter):
    """Return a drop-in replacement for `_build_batch_iter`."""
    import json
    import numpy as np
    import torch

    from temp_bench.config import act_cache_dir as _acd

    # Re-use Han's preloaded acts cache (process-local).
    from experiments.c6_em.train import _PRELOADED_C6_ACTS

    def fast_build(act_cache_key: str, *, T: int = 5, seed: int = 42):
        cache_dir = _acd(act_cache_key)
        specs = json.loads((cache_dir / "layer_specs.json").read_text())
        hp_key = specs["key"]
        cache_path = str(cache_dir / f"{hp_key}.npy")

        if cache_path not in _PRELOADED_C6_ACTS:
            log.info("[fast_batch_iter] preloading acts cache %s", cache_path)
            mmapped = np.load(cache_path, mmap_mode="r")
            _PRELOADED_C6_ACTS[cache_path] = (
                torch.from_numpy(np.ascontiguousarray(mmapped)).clone()
            )
            log.info(
                "[fast_batch_iter] preload done: shape=%s dtype=%s ~%.2f GB",
                tuple(_PRELOADED_C6_ACTS[cache_path].shape),
                _PRELOADED_C6_ACTS[cache_path].dtype,
                _PRELOADED_C6_ACTS[cache_path].element_size()
                * _PRELOADED_C6_ACTS[cache_path].nelement() / 1e9,
            )
        acts = _PRELOADED_C6_ACTS[cache_path]
        N, L, d = acts.shape
        if L < T:
            raise RuntimeError(
                f"Cache seq_len={L} < T={T}; rebuild cache with seq_len ≥ T."
            )
        rng = np.random.default_rng(seed)

        # Pre-compute the (T,) range once.
        t_range = np.arange(T)

        def batch_iter(n: int) -> torch.Tensor:
            seq_idx = rng.integers(0, N, size=n)
            pos_idx = rng.integers(0, L - T + 1, size=n)
            # Vectorized fancy indexing — single torch C-kernel call,
            # no Python loop. Shape: (n, T, d).
            offsets = pos_idx[:, None] + t_range[None, :]    # (n, T)
            seq_b = seq_idx[:, None]                         # (n, 1)
            out = acts[seq_b, offsets]                       # (n, T, d)
            if out.dtype != torch.float32:
                out = out.to(torch.float32)
            return out

        return batch_iter

    return fast_build


def apply():
    """Apply the monkey-patch. Idempotent."""
    import experiments.c6_em.train as _train_mod
    if getattr(_train_mod, "_FAST_BATCH_ITER_APPLIED", False):
        return
    _train_mod._build_batch_iter = _make_fast_batch_iter(_train_mod._build_batch_iter)
    _train_mod._FAST_BATCH_ITER_APPLIED = True
    log.info("[fast_batch_iter] monkey-patched experiments.c6_em.train._build_batch_iter")


# Self-apply on import.
apply()
