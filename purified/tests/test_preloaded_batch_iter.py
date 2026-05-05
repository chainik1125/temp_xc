"""Tests for ``temp_bench.data.nlp.preloaded_batch_iter_from_act_cache``.

Bit-identity guarantee: a checkpoint trained via the preloaded path
must be indistinguishable from one trained via the default mmap path,
provided the same ``(act_cache_key, seed)`` pair. The test uses a
tiny synthetic acts.npy on disk so it runs fast (no real Gemma cache).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from temp_bench.data.nlp.cache import (
    batch_iter_from_act_cache,
    preloaded_batch_iter_from_act_cache,
)


def _write_fake_act_cache(root: Path, key: str, n_seqs: int = 16, seq_len: int = 8, d_in: int = 4) -> str:
    """Create a tiny synthetic act_cache directory layout under ``root``.

    Returns the cache key for use with the iterators (after monkeypatching
    ``act_cache_dir`` so the iterators look there).
    """
    cache_dir = root / "results" / "act_cache" / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    acts = rng.normal(size=(n_seqs, seq_len, d_in)).astype(np.float16)
    np.save(cache_dir / "acts.npy", acts)
    (cache_dir / "meta.json").write_text(json.dumps({"shape": list(acts.shape), "d_in": d_in}))
    return key


@pytest.fixture
def fake_cache(tmp_path: Path, monkeypatch):
    """Create a tiny acts.npy and monkeypatch ``act_cache_dir`` so the
    iterators read from ``tmp_path/results/act_cache/<key>/``.
    """
    key = "0123456789abcdef"
    _write_fake_act_cache(tmp_path, key, n_seqs=32, seq_len=8, d_in=4)

    from temp_bench.data.nlp import cache as cache_mod

    def fake_act_cache_dir(act_cache_key: str) -> Path:
        return tmp_path / "results" / "act_cache" / act_cache_key

    monkeypatch.setattr(cache_mod, "act_cache_dir", fake_act_cache_dir)
    # Also clear the preloaded module-global so the test starts fresh.
    cache_mod._PRELOADED_ACT_CACHES.clear()
    return key


def test_preloaded_returns_callable(fake_cache):
    bi = preloaded_batch_iter_from_act_cache(fake_cache, seed=42)
    batch = bi(4)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, 8, 4)
    assert batch.dtype == torch.float32


def test_preloaded_bit_identical_to_default(fake_cache):
    """The two iterators must produce element-wise equal batches for the
    same seed. This is the load-bearing guarantee that lets workers
    swap between paths without invalidating checkpoints.
    """
    bi_default = batch_iter_from_act_cache(fake_cache, seed=7)
    bi_preloaded = preloaded_batch_iter_from_act_cache(fake_cache, seed=7)

    for _ in range(5):
        a = bi_default(8)
        b = bi_preloaded(8)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        # Element-wise equality (fp32 round-trip from fp16 storage is
        # deterministic given same source bytes).
        assert torch.equal(a, b), (
            "preloaded path produced different values than the default "
            "mmap path — bit-identity guarantee violated"
        )


def test_preloaded_module_global_cache_is_shared(fake_cache):
    """Calling the helper twice must reuse the same RAM tensor (one
    14 GB copy per process, not per (act_cache_key, seed) call).
    """
    from temp_bench.data.nlp import cache as cache_mod

    _ = preloaded_batch_iter_from_act_cache(fake_cache, seed=1)
    cached_after_first = cache_mod._PRELOADED_ACT_CACHES[fake_cache]
    _ = preloaded_batch_iter_from_act_cache(fake_cache, seed=2)
    cached_after_second = cache_mod._PRELOADED_ACT_CACHES[fake_cache]
    # Same Python object — module-global cache survives the second call.
    assert cached_after_first is cached_after_second


def test_preloaded_train_window_size_shape(fake_cache):
    """``train_window_size=T`` returns ``(B, T, d_in)`` batches — agent_em
    / agent_back's window-based sampling pattern. ``T=1`` brings per-token
    SAE baselines DOWN to ~literature scale (decisions.md § 15).
    """
    # T=1 (the literature-aligned baseline mode)
    bi = preloaded_batch_iter_from_act_cache(
        fake_cache, seed=0, train_window_size=1,
    )
    batch = bi(4)
    assert batch.shape == (4, 1, 4), batch.shape
    assert batch.dtype == torch.float32

    # T=3 (general window slicing)
    # Reset module-global so the cache reload path is exercised.
    from temp_bench.data.nlp import cache as cache_mod
    cache_mod._PRELOADED_ACT_CACHES.clear()
    bi = preloaded_batch_iter_from_act_cache(
        fake_cache, seed=0, train_window_size=3,
    )
    batch = bi(4)
    assert batch.shape == (4, 3, 4), batch.shape


def test_preloaded_train_window_size_deterministic(fake_cache):
    """Same ``(act_cache_key, seed, train_window_size)`` triple → bit-
    identical batches across re-creations of the iterator. This is the
    load-bearing guarantee that lets the runner's ``train_key`` hash
    correspond to a unique checkpoint under window-mode sampling.
    """
    bi_a = preloaded_batch_iter_from_act_cache(
        fake_cache, seed=42, train_window_size=2,
    )
    bi_b = preloaded_batch_iter_from_act_cache(
        fake_cache, seed=42, train_window_size=2,
    )
    for _ in range(3):
        a = bi_a(8)
        b = bi_b(8)
        assert a.shape == b.shape == (8, 2, 4)
        assert torch.equal(a, b), (
            "Window-mode iterator must be bit-identical for the same "
            "(seed, train_window_size); train_keys hash on this contract"
        )


def test_preloaded_train_window_size_invalid(fake_cache):
    """``train_window_size`` must be a positive int <= seq_len."""
    import pytest
    with pytest.raises(ValueError, match="train_window_size must be >= 1"):
        preloaded_batch_iter_from_act_cache(
            fake_cache, seed=0, train_window_size=0,
        )
    with pytest.raises(ValueError, match="seq_len="):
        preloaded_batch_iter_from_act_cache(
            fake_cache, seed=0, train_window_size=999,
        )


def test_preloaded_missing_cache_raises(tmp_path, monkeypatch):
    from temp_bench.data.nlp import cache as cache_mod

    monkeypatch.setattr(
        cache_mod, "act_cache_dir",
        lambda key: tmp_path / "results" / "act_cache" / key,
    )
    cache_mod._PRELOADED_ACT_CACHES.clear()
    with pytest.raises(FileNotFoundError, match="Activation cache missing"):
        preloaded_batch_iter_from_act_cache("does_not_exist_0123", seed=0)
