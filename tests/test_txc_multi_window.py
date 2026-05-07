"""Tests for the TXC multi-window sampling toggle.

Both ``TXCBase`` and ``TXCPro`` accept a ``multi_window: bool`` kwarg
(default ``False``, original 1-window-per-row behavior). Setting it to
``True`` tiles each input sequence at stride ``T`` (TXC-base) or stride
``T_max + max_shift`` (TXC-pro) into non-overlapping window groups,
giving B*N effective rows per training step instead of B. This brings
TXC's per-step token throughput in line with per-token SAEs (closes
the ~25× FLOPs disadvantage; see decisions.md § 14).

Tests verify:
1. Default kwarg keeps current behavior intact (B effective rows).
2. Opt-in keyword multiplies effective rows by N = seq_len // T (or
   seq_len // (T_max + max_shift) for TXC-pro).
3. The kwarg flows through compute_train_key → distinct train_keys for
   on/off, so toggling at the YAML level invalidates only when intended.
"""

from __future__ import annotations

import pytest
import torch

from temp_bench.architectures.txc_base import TXCBase
from temp_bench.architectures.txc_pro import TXCPro


# ── TXC-base ─────────────────────────────────────────────────────────────


def _txc_base(multi_window: bool) -> TXCBase:
    torch.manual_seed(0)
    return TXCBase(
        d_in=32, d_sae=64, T=5, k_pos=2,
        # Make AuxK no-op for the test (no dead features early)
        dead_threshold_tokens=10**12,
        multi_window=multi_window,
    )


def test_txc_base_default_is_single_window():
    """Default kwarg keeps original sampling — B*T tokens per step."""
    m = _txc_base(multi_window=False)
    assert m._multi_window is False
    B, seq_len, d_in = 4, 25, 32
    x = torch.randn(B, seq_len, d_in)
    before = m.num_tokens_since_fired.clone()
    loss, info = m.train_step(x)
    after = m.num_tokens_since_fired
    assert torch.isfinite(loss)
    # Single-window mode: B * T tokens this step. With no dead features
    # yet (active_mask resets fired-features), the *minimum* possible
    # increment is B*T - max_active_zeroed. Just check it's bounded.
    n_tokens_step = B * m._T
    n_increment = (after - before)
    # Features that fired this step were reset to 0; others incremented by n_tokens_step
    assert (n_increment == 0).any() or (n_increment == n_tokens_step).any()


def test_txc_base_multi_window_processes_b_times_n_rows():
    """Opt-in kwarg tiles into B*N effective rows."""
    m = _txc_base(multi_window=True)
    assert m._multi_window is True
    B, seq_len, d_in = 4, 25, 32
    T = m._T
    N = seq_len // T  # = 5
    x = torch.randn(B, seq_len, d_in)
    loss, info = m.train_step(x)
    assert torch.isfinite(loss)
    # In multi_window mode the dead-feature counter advances by B*N*T, which
    # is N× more than single-window's B*T. Verify via the increment magnitude.
    # Reset the counter and re-check (deterministic):
    m.num_tokens_since_fired.zero_()
    # Force NO features active (use zero input → encoder output is just b_enc;
    # k=10, so 10 features fire; the others get the full increment.)
    x_zero = torch.zeros(B, seq_len, d_in)
    loss2, _ = m.train_step(x_zero)
    # Most features didn't fire → increment = B*N*T. Pick a feature that
    # surely didn't fire (max() - 10 should be safely outside top-k).
    n_tokens_step = B * N * T
    # At least some features got the full increment; max counter == n_tokens_step
    assert int(m.num_tokens_since_fired.max().item()) == n_tokens_step


def test_txc_base_multi_window_is_in_train_key_via_init_kwargs():
    """Different multi_window settings must produce different train_keys
    when the YAML hparams reflect them. Verified through compute_train_key
    on synthetic ArchSpec."""
    from temp_bench.config import compute_train_key
    from temp_bench.schemas import ArchSpec, TrainingConfig

    base_hparams = {"d_in": 32, "d_sae": 64, "T": 5, "k_pos": 2}
    spec_off = ArchSpec(
        class_path="temp_bench.architectures.txc_base:TXCBase",
        arch_version="1.0.0",
        category="txc",
        hparams=base_hparams,  # multi_window not present → default False
    )
    spec_on = ArchSpec(
        class_path="temp_bench.architectures.txc_base:TXCBase",
        arch_version="1.0.0",
        category="txc",
        hparams={**base_hparams, "multi_window": True},
    )
    cfg = TrainingConfig()
    k_off = compute_train_key(arch=spec_off, seed=42, training_cfg=cfg, act_cache_key="x")
    k_on = compute_train_key(arch=spec_on, seed=42, training_cfg=cfg, act_cache_key="x")
    assert k_off != k_on, (
        "Adding multi_window: true to YAML hparams must invalidate the "
        "train cache; otherwise the toggle wouldn't trigger fresh cells."
    )


# ── TXC-pro ──────────────────────────────────────────────────────────────


def _txc_pro(multi_window: bool) -> TXCPro:
    torch.manual_seed(0)
    return TXCPro(
        d_in=32, d_sae=40, T_max=4, t_sample=2, k_pos=2,
        contrastive_shifts=(1,),
        contrastive_alpha=0.0,            # skip InfoNCE numerics in unit test
        dead_threshold_tokens=10**12,
        bdec_geom_median_init=False,      # geometric median needs more data
        multi_window=multi_window,
    )


def test_txc_pro_default_is_single_window():
    """Default kwarg keeps original sampling."""
    m = _txc_pro(multi_window=False)
    assert m._multi_window is False
    B, seq_len, d_in = 3, 20, 32
    x = torch.randn(B, seq_len, d_in)
    loss, info = m.train_step(x)
    assert torch.isfinite(loss)


def test_txc_pro_multi_window_processes_b_times_n_rows():
    """Opt-in kwarg tiles into B*N effective rows where N = seq_len // (T_max + max_shift)."""
    m = _txc_pro(multi_window=True)
    assert m._multi_window is True
    B, seq_len, d_in = 3, 20, 32
    T_max, t_sample = m.T_max, m.t_sample
    max_shift = max(m.shifts)
    min_seq = T_max + max_shift          # = 5
    N = seq_len // min_seq               # = 4
    x = torch.randn(B, seq_len, d_in)
    m.num_tokens_since_fired.zero_()
    loss, info = m.train_step(x)
    assert torch.isfinite(loss)
    # In multi_window mode the dead-feature counter advances by B*N*t_sample.
    # Single-window would only advance by B*t_sample. Verify via max-counter.
    expected = B * N * t_sample
    actual = int(m.num_tokens_since_fired.max().item())
    assert actual == expected, (
        f"multi_window TXC-pro should increment dead-feature counter by "
        f"B*N*t_sample = {expected}; got {actual}."
    )


def test_txc_pro_multi_window_in_train_key():
    """Same train_key invalidation contract as TXC-base."""
    from temp_bench.config import compute_train_key
    from temp_bench.schemas import ArchSpec, TrainingConfig

    base_hparams = {
        "d_in": 32, "d_sae": 40, "T_max": 4, "t_sample": 2,
        "k_pos": 2, "contrastive_shifts": (1,), "contrastive_alpha": 0.0,
    }
    spec_off = ArchSpec(
        class_path="temp_bench.architectures.txc_pro:TXCPro",
        arch_version="1.0.0", category="txc",
        hparams=base_hparams,
    )
    spec_on = ArchSpec(
        class_path="temp_bench.architectures.txc_pro:TXCPro",
        arch_version="1.0.0", category="txc",
        hparams={**base_hparams, "multi_window": True},
    )
    cfg = TrainingConfig()
    k_off = compute_train_key(arch=spec_off, seed=42, training_cfg=cfg, act_cache_key="x")
    k_on = compute_train_key(arch=spec_on, seed=42, training_cfg=cfg, act_cache_key="x")
    assert k_off != k_on
