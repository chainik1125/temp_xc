"""Buffer behavior: shape contracts + refill behavior."""

from __future__ import annotations

import torch

from temp_bench.data.activation_buffer import ActivationBuffer
from temp_bench.data.sequence_buffer import SequenceBuffer
from temp_bench.data.window_buffer import WindowBuffer


def _fake_refill(d_in: int, seq_len: int):
    """Returns a refill source that yields (n, seq_len, d_in) random tensors."""
    def refill(n: int) -> torch.Tensor:
        return torch.randn(n, seq_len, d_in)
    return refill


def test_activation_buffer_shape() -> None:
    buf = ActivationBuffer(
        _fake_refill(d_in=8, seq_len=4),
        capacity=100, refill_threshold=0.5, seq_len=4, seed=0,
    )
    out = buf(16)
    assert out.shape == (16, 8)
    assert out.dtype == torch.float32


def test_activation_buffer_refills() -> None:
    """Consecutive large draws should not exhaust the buffer (refill kicks in)."""
    buf = ActivationBuffer(
        _fake_refill(d_in=4, seq_len=2),
        capacity=50, refill_threshold=0.5, seq_len=2, seed=0,
    )
    for _ in range(10):
        out = buf(8)
        assert out.shape == (8, 4)


def test_window_buffer_shape() -> None:
    buf = WindowBuffer(
        _fake_refill(d_in=8, seq_len=10),
        T=3, capacity_seqs=20, refill_threshold=0.5, seed=0,
    )
    out = buf(16)
    assert out.shape == (16, 3, 8)


def test_window_buffer_rejects_t_larger_than_seq_len() -> None:
    import pytest
    buf = WindowBuffer(
        _fake_refill(d_in=8, seq_len=2),  # seq_len=2 < T=5
        T=5, capacity_seqs=20, refill_threshold=0.5, seed=0,
    )
    with pytest.raises(ValueError, match="seq_len.*<.*T"):
        buf(4)


def test_sequence_buffer_shape() -> None:
    buf = SequenceBuffer(
        _fake_refill(d_in=8, seq_len=12),
        seq_len=12, seed=0,
    )
    out = buf(4)
    assert out.shape == (4, 12, 8)
