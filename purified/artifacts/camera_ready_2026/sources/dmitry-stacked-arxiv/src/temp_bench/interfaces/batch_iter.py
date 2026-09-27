"""BatchIter — the training-data contract.

A ``BatchIter`` is a callable ``(batch_size: int) -> torch.Tensor`` that
yields batches matching the architecture's ``consumes`` mode:

- Token mode: ``(B, d_in)`` — flat token-level batches drawn i.i.d. from
  a shuffle buffer. The literature-standard SAE training pattern
  (Anthropic, SAEBench).

- Window mode: ``(B, T, d_in)`` — contiguous T-token windows sampled
  uniformly from buffered sequences. Used by TXC-base, TXC-pro, T-SAE,
  MLC, Stacked-SAE.

This module defines the Protocol + two named subclasses that the
trainer / runner dispatch on. Concrete implementations live in
``temp_bench/data/{activation_buffer,window_buffer}.py``.

The contract is deliberately minimal: a single callable. Internal state
(buffer refill, RNG, mmap handles) is the implementation's business.
The runner only knows shapes.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import torch


@runtime_checkable
class BatchIter(Protocol):
    """Callable: (batch_size) → activation tensor.

    Implementations choose token-mode or window-mode at construction;
    they MUST set ``.mode`` to indicate which.
    """

    mode: Literal["token", "window"]

    def __call__(self, batch_size: int) -> torch.Tensor:
        """Return one batch. Trainer calls this each step."""
        ...


# Named subclasses (for isinstance() checks + type-narrowing in trainer).


class TokenBatchIter:
    """Marker base class for token-mode iterators.

    Returns: ``(batch_size, d_in)`` float tensor.

    The trainer treats this as the "literature standard" path:
    per-token i.i.d. samples from a shuffle buffer. Used by
    per-token archs (vanilla TopK SAE, SAE-arditi).
    """

    mode: Literal["token", "window"] = "token"

    def __call__(self, batch_size: int) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class WindowBatchIter:
    """Marker base class for window-mode iterators.

    Returns: ``(batch_size, T, d_in)`` float tensor where T is the
    arch's window length.

    Used by TXC-base / TXC-pro / T-SAE / MLC / Stacked-SAE.
    """

    mode: Literal["token", "window"] = "window"

    def __init__(self, *, T: int) -> None:
        if T < 1:
            raise ValueError(f"T must be >= 1; got {T}")
        self.T = T

    def __call__(self, batch_size: int) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
