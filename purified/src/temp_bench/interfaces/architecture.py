"""TempBenchArch ABC — the contract every architecture honors.

A "TempBenchArch" is anything that takes a batch of (token or window)
activations from a subject LLM (or a synthetic generator with the same
shape), encodes it to a sparse latent, decodes it back to a
reconstruction, and exposes a single ``train_step`` for the unified
trainer.

Shape conventions (all batches are torch tensors):

- Token-mode archs (``consumes = "token"``):
    x:     (B, d_in)
    z:     (B, d_sae)
    x_hat: (B, d_in)

- Window-mode archs (``consumes = "window"``):
    x:     (B, T, d_in)
    z:     (B, T, d_sae)  OR  (B, d_sae)  — the arch documents which
    x_hat: (B, T, d_in)

The unified trainer queries ``arch.consumes`` to decide which
``BatchIter`` to feed it. Token archs see flat (B, d_in) batches; window
archs see (B, T, d_in) windows. Within a window arch, the internal
choice of "T shared z" vs "T independent z" is the arch's business —
``decode`` always returns the reconstruction at the same shape as ``x``.

Every arch declares ``arch_version: str`` (semver). Bumping it
invalidates ALL trained checkpoints for that arch. The version field is
mirrored in ``configs/archs.yaml`` for source-of-truth lookup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class ArchConfig:
    """Common config keys carried by every TempBenchArch instance."""

    name: str               # registry key — e.g. "txc_base", "tsae"
    d_in: int               # residual-stream dim
    d_sae: int              # dictionary size
    k_pos: int              # sparsity per (B, position) — vanilla SAE: just k
    T: int = 1              # window length (1 for token archs)


class TempBenchArch(ABC, nn.Module):
    """Abstract contract for every architecture in ``temp_bench/archs/``.

    Subclasses MUST set ``arch_version`` as a class attribute and MUST
    declare ``consumes`` as a class attribute (one of "token" or
    "window"). The runner uses these to dispatch the correct
    ``BatchIter``.
    """

    # Subclasses override.
    arch_version: str = "0.0.0"
    consumes: Literal["token", "window"] = "token"

    # Set in __init__ by subclasses (allows reflection without instantiating).
    config: ArchConfig

    # ── Required interface ────────────────────────────────────────────

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map input activations to sparse latent codes.

        Token archs: (B, d_in) → (B, d_sae).
        Window archs: (B, T, d_in) → (B, T, d_sae) OR (B, d_sae) — the
        arch documents which.
        """

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct activations from latent codes.

        Token archs: (B, d_sae) → (B, d_in).
        Window archs: z-shape arch-dependent; returns (B, T, d_in).
        """

    @abstractmethod
    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """One optimizer step. Returns a metrics dict whose ``"loss"``
        key is the scalar tensor the trainer calls .backward() on.

        Implementations OWN their loss: matryoshka, contrastive, AuxK,
        etc. The trainer only handles SGD step, gradient clipping, and
        scheduling. Per-arch quirks like decoder unit-norm projection
        live in ``post_step``.
        """

    # ── Optional hooks ────────────────────────────────────────────────

    def post_step(self) -> None:
        """Called by trainer AFTER ``optimizer.step()``. Default no-op.

        Used by T-SAE for decoder unit-norm projection, by TXC archs for
        any post-step renormalisation, etc. Differs from a training-step
        post-hook in that it runs OUTSIDE the gradient path.
        """

    def pre_step(self) -> None:
        """Called by trainer BEFORE ``optimizer.zero_grad()``. Default
        no-op. Used by Bricken-style resamplers for dead-feature
        diagnostics on a check batch."""

    # ── Forward (recon) ───────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def d_sae(self) -> int:
        return self.config.d_sae

    @property
    def T(self) -> int:
        return self.config.T

    def decoder_directions(self) -> torch.Tensor:
        """Return decoder columns as ``(d_sae, d_in)``.

        Used by § 4 synthetic feature-recovery AUC. For window archs
        the conventional reduction is to average the per-position
        decoders into a single (d_sae, d_in) matrix; subclasses
        override if a different reduction is the right one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement decoder_directions()"
        )
