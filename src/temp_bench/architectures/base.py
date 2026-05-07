"""Abstract base class for all architectures in ``temp-bench``.

Every TXC and SAE in this paper exposes the same minimal interface:

    encode(x) -> z          # required
    decode(z) -> x_hat      # required
    train_step(x) -> loss, info  # default = MSE; archs override for
                                 # auxK / contrastive / matryoshka
    post_step()             # default = no-op; archs override for
                            # decoder-norm projection etc.

This unifies probing, qualitative analysis, steering, case-study eval,
**and training**: one canonical trainer in ``temp_bench.training`` calls
``train_step`` and ``post_step`` regardless of arch family. Components
do NOT write training loops; see PROTOCOL.md § 11 *Code reuse contract*.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ArchConfig:
    """Common config keys. Specific archs subclass and add fields."""
    name: str
    d_in: int
    d_sae: int
    k_pos: int          # sparsity per (token, position) — 1 for SAEs, T*k_pos for window total
    T: int = 1          # window length; 1 for per-token SAEs


class TempBenchArch(ABC, nn.Module):
    """Unified interface across SAE / T-SAE / TFA / MLC / TXC-base / TXC-pro.

    Shape conventions:

        x:      (B, T, d_in)        — windowed input (T=1 for per-token SAEs)
        z:      (B, T, d_sae)       — latent codes (T may be 1 for shared-z TXCs;
                                       implementations document their own shape)
        x_hat:  (B, T, d_in)        — reconstruction
    """

    config: ArchConfig

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # ── Training contract ──────────────────────────────────────────────

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute training loss + diagnostics for one batch.

        Default = pure MSE reconstruction. Archs with auxK, multi-distance
        contrastive, matryoshka splits, etc. override this. The canonical
        trainer in :mod:`temp_bench.training.sae_trainer` calls only this
        method and :meth:`post_step` — never reaches into per-arch
        internals.

        Returns:
            loss:  scalar tensor with grad
            info:  dict with at least ``mse`` (float), ``l0`` (float),
                ``z`` (detached latents tensor). Trainer logs these and
                Bricken reads ``z`` to track dead features.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        mse = (x_hat - x).pow(2).sum(dim=-1).mean()
        z_flat = z.reshape(-1, z.shape[-1])
        l0 = (z_flat != 0).float().sum(dim=-1).mean()
        return mse, {
            "mse": mse.detach(),
            "l0": l0.detach(),
            "z": z.detach(),
        }

    def post_step(self) -> None:
        """Hook called after ``optimizer.step()``.

        Default = no-op. Archs that need decoder unit-norm projection
        (TXC-base / TXC-pro / T-SAE with ``decoder_grad_orthogonalize``)
        override this. Called BEFORE the next training step.
        """
        return None

    # ── Inspection ─────────────────────────────────────────────────────

    @property
    def d_sae(self) -> int:
        return self.config.d_sae

    @property
    def T(self) -> int:
        return self.config.T

    def decoder_directions(self) -> torch.Tensor:
        """Return decoder columns as ``(d_sae, d_in)`` (T-averaged for window archs).

        Used by C1/C2 feature-recovery AUC and C4 qualitative ranking.
        Subclasses override if averaging is not the right reduction.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement decoder_directions()"
        )
