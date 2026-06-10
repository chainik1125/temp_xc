"""Stacked-SAE — T independent TopK SAEs, one per token position.

Source: ``origin/han-phase7-unification @ 94119bc0:src/architectures/stacked_sae.py``
("per-layer SAEs" baseline from the crosscoders paper, adapted to
the temporal setting).

Adapted to the unified :class:`TempBenchArch` framework. Each of the
``T`` positions has its own SAE with independent weights. Window-level
L0 = ``k_pos * T``.

- ``encode(x)`` accepts ``(B, T, d_in)`` and returns
  ``(B, T, d_sae)`` per-position latents.
- ``train_step(x)`` accepts ``(B, seq_len, d_in)`` from the canonical
  batch_iter (full sequences from the activation cache) and randomly
  extracts a single T-window per batch element. Loss is mean MSE
  reconstruction across the T positions.
- Decoder unit-norm renormalisation in ``post_step()`` (per-SAE).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


class _PerPosSAE(nn.Module):
    """Per-position TopK SAE — analogue of TopKSAE but flat tensors so
    the StackedSAE can hold T of them in nn.ModuleList."""

    def __init__(self, *, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.k = k
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - self.b_dec
        pre = x_c @ self.W_enc.T + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, F.relu(vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_dec


class StackedSAE(TempBenchArch):
    """T independent TopK SAEs, one per window position.

    Args (from ``configs/locked_archs.yaml::stacked_sae``):
        d_in:    residual width.
        d_sae:   per-position dictionary size (each SAE has d_sae features).
        T:       window length.
        k_pos:   per-token sparsity. Window L0 = k_pos * T.
    """


    # v2 framework attrs (added during arxiv migration).
    # consumes='sequence': train_step does internal T-window sampling on
    # (B, seq_len, d_in) batches; needs full sequences.
    arch_version: str = "2.0.0"
    consumes: str = 'sequence'

    def __init__(self, *, d_in: int, d_sae: int, k_pos: int, T: int = 5):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name="stacked_sae", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        self.k_pos = k_pos
        self.saes = nn.ModuleList([
            _PerPosSAE(d_in=d_in, d_sae=d_sae, k=k_pos) for _ in range(T)
        ])

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        for sae in self.saes:
            sae._normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, T, d_in) → (B, T, d_sae)`` per-position TopK encoding."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, T, d = x.shape
        if T != self._T:
            raise ValueError(
                f"StackedSAE.encode expects T={self._T}; got T={T}."
            )
        out = torch.empty(B, T, self._d_sae, device=x.device, dtype=x.dtype)
        for t in range(T):
            out[:, t, :] = self.saes[t].encode(x[:, t, :])
        return out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``z: (B, T, d_sae) → (B, T, d_in)`` per-position decoding."""
        if z.dim() == 2:
            z = z.unsqueeze(1)
        B, T, _ = z.shape
        if T != self._T:
            raise ValueError(
                f"StackedSAE.decode expects T={self._T}; got T={T}."
            )
        out = torch.empty(B, T, self.d_in, device=z.device, dtype=z.dtype)
        for t in range(T):
            out[:, t, :] = self.saes[t].decode(z[:, t, :])
        return out

    def post_step(self) -> None:
        self._normalize_decoder()

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Pull a random T-window per batch element + per-position MSE."""
        if x.dim() != 3 or x.shape[1] < self._T:
            raise ValueError(
                f"StackedSAE.train_step expects (B, seq_len>={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, seq_len, _ = x.shape
        offsets = torch.randint(0, seq_len - self._T + 1, (B,), device=x.device)
        idx_t = offsets.unsqueeze(1) + torch.arange(self._T, device=x.device).unsqueeze(0)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, self._T)
        windows = x[batch_idx, idx_t]                                    # (B, T, d_in)
        z = self.encode(windows)
        x_hat = self.decode(z)
        mse = (x_hat - windows).pow(2).sum(dim=-1).mean()
        l0 = (z != 0).float().sum(dim=-1).mean()
        return mse, {"mse": mse.detach(), "l0": l0.detach(), "z": z.detach()}

    def decoder_directions(self) -> torch.Tensor:
        """T-averaged decoder columns, ``(d_sae, d_in)`` to match TempBenchArch
        convention. Per-position decoders accessible via ``saes[t].W_dec``."""
        # W_dec is (d_in, d_sae); transpose to (d_sae, d_in) and mean over T.
        return torch.stack([sae.W_dec.data.T for sae in self.saes]).mean(dim=0)
