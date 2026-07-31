"""Learned one-code bottleneck over a frozen shared per-token SAE.

The first stage is an ordinary shared TopK SAE.  Given its sparse codes
``z[b, t, f]``, this module learns

1. a per-feature temporal encoder (the depthwise path);
2. an optional low-rank, cross-feature temporal residual; and
3. a per-feature temporal decoder.

The output is one sparse code per window.  Output feature ``f`` decodes through
the frozen first-stage SAE direction ``f`` at every position, with a learned
position-dependent scale.  This makes the comparison deliberately stronger
than fixed max-pooling while preserving the semantic dictionary learned by the
per-token SAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BottleneckShape:
    d_in: int
    d_sae: int
    window: int
    k_window: int
    rank: int


class TrajectoryBottleneck(nn.Module):
    """Compress sparse per-token SAE trajectories into one sparse code."""

    def __init__(
        self,
        *,
        base_decoder: torch.Tensor,
        base_decoder_bias: torch.Tensor,
        window: int,
        k_window: int,
        rank: int,
        auxk_alpha: float = 1.0 / 32.0,
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
    ) -> None:
        super().__init__()
        if base_decoder.ndim != 2:
            raise ValueError("base_decoder must have shape (d_sae, d_in)")
        if base_decoder_bias.shape != (base_decoder.shape[1],):
            raise ValueError("base_decoder_bias must have shape (d_in,)")
        if window < 1 or k_window < 1 or rank < 0:
            raise ValueError("window/k_window must be positive and rank nonnegative")

        d_sae, d_in = base_decoder.shape
        self.shape = BottleneckShape(
            d_in=int(d_in),
            d_sae=int(d_sae),
            window=int(window),
            k_window=min(int(k_window), int(d_sae)),
            rank=int(rank),
        )
        self.auxk_alpha = float(auxk_alpha)
        self.aux_k = min(int(aux_k), int(d_sae))
        self.dead_threshold_tokens = int(dead_threshold_tokens)

        decoder = base_decoder.detach().clone()
        decoder = decoder / decoder.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("base_decoder", decoder, persistent=True)
        self.register_buffer(
            "base_decoder_bias",
            base_decoder_bias.detach().clone(),
            persistent=True,
        )

        scale = float(window) ** -0.5
        self.temporal_encoder = nn.Parameter(
            torch.full((window, d_sae), scale, dtype=decoder.dtype)
        )
        self.temporal_decoder = nn.Parameter(
            torch.full((window, d_sae), scale, dtype=decoder.dtype)
        )
        self.code_bias = nn.Parameter(torch.zeros(d_sae, dtype=decoder.dtype))

        if rank:
            self.cross_in = nn.Parameter(
                torch.empty(window, d_sae, rank, dtype=decoder.dtype)
            )
            self.cross_out = nn.Parameter(
                torch.zeros(rank, d_sae, dtype=decoder.dtype)
            )
            nn.init.normal_(self.cross_in, std=rank**-0.5)
        else:
            self.register_parameter("cross_in", None)
            self.register_parameter("cross_out", None)

        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(d_sae, dtype=torch.long),
            persistent=True,
        )
        self.normalize_decoder_profiles()

    @property
    def d_sae(self) -> int:
        return self.shape.d_sae

    @property
    def k_window(self) -> int:
        return self.shape.k_window

    @torch.no_grad()
    def normalize_decoder_profiles(self) -> None:
        """Give each composite temporal decoder feature unit norm."""

        norms = self.temporal_decoder.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.temporal_decoder.div_(norms)

    def preactivations(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Return dense bottleneck preactivations from sparse SAE codes.

        ``indices`` and ``values`` have shape ``(B, T, k_pos)``.
        """

        if indices.shape != values.shape or indices.ndim != 3:
            raise ValueError("indices/values must share shape (B, T, k_pos)")
        batch, window, _ = indices.shape
        if window != self.shape.window:
            raise ValueError(
                f"expected T={self.shape.window}, got T={window}"
            )

        pre = self.code_bias.unsqueeze(0).expand(batch, -1).clone()
        cross_summary = None
        if self.shape.rank:
            cross_summary = values.new_zeros((batch, self.shape.rank))

        for position in range(window):
            idx = indices[:, position].long()
            val = values[:, position]
            depthwise = val * self.temporal_encoder[position][idx]
            pre.scatter_add_(1, idx, depthwise)
            if cross_summary is not None:
                selected = self.cross_in[position][idx]
                cross_summary = cross_summary + (
                    selected * val.unsqueeze(-1)
                ).sum(dim=1)

        if cross_summary is not None:
            pre = pre + cross_summary @ self.cross_out
        return pre

    def encode_sparse(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(active_values, active_indices, dense_preactivations)``."""

        pre = self.preactivations(indices, values)
        active_values, active_indices = pre.topk(self.k_window, dim=-1)
        return F.relu(active_values), active_indices, pre

    def decode_sparse(
        self,
        active_values: torch.Tensor,
        active_indices: torch.Tensor,
        *,
        add_bias: bool,
    ) -> torch.Tensor:
        """Decode a sparse one-code representation to ``(B, T, d_in)``."""

        if active_values.shape != active_indices.shape:
            raise ValueError("active values and indices must share shape")
        decoder_rows = self.base_decoder[active_indices.long()]
        positions = []
        for position in range(self.shape.window):
            profile = self.temporal_decoder[position][active_indices.long()]
            weights = active_values * profile
            reconstructed = torch.einsum("bk,bkd->bd", weights, decoder_rows)
            if add_bias:
                reconstructed = reconstructed + self.base_decoder_bias
            positions.append(reconstructed)
        return torch.stack(positions, dim=1)

    def loss(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        target: torch.Tensor,
        *,
        update_dead: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Reconstruct the raw activation window and apply TXC-style AuxK."""

        active_values, active_indices, pre = self.encode_sparse(indices, values)
        reconstruction = self.decode_sparse(
            active_values, active_indices, add_bias=True
        )
        recon_loss = (target - reconstruction).pow(2).sum(dim=-1).mean()

        if update_dead:
            with torch.no_grad():
                n_tokens = int(target.shape[0] * target.shape[1])
                fired_features = active_indices[active_values > 0]
                self.num_tokens_since_fired.add_(n_tokens)
                if fired_features.numel():
                    self.num_tokens_since_fired[fired_features.unique()] = 0

        dead = self.num_tokens_since_fired >= self.dead_threshold_tokens
        n_dead = int(dead.sum().item())
        if n_dead:
            aux_k = min(self.aux_k, n_dead)
            aux_pre = F.relu(pre).masked_fill(~dead.unsqueeze(0), 0)
            aux_values, aux_indices = aux_pre.topk(aux_k, dim=-1, sorted=False)
            aux_reconstruction = self.decode_sparse(
                aux_values, aux_indices, add_bias=False
            )
            residual = (target - reconstruction).detach()
            aux_numerator = (
                residual - aux_reconstruction
            ).pow(2).sum(dim=-1).mean()
            residual_mean = residual.mean(dim=(0, 1), keepdim=True)
            aux_denominator = (
                residual - residual_mean
            ).pow(2).sum(dim=-1).mean().clamp(min=1e-8)
            aux_loss = (aux_numerator / aux_denominator).nan_to_num(0.0)
        else:
            aux_loss = recon_loss.new_zeros(())

        total = recon_loss + self.auxk_alpha * aux_loss
        with torch.no_grad():
            effective_l0 = (active_values > 0).float().sum(dim=-1).mean()
        return {
            "loss": total,
            "mse": recon_loss.detach(),
            "auxk": aux_loss.detach(),
            "l0": effective_l0.detach(),
            "dead": recon_loss.new_tensor(float(n_dead)),
        }

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
