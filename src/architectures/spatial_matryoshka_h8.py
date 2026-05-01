"""Spatial Matryoshka H8 — Han's deadzone-escape variant.

Standard Matryoshka SAE: feature-prefix levels reconstruct the FULL window.
Standard Temporal Matryoshka: feature-prefix levels reconstruct nested
position subsets (e.g. positions [0], [0,1], [0..T-1]) — but features
become tied to specific position indices.

Spatial Matryoshka (Han's idea): each feature-prefix level reconstructs
a RANDOM subset of positions, sampled fresh each training step:

  H prefix       → reconstruct random subset of size 1 (or |level_subset_sizes[0]|)
  H+M prefix     → reconstruct random subset of size 3
  All d_sae      → reconstruct full T_max

This forces the H prefix to be position-flexible "per-token" features
(works at ANY single position), and the deeper feature levels to add
compositional/cross-position information.

Two design knobs:
- nested: if True, level i+1's subset is a superset of level i's. Else
  independent random subsets per level.
- subset sampling mode: uniform random vs Gaussian-mixture (mimics the
  "language features have spatial locality" hypothesis).

Stack: H8 = anti-dead + matryoshka H/L (existing prefix mechanism) +
multi-distance contrastive InfoNCE. Subclass of
TXCBareMultiDistanceContrastiveAntidead, override forward() to add the
spatial-matryoshka loss term.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.txc_bare_multidistance_contrastive_antidead import (
    TXCBareMultiDistanceContrastiveAntidead,
    _info_nce,
)
from src.architectures.phase5b_subseq_sampling_txcdr import _sample_subset_indices


def _gather_positions(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x: (B, T_max, d). idx: (B, k). Returns (B, k, d) — positions selected per row."""
    B, T_max, d = x.shape
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))


class SpatialMatryoshkaH8(TXCBareMultiDistanceContrastiveAntidead):
    """H8 stack with spatial-matryoshka decoder loss.

    Args
    ----
    level_prefix_sizes: tuple[int, ...] — feature-count cutoffs for each
        Matryoshka level (e.g. (matryoshka_h_size, d_sae // 2, d_sae)).
        Each is a number of features; the corresponding decoder slice uses
        z[:, :prefix_size].

    level_subset_sizes: tuple[int, ...] — number of positions to reconstruct
        at each level. Must match level_prefix_sizes length. Levels with
        subset_size = T_max reconstruct the full window.

    nested: bool — if True, level (i+1)'s subset is constrained to contain
        level i's subset. Else independent samples per level.

    subset_sampling_mode: str — "uniform" or "gaussian". Gaussian uses
        sigma_range and n_gaussians.

    enable_contrastive: bool — if False, drops the H8 multi-distance
        InfoNCE loss. The spatial-matryoshka loss replaces the consistency
        signal in that case. Useful for testing whether contrastive is
        still needed when spatial-matryoshka is enforcing a different prior.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int,
        *,
        level_prefix_sizes: tuple[int, ...] | None = None,
        level_subset_sizes: tuple[int, ...] | None = None,
        nested: bool = False,
        subset_sampling_mode: str = "uniform",
        sigma_range: tuple[float, float] | None = None,
        n_gaussians: int = 1,
        enable_contrastive: bool = True,
        shifts: tuple[int, ...] | None = None,
        weights: tuple[float, ...] | None = None,
        matryoshka_h_size: int | None = None,
        alpha: float = 1.0,
        spatial_matryoshka_alpha: float = 1.0,
        **kw,
    ):
        super().__init__(
            d_in, d_sae, T, k,
            shifts=shifts, weights=weights,
            matryoshka_h_size=matryoshka_h_size, alpha=alpha,
            **kw,
        )

        # Default levels: (H, d_sae) with subsets (1, T)
        if level_prefix_sizes is None:
            level_prefix_sizes = (matryoshka_h_size or int(d_sae * 0.2), d_sae)
        if level_subset_sizes is None:
            level_subset_sizes = (1, T)

        assert len(level_prefix_sizes) == len(level_subset_sizes), (
            "level_prefix_sizes and level_subset_sizes must have the same length"
        )
        for n in level_prefix_sizes:
            assert 1 <= n <= d_sae, f"prefix size {n} out of [1, {d_sae}]"
        for ks in level_subset_sizes:
            assert 1 <= ks <= T, f"subset size {ks} out of [1, {T}]"

        self.level_prefix_sizes = tuple(int(n) for n in level_prefix_sizes)
        self.level_subset_sizes = tuple(int(k) for k in level_subset_sizes)
        self.nested = bool(nested)
        self.subset_sampling_mode = str(subset_sampling_mode)
        self.sigma_range = sigma_range
        self.n_gaussians = int(n_gaussians)
        self.enable_contrastive = bool(enable_contrastive)
        self.spatial_matryoshka_alpha = float(spatial_matryoshka_alpha)
        assert self.subset_sampling_mode in {"uniform", "gaussian"}

    # ---- subset sampling helpers ---------------------------------

    def _sample_level_subset(self, T_max: int, k: int, B: int, device) -> torch.Tensor:
        """Sample (B, k) integer indices into [0, T_max) per the configured mode."""
        if self.subset_sampling_mode == "uniform":
            # random non-contiguous
            return _sample_subset_indices(T_max, k, B, "random", device)
        # gaussian mixture
        return _sample_subset_indices(
            T_max, k, B, "gaussian", device,
            sigma_range=self.sigma_range, n_gaussians=self.n_gaussians,
        )

    def _sample_nested_subsets(self, T_max: int, B: int, device) -> list[torch.Tensor]:
        """For nested mode: largest subset first (sampled freely), inner levels
        are random subsets of the outer one."""
        sizes = list(self.level_subset_sizes)
        # Sample largest level first
        ordered = sorted(range(len(sizes)), key=lambda i: -sizes[i])  # largest → smallest
        idx_per_level = {}
        outer_idx = self._sample_level_subset(T_max, sizes[ordered[0]], B, device)  # (B, max_k)
        idx_per_level[ordered[0]] = outer_idx
        for j in ordered[1:]:
            inner_size = sizes[j]
            outer_size = outer_idx.shape[1]
            # For each row, pick a random subset of size inner_size from outer_idx
            keys = torch.rand(B, outer_size, device=device)
            _, sel = keys.topk(inner_size, dim=-1, largest=True)
            sel, _ = sel.sort(dim=-1)
            inner_idx = outer_idx.gather(1, sel)
            idx_per_level[j] = inner_idx
            outer_idx = inner_idx  # next inner uses this as the new outer
        return [idx_per_level[i] for i in range(len(sizes))]

    def _sample_independent_subsets(self, T_max: int, B: int, device) -> list[torch.Tensor]:
        """Non-nested: each level independently samples its subset."""
        return [
            self._sample_level_subset(T_max, k, B, device)
            for k in self.level_subset_sizes
        ]

    # ---- decoder + loss helpers ----------------------------------

    def _decode_with_prefix(self, z: torch.Tensor, prefix_size: int) -> torch.Tensor:
        """Decode using only first `prefix_size` features. z: (B, d_sae)."""
        z_pref = torch.zeros_like(z)
        z_pref[:, :prefix_size] = z[:, :prefix_size]
        return self.decode(z_pref)  # (B, T_max, d_in)

    def _spatial_matryoshka_loss(
        self, x: torch.Tensor, z: torch.Tensor, subsets: list[torch.Tensor],
    ) -> torch.Tensor:
        """Sum of MSE per level — each level reconstructs only its position subset."""
        loss = torch.zeros((), device=x.device, dtype=x.dtype)
        for prefix_size, sub_idx in zip(self.level_prefix_sizes, subsets):
            x_hat_full = self._decode_with_prefix(z, prefix_size)
            x_S = _gather_positions(x, sub_idx)
            x_hat_S = _gather_positions(x_hat_full, sub_idx)
            loss = loss + (x_S - x_hat_S).pow(2).sum(dim=-1).mean()
        return loss

    # ---- forward -------------------------------------------------

    def forward(self, x: torch.Tensor, alpha: float | None = None):
        """Accepts (B, 1+K, T, d) for H8-style multi-distance training, or
        (B, T, d) single-window. Adds spatial-matryoshka loss on the anchor."""
        eff_alpha = self.alpha if alpha is None else alpha

        if x.ndim == 4 and x.shape[1] >= 2:
            K = x.shape[1] - 1
            B = x.shape[0]
            T_max = x.shape[2]

            x_anchor = x[:, 0]
            pre_a = torch.einsum("btd,tds->bs", x_anchor, self.W_enc) + self.b_enc
            vals_a, idx_a = pre_a.topk(self.k, dim=-1)
            z_anchor = torch.zeros_like(pre_a)
            z_anchor.scatter_(1, idx_a, F.relu(vals_a))

            # Standard H8 recon: full window with full features
            x_hat = self.decode(z_anchor)
            l_recon = (x_anchor - x_hat).pow(2).sum(dim=-1).mean()

            # Standard H8 matryoshka prefix: full-window recon with H prefix
            if self.matryoshka_h_size:
                x_hat_h = self._decode_with_prefix(z_anchor, self.matryoshka_h_size)
                l_recon = l_recon + (x_anchor - x_hat_h).pow(2).sum(dim=-1).mean()

            # NEW: spatial-matryoshka loss
            if self.nested:
                subsets = self._sample_nested_subsets(T_max, B, x.device)
            else:
                subsets = self._sample_independent_subsets(T_max, B, x.device)
            l_sm = self._spatial_matryoshka_loss(x_anchor, z_anchor, subsets)

            # Contrastive
            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            if self.enable_contrastive:
                for k_idx, w_s in enumerate(self.loss_weights):
                    x_pos = x[:, 1 + k_idx]
                    pre_p = torch.einsum("btd,tds->bs", x_pos, self.W_enc) + self.b_enc
                    vals_p, idx_p = pre_p.topk(self.k, dim=-1)
                    z_pos = torch.zeros_like(pre_p)
                    z_pos.scatter_(1, idx_p, F.relu(vals_p))
                    if eff_alpha > 0.0:
                        h = self.contr_prefix
                        l_contr = l_contr + w_s * _info_nce(z_anchor[:, :h], z_pos[:, :h])

            # AuxK (dead-feature recovery) on anchor
            l_auxk = self._update_dead_and_auxk(x_anchor, x_hat, pre_a, z_anchor)

            total = (
                l_recon
                + self.spatial_matryoshka_alpha * l_sm
                + (eff_alpha * l_contr if self.enable_contrastive else 0.0)
                + self.auxk_alpha * l_auxk
            )
            return total, x_hat, z_anchor

        # Single-window fallback (no contrastive)
        if x.ndim == 3:
            B = x.shape[0]
            T_max = x.shape[1]
            pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
            x_hat = self.decode(z)
            l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
            if self.matryoshka_h_size:
                x_hat_h = self._decode_with_prefix(z, self.matryoshka_h_size)
                l_recon = l_recon + (x - x_hat_h).pow(2).sum(dim=-1).mean()
            subsets = (self._sample_nested_subsets if self.nested
                       else self._sample_independent_subsets)(T_max, B, x.device)
            l_sm = self._spatial_matryoshka_loss(x, z, subsets)
            l_auxk = self._update_dead_and_auxk(x, x_hat, pre, z)
            total = l_recon + self.spatial_matryoshka_alpha * l_sm + self.auxk_alpha * l_auxk
            return total, x_hat, z

        raise ValueError(f"unexpected input shape {tuple(x.shape)}")
