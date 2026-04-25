"""Dead-feature detection and reinitialization for TemporalCrosscoder /
MultiLayerCrosscoder training.

A feature is "dead" if it never falls into the TopK on a held-out batch of
activations. Once dead, it receives no gradient (TopK zeroes out inactive
features), so without intervention it stays dead forever. Andy RDT's
dictionary_learning and other SAE libraries resample these periodically.

Our implementation is intentionally simple:
  - Every ``resample_every`` steps, run the encoder on ``n_check`` windows,
    count per-feature fires.
  - Features that fire fewer than ``min_fires`` times are marked dead.
  - Dead features get a new random encoder column (kaiming_uniform), tied
    decoder rows, zero bias, and the decoder is re-unit-normalized.

Usage:
    resampler = DeadFeatureResampler(txc, resample_every=500, min_fires=1)
    for step, batch in enumerate(...):
        ...
        if resampler.maybe_resample(step, buffer_sample_fn):
            print(f"resampled N={resampler.last_n_resampled}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class ResampleStats:
    step: int
    n_dead: int
    n_features: int
    n_resampled: int
    fire_hist_quantiles: dict  # {q: int}


class DeadFeatureResampler:
    def __init__(
        self,
        crosscoder,
        *,
        resample_every: int = 500,
        min_fires: int = 1,
        n_check: int = 2048,
        max_resample_fraction: float = 0.5,
        seed: int = 0,
    ):
        self.cc = crosscoder
        self.resample_every = resample_every
        self.min_fires = min_fires
        self.n_check = n_check
        self.max_resample_fraction = max_resample_fraction
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        self.history: list[ResampleStats] = []
        self.last_n_resampled = 0

    @torch.no_grad()
    def _measure_fire_counts(self, sample_fn) -> torch.Tensor:
        """sample_fn(n_samples) -> tensor acceptable by self.cc.encode."""
        sample = sample_fn(self.n_check).to(next(self.cc.parameters()).device)
        z = self.cc.encode(sample)
        return (z != 0).sum(dim=0)  # (d_sae,)

    @torch.no_grad()
    def _resample_indices(self, dead_idx: torch.Tensor, alive_idx: torch.Tensor) -> int:
        """Reinitialize W_enc columns + W_dec rows + b_enc at dead indices."""
        device = next(self.cc.parameters()).device
        n_dead = dead_idx.numel()
        if n_dead == 0:
            return 0
        # Cap fraction so we don't reset ~everything late in training.
        max_n = max(1, int(self.cc.d_sae * self.max_resample_fraction))
        if n_dead > max_n:
            # Keep the deadest (lowest fire count) — here all dead == fire_count < min_fires.
            dead_idx = dead_idx[:max_n]
            n_dead = max_n

        # Target encoder column norm: the median across currently-alive columns.
        enc = self.cc.W_enc.data  # TemporalCrosscoder: (T, d_in, d_sae)
        # If it's MultiLayerCrosscoder the shape is (L, d_in, d_sae) — same layout.
        alive_col_norms = enc[:, :, alive_idx].pow(2).sum(dim=(0, 1)).sqrt()
        target = float(alive_col_norms.median().item()) if alive_col_norms.numel() > 0 else 1.0

        # New encoder columns — fan_in = T * d_in (or L * d_in).
        T_or_L = enc.shape[0]
        d_in = enc.shape[1]
        new_enc = torch.randn(T_or_L, d_in, n_dead, generator=self._gen).to(device)
        # Normalize each column to `target`.
        norms = new_enc.pow(2).sum(dim=(0, 1)).sqrt().clamp(min=1e-8)
        new_enc = new_enc * (target / norms)
        enc[:, :, dead_idx] = new_enc.to(enc.dtype)

        # Tie decoder to encoder transpose. Detect layout:
        #   sae_day TemporalCrosscoder / MultiLayerCrosscoder: (T, d_sae, d_in)
        #   han TXCBareAntidead:                              (d_sae, T, d_in)
        dec = self.cc.W_dec.data
        if dec.shape[0] == T_or_L and dec.shape[1] == self.cc.d_sae:
            # sae_day layout
            for t in range(T_or_L):
                dec[t, dead_idx, :] = new_enc[t].T.to(dec.dtype)
        elif dec.shape[0] == self.cc.d_sae and dec.shape[1] == T_or_L:
            # han layout: W_dec[feature_idx, position, :]
            dec[dead_idx, :, :] = new_enc.permute(2, 0, 1).to(dec.dtype)
        else:
            raise RuntimeError(
                f"unknown W_dec layout {tuple(dec.shape)} (expected (T, d_sae, d_in) "
                f"or (d_sae, T, d_in) with T={T_or_L}, d_sae={self.cc.d_sae})"
            )

        # Zero the encoder bias at dead indices so their pre-activations start fresh.
        self.cc.b_enc.data[dead_idx] = 0.0

        # Re-unit-normalize decoder rows jointly across slots.
        if hasattr(self.cc, "normalize_decoder"):
            self.cc.normalize_decoder()

        return n_dead

    @torch.no_grad()
    def maybe_resample(self, step: int, sample_fn) -> bool:
        """Returns True if a resample happened at this step."""
        if step == 0 or step % self.resample_every != 0:
            return False
        fire = self._measure_fire_counts(sample_fn)
        dead_mask = fire < self.min_fires
        dead_idx = dead_mask.nonzero(as_tuple=True)[0]
        alive_idx = (~dead_mask).nonzero(as_tuple=True)[0]
        n_resampled = self._resample_indices(dead_idx, alive_idx)
        self.last_n_resampled = n_resampled
        quantiles = {q: int(torch.quantile(fire.float(), q).item()) for q in (0.01, 0.1, 0.5, 0.9, 0.99)}
        self.history.append(ResampleStats(
            step=step, n_dead=int(dead_mask.sum().item()),
            n_features=self.cc.d_sae, n_resampled=n_resampled,
            fire_hist_quantiles=quantiles,
        ))
        return True

    @torch.no_grad()
    def diagnostic(self, sample_fn) -> dict:
        """Standalone diagnostic (no resample); returns counts."""
        fire = self._measure_fire_counts(sample_fn)
        return {
            "n_dead": int((fire < self.min_fires).sum().item()),
            "n_features": int(self.cc.d_sae),
            "max_fire": int(fire.max().item()),
            "median_fire": int(fire.median().item()),
            "quantiles": {q: int(torch.quantile(fire.float(), q).item())
                          for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
        }
