"""Bricken-style dead-feature resample — **opt-in per experiment**.

This is NOT part of the locked TXC-base / TXC-pro architecture spec.
Components opt in by passing :class:`BrickenConfig` to the trainer; the
default is OFF. See `docs/paper/architecture.md` § *Per-experiment
training knobs* and the per-component A/B verdict in
`docs/components/cN.md` before turning it on.

Why opt-in: the prior author's winning recipe `brickenauxk_a8` co-tunes six knobs
together (resample_every, min_fires, n_check, max_resample_fraction,
EMA-AuxK alpha, dead_threshold). The recipe is coherent on Qwen-7B
medical activations; whether it transfers to Gemma activations, the
matryoshka × multi-distance InfoNCE objective in TXC-pro, or low-d
toy data is an empirical question, not a default to assume.

Trainer-level augmentation that periodically hard-resets features which
have not fired on a held-out check batch. Complementary to the AuxK
loss in the anti-dead stack: AuxK gives dead features a gradient signal
(so they learn from the residual), Bricken hard-resets the truly stuck
ones.

Ported from
``origin/case-em-prior @ [scrubbed-sha]:experiments/em_features/dead_feature_resample.py``.
The measurement loop is unchanged. The reset logic is implemented
directly on the arch's ``W_enc`` / ``W_dec`` / ``b_enc`` parameters
because :class:`temp_bench.architectures.txc_base.TXCBase` (the only
arch C6 turns Bricken on for) doesn't expose a per-arch reset hook —
[pipeline]'s port deliberately stayed minimal.

Supported layout: TXC-3D (``W_enc[T, d_in, d_sae]``, ``W_dec[d_sae,
T, d_in]``, ``b_enc[d_sae]``). Other archs raise ``RuntimeError`` —
which is the intended fail-fast: Bricken is opt-in, so the only path
that hits this is a deliberate ``training_cfg.bricken_enabled=True``.

Recipe (default values from the prior author's brickenauxk_a8 winning config):

- ``resample_every = 500`` steps
- ``min_fires = 1`` on a held-out check batch (a feature is "dead" if
  it fired fewer than ``min_fires`` times across the check)
- ``n_check = 2048`` windows per check
- ``max_resample_fraction = 0.5`` cap on per-call resets

Dead-feature trajectory on Qwen-7B medical (the prior author's
``summary_brickenauxk_a8_frontier.md``):

    step    dead    loss
     500    58.5%   2581
    5000    51.1%   3722    ← minimum
   10000    67.7%   2566
   21500    72.6%   2212
   40000    75.7%   2643

Dead-fraction recovers to ~51% around step 5k then regresses to ~75%
by step 40k as Bricken can't keep up with newly-collapsing features.
Loss plateaus around step 21k.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class BrickenConfig:
    """Trainer-level config for Bricken resample. Defaults match the
    winning Qwen-7B medical recipe (`brickenauxk_a8`)."""
    resample_every: int = 500
    min_fires: int = 1
    n_check: int = 2048
    max_resample_fraction: float = 0.5
    seed: int = 0


@dataclass
class ResampleStats:
    step: int
    n_dead: int
    n_features: int
    n_resampled: int
    fire_hist_quantiles: dict[float, int]


class BrickenResampler:
    """Stateful Bricken resampler for the TXC-3D layout.

    Measurement is arch-agnostic on the surface: ``arch.encode(check)``
    is called and non-zero entries per feature are counted. The reset
    requires the TXC-3D parameter layout (``W_enc[T, d_in, d_sae]``,
    ``W_dec[d_sae, T, d_in]``, ``b_enc[d_sae]``) — anything else
    raises ``RuntimeError``, by design.

    The check batch may arrive in either ``(B, T, d_in)`` or
    ``(B, seq_len, d_in)`` shape (the canonical trainer in
    :mod:`temp_bench.training.sae_trainer` calls ``check_fn = lambda:
    batch_iter(n_check)`` and ``batch_iter`` may produce full sequences
    rather than windows). When ``seq_len != T`` we extract one random
    T-window per batch element, matching the convention used inside
    :meth:`temp_bench.architectures.txc_base.TXCBase.train_step`.
    """

    def __init__(self, arch, cfg: BrickenConfig | None = None):
        self.arch = arch
        self.cfg = cfg or BrickenConfig()
        self.last_stats: Optional[ResampleStats] = None
        self._gen = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        self.history: list[ResampleStats] = []

    # ── Internals ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _to_windows(self, sample: torch.Tensor) -> torch.Tensor:
        """Adapt a check batch to the arch's encode contract.

        TXC archs need ``(B, T, d_in)``. If the trainer hands us
        ``(B, seq_len, d_in)`` with ``seq_len >= T`` (the typical case
        for batch_iter producing full-sequence activations), sample
        one random T-window per batch element and return the resulting
        ``(B, T, d_in)`` tensor.
        """
        T = int(getattr(self.arch, "_T", 1))
        if sample.dim() != 3:
            return sample
        if sample.shape[1] == T:
            return sample
        if sample.shape[1] < T:
            raise RuntimeError(
                f"Bricken check batch seq_len={sample.shape[1]} < T={T}; "
                "increase batch_iter's seq_len or T."
            )
        seq_len = sample.shape[1]
        B = sample.shape[0]
        device = sample.device
        offsets = torch.randint(0, seq_len - T + 1, (B,), device=device)
        idx_t = offsets.unsqueeze(1) + torch.arange(T, device=device).unsqueeze(0)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, T)
        return sample[batch_idx, idx_t]

    @torch.no_grad()
    def _measure_fire_counts(
        self,
        check_batch_fn: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Encode the check batch and count non-zero entries per feature.

        Returns a 1-D tensor of length ``d_sae``.
        """
        sample = check_batch_fn()
        device = next(self.arch.parameters()).device
        if sample.device != device:
            sample = sample.to(device)
        sample = self._to_windows(sample)
        z = self.arch.encode(sample)
        # z may be (B, d_sae) or (B, 1, d_sae) (TXCBase squeezes/unsqueezes
        # the T axis); flatten everything except the d_sae trailing dim.
        z_flat = z.reshape(-1, z.shape[-1])
        return (z_flat != 0).sum(dim=0)

    @torch.no_grad()
    def _validate_layout(self) -> tuple[int, int, int]:
        """Verify the TXC-3D parameter layout. Returns (T, d_in, d_sae)."""
        if not all(hasattr(self.arch, n) for n in ("W_enc", "W_dec", "b_enc")):
            raise RuntimeError(
                f"{type(self.arch).__name__} missing W_enc/W_dec/b_enc; "
                "BrickenResampler can only resample arches with these "
                "parameter names."
            )
        d_sae = int(getattr(self.arch.config, "d_sae",
                            getattr(self.arch, "_d_sae", 0)))
        enc = self.arch.W_enc.data
        dec = self.arch.W_dec.data
        if enc.dim() != 3 or enc.shape[2] != d_sae:
            raise RuntimeError(
                f"BrickenResampler expects W_enc layout (T, d_in, d_sae); "
                f"got {tuple(enc.shape)} with d_sae={d_sae}. The TXC-3D "
                "layout used by temp_bench.architectures.txc_base.TXCBase "
                "is the only currently supported target."
            )
        if dec.dim() != 3 or dec.shape[0] != d_sae:
            raise RuntimeError(
                f"BrickenResampler expects W_dec layout (d_sae, T, d_in); "
                f"got {tuple(dec.shape)} with d_sae={d_sae}."
            )
        T_e, d_in, _ = enc.shape
        T_d = dec.shape[1]
        if T_e != T_d:
            raise RuntimeError(
                f"W_enc T={T_e} ≠ W_dec T={T_d}; layout corrupt."
            )
        return T_e, d_in, d_sae

    @torch.no_grad()
    def _reset_features(
        self,
        dead_idx: torch.Tensor,
        T: int,
        d_in: int,
        d_sae: int,
    ) -> int:
        """Reinit ``W_enc[:, :, dead_idx]`` and ``W_dec[dead_idx, :, :]``,
        zero ``b_enc[dead_idx]``, and re-unit-normalise the decoder.
        Returns the number of features actually reset.
        """
        n_dead = int(dead_idx.numel())
        if n_dead == 0:
            return 0
        device = self.arch.W_enc.device

        # Target encoder column norm: median of currently-alive columns.
        alive_mask = torch.ones(d_sae, dtype=torch.bool, device=device)
        alive_mask[dead_idx] = False
        alive_idx = alive_mask.nonzero(as_tuple=True)[0]
        if alive_idx.numel() > 0:
            alive_norms = self.arch.W_enc.data[:, :, alive_idx].pow(2).sum(dim=(0, 1)).sqrt()
            target = float(alive_norms.median().item())
        else:
            target = 1.0

        # New encoder columns. CPU generator; move to device after sampling
        # so the generator state is deterministic across CUDA / CPU runs.
        new_enc = torch.randn(T, d_in, n_dead, generator=self._gen)
        new_enc = new_enc.to(device=device, dtype=self.arch.W_enc.dtype)
        norms = new_enc.pow(2).sum(dim=(0, 1)).sqrt().clamp(min=1e-8)
        new_enc = new_enc * (target / norms)
        self.arch.W_enc.data[:, :, dead_idx] = new_enc

        # Tie decoder rows to encoder transpose:
        #   W_dec[d, t, k] = W_enc[t, k, d] for d ∈ dead_idx
        # so we permute (T, d_in, n_dead) → (n_dead, T, d_in).
        self.arch.W_dec.data[dead_idx, :, :] = (
            new_enc.permute(2, 0, 1).to(self.arch.W_dec.dtype)
        )

        # Zero encoder bias at dead indices so freshly-reset features
        # start from a clean pre-activation.
        self.arch.b_enc.data[dead_idx] = 0.0

        # Re-unit-normalise decoder atoms (TXCBase exposes the helper).
        if hasattr(self.arch, "_normalize_decoder"):
            self.arch._normalize_decoder()

        # Reset the dead-tracker so resampled features get a clean
        # token budget before being marked dead again.
        if hasattr(self.arch, "num_tokens_since_fired"):
            self.arch.num_tokens_since_fired[dead_idx] = 0

        return n_dead

    # ── Public API ─────────────────────────────────────────────────────

    @torch.no_grad()
    def maybe_resample(
        self,
        step: int,
        check_batch_fn: Callable[[], torch.Tensor],
    ) -> bool:
        """If ``step % resample_every == 0`` and step > 0, do a resample.

        Returns True if a check fired this call (the trainer should log
        ``self.last_stats``). False if the call was skipped because the
        resample schedule didn't trigger.
        """
        if step == 0 or step % self.cfg.resample_every != 0:
            return False

        T, d_in, d_sae = self._validate_layout()
        fire_counts = self._measure_fire_counts(check_batch_fn)
        dead_mask = fire_counts < self.cfg.min_fires
        n_dead_total = int(dead_mask.sum().item())
        dead_idx = dead_mask.nonzero(as_tuple=True)[0]

        # Cap per-call resets so we don't hard-reset ~the whole dictionary
        # late in training (which would erase prior learning).
        max_n = max(1, int(d_sae * self.cfg.max_resample_fraction))
        if dead_idx.numel() > max_n:
            dead_idx = dead_idx[:max_n]

        n_resampled = self._reset_features(dead_idx, T=T, d_in=d_in, d_sae=d_sae)

        quantiles = {
            q: int(torch.quantile(fire_counts.float(), q).item())
            for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        }
        self.last_stats = ResampleStats(
            step=step, n_dead=n_dead_total, n_features=d_sae,
            n_resampled=n_resampled, fire_hist_quantiles=quantiles,
        )
        self.history.append(self.last_stats)
        return True

    @torch.no_grad()
    def diagnostic(self, check_batch_fn: Callable[[], torch.Tensor]) -> dict:
        """Standalone diagnostic (no resample); returns counts. Useful
        for the trainer's startup health check or for unit tests."""
        fire_counts = self._measure_fire_counts(check_batch_fn)
        return {
            "n_dead": int((fire_counts < self.cfg.min_fires).sum().item()),
            "n_features": int(fire_counts.numel()),
            "max_fire": int(fire_counts.max().item()),
            "median_fire": int(fire_counts.median().item()),
            "quantiles": {
                q: int(torch.quantile(fire_counts.float(), q).item())
                for q in (0.01, 0.1, 0.5, 0.9, 0.99)
            },
        }
