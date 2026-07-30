"""Controlled spectral-TXC ablations for the power-spectrum experiment.

This module deliberately lives under ``experiments/``: it is an experimental
extension of :class:`temp_bench.archs.spectral_txc.SpectralTXCBatchTopK`, not a
new registry architecture.  With every new option at its default, ``train_step``
delegates to the parent implementation and is numerically backward-compatible.

The v1 architecture bundles two choices with the DCT parameterisation:

* per-band BatchTopK reserves support for every band, whereas a monolithic TXC
  allocates support globally;
* equal atoms/support per band over-represent a singleton DC band relative to
  wider AC bands.

``selection_mode="global"`` isolates the first choice.  ``dc_mode="remove"``
isolates temporal variation by centring every input window and masking the
decoder's DCT-0 coefficients.  Its reconstruction target remains the original
window, so the reported MSE honestly includes DC information the model can no
longer encode; it is not an easier centred-target metric.

Two opt-in objectives implement the proposed spectral penalties:

``dominance_alpha``
    Penalises only band reconstruction-power shares above
    ``dominance_cap * target_share``.  Width-proportional targets avoid treating
    a three-bin band and a singleton band as equally large by definition.

``frequency_matryoshka_alpha``
    Retains the original nested low-to-high ablation for provenance.  Because
    earlier bands occur in more prefixes, this is a *fixed low-frequency
    weighting*, not the learned-frequency objective.

``adaptive_frequency_alpha``
    Adds order-free, independently projected band losses.  Their positive
    relative weights are learned on the simplex.  The model minimizes the
    weighted reconstruction loss while the weight logits adversarially
    emphasize bands with high power-normalized residual.  A bandwidth prior,
    entropy regularization, and a positive prior floor prevent collapse.  The
    two gradient directions are separated with stop-gradient operations, so a
    single ordinary optimizer performs the min-max update without allowing the
    weights to hide a difficult band.

``temporal_basis="fourier"``
    Uses a real orthonormal DFT basis and keeps every sine/cosine quadrature
    pair in one band.  Unlike a DCT-only branch, a translated sinusoid remains
    inside the same frequency subspace.

``adaptive_frequency_routing_strength``
    Uses the learned relative weights as detached score multipliers during
    global BatchTopK support competition.  Selected code amplitudes are left
    unchanged.  When a synthetic noise variance is supplied, the adaptive
    statistics subtract its analytic per-basis-row contribution so the
    adversary does not merely chase irreducible white noise.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from temp_bench.archs.spectral_txc import (
    SpectralTXCBatchTopK,
    _dct_basis,
    _split_evenly,
)


def _real_fourier_basis(length: int) -> torch.Tensor:
    """Real orthonormal DFT basis with sine/cosine pairs kept explicit."""

    time = torch.arange(length, dtype=torch.float32)
    rows = [torch.ones(length) / float(length) ** 0.5]
    for frequency in range(1, (length - 1) // 2 + 1):
        angle = 2.0 * torch.pi * frequency * time / length
        scale = (2.0 / length) ** 0.5
        rows.extend((scale * torch.cos(angle), scale * torch.sin(angle)))
    if length % 2 == 0:
        rows.append(torch.cos(torch.pi * time) / float(length) ** 0.5)
    return torch.stack(rows)


def _fourier_bands(length: int, mode: str) -> tuple[list[list[int]], list[list[int]]]:
    """Return basis-row bands and their physical non-negative Fourier bins."""

    if mode == "full":
        return [list(range(length))], [list(range(length // 2 + 1))]
    if mode == "dcac":
        ac_rows = list(range(1, length))
        return [[0]] + ([ac_rows] if ac_rows else []), [[0]] + (
            [list(range(1, length // 2 + 1))] if ac_rows else []
        )
    if mode != "multiband":
        raise ValueError(f"unknown bands mode {mode!r} (use multiband|dcac|full)")

    units: list[tuple[list[int], int]] = []
    row = 1
    for frequency in range(1, (length - 1) // 2 + 1):
        units.append(([row, row + 1], frequency))
        row += 2
    if length % 2 == 0:
        units.append(([row], length // 2))
    if not units:
        return [[0]], [[0]]
    group_count = min(3, len(units))
    base, remainder = divmod(len(units), group_count)
    sizes = [base] * group_count
    for offset in range(remainder):
        sizes[group_count - 1 - offset] += 1
    row_bands = [[0]]
    frequency_bands = [[0]]
    start = 0
    for size in sizes:
        group = units[start : start + size]
        row_bands.append([index for rows, _ in group for index in rows])
        frequency_bands.append([frequency for _, frequency in group])
        start += size
    return row_bands, frequency_bands


class SpectralTXCV2(SpectralTXCBatchTopK):
    """Spectral TXC with controlled DC, support-allocation, and loss ablations."""

    arch_version = "2.2.0-experimental"
    _registry_name = "spectral_txc_v2"

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        T: int = 8,
        k_pos: int = 1,
        bands: str = "multiband",
        auxk_alpha: float = 1.0 / 32.0,
        dead_threshold_tokens: int = 10_000_000,
        aux_k: int | None = None,
        threshold_start_step: int = 1000,
        threshold_beta: float = 0.999,
        temporal_basis: str = "dct",
        dc_mode: str = "keep",
        selection_mode: str = "per_band",
        dominance_alpha: float = 0.0,
        dominance_cap: float = 1.5,
        dominance_target: str = "bandwidth",
        frequency_matryoshka_alpha: float = 0.0,
        frequency_matryoshka_decay: float = 1.0,
        adaptive_frequency_alpha: float = 0.0,
        adaptive_frequency_adversary_alpha: float = 0.0,
        adaptive_frequency_entropy: float = 0.1,
        adaptive_frequency_floor: float = 0.2,
        adaptive_frequency_ema_beta: float = 0.99,
        adaptive_frequency_power_floor: float = 0.05,
        adaptive_frequency_noise_variance: float = 0.0,
        adaptive_frequency_warmup_steps: int = 0,
        adaptive_frequency_routing_strength: float = 0.0,
        adaptive_frequency_routing_min: float = 0.5,
        adaptive_frequency_routing_max: float = 2.0,
    ):
        if temporal_basis not in {"dct", "fourier"}:
            raise ValueError("temporal_basis must be 'dct' or 'fourier'")
        if dc_mode not in {"keep", "remove"}:
            raise ValueError("dc_mode must be 'keep' or 'remove'")
        if selection_mode not in {"per_band", "global"}:
            raise ValueError("selection_mode must be 'per_band' or 'global'")
        if dominance_alpha < 0:
            raise ValueError("dominance_alpha must be non-negative")
        if dominance_cap < 1:
            raise ValueError("dominance_cap must be at least 1")
        if dominance_target not in {"bandwidth", "uniform"}:
            raise ValueError("dominance_target must be 'bandwidth' or 'uniform'")
        if frequency_matryoshka_alpha < 0:
            raise ValueError("frequency_matryoshka_alpha must be non-negative")
        if frequency_matryoshka_decay <= 0:
            raise ValueError("frequency_matryoshka_decay must be positive")
        if adaptive_frequency_alpha < 0:
            raise ValueError("adaptive_frequency_alpha must be non-negative")
        if adaptive_frequency_adversary_alpha < 0:
            raise ValueError("adaptive_frequency_adversary_alpha must be non-negative")
        if (adaptive_frequency_alpha > 0) != (adaptive_frequency_adversary_alpha > 0):
            raise ValueError(
                "adaptive frequency model and adversary alphas must either both "
                "be positive or both be zero"
            )
        if adaptive_frequency_entropy <= 0:
            raise ValueError("adaptive_frequency_entropy must be positive")
        if not 0 <= adaptive_frequency_floor < 1:
            raise ValueError("adaptive_frequency_floor must lie in [0, 1)")
        if not 0 <= adaptive_frequency_ema_beta < 1:
            raise ValueError("adaptive_frequency_ema_beta must lie in [0, 1)")
        if not 0 < adaptive_frequency_power_floor <= 1:
            raise ValueError("adaptive_frequency_power_floor must lie in (0, 1]")
        if adaptive_frequency_noise_variance < 0:
            raise ValueError("adaptive_frequency_noise_variance must be non-negative")
        if adaptive_frequency_warmup_steps < 0:
            raise ValueError("adaptive_frequency_warmup_steps must be non-negative")
        if not 0 <= adaptive_frequency_routing_strength <= 1:
            raise ValueError("adaptive_frequency_routing_strength must lie in [0, 1]")
        if adaptive_frequency_routing_strength > 0 and adaptive_frequency_alpha == 0:
            raise ValueError("adaptive frequency routing requires positive adaptive alphas")
        if adaptive_frequency_routing_min <= 0:
            raise ValueError("adaptive_frequency_routing_min must be positive")
        if adaptive_frequency_routing_max < adaptive_frequency_routing_min:
            raise ValueError("adaptive_frequency_routing_max must be at least the routing minimum")
        if dc_mode == "remove" and T < 2:
            raise ValueError("dc_mode='remove' needs T >= 2 (at least one AC bin)")

        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            T=T,
            k_pos=k_pos,
            bands=bands,
            auxk_alpha=auxk_alpha,
            dead_threshold_tokens=dead_threshold_tokens,
            aux_k=aux_k,
            threshold_start_step=threshold_start_step,
            threshold_beta=threshold_beta,
        )
        if temporal_basis == "fourier":
            self._rebuild_fourier_parameterization(
                d_in=d_in,
                d_sae=d_sae,
                length=T,
                bands_mode=bands,
            )
        else:
            self.frequency_bin_bands = [list(band) for band in self.bands]
        self.temporal_basis_name = temporal_basis
        temporal_basis_tensor = _dct_basis(T) if temporal_basis == "dct" else _real_fourier_basis(T)
        self.register_buffer(
            "temporal_basis_v2",
            temporal_basis_tensor,
            persistent=False,
        )
        self.dc_mode = dc_mode
        self.selection_mode = selection_mode
        self.dominance_alpha = float(dominance_alpha)
        self.dominance_cap = float(dominance_cap)
        self.dominance_target = dominance_target
        self.frequency_matryoshka_alpha = float(frequency_matryoshka_alpha)
        self.frequency_matryoshka_decay = float(frequency_matryoshka_decay)
        self.adaptive_frequency_alpha = float(adaptive_frequency_alpha)
        self.adaptive_frequency_adversary_alpha = float(adaptive_frequency_adversary_alpha)
        self.adaptive_frequency_entropy = float(adaptive_frequency_entropy)
        self.adaptive_frequency_floor = float(adaptive_frequency_floor)
        self.adaptive_frequency_ema_beta = float(adaptive_frequency_ema_beta)
        self.adaptive_frequency_power_floor = float(adaptive_frequency_power_floor)
        self.adaptive_frequency_noise_variance = float(adaptive_frequency_noise_variance)
        self.adaptive_frequency_warmup_steps = int(adaptive_frequency_warmup_steps)
        self.adaptive_frequency_routing_strength = float(adaptive_frequency_routing_strength)
        self.adaptive_frequency_routing_min = float(adaptive_frequency_routing_min)
        self.adaptive_frequency_routing_max = float(adaptive_frequency_routing_max)
        self.k_win = self.k_pos * self._T

        # A coefficient mask is preferable to merely omitting the DC-only
        # branch: it also removes w=0 from "full" and "dcac" mixed atoms.
        for b, band in enumerate(self.bands):
            values = [1.0 if (dc_mode == "keep" or w != 0) else 0.0 for w in band]
            self.register_buffer(
                f"frequency_mask_{b}",
                torch.tensor(values, dtype=self.enc_coef[b].dtype),
                persistent=False,
            )

        active_bands = [b for b in range(self.n_bands) if self._frequency_count(b) > 0]
        self.active_bands = tuple(active_bands)
        active_features = torch.zeros(d_sae, dtype=torch.bool)
        for b in active_bands:
            s, e = self.band_slices[b]
            active_features[s:e] = True
        self.register_buffer("active_features", active_features, persistent=False)
        self.register_buffer("global_threshold", torch.tensor(-1.0))

        widths = torch.tensor(
            [self._frequency_count(b) for b in self.active_bands],
            dtype=torch.float32,
        )
        self.register_buffer(
            "frequency_weight_prior",
            widths / widths.sum(),
            persistent=False,
        )
        self.register_buffer(
            "frequency_error_ema",
            torch.ones(len(self.active_bands), dtype=torch.float32),
        )
        self.register_buffer(
            "frequency_power_ema",
            torch.ones(len(self.active_bands), dtype=torch.float32),
        )
        self.register_buffer(
            "frequency_ema_initialized",
            torch.tensor(False),
        )
        if self.adaptive_frequency_alpha > 0:
            self.frequency_weight_logits = torch.nn.Parameter(
                torch.zeros(len(self.active_bands), dtype=torch.float32)
            )
        else:
            self.register_parameter("frequency_weight_logits", None)

        # DC-only branches receive no reserved support.  Reassign their legacy
        # share over the active branches while preserving the total k_win.
        self.selection_k_per_band = [0] * self.n_bands
        base, rem = divmod(self.k_win, len(active_bands))
        for i, b in enumerate(active_bands):
            self.selection_k_per_band[b] = base + int(i < rem)
            if self.selection_k_per_band[b] > self.h_per_band[b]:
                raise ValueError(
                    f"band {b}: h_b ({self.h_per_band[b]}) is smaller than its "
                    f"v2 support budget ({self.selection_k_per_band[b]})"
                )

        if dc_mode == "remove":
            with torch.no_grad():
                self._enforce_frequency_masks()
                self._normalize_decoder()
                for b in range(self.n_bands):
                    self.enc_coef[b].copy_(self.dec_coef[b])

    def _rebuild_fourier_parameterization(
        self,
        *,
        d_in: int,
        d_sae: int,
        length: int,
        bands_mode: str,
    ) -> None:
        """Replace the parent's DCT branches with real-Fourier branches."""

        basis = _real_fourier_basis(length)
        row_bands, frequency_bands = _fourier_bands(length, bands_mode)
        self.bands = row_bands
        self.frequency_bin_bands = frequency_bands
        self.n_bands = len(row_bands)
        self.h_per_band = _split_evenly(d_sae, self.n_bands, minimum=1)
        self.k_per_band = _split_evenly(
            self.k_pos * length,
            self.n_bands,
            minimum=1,
        )
        for b, (h_b, k_b) in enumerate(zip(self.h_per_band, self.k_per_band)):
            if h_b < k_b:
                raise ValueError(f"band {b}: h_b ({h_b}) < k_b ({k_b}); raise d_sae or lower k_pos")
        self.band_slices = []
        start = 0
        for width in self.h_per_band:
            self.band_slices.append((start, start + width))
            start += width

        self.enc_coef = nn.ParameterList()
        self.dec_coef = nn.ParameterList()
        self.b_enc = nn.ParameterList()
        for b, (band, h_b) in enumerate(zip(row_bands, self.h_per_band)):
            band_width = len(band)
            scale = 1.0 / float(band_width * d_in) ** 0.5
            self.enc_coef.append(nn.Parameter(torch.randn(h_b, band_width, d_in) * scale))
            self.dec_coef.append(nn.Parameter(torch.randn(h_b, band_width, d_in) * scale))
            self.b_enc.append(nn.Parameter(torch.zeros(h_b)))
            name = f"psi_{b}"
            value = basis[band].clone()
            if name in self._buffers:
                setattr(self, name, value)
            else:
                self.register_buffer(name, value, persistent=False)
        self.b_dec = nn.Parameter(torch.zeros(length, d_in))
        self.threshold = torch.full((self.n_bands,), -1.0)
        self.num_tokens_since_fired = torch.zeros(d_sae, dtype=torch.long)
        self.global_step = torch.tensor(0, dtype=torch.long)

        with torch.no_grad():
            self._normalize_decoder()
            for b in range(self.n_bands):
                self.enc_coef[b].copy_(self.dec_coef[b])
        for decoder in self.dec_coef:
            decoder.register_post_accumulate_grad_hook(self._project_dec_grad)

    def _frequency_mask(self, b: int) -> torch.Tensor:
        return getattr(self, f"frequency_mask_{b}")

    def _frequency_count(self, b: int) -> int:
        return int(self._frequency_mask(b).sum().item())

    def _enc_kernel(self, b: int) -> torch.Tensor:
        coef = self.enc_coef[b] * self._frequency_mask(b)[None, :, None]
        return torch.einsum("wt,hwd->htd", self._psi(b), coef)

    def _dec_kernel(self, b: int) -> torch.Tensor:
        coef = self.dec_coef[b] * self._frequency_mask(b)[None, :, None]
        return torch.einsum("wt,hwd->htd", self._psi(b), coef)

    @torch.no_grad()
    def _enforce_frequency_masks(self) -> None:
        for b in range(self.n_bands):
            mask = self._frequency_mask(b)[None, :, None]
            self.enc_coef[b].mul_(mask)
            self.dec_coef[b].mul_(mask)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.dc_mode == "remove":
            return x - x.mean(dim=1, keepdim=True)
        return x

    def frequency_routing_scales(self) -> torch.Tensor:
        """Per-band score multipliers used only for support competition."""

        full_weights = self.learned_frequency_weights()
        full_prior = full_weights.new_zeros(self.n_bands)
        full_prior[list(self.active_bands)] = self.frequency_weight_prior
        ratio = torch.ones_like(full_weights)
        active = full_prior > 0
        ratio[active] = full_weights[active] / full_prior[active]
        exponent = 0.5 * self.adaptive_frequency_routing_strength
        return (
            ratio.detach()
            .pow(exponent)
            .clamp(
                self.adaptive_frequency_routing_min,
                self.adaptive_frequency_routing_max,
            )
        )

    def _selection_scores(self, pre: torch.Tensor) -> torch.Tensor:
        if self.adaptive_frequency_routing_strength == 0:
            return pre
        scales = self.frequency_routing_scales().to(pre)
        feature_scales = pre.new_ones(self._d_sae)
        for b, (start, stop) in enumerate(self.band_slices):
            feature_scales[start:stop] = scales[b]
        return pre * feature_scales.unsqueeze(0)

    def _batchtopk_global(self, pre: torch.Tensor) -> torch.Tensor:
        """One support pool over all active bands, with total budget ``k_win``."""
        pre = pre * self.active_features.unsqueeze(0)
        scores = self._selection_scores(pre)
        k_total = self.k_win * pre.shape[0]
        flat_values = pre.reshape(-1)
        flat_scores = scores.reshape(-1)
        if k_total >= flat_values.numel():
            return pre
        indices = flat_scores.topk(k_total, sorted=False).indices
        values = flat_values[indices]
        return torch.zeros_like(flat_values).scatter_(-1, indices, values).reshape_as(pre)

    def _select(self, pre: torch.Tensor) -> torch.Tensor:
        if self.selection_mode == "global":
            if (not self.training) and self.global_threshold.item() >= 0:
                active = self.active_features.unsqueeze(0)
                scores = self._selection_scores(pre)
                return pre * (scores > self.global_threshold) * active
            return self._batchtopk_global(pre)

        z = torch.zeros_like(pre)
        thresholds_ready = all(self.threshold[b].item() >= 0 for b in self.active_bands)
        use_threshold = (not self.training) and thresholds_ready
        for b in self.active_bands:
            s, e = self.band_slices[b]
            pre_b = pre[:, s:e]
            if use_threshold:
                z[:, s:e] = pre_b * (pre_b > self.threshold[b])
            else:
                z[:, s:e] = self._batchtopk_band(pre_b, self.selection_k_per_band[b])
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"{type(self).__name__}.encode expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        return self._select(self._pre(self._prepare_input(x))).unsqueeze(1)

    def _update_thresholds(self, z: torch.Tensor) -> None:
        if int(self.global_step.item()) <= self.threshold_start_step:
            return
        with torch.no_grad():
            if self.selection_mode == "global":
                scores = self._selection_scores(z)
                active = scores[z > 0]
                current = active.min().float() if active.numel() else z.new_tensor(0.0)
                if self.global_threshold.item() < 0:
                    self.global_threshold.copy_(current)
                else:
                    self.global_threshold.copy_(
                        self.threshold_beta * self.global_threshold
                        + (1 - self.threshold_beta) * current
                    )
                return
            for b in self.active_bands:
                s, e = self.band_slices[b]
                active = z[:, s:e]
                active = active[active > 0]
                current = active.min().float() if active.numel() else z.new_tensor(0.0)
                if self.threshold[b].item() < 0:
                    self.threshold[b].copy_(current)
                else:
                    self.threshold[b].copy_(
                        self.threshold_beta * self.threshold[b]
                        + (1 - self.threshold_beta) * current
                    )

    def _band_reconstructions(self, z: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for b in range(self.n_bands):
            s, e = self.band_slices[b]
            out.append(torch.einsum("bs,std->btd", z[:, s:e], self._dec_kernel(b)))
        return out

    def _dominance_loss(
        self, band_recons: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        powers = torch.stack(
            [band_recons[b].square().sum(dim=(1, 2)).mean() for b in self.active_bands]
        )
        total = powers.sum()
        shares = torch.where(total > 0, powers / total.clamp_min(1e-12), powers)
        if self.dominance_target == "uniform":
            target = torch.full_like(shares, 1.0 / len(self.active_bands))
        else:
            widths = shares.new_tensor([self._frequency_count(b) for b in self.active_bands])
            target = widths / widths.sum()
        excess = torch.relu(shares - self.dominance_cap * target)
        loss = (excess.square() / target.clamp_min(1e-12)).sum()
        return loss, shares

    def _project_frequencies(self, x: torch.Tensor, frequencies: Sequence[int]) -> torch.Tensor:
        if not frequencies:
            return torch.zeros_like(x)
        basis = self.temporal_basis_v2.to(dtype=x.dtype)
        rows = basis[list(frequencies)]
        coefficients = torch.einsum("wt,btd->bwd", rows, x)
        return torch.einsum("wt,bwd->btd", rows, coefficients)

    def _frequency_matryoshka_loss(
        self, x: torch.Tensor, band_recons: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        # The last prefix is the full active spectrum and would mostly duplicate
        # the main reconstruction term, so only proper prefixes are added.
        if len(self.active_bands) < 2:
            return x.new_zeros(())
        cumulative = torch.zeros_like(x)
        frequencies: list[int] = []
        terms: list[torch.Tensor] = []
        weights: list[float] = []
        for prefix, b in enumerate(self.active_bands[:-1]):
            cumulative = cumulative + band_recons[b]
            mask = self._frequency_mask(b).tolist()
            frequencies.extend(w for w, keep in zip(self.bands[b], mask) if keep)
            target = self._project_frequencies(x, frequencies)
            bias = self._project_frequencies(self.b_dec.unsqueeze(0), frequencies)
            terms.append((target - cumulative - bias).square().sum(dim=-1).mean())
            weights.append(self.frequency_matryoshka_decay**prefix)
        weight_tensor = x.new_tensor(weights)
        return (torch.stack(terms) * weight_tensor).sum() / weight_tensor.sum()

    def learned_frequency_weights(self) -> torch.Tensor:
        """Return positive normalized weights in full band order.

        The trainable logits only cover active bands.  Inactive bands (the DC
        branch under ``dc_mode="remove"``) receive zero weight.
        """

        prior = self.frequency_weight_prior
        if self.frequency_weight_logits is None:
            active_weights = prior
        else:
            learned = torch.softmax(
                torch.log(prior) + self.frequency_weight_logits,
                dim=0,
            )
            active_weights = (
                self.adaptive_frequency_floor * prior
                + (1.0 - self.adaptive_frequency_floor) * learned
            )
        full = active_weights.new_zeros(self.n_bands)
        full[list(self.active_bands)] = active_weights
        return full

    def _active_frequency_weights(self) -> torch.Tensor:
        return self.learned_frequency_weights()[list(self.active_bands)]

    def _frequency_band_errors_and_power(
        self,
        x: torch.Tensor,
        band_recons: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-frequency residual and target power for every active band."""

        errors: list[torch.Tensor] = []
        powers: list[torch.Tensor] = []
        for b in self.active_bands:
            kept = [
                frequency
                for frequency, keep in zip(
                    self.bands[b],
                    self._frequency_mask(b).tolist(),
                )
                if keep
            ]
            width = float(len(kept))
            target = self._project_frequencies(x, kept)
            bias = self._project_frequencies(self.b_dec.unsqueeze(0), kept)
            residual = target - band_recons[b] - bias
            noise_per_basis_row = self.adaptive_frequency_noise_variance / self._T
            errors.append(
                (residual.float().square().mean() / width - noise_per_basis_row).clamp_min(0)
            )
            powers.append(
                (target.float().square().mean() / width - noise_per_basis_row).clamp_min(0)
            )
        return torch.stack(errors), torch.stack(powers)

    @torch.no_grad()
    def _update_frequency_emas(
        self,
        errors: torch.Tensor,
        powers: torch.Tensor,
    ) -> None:
        errors = errors.detach().to(self.frequency_error_ema)
        powers = powers.detach().to(self.frequency_power_ema)
        if not bool(self.frequency_ema_initialized):
            self.frequency_error_ema.copy_(errors)
            self.frequency_power_ema.copy_(powers)
            self.frequency_ema_initialized.fill_(True)
            return
        beta = self.adaptive_frequency_ema_beta
        self.frequency_error_ema.lerp_(errors, 1.0 - beta)
        self.frequency_power_ema.lerp_(powers, 1.0 - beta)

    def _normalized_frequency_errors(
        self,
        errors: torch.Tensor,
    ) -> torch.Tensor:
        prior = self.frequency_weight_prior
        global_power = (prior * self.frequency_power_ema).sum()
        denominator = torch.maximum(
            self.frequency_power_ema,
            self.adaptive_frequency_power_floor * global_power,
        ).clamp_min(1e-12)
        return errors / denominator.detach().to(errors)

    def _frequency_weight_reward(
        self,
        normalized_errors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adversary reward and KL; gradients flow only to weight logits."""

        weights = self._active_frequency_weights()
        prior = self.frequency_weight_prior.to(weights)
        learned = (
            torch.softmax(
                torch.log(prior) + self.frequency_weight_logits,
                dim=0,
            )
            if self.frequency_weight_logits is not None
            else prior
        )
        kl = (
            learned * (torch.log(learned.clamp_min(1e-12)) - torch.log(prior.clamp_min(1e-12)))
        ).sum()
        reward = (
            weights * normalized_errors.detach().to(weights)
        ).sum() - self.adaptive_frequency_entropy * kl
        return reward, kl

    def _adaptive_frequency_loss(
        self,
        x: torch.Tensor,
        band_recons: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return model loss, zero-value adversary surrogate, reward, and KL."""

        if self.frequency_weight_logits is None:
            zero = x.new_zeros(())
            return zero, zero, zero, zero
        errors, powers = self._frequency_band_errors_and_power(x, band_recons)
        self._update_frequency_emas(errors, powers)
        normalized = self._normalized_frequency_errors(errors)
        weights = self._active_frequency_weights().detach().to(normalized)
        model_loss = (weights * normalized).sum()
        reward, kl = self._frequency_weight_reward(
            self._normalized_frequency_errors(self.frequency_error_ema)
        )
        if int(self.global_step.item()) < self.adaptive_frequency_warmup_steps:
            adversary = reward.new_zeros(())
        else:
            # Forward value is exactly zero.  Its gradient makes an ordinary
            # minimizing optimizer ascend the detached adversary reward.
            adversary = -self.adaptive_frequency_adversary_alpha * (reward - reward.detach())
        return model_loss, adversary, reward, kl

    def _uses_parent_objective(self) -> bool:
        return (
            self.dc_mode == "keep"
            and self.selection_mode == "per_band"
            and self.dominance_alpha == 0
            and self.frequency_matryoshka_alpha == 0
            and self.adaptive_frequency_alpha == 0
            and self.adaptive_frequency_adversary_alpha == 0
        )

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # This fast path is intentional: the default variant is an exact v1
        # control, including threshold and AuxK bookkeeping.
        if self._uses_parent_objective():
            return super().train_step(x)
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"{type(self).__name__}.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        batch, length, _ = x.shape
        pre = self._pre(self._prepare_input(x))
        z = self._select(pre)
        self._update_thresholds(z)

        band_recons = self._band_reconstructions(z)
        x_hat = sum(band_recons, torch.zeros_like(x)) + self.b_dec
        reconstruction = (x - x_hat).square().sum(dim=-1).mean()

        with torch.no_grad():
            active = (z > 0).any(dim=0)
            self.num_tokens_since_fired += batch * length
            self.num_tokens_since_fired[active] = 0
            dead_mask = (
                self.num_tokens_since_fired >= self.dead_threshold_tokens
            ) & self.active_features
            n_dead = int(dead_mask.sum().item())

        if n_dead:
            k_aux = min(self.aux_k, n_dead)
            aux_pre = pre.masked_fill(~dead_mask.unsqueeze(0), 0.0)
            values, indices = aux_pre.topk(k_aux, dim=-1, sorted=False)
            aux_code = torch.zeros_like(pre).scatter_(-1, indices, values)
            aux_hat = torch.einsum("bs,std->btd", aux_code, self._dec_full())
            residual = (x - x_hat).detach()
            aux_error = (residual - aux_hat).square().sum(dim=-1).mean()
            residual_mean = residual.mean(dim=(0, 1), keepdim=True)
            denominator = (residual - residual_mean).square().sum(dim=-1).mean()
            aux_loss = (aux_error / denominator.clamp_min(1e-8)).nan_to_num(0.0)
        else:
            aux_loss = x.new_zeros(())

        dominance, shares = self._dominance_loss(band_recons)
        matryoshka = self._frequency_matryoshka_loss(x, band_recons)
        adaptive, adversary, weight_reward, weight_kl = self._adaptive_frequency_loss(
            x, band_recons
        )
        loss = (
            reconstruction
            + self.auxk_alpha * aux_loss
            + self.dominance_alpha * dominance
            + self.frequency_matryoshka_alpha * matryoshka
            + self.adaptive_frequency_alpha * adaptive
            + adversary
        )

        with torch.no_grad():
            self.global_step += 1
            l0 = (z != 0).float().sum(dim=-1).mean()
            max_share = shares.max() if shares.numel() else x.new_zeros(())
            threshold = (
                self.global_threshold
                if self.selection_mode == "global"
                else self.threshold[list(self.active_bands)].mean()
            )

        metrics = {
            "loss": loss,
            "mse": reconstruction.detach(),
            "l0": l0.detach(),
            "auxk": aux_loss.detach(),
            "dead": x.new_tensor(float(n_dead)),
            "threshold": threshold.detach().clone(),
            "dominance": dominance.detach(),
            "frequency_matryoshka": matryoshka.detach(),
            "max_band_power_share": max_share.detach(),
            "adaptive_frequency": adaptive.detach(),
            "frequency_weight_reward": weight_reward.detach(),
            "frequency_weight_kl": weight_kl.detach(),
        }
        for b, weight in enumerate(self.learned_frequency_weights()):
            metrics[f"frequency_weight_{b}"] = weight.detach()
        for b, scale in enumerate(self.frequency_routing_scales()):
            metrics[f"frequency_routing_scale_{b}"] = scale.detach()
        return metrics

    def post_step(self) -> None:
        if self.dc_mode == "remove":
            self._enforce_frequency_masks()
        super().post_step()


__all__ = ["SpectralTXCV2"]
