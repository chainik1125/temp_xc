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
    Adds nested low-to-high reconstruction terms at DCT-band boundaries.  Each
    prefix is compared with the matching projection of the target, rather than
    asking low-frequency atoms to reproduce impossible high-frequency detail.
    Under strictly disjoint per-band support this is mathematically a
    low-frequency reweighting; with global support selection it also changes
    competition for the shared active-atom budget.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from temp_bench.archs.spectral_txc import SpectralTXCBatchTopK, _dct_basis


class SpectralTXCV2(SpectralTXCBatchTopK):
    """Spectral TXC with controlled DC, support-allocation, and loss ablations."""

    arch_version = "2.0.0-experimental"
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
        dc_mode: str = "keep",
        selection_mode: str = "per_band",
        dominance_alpha: float = 0.0,
        dominance_cap: float = 1.5,
        dominance_target: str = "bandwidth",
        frequency_matryoshka_alpha: float = 0.0,
        frequency_matryoshka_decay: float = 1.0,
    ):
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
        self.dc_mode = dc_mode
        self.selection_mode = selection_mode
        self.dominance_alpha = float(dominance_alpha)
        self.dominance_cap = float(dominance_cap)
        self.dominance_target = dominance_target
        self.frequency_matryoshka_alpha = float(frequency_matryoshka_alpha)
        self.frequency_matryoshka_decay = float(frequency_matryoshka_decay)
        self.k_win = self.k_pos * self._T
        self.register_buffer("dct_basis_v2", _dct_basis(T), persistent=False)

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

    def _batchtopk_global(self, pre: torch.Tensor) -> torch.Tensor:
        """One support pool over all active bands, with total budget ``k_win``."""
        pre = pre * self.active_features.unsqueeze(0)
        k_total = self.k_win * pre.shape[0]
        flat = pre.reshape(-1)
        if k_total >= flat.numel():
            return pre
        top = flat.topk(k_total, sorted=False)
        return torch.zeros_like(flat).scatter_(-1, top.indices, top.values).reshape_as(pre)

    def _select(self, pre: torch.Tensor) -> torch.Tensor:
        if self.selection_mode == "global":
            if (not self.training) and self.global_threshold.item() >= 0:
                active = self.active_features.unsqueeze(0)
                return pre * (pre > self.global_threshold) * active
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
                z[:, s:e] = self._batchtopk_band(
                    pre_b, self.selection_k_per_band[b]
                )
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
                active = z[z > 0]
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

    def _project_frequencies(
        self, x: torch.Tensor, frequencies: Sequence[int]
    ) -> torch.Tensor:
        if not frequencies:
            return torch.zeros_like(x)
        basis = self.dct_basis_v2.to(dtype=x.dtype)
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

    def _uses_parent_objective(self) -> bool:
        return (
            self.dc_mode == "keep"
            and self.selection_mode == "per_band"
            and self.dominance_alpha == 0
            and self.frequency_matryoshka_alpha == 0
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
        loss = (
            reconstruction
            + self.auxk_alpha * aux_loss
            + self.dominance_alpha * dominance
            + self.frequency_matryoshka_alpha * matryoshka
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

        return {
            "loss": loss,
            "mse": reconstruction.detach(),
            "l0": l0.detach(),
            "auxk": aux_loss.detach(),
            "dead": x.new_tensor(float(n_dead)),
            "threshold": threshold.detach().clone(),
            "dominance": dominance.detach(),
            "frequency_matryoshka": matryoshka.detach(),
            "max_band_power_share": max_share.detach(),
        }

    def post_step(self) -> None:
        if self.dc_mode == "remove":
            self._enforce_frequency_masks()
        super().post_step()


__all__ = ["SpectralTXCV2"]
