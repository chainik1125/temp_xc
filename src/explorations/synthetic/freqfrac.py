"""FreqFrac — per-atom temporal frequency profiles of trained encoder atoms.

The FreqBench axis-1 lens (``experiments/explorations/synthetic/freqbench/PORT.md``
§ C): for each encoder atom with time-domain taps ``e_{h,τ,:} ∈ R^{d_in}``
(τ = 0..T−1), transform the tap sequence into the orthonormal DCT-II basis,
take the per-frequency energy summed over ``d_in``, and normalize per atom.
``FreqFrac_h(w)`` is then a distribution over temporal-frequency indices
``w = 0..T−1`` — the fraction of atom ``h``'s encoder energy at frequency
``w``. Port of ``freq_profile`` from Dmitry's FreqBench sprint
(``origin/dmitry-spectral-sprint2 … fb_core.py``), generalized to the v2
fair-backbone panel.

Methodology rules carried from the sprint (both mandatory in any report):

1. **Firing-weighted aggregation** — population means dilute; weight the
   per-arch curve by each atom's realized activity on the shared eval
   windows (:func:`firing_weights`).
2. **Untrained same-arch null** — random-init profiles are the baseline any
   claim of *learned* spectral structure must clear.

Tap extraction is duck-typed on the panel's weight layouts:

- ``W_enc (T, d_in, d_sae)`` — ``txc_batchtopk_pre`` / ``txc_batchtopk_post``
  (identical encoder tensor; they differ only in where BatchTopK applies) and
  ``stacked_batchtopk`` (independent per-position dicts).
- ``enc_coef`` band coefficients — ``spectral_txc`` (already
  DCT-parameterized; we synthesize time-domain kernels and transform back,
  which recovers the band support exactly — Parseval).
- ``W_enc (d_in, d_sae)`` — ``batchtopk_sae`` / ``tsae`` (T = 1): a single
  tap has all its energy at DC by construction; these archs have no temporal
  response (their window behaviour lives in the stacked *probe*, not the
  code — proofs registry P2).

No leaderboard interaction and no ``temp_bench/core`` imports — this is a
weight-space diagnostic, not an evaluator (PORT.md § D).
"""

from __future__ import annotations

import numpy as np
import torch


def dct_basis(T: int) -> torch.Tensor:
    """(T, T) orthonormal DCT-II basis; row ``w`` = temporal-frequency index.

    Kept numerically identical to ``temp_bench.archs.spectral_txc._dct_basis``
    (cross-checked in ``tests/test_freqfrac.py``) without importing the
    plugin's private helper.
    """
    tau = np.arange(T)
    psi = np.zeros((T, T), dtype=np.float32)
    for w in range(T):
        if w == 0:
            psi[w] = np.sqrt(1.0 / T)
        else:
            psi[w] = np.sqrt(2.0 / T) * np.cos(np.pi * (tau + 0.5) * w / T)
    return torch.from_numpy(psi)


def encoder_taps(model: torch.nn.Module) -> torch.Tensor:
    """(H, T, d_in) time-domain encoder kernels, one row per atom.

    Duck-typed on the panel layouts (see module docstring). Atom order matches
    the model's code order (for ``spectral_txc`` the per-band kernels are
    concatenated in ``band_slices`` order, which *is* the code order).
    """
    with torch.no_grad():
        if hasattr(model, "bands") and hasattr(model, "_enc_kernel"):
            return torch.cat(
                [model._enc_kernel(b) for b in range(model.n_bands)], dim=0
            ).detach()
        W = getattr(model, "W_enc", None)
        if W is None:
            raise TypeError(
                f"{type(model).__name__}: no recognised encoder layout "
                "(expected spectral enc_coef bands, or W_enc of dim 2/3)."
            )
        if W.dim() == 3:  # (T, d_in, d_sae) — txc pre/post, stacked
            return W.detach().permute(2, 0, 1).contiguous()
        if W.dim() == 2:  # (d_in, d_sae) — token archs, T = 1
            return W.detach().T.unsqueeze(1).contiguous()
    raise TypeError(
        f"{type(model).__name__}: W_enc has unsupported dim {W.dim()}."
    )


def freq_profile(
    model: torch.nn.Module | None = None, *, taps: torch.Tensor | None = None
) -> torch.Tensor:
    """(H, T) per-atom FreqFrac: normalized DCT energy of the encoder taps.

    Row ``h`` sums to 1; column ``w`` is the fraction of atom ``h``'s encoder
    energy at temporal-frequency index ``w``. Pass either a panel model or
    precomputed ``taps`` ``(H, T, d_in)``.
    """
    if taps is None:
        if model is None:
            raise ValueError("pass a model or taps")
        taps = encoder_taps(model)
    _H, T, _d = taps.shape
    psi = dct_basis(T).to(dtype=taps.dtype, device=taps.device)
    coef = torch.einsum("wt,htd->hwd", psi, taps)
    energy = coef.pow(2).sum(dim=-1)                              # (H, T)
    return energy / energy.sum(dim=-1, keepdim=True).clamp(min=1e-12)


@torch.no_grad()
def firing_weights(
    model: torch.nn.Module, x: torch.Tensor, batch_size: int = 1024
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-atom activity on eval windows: (mean_activation, firing_rate).

    ``x`` is ``(N, T, d_in)`` windows in the arch's native ``T``;
    ``model.encode`` output may keep a window axis (shared-code archs return
    ``(B, 1, d_sae)``, stacked returns ``(B, T, d_sae)``) — all leading axes
    are pooled. The model is put in eval mode (JumpReLU threshold if trained;
    BatchTopK fallback for untrained nulls). Restricted to window archs;
    T = 1 archs have a constant DC profile, so weighting cannot change their
    curve — use uniform weights there.
    """
    was_training = model.training
    model.eval()
    try:
        total = None
        fired = None
        n_rows = 0
        for i in range(0, x.shape[0], batch_size):
            z = model.encode(x[i : i + batch_size])
            z2 = z.reshape(-1, z.shape[-1])
            total = z2.sum(0) if total is None else total + z2.sum(0)
            fired = ((z2 > 0).float().sum(0) if fired is None
                     else fired + (z2 > 0).float().sum(0))
            n_rows += z2.shape[0]
    finally:
        if was_training:
            model.train()
    return total / max(n_rows, 1), fired / max(n_rows, 1)


def arch_curve(
    profile: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """(T,) per-arch frequency-response curve: weighted mean of atom profiles.

    ``weights`` are non-negative per-atom weights (e.g. mean activation from
    :func:`firing_weights`); ``None`` → uniform (the population mean — report
    it only next to the firing-weighted curve, never alone).
    """
    if weights is None:
        weights = torch.ones(profile.shape[0], dtype=profile.dtype)
    w = weights.to(dtype=profile.dtype, device=profile.device).clamp(min=0)
    if float(w.sum()) <= 0:
        w = torch.ones_like(w)
    w = w / w.sum()
    return (profile * w[:, None]).sum(dim=0)


def spectral_concentration(profile: torch.Tensor) -> torch.Tensor:
    """(H,) top-2-adjacent concentration: max over w of p[w] + p[w+1].

    The sprint's tone-likeness statistic — a pure single tone scores ≈ 1, a
    flat profile scores ≈ 2/T. Compare against the untrained-init null before
    claiming learned structure (random-init concentration ≈ 0.205 at T = 16
    in the sprint). T = 1 profiles score 1 by convention.
    """
    if profile.shape[1] == 1:
        return torch.ones(profile.shape[0], dtype=profile.dtype,
                          device=profile.device)
    pair = profile[:, :-1] + profile[:, 1:]
    return pair.max(dim=1).values


def band_fractions(
    profile: torch.Tensor, bands: list[list[int]]
) -> torch.Tensor:
    """(H, n_bands) per-atom energy mass inside each DCT-index set.

    ``bands`` is a list of index lists (e.g. ``[[0], [1..T-1]]`` for the
    DC/AC split); rows sum to 1 when the bands partition ``0..T-1``. This is
    the axis-1 coordinate aggregator (the sprint's "lowfrac" is the low-band
    column).
    """
    cols = [profile[:, idx].sum(dim=1) for idx in bands]
    return torch.stack(cols, dim=1)
