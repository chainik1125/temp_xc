"""Colored-sources feature-direction recovery — FreqBench FB-3 add-on.

Like the other add-ons, NOT a routed evaluator: :class:`SyntheticRecovery`
calls :func:`colored_metrics` whenever the materialised data carries
``extra['rho_schedule']`` (the ``toy_colored_sources_*`` datasources). No-op
for every other bench → ``protocol_version`` stays put.

Why ``eauc`` cannot serve as the primary: the CS-1/CS-2 proofs
(freqbench/cards/FB-3.md § 3) are stated in squared-cosine recovery
``Rec = (1/N) Σ_i max_j cos²(f_i, w_j)`` with an explicit chance floor
(random directions score ~log(H)/N, NOT 0), so the headline must be
chance-adjusted — and a window atom's direction content may live at any
decoder tap, so the max must run over per-position decoder slices. The
stored eauc scalars (threshold-AUC of |cos| over T-averaged directions)
cannot express either. This is a **weight-space** metric: no probe, no
data pass — memorization-free by construction.

Metrics:

- ``colored_rec_adj`` (**headline**): ``(rec_sq − chance)/(1 − chance)``
  where ``chance`` is the empirical mean of ``rec_sq`` over seeded random
  unit dictionaries of the SAME candidate count (16 draws; the analytic
  scale is ~log(n_cand)/d_in).
- ``colored_rec_sq``, ``colored_chance``, ``colored_chance_std``.
- ``colored_rec_q1..q4``: per-source max-cos² averaged within ρ-quartiles
  (q4 = highest ρ) — recovery should be ρ-ordered if the temporal route is
  used (CS-2: signal strength ∝ ρ).
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.interfaces.architecture import TempBenchArch


@torch.no_grad()
def _candidate_directions(model: TempBenchArch) -> torch.Tensor:
    """All per-position decoder slices, ``(n_cand, d_in)``.

    - spectral archs expose ``_dec_full()`` → (d_sae, T, d_in);
    - crosscoders / stacked hold a 3-D ``W_dec`` ((d_sae, T, d_in) or
      (T, d_sae, d_in) — both flatten to per-position slices);
    - token archs hold a 2-D ``W_dec`` (d_sae, d_in).
    """
    if hasattr(model, "_dec_full"):
        K = model._dec_full().detach().cpu().float()
    else:
        K = model.W_dec.detach().cpu().float()
    d_in = int(model.config.d_in)
    return K.reshape(-1, d_in)


def _rec_per_source(cand: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """max_j cos²(f_i, cand_j) per source row i → (N,)."""
    C = cand / cand.norm(dim=1, keepdim=True).clamp(min=1e-8)
    Fn = F / F.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos2 = (C @ Fn.T).pow(2)                          # (n_cand, N)
    return cos2.max(dim=0).values


def _empirical_chance(n_cand: int, d_in: int, N: int, *, n_draws: int = 16,
                      seed: int = 0) -> tuple[float, float]:
    """Mean/std of rec_sq for random unit dictionaries of the same shape."""
    g = torch.Generator().manual_seed(seed)
    F = torch.linalg.qr(torch.randn(d_in, d_in, generator=g))[0][:, :N].T
    vals = []
    for _ in range(n_draws):
        R = torch.randn(n_cand, d_in, generator=g)
        vals.append(float(_rec_per_source(R, F).mean()))
    return float(np.mean(vals)), float(np.std(vals))


def colored_metrics(model: TempBenchArch, data, **_kw) -> dict[str, float]:
    """Chance-adjusted squared-cosine recovery of F from decoder geometry."""
    F = data.emission_features.detach().cpu().float()          # (N, d_in)
    N, d_in = F.shape
    cand = _candidate_directions(model)
    rec = _rec_per_source(cand, F)                             # (N,)
    rec_sq = float(rec.mean())
    chance, chance_std = _empirical_chance(cand.shape[0], d_in, N)
    denom = max(1.0 - chance, 1e-9)

    out = {
        "colored_rec_sq": rec_sq,
        "colored_chance": chance,
        "colored_chance_std": chance_std,
        "colored_rec_adj": float((rec_sq - chance) / denom),
    }
    # ρ-quartile curve (ρ_i ascending by construction of the schedule; sort
    # defensively anyway).
    rho = data.extra["rho_schedule"].detach().cpu().numpy()
    order = np.argsort(rho)
    quarts = np.array_split(rec.numpy()[order], 4)
    for q, vals in enumerate(quarts, start=1):
        out[f"colored_rec_q{q}"] = float(np.mean(vals))
    return out
