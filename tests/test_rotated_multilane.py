"""Rotated multilane (FB-4) — generator contract tests.

Covers the FreqBench FB-4 card (freqbench/cards/FB-4.md § 2): the generator is
*exactly* the FB-2 generator composed with ONE fixed orthogonal rotation —
- the realized ``Q`` is orthogonal and identical across data seeds (drawn from
  ``rotation_seed``, not ``seed``),
- ``x``, ``emission_features``, ``lane_planes`` are the base generator's
  ``Q``-images to float precision; labels and hidden latents are untouched,
- oracle consistency: the per-lane periodogram through the *rotated* planes
  reproduces the base oracle's decisions exactly (P5 restated under rotation),
- the P1 premise survives: per-token class-conditional means ≈ 0 per lane.
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.data.synthetic import multilane_tones, multilane_tones_rotated

M = 101
KW = dict(M=M, d_in=24, sigma=0.25, seq_len=64, n_seqs=64)


def test_rotation_is_orthogonal_and_seed_independent() -> None:
    a = multilane_tones_rotated(seed=1, **KW)
    b = multilane_tones_rotated(seed=2, **KW)
    Q = a.extra["rotation_Q"].numpy().astype(np.float64)
    assert Q.shape == (24, 24)
    assert np.allclose(Q.T @ Q, np.eye(24), atol=1e-5)
    assert torch.equal(a.extra["rotation_Q"], b.extra["rotation_Q"])
    assert a.extra["rotation_seed"] == 777


def test_rotated_fields_are_exact_Q_images_of_base() -> None:
    base = multilane_tones(seed=1, **KW)
    rot = multilane_tones_rotated(seed=1, **KW)
    Q = rot.extra["rotation_Q"]
    assert torch.allclose(rot.x, base.x @ Q.T, atol=1e-5)
    assert torch.allclose(rot.emission_features, base.emission_features @ Q.T,
                          atol=1e-5)
    assert torch.allclose(rot.extra["lane_planes"],
                          torch.einsum("ij,kjl->kil", Q,
                                       base.extra["lane_planes"]), atol=1e-5)
    # labels and hidden latents untouched
    for key in ("lane_velocity_labels", "lane_velocity", "lane_phase_labels",
                "lane_of_atom"):
        assert torch.equal(rot.extra[key], base.extra[key])
    assert rot.extra["omega"] == base.extra["omega"]
    # rotated planes stay orthonormal (P5's orthogonal-separation premise)
    planes = rot.extra["lane_planes"].numpy().astype(np.float64)
    flat = np.concatenate([planes[k] for k in range(3)], axis=1)  # (24, 6)
    assert np.allclose(flat.T @ flat, np.eye(6), atol=1e-5)


def test_oracle_decisions_invariant_under_rotation() -> None:
    """Per-lane periodogram via the rotated planes on rotated data ≡ the base
    oracle on base data — decision for decision (P5 restated)."""
    from temp_bench.evals.multilane_recovery import _lane_periodogram_pred

    base = multilane_tones(seed=42, **KW)
    rot = multilane_tones_rotated(seed=42, **KW)
    omega = list(base.extra["omega"])
    xb = base.x[:, :16, :].numpy().astype(np.float64)
    xr = rot.x[:, :16, :].numpy().astype(np.float64)
    for k in range(3):
        pb = _lane_periodogram_pred(
            xb, base.extra["lane_planes"][k].numpy().astype(np.float64),
            omega, M)
        pr = _lane_periodogram_pred(
            xr, rot.extra["lane_planes"][k].numpy().astype(np.float64),
            omega, M)
        assert (pb == pr).all()


def test_per_token_means_stay_degenerate() -> None:
    """P1 premise under rotation: per-lane class-conditional per-token means
    vanish (velocity stays 2nd-moment; the T=1 falsifier's analytic side)."""
    rot = multilane_tones_rotated(seed=1, M=M, d_in=24, sigma=0.25,
                                  seq_len=64, n_seqs=2048)
    x = rot.x.numpy().reshape(-1, 24)
    lab = rot.extra["lane_velocity_labels"].numpy().reshape(-1, 3)
    # MC floor: for the Y=0 class the phase is frozen per sequence, so the
    # effective count is ~n_seqs/10 sequences → per-coord std ≈ 1/√205 ≈ 0.07.
    for k in range(3):
        mus = np.stack([x[lab[:, k] == c].mean(axis=0) for c in range(10)])
        assert float(np.abs(mus).max()) < 0.10          # measured 0.0496
