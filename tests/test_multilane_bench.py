"""Multilane superposition bench (FB-2) — generator + ground-truth tests.

Covers the FreqBench FB-2 card (freqbench/cards/FB-2.md):
- ``multilane_tones`` shapes + ground-truth labels + plane orthogonality,
- per-lane independence of the hidden latents,
- the **per-lane tone property**: projecting a noiseless window onto lane k's
  plane and periodogram-peak-picking recovers Y_k exactly (P5: orthogonal
  planes ⇒ exact lane separation — the other lanes vanish under projection),
- the per-token / raw-linear premises: ``E[x_t | Y_k] ≈ 0`` per lane
  (velocity is 2nd-moment; the equality-variant § 8 situation),
- codebook geometry (3M atoms, unit-norm, rank-6 total),
- dispatch contract (other extra-bearing generators don't carry the key),
- evaluator add-on keys for token + spectral archs.
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.data.synthetic import cyclic_tones, multilane_tones

OMEGA = (0, 1, 2, 4, 8, 16, 24, 32, 40, 50)
M = 101


def test_multilane_shapes_and_extra() -> None:
    data = multilane_tones(M=M, d_in=24, sigma=0.1, seq_len=64, n_seqs=64, seed=0)
    assert data.x.shape == (64, 64, 24)
    assert data.emission_features.shape == (3 * M, 24)
    assert data.hidden_features is None
    ll = data.extra["lane_velocity_labels"]
    assert ll.shape == (64, 64, 3)
    assert int(ll.min()) >= 0 and int(ll.max()) < len(OMEGA)
    assert data.extra["lane_planes"].shape == (3, 24, 2)
    assert data.extra["lane_phase_labels"].shape == (64, 3)
    assert data.extra["n_lanes"] == 3
    assert list(data.extra["omega"]) == list(OMEGA)


def test_multilane_planes_orthonormal_and_disjoint() -> None:
    data = multilane_tones(seq_len=8, n_seqs=8, seed=1)
    P = data.extra["lane_planes"].numpy()               # (3, d_in, 2)
    flat = P.transpose(0, 2, 1).reshape(6, -1)          # 6 axis vectors
    G = flat @ flat.T
    assert np.abs(G - np.eye(6)).max() < 1e-5, "plane axes must be orthonormal"


def test_multilane_labels_constant_per_seq_and_independent() -> None:
    data = multilane_tones(seq_len=64, n_seqs=2048, seed=2)
    ll = data.extra["lane_velocity_labels"].numpy()
    assert (ll == ll[:, :1, :]).all(), "velocity constant along the sequence"
    lab = ll[:, 0, :]                                   # (n, 3)
    # lanes are independent draws: cross-lane label correlation ≈ 0
    for a in range(3):
        for b in range(a + 1, 3):
            r = np.corrcoef(lab[:, a], lab[:, b])[0, 1]
            assert abs(r) < 0.08, f"lanes {a},{b} correlated ({r:.3f})"


def test_multilane_per_lane_tone_property_noiseless() -> None:
    """P5 discharge shape: per-lane periodogram on the true plane is exact
    (noiseless, T=16): orthogonal planes remove the other two lanes."""
    data = multilane_tones(M=M, d_in=24, sigma=0.0, seq_len=64, n_seqs=256, seed=3)
    P = data.extra["lane_planes"].numpy()
    lab = data.extra["lane_velocity_labels"].numpy()[:, 0, :]
    T = 16
    win = data.x.numpy()[:, :T, :]
    t = np.arange(T)
    basis = np.exp(-2j * np.pi * np.asarray(OMEGA)[:, None] * t[None, :] / M)
    for k in range(3):
        proj = win @ P[k]                               # (n, T, 2)
        c = proj[..., 0] + 1j * proj[..., 1]
        pred = np.abs(c @ basis.T).argmax(axis=1)
        assert (pred == lab[:, k]).mean() > 0.999, f"lane {k} oracle not exact"


def test_multilane_class_conditional_mean_near_zero() -> None:
    """E[x_t | Y_k] ≈ 0 per lane (phases uniform) — velocity is 2nd-moment."""
    data = multilane_tones(M=M, d_in=24, sigma=0.0, seq_len=64, n_seqs=4000, seed=4)
    x = data.x.numpy().reshape(-1, 24)
    ll = data.extra["lane_velocity_labels"].numpy().reshape(-1, 3)
    for k in range(3):
        for c in (0, 5, 9):                             # spot-check classes
            mu = x[ll[:, k] == c].mean(axis=0)
            assert np.linalg.norm(mu) < 0.10, (
                f"lane {k} class {c} mean not ~0 ({np.linalg.norm(mu):.3f})")


def test_multilane_codebook_geometry() -> None:
    data = multilane_tones(seq_len=8, n_seqs=8, seed=5)
    U = data.emission_features
    assert torch.allclose(U.norm(dim=1), torch.ones(3 * M), atol=1e-4)
    sv = torch.linalg.svdvals(U)
    assert int((sv > 1e-3).sum()) == 6, "codebook must span exactly the 6 plane axes"
    lane_tag = data.extra["lane_of_atom"].numpy()
    assert (np.bincount(lane_tag) == M).all()


def test_multilane_d_in_check() -> None:
    import pytest
    with pytest.raises(ValueError, match="d_in"):
        multilane_tones(d_in=4, n_lanes=3, seed=0)


def test_other_generators_do_not_trip_multilane_dispatch() -> None:
    freq = cyclic_tones(embedding="circle", seq_len=8, n_seqs=8, seed=0)
    assert "lane_velocity_labels" not in freq.extra


def test_multilane_metrics_keys_token_and_spectral() -> None:
    from temp_bench.archs.batchtopk_sae import BatchTopKSAE
    from temp_bench.archs.spectral_txc import SpectralTXCBatchTopK
    from temp_bench.evals.multilane_recovery import multilane_metrics

    data = multilane_tones(M=M, d_in=24, sigma=0.25, seq_len=64, n_seqs=512, seed=0)
    base = {"multilane_recovery", "multilane_oracle", "multilane_chance"} | {
        f"lane{k}_{s}" for k in range(3) for s in ("recovery", "balacc", "oracle", "chance")
    }
    out = multilane_metrics(BatchTopKSAE(d_in=24, d_sae=50, k_pos=1),
                            data, eval_window_L=32, n_windows=64)
    assert base <= set(out)
    assert not any(k.startswith("band") for k in out)
    assert -0.5 < out["multilane_recovery"] < 0.5      # per-token ≈ chance
    sp = SpectralTXCBatchTopK(d_in=24, d_sae=101, T=8, k_pos=1, bands="multiband")
    outw = multilane_metrics(sp, data, eval_window_L=32, n_windows=64)
    assert base <= set(outw)
    assert "band0_ml_recovery" in outw
    # per-lane periodogram oracle well above chance at T=8 on raw tiles
    assert outw["multilane_oracle"] > 0.5
