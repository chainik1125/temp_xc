"""Cyclic-tone frequency bench — generator + ground-truth tests.

Covers the autoresearch #3 add-on (frequency/bench_spec.md, amendments A1–A4):
- the ``cyclic_tones`` generator's shapes + ground-truth labels (circle+random),
- the walk algebra (velocity constant/seq; label = index into Ω),
- the **tone property**: under the circle embedding the noiseless window's
  periodogram peaks at the true velocity (the ML oracle recovers Y),
- the per-token / raw-linear premises: the class-conditional mean ``E[x_t|Y]≈0``
  (velocity is 2nd-moment, not linear — amendment A4),
- the codebook geometry (circle atoms 2-D & unit-norm; random atoms orthonormal
  F=M) and the exposed circle plane R,
- that other generators don't carry ``velocity_labels`` (dispatch contract).
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.data.synthetic import cyclic_tones, semi_markov_modes

OMEGA = (0, 1, 2, 4, 8, 16, 24, 32, 40, 50)


def test_cyclic_tones_shapes_and_extra() -> None:
    for emb in ("circle", "random"):
        data = cyclic_tones(embedding=emb, M=101, d_in=128, sigma=0.1,
                            seq_len=64, n_seqs=64, seed=0)
        assert data.x.shape == (64, 64, 128)
        assert data.emission_features.shape == (101, 128)   # M symbol atoms
        assert data.hidden_features is None                 # velocity not a direction
        vl = data.extra["velocity_labels"]
        assert vl.shape == (64, 64)
        assert int(vl.min()) >= 0 and int(vl.max()) < len(OMEGA)
        assert data.extra["M"] == 101
        assert data.extra["embedding"] == emb
        assert list(data.extra["omega"]) == list(OMEGA)
        if emb == "circle":
            assert data.extra["circle_plane"].shape == (128, 2)
        else:
            assert "circle_plane" not in data.extra


def test_cyclic_tones_velocity_constant_per_seq() -> None:
    data = cyclic_tones(embedding="circle", seq_len=64, n_seqs=128, seed=1)
    vl = data.extra["velocity_labels"].numpy()
    vv = data.extra["velocity"].numpy()
    # constant along the sequence (single-tone task)
    assert (vl == vl[:, :1]).all()
    assert (vv == vv[:, :1]).all()
    # velocity value == Ω[class index]
    assert np.array_equal(vv[:, 0], np.asarray(OMEGA)[vl[:, 0]])


def test_cyclic_tones_circle_tone_property() -> None:
    """Noiseless circle: the periodogram of a window peaks at the true velocity."""
    data = cyclic_tones(embedding="circle", M=101, d_in=128, sigma=0.0,
                        seq_len=64, n_seqs=256, seed=2)
    R = data.extra["circle_plane"].numpy()          # (d_in, 2)
    lab = data.extra["velocity_labels"].numpy()[:, 0]
    T = 16
    win = data.x.numpy()[:, :T, :]                  # (n, T, d_in)
    proj = win @ R                                  # (n, T, 2)
    c = proj[..., 0] + 1j * proj[..., 1]
    t = np.arange(T)
    basis = np.exp(-2j * np.pi * np.asarray(OMEGA)[:, None] * t[None, :] / 101)
    scores = np.abs(c @ basis.T)                    # (n, |Ω|)
    pred = scores.argmax(axis=1)
    assert (pred == lab).mean() > 0.999             # noiseless oracle is exact at T=16


def test_cyclic_tones_class_conditional_mean_near_zero() -> None:
    """E[x_t|Y] ≈ 0 (circle centred, B uniform) — velocity is 2nd-moment (A4)."""
    data = cyclic_tones(embedding="circle", M=101, d_in=128, sigma=0.0,
                        seq_len=64, n_seqs=4000, seed=3)
    x = data.x.numpy().reshape(-1, 128)
    lab = data.extra["velocity_labels"].numpy().reshape(-1)
    # ‖E[x_t|Y]‖ ≪ signal magnitude (1.0). The DC class Y=0 has only ~n_seqs/10
    # independent phase draws (all positions share one symbol), so its empirical
    # mean has a ~0.05 sampling floor — still a >10× separation from the signal.
    for c in range(len(OMEGA)):
        mu = x[lab == c].mean(axis=0)
        assert np.linalg.norm(mu) < 0.10, f"class {c} mean not ~0 ({np.linalg.norm(mu)})"


def test_cyclic_tones_codebook_geometry() -> None:
    # circle atoms: unit-norm, and span a 2-D subspace (rank 2)
    circ = cyclic_tones(embedding="circle", M=101, d_in=128, seed=4)
    U = circ.emission_features
    assert torch.allclose(U.norm(dim=1), torch.ones(101), atol=1e-4)
    sv = torch.linalg.svdvals(U)
    assert int((sv > 1e-3).sum()) == 2, "circle atoms should live in a 2-D plane"
    # random atoms: orthonormal (F = M)
    rand = cyclic_tones(embedding="random", M=101, d_in=128, seed=4)
    Ur = rand.emission_features
    G = Ur @ Ur.T
    assert (G - torch.eye(101)).abs().max().item() < 1e-4


def test_cyclic_tones_random_needs_d_in_ge_M() -> None:
    import pytest
    with pytest.raises(ValueError, match="d_in"):
        cyclic_tones(embedding="random", M=101, d_in=64, seed=0)


def test_other_generators_do_not_trip_frequency_dispatch() -> None:
    """The evaluator dispatch keys off extra['velocity_labels']; the changepoint
    generator (also extra-bearing) must not carry it."""
    cp = semi_markov_modes(seq_len=8, n_seqs=8, seed=0)
    assert "velocity_labels" not in cp.extra


# ── spectral_txc arch (DCT-band BatchTopK) ──────────────────────────────


def test_spectral_txc_bands_adapt_to_T() -> None:
    from temp_bench.archs.spectral_txc import _build_bands
    # multiband degenerates to the available DCT indices at small T
    assert _build_bands(2, "multiband") == [[0], [1]]
    assert _build_bands(8, "multiband") == [[0], [1, 2], [3, 4], [5, 6, 7]]
    assert _build_bands(16, "multiband") == [[0], [1, 2, 3, 4, 5],
                                             [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    assert _build_bands(8, "dcac") == [[0], [1, 2, 3, 4, 5, 6, 7]]
    assert _build_bands(8, "full") == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_spectral_txc_shapes_budget_and_parseval() -> None:
    from temp_bench.archs.spectral_txc import SpectralTXCBatchTopK
    for T in (2, 4, 8, 16):
        m = SpectralTXCBatchTopK(d_in=128, d_sae=101, T=T, k_pos=1, bands="multiband")
        # per-band budget sums to the total window budget k_win = k_pos·T
        assert sum(m.k_per_band) == 1 * T
        assert sum(m.h_per_band) == 101
        x = torch.randn(64, T, 128)
        z = m.encode(x)
        assert z.shape == (64, 1, 101)
        # L0 == k_win (each band contributes its per-window budget)
        assert abs(float((z != 0).float().sum(-1).mean()) - T) < 1e-6
        assert m.decode(z).shape == (64, T, 128)
        # Parseval decoder unit-norm: time-domain atom L2 == coefficient L2 == 1
        m.post_step()
        W = m.decoder_directions()          # averaged over T (not unit-norm itself)
        full = m._dec_full()                # (d_sae, T, d_in) time-domain kernels
        norms = full.flatten(1).norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
        assert W.shape == (101, 128)


def test_spectral_txc_full_band_matches_window_crosscoder_shape() -> None:
    """The 'full' single-band mode is the vanilla DCT crosscoder (1 band)."""
    from temp_bench.archs.spectral_txc import SpectralTXCBatchTopK
    m = SpectralTXCBatchTopK(d_in=64, d_sae=40, T=8, k_pos=2, bands="full")
    assert m.n_bands == 1 and m.k_per_band == [16] and m.h_per_band == [40]


# ── evaluator dispatch (velocity_recovery + S(f) + bands) ───────────────


def test_frequency_metrics_keys_token_and_window() -> None:
    from temp_bench.archs.batchtopk_sae import BatchTopKSAE
    from temp_bench.archs.spectral_txc import SpectralTXCBatchTopK
    from temp_bench.evals.frequency_recovery import frequency_metrics

    data = cyclic_tones(embedding="circle", M=101, d_in=128, sigma=0.1,
                        seq_len=64, n_seqs=512, seed=0)
    base = {"velocity_recovery", "velocity_balacc", "velocity_oracle",
            "velocity_chance"} | {f"vel_recall_c{c}" for c in range(10)} \
        | {f"vel_oracle_c{c}" for c in range(10)}
    # per-token: no band keys
    out = frequency_metrics(BatchTopKSAE(d_in=128, d_sae=64, k_pos=1),
                            data, eval_window_L=32, n_windows=64)
    assert base <= set(out)
    assert not any(k.startswith("band") for k in out)
    # window spectral: adds per-band keys, oracle in [chance, 1]
    sp = SpectralTXCBatchTopK(d_in=128, d_sae=101, T=8, k_pos=1, bands="multiband")
    outw = frequency_metrics(sp, data, eval_window_L=32, n_windows=64)
    assert base <= set(outw)
    assert {"band0_recovery", "band1_recovery"} <= set(outw)
    assert 0.1 - 1e-6 <= outw["velocity_oracle"] <= 1.0 + 1e-6
