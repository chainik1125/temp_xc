"""Phasepair bench (FB-1) — pair detection + metric tests.

Covers the FreqBench FB-1 card (freqbench/cards/FB-1.md):
- ``find_pairs`` identifies ± pairs and returns none for the frequency Ω
  (the dispatch no-op contract — old frequency rows stay byte-identical),
- the exact bag-null premise: within a pair, the window symbol-SET
  distributions coincide (spot-check via sorted-set statistics),
- the signed periodogram oracle separates ±y on noiseless windows,
- ``phasepair_metrics`` keys for a token arch and a window arch, with the
  sign floor ≈ 0 for the token arch (P1).
"""

from __future__ import annotations

import numpy as np

from temp_bench.data.synthetic import cyclic_tones
from temp_bench.evals.phasepair_recovery import find_pairs, phasepair_metrics

M = 101
OMEGA_PP = (3, 98, 12, 89, 30, 71)
OMEGA_FREQ = (0, 1, 2, 4, 8, 16, 24, 32, 40, 50)


def _data(sigma=0.10, n_seqs=512, seed=0):
    return cyclic_tones(M=M, omega=OMEGA_PP, embedding="circle", d_in=24,
                        sigma=sigma, seq_len=64, n_seqs=n_seqs, seed=seed)


def test_find_pairs() -> None:
    assert find_pairs(list(OMEGA_PP), M) == [(0, 1), (2, 3), (4, 5)]
    assert find_pairs(list(OMEGA_FREQ), M) == []          # frequency: no-op


def test_pair_symbol_sets_identical_distribution() -> None:
    """±y windows have identical symbol-SET distributions (the exact bag
    null): compare the distribution of sorted circular gaps."""
    data = _data(sigma=0.0, n_seqs=4000, seed=1)
    lab = data.extra["velocity_labels"].numpy()[:, 0]
    x = data.x.numpy()
    R = data.extra["circle_plane"].numpy()
    T = 8
    # bag statistic: sorted eigenvalue-free summary — mean pairwise distance
    # of the projected points (order-free, reflection-invariant)
    proj = x[:, :T, :] @ R                                # (n, T, 2)
    d2 = ((proj[:, :, None, :] - proj[:, None, :, :]) ** 2).sum(-1)
    stat = d2.reshape(len(x), -1).mean(axis=1)
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        mi, mj = stat[lab == i], stat[lab == j]
        # noiseless: the statistic is deterministic given |y| (phase-free),
        # so within-pair means must agree to numerical precision
        assert abs(mi.mean() - mj.mean()) < 1e-4, \
            f"pair ({i},{j}) bag statistic differs ({mi.mean()} vs {mj.mean()})"
        assert mi.std() < 1e-4 and mj.std() < 1e-4


def test_signed_oracle_separates_pairs_noiseless() -> None:
    data = _data(sigma=0.0, n_seqs=512, seed=2)
    lab = data.extra["velocity_labels"].numpy()[:, 0]
    x = data.x.numpy()
    R = data.extra["circle_plane"].numpy()
    T = 8
    proj = x[:, :T, :] @ R
    c = proj[..., 0] + 1j * proj[..., 1]
    t = np.arange(T)
    basis = np.exp(-2j * np.pi * np.asarray(OMEGA_PP)[:, None] * t / M)
    pred = np.abs(c @ basis.T).argmax(axis=1)
    assert (pred == lab).mean() > 0.99                    # ± separated exactly


def test_phasepair_metrics_keys_and_token_floor() -> None:
    from temp_bench.archs.batchtopk_sae import BatchTopKSAE
    from temp_bench.archs.txc_batchtopk import TXCBatchTopKPost

    data = _data(n_seqs=512, seed=3)
    out = phasepair_metrics(BatchTopKSAE(d_in=24, d_sae=50, k_pos=2),
                            data, eval_window_L=32, n_windows=64)
    for k in ("pair_recovery", "sign_recovery", "sign_balacc", "sign_oracle"):
        assert k in out, k
    assert abs(out["sign_recovery"]) < 0.35               # token ≈ chance (P1)
    outw = phasepair_metrics(TXCBatchTopKPost(d_in=24, d_sae=101, T=8, k_pos=2),
                             data, eval_window_L=32, n_windows=64)
    assert "sign_oracle_pair0" in outw
    assert outw["sign_oracle"] > 0.8                      # raw phase info present
