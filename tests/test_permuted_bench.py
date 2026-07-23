"""Permuted tones bench (FB-5) — generator + ground-truth + add-on tests.

Covers the FreqBench FB-5 card (freqbench/cards/FB-5.md § 2 build spec):
- schedule table: K valid permutations of Z_M, re-drawn per data seed,
  reproducible at fixed seed,
- trajectory construction ``z_t = π_Y((t+B) mod M)`` exact against the table,
- **P1 marginal uniformity**: per-class pooled symbol histogram ≈ uniform
  (the exact per-token deadness premise, numerically),
- per-token class-conditional means ≈ 0 (the raw-linear § 8 premise),
- matched-filter oracle: saturates on the built generator at T=8 (window
  uniqueness across all K·M (k,s) pairs),
- evaluator add-on: ``permuted_metrics`` produces the § 2 keys and its
  oracle reads the task on an untrained token arch's raw tiles,
- dispatch contract: tone datasources do not carry ``schedule_table``.
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.data.synthetic import cyclic_tones, permuted_tones

M, K = 101, 10


def test_schedule_table_valid_and_seed_behavior() -> None:
    a = permuted_tones(M=M, K=K, d_in=32, sigma=0.1, seq_len=64, n_seqs=32, seed=1)
    a2 = permuted_tones(M=M, K=K, d_in=32, sigma=0.1, seq_len=64, n_seqs=32, seed=1)
    b = permuted_tones(M=M, K=K, d_in=32, sigma=0.1, seq_len=64, n_seqs=32, seed=2)
    P = a.extra["schedule_table"].numpy()
    assert P.shape == (K, M)
    for k in range(K):
        assert np.array_equal(np.sort(P[k]), np.arange(M))      # bijective
    assert torch.equal(a.extra["schedule_table"], a2.extra["schedule_table"])
    assert not torch.equal(a.extra["schedule_table"], b.extra["schedule_table"])


def test_trajectory_matches_table_exactly() -> None:
    d = permuted_tones(M=M, K=K, d_in=16, sigma=0.0, seq_len=64, n_seqs=64, seed=3)
    P = d.extra["schedule_table"].numpy()
    Y = d.extra["schedule_labels"][:, 0].numpy()
    B = d.extra["offset_labels"].numpy()
    U = d.emission_features.numpy()
    t = np.arange(64)
    for i in (0, 5, 63):
        z = P[Y[i]][(B[i] + t) % M]
        assert np.allclose(d.x[i].numpy(), U[z], atol=1e-6)


def test_p1_marginal_uniform_per_class() -> None:
    d = permuted_tones(M=M, K=K, d_in=16, sigma=0.0, seq_len=64, n_seqs=2000, seed=0)
    P = d.extra["schedule_table"].numpy()
    Y = d.extra["schedule_labels"][:, 0].numpy()
    B = d.extra["offset_labels"].numpy()
    t = np.arange(64)
    for k in range(K):
        z = P[k][(B[Y == k][:, None] + t[None, :]) % M].ravel()
        hist = np.bincount(z, minlength=M) / len(z)
        tv = 0.5 * np.abs(hist - 1.0 / M).sum()
        assert tv < 0.10, (k, tv)          # MC scale ~0.04 at ~12.8k samples


def test_per_token_class_means_vanish() -> None:
    d = permuted_tones(M=M, K=K, d_in=32, sigma=0.10, seq_len=64, n_seqs=2000, seed=0)
    x = d.x.numpy().reshape(-1, 32)
    y = d.extra["schedule_labels"].numpy().ravel()
    mus = np.stack([x[y == k].mean(axis=0) for k in range(K)])
    # every sequence visits 64 DISTINCT symbols (t+B distinct mod prime M),
    # so the pooled means concentrate fast; no frozen-phase class exists.
    assert float(np.abs(mus).max()) < 0.05


def test_matched_filter_oracle_saturates() -> None:
    from temp_bench.evals.permuted_recovery import _matched_filter_pred
    d = permuted_tones(M=M, K=K, d_in=32, sigma=0.10, seq_len=64, n_seqs=256, seed=1)
    U = d.emission_features.numpy().astype(np.float64)
    P = d.extra["schedule_table"].numpy()
    Y = d.extra["schedule_labels"][:, 0].numpy()
    T = 8
    tiles = d.x[:, :T, :].numpy().astype(np.float64)
    pred = _matched_filter_pred(tiles, U, P, M)
    assert float((pred == Y).mean()) >= 0.99


def test_eval_addon_keys_and_dispatch_contract() -> None:
    from temp_bench.archs.batchtopk_sae import BatchTopKSAE
    from temp_bench.evals.permuted_recovery import permuted_metrics
    d = permuted_tones(M=M, K=K, d_in=32, sigma=0.10, seq_len=64,
                       n_seqs=256, seed=1)
    model = BatchTopKSAE(d_in=32, d_sae=48, k_pos=2)
    out = permuted_metrics(model, d, eval_window_L=32, n_windows=64)
    for key in ("schedule_recovery", "schedule_balacc", "schedule_oracle",
                "schedule_chance", "sched_recall_c0", "sched_oracle_c9"):
        assert key in out, key
    # token archs tile at T=1, where a single token carries zero schedule
    # information (P1) — the matched filter must sit at ~chance there.
    assert out["schedule_oracle"] <= 0.2
    assert 0.0 <= out["schedule_balacc"] <= 1.0
    # tone datasources must NOT carry the dispatch key
    tone = cyclic_tones(M=M, embedding="circle", d_in=32, sigma=0.1,
                        seq_len=64, n_seqs=16, seed=0)
    assert "schedule_table" not in tone.extra
