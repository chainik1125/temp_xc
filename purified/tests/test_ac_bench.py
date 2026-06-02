"""AC-only signed-motion bench — generator + metric-wiring smoke tests.

Covers the FrequencyBench § 5 add-on:
- the ``signed_motion`` generator's shapes + ground-truth labels,
- the data-processing-inequality premise (per-token marginal of the symbol
  is uniform regardless of the hidden sign),
- the order-sensitive signal (the consecutive step encodes the sign),
- that :func:`signed_motion_metrics` returns the right keys and that the
  ``atom_dc_fraction`` diagnostic is defined only for the window decoder.
"""

from __future__ import annotations

import torch

from temp_bench.data.synthetic import (
    coupled_hmm,
    markov_chain_support,
    signed_motion,
)


def test_signed_motion_generator_shapes() -> None:
    data = signed_motion(M=7, v=3, d_in=10, seq_len=8, n_seqs=16, seed=0)
    assert data.x.shape == (16, 8, 10)
    assert data.emission_features.shape == (7, 10)
    assert data.hidden_features is None
    assert data.support is None
    signs = data.extra["sign_labels"]
    assert set(signs.unique().tolist()).issubset({-1, 1})
    phases = data.extra["phase_labels"]
    assert int(phases.min()) >= 0 and int(phases.max()) <= 6


def test_signed_motion_alphabet_orthonormal() -> None:
    data = signed_motion(M=19, v=9, d_in=40, seq_len=8, n_seqs=8, seed=1)
    G = data.emission_features @ data.emission_features.T
    off = (G - torch.eye(19)).abs().max().item()
    assert off < 1e-4, f"alphabet not orthonormal (max off-diag {off})"


def test_signed_motion_dpi_premise_uniform_marginal() -> None:
    """Per-token symbol marginal is ~uniform AND independent of the sign.

    This is the I(S; Q_t)=0 premise the impossibility result rests on.
    """
    M = 7
    data = signed_motion(M=M, v=3, d_in=10, seq_len=64, n_seqs=16384, seed=0)
    feats = data.emission_features
    sims = torch.einsum("ntd,md->ntm", data.x, feats)
    q = sims.argmax(dim=-1)                                  # (N, T)
    S = data.extra["sign_labels"]

    # Overall marginal ~ uniform.
    counts = torch.bincount(q.flatten(), minlength=M).float()
    counts /= counts.sum()
    assert torch.allclose(counts, torch.full((M,), 1 / M), atol=0.02)

    # Conditional on each sign, the marginal at a fixed t is still uniform.
    for s in (-1, 1):
        qs = q[S == s][:, 5]
        c = torch.bincount(qs, minlength=M).float()
        c /= c.sum()
        assert torch.allclose(c, torch.full((M,), 1 / M), atol=0.05)


def test_signed_motion_step_encodes_sign() -> None:
    """The order-sensitive signal: Q_{t+1} - Q_t == S*v (mod M) exactly."""
    M, v = 19, 9
    data = signed_motion(M=M, v=v, d_in=40, seq_len=32, n_seqs=256, seed=2)
    feats = data.emission_features
    q = torch.einsum("ntd,md->ntm", data.x, feats).argmax(dim=-1)
    step = (q[:, 1:] - q[:, :-1]) % M
    expected = (data.extra["sign_labels"].unsqueeze(1) * v) % M
    assert bool((step == expected).all())


def test_other_generators_have_no_sign_labels() -> None:
    """Gating contract: markov/coupled leave `extra` None so the s_temp
    add-on never fires for the committed § 4 benches."""
    mk = markov_chain_support(n_features=4, d_in=8, seq_len=8, n_seqs=8, seed=0)
    cp = coupled_hmm(K_hidden=3, M_emissions=4, n_parents=2, d_in=16,
                     seq_len=8, n_seqs=8, seed=0)
    assert mk.extra is None
    assert cp.extra is None


def test_signed_motion_metrics_keys_and_dc_fraction() -> None:
    """signed_motion_metrics returns s_temp + acc for any arch, and
    atom_dc_fraction only for the (d_sae, T, d_in) window decoder."""
    from temp_bench.archs.txc_base import TXCBase
    from temp_bench.archs.topk_sae import TopKSAE
    from temp_bench.evals.signed_motion_recovery import signed_motion_metrics

    data = signed_motion(M=19, v=9, d_in=40, seq_len=16, n_seqs=512, seed=0)

    txc = TXCBase(d_in=40, d_sae=20, T=5, k_pos=2)
    m_txc = signed_motion_metrics(txc, data)
    assert "s_temp" in m_txc and "sign_probe_acc" in m_txc
    assert -1.0 <= m_txc["s_temp"] <= 1.0
    assert 0.0 <= m_txc["sign_probe_acc"] <= 1.0
    # Window decoder → DC fraction defined and in [0, 1].
    assert "atom_dc_fraction" in m_txc
    assert 0.0 <= m_txc["atom_dc_fraction"] <= 1.0

    topk = TopKSAE(d_in=40, d_sae=20, k_pos=2)
    m_topk = signed_motion_metrics(topk, data)
    assert "s_temp" in m_topk and "sign_probe_acc" in m_topk
    # Token arch → no aligned window decoder → metric omitted.
    assert "atom_dc_fraction" not in m_topk
