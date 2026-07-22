"""Tests for the expansion factory harness (pure-math parts — no network).

Covers: the generalized signature toolkit + null battery (a planted
self-exciting process must beat its own shuffle; an iid process must not),
the Appendix-B mirror fits (parameter recovery + round-trip validation), the
labeler parsing/agreement math, the spend meter's hard cap, and the corpus
sentence splitter. The Claude-calling paths are exercised only up to the
request boundary.
"""

import numpy as np
import pytest

from explorations.synthetic.expansion import corpus, labeler, mirrors
from explorations.synthetic.expansion import signature as sig
from explorations.synthetic.expansion.client import Meter, SpendCapExceeded


def _selfexciting(n=120, L=90, seed=0):
    rng = np.random.default_rng(seed)
    params = {"process": "logistic_ar", "K": 2, "position": True,
              "intercept": -2.6, "coef_position": 0.5, "kernel_w": [1.8, 0.8]}
    return mirrors.gen_logistic_ar(params, [L] * n, rng), rng


def test_signature_binary_gate_positive():
    seqs, _ = _selfexciting()
    stats = sig.measure(seqs, "binary", seed=0, n_null=60, n_boot=60,
                        noise_eps=(0.05,))
    acf1 = stats["real"]["acf"][0]
    n1_hi = stats["nulls"]["N1_permute"]["acf"]["hi"][0]
    n2_hi = stats["nulls"]["N2_trend"]["acf"]["hi"][0]
    assert acf1 > n1_hi and acf1 > n2_hi          # planted order beats both nulls
    assert stats["real"]["excite_ratio"] > 1.5
    assert stats["markov"]["p_order1_vs_0"] < 1e-6
    assert "eps=0.05" in stats["label_noise"]


def test_signature_binary_gate_negative_on_iid():
    rng = np.random.default_rng(1)
    seqs = [(rng.random(90) < 0.12).astype(np.int8) for _ in range(120)]
    stats = sig.measure(seqs, "binary", seed=0, n_null=60, n_boot=60)
    acf1 = stats["real"]["acf"][0]
    assert stats["nulls"]["N1_permute"]["acf"]["lo"][0] <= acf1 \
        <= stats["nulls"]["N1_permute"]["acf"]["hi"][0]  # iid sits inside its null


def test_signature_categorical_sticky_markov():
    rng = np.random.default_rng(2)
    P = np.array([[0.95, 0.05], [0.05, 0.95]])
    seqs = []
    for _ in range(80):
        s = np.zeros(120, dtype=np.int8)
        for i in range(1, 120):
            s[i] = rng.choice(2, p=P[s[i - 1]])
        seqs.append(s)
    stats = sig.measure(seqs, "categorical", seed=0, n_null=40, n_boot=40)
    assert stats["real"]["acf"][0] > stats["nulls"]["N1_permute"]["acf"]["hi"][0]
    assert stats["real"]["dwell_mean"] > 5      # sticky: mean dwell ~20 >> iid ~2
    assert stats["n_symbols"] == 2


def test_nulls_preserve_marginals():
    seqs, rng = _selfexciting(n=40)
    perm = sig.null_permute(seqs, rng)
    assert all(int(a.sum()) == int(b.sum()) for a, b in zip(seqs, perm))
    prof = sig.position_profile(seqs, 10)
    trend = sig.null_trend(seqs, rng, "binary", 10)
    assert abs(sig.base_rate(trend) - sig.base_rate(seqs)) < 0.05
    assert np.corrcoef(prof, sig.position_profile(trend, 10))[0, 1] > 0.5


def test_mirror_markov_and_ar1_recovery():
    rng = np.random.default_rng(3)
    P = np.array([[0.9, 0.1], [0.3, 0.7]])
    seqs = []
    for _ in range(60):
        s = np.zeros(200, dtype=np.int8)
        for i in range(1, 200):
            s[i] = rng.choice(2, p=P[s[i - 1]])
        seqs.append(s)
    fit = mirrors.fit_markov(seqs)
    assert abs(fit["P"][0][0] - 0.9) < 0.03 and abs(fit["P"][1][1] - 0.7) < 0.04

    ar = mirrors.gen_ar1({"process": "ar1", "mu": 1.0, "rho": 0.8, "sigma": 0.5},
                         [300] * 40, rng)
    fit2 = mirrors.fit_ar1(ar)
    assert abs(fit2["rho"] - 0.8) < 0.05 and abs(fit2["mu"] - 1.0) < 0.15


def test_mirror_roundtrip_validation():
    seqs, rng = _selfexciting(n=150)
    train, ev = seqs[:100], seqs[100:]
    fit = mirrors.fit_logistic_ar(train, K=2)
    syn = mirrors.gen_logistic_ar(fit, [s.size for s in ev], rng)
    v = mirrors.validate_mirror(ev, syn, "binary")
    assert v["abs_err"]["acf_lag1_5"] < 0.08
    assert v["abs_err"]["excite_ratio"] < 1.0


def test_directed_transition_asymmetry():
    rng = np.random.default_rng(4)
    # planted convention: symbol 1 is always followed by symbol 2
    seqs = []
    for _ in range(60):
        s = rng.choice([0, 0, 0, 1], size=100).astype(np.int8)
        s[1:][s[:-1] == 1] = 2
        seqs.append(s)
    d = sig.directed_transition(seqs, 1, 2)
    assert d["fwd_rate"] > 0.9 and d["asym"] > 0.3
    perm = sig.null_permute(seqs, rng)
    dp = sig.directed_transition(perm, 1, 2)
    assert abs(dp["asym"]) < 0.15                      # permutation kills direction
    stats = sig.measure(seqs, "categorical", seed=0, n_null=40, n_boot=40, pair=(1, 2))
    assert stats["real"]["asym"] > stats["nulls"]["N1_permute"]["asym"]["hi"]


def test_perturb_categorical_and_ar1_trend():
    rng = np.random.default_rng(5)
    seqs = [rng.choice([0, 1, 2], size=80).astype(np.int8) for _ in range(20)]
    pert = sig.perturb_categorical(seqs, 0.1, rng)
    frac = np.mean([float((a != b).mean()) for a, b in zip(seqs, pert)])
    assert 0.02 < frac < 0.12                          # ~eps*(1-1/k)
    ar = mirrors.gen_ar1({"process": "ar1", "mu": 0.5, "beta_position": 1.0,
                          "rho": 0.6, "sigma": 0.3}, [200] * 40, rng)
    fit = mirrors.fit_ar1(ar, position=True)
    assert abs(fit["beta_position"] - 1.0) < 0.15 and abs(fit["rho"] - 0.6) < 0.08


def test_mirror_hier_ar1_plateau_recovery():
    rng = np.random.default_rng(6)
    true = {"process": "hier_ar1", "mu": 1.0, "beta_position": 0.3, "rho": 0.3,
            "sigma": 0.5, "levels": (0.8 * rng.standard_normal(200)).tolist()}
    seqs = mirrors.gen_hier_ar1(true, [120] * 200, rng)
    fit = mirrors.fit_hier_ar1(seqs, position=True)
    assert abs(fit["rho"] - 0.3) < 0.06
    assert abs(np.std(fit["levels"]) - 0.8) < 0.15
    # the signature the extension exists for: a pooled-ACF plateau at long lags
    # that a plain AR(1) with the same rho cannot hold up
    syn = mirrors.gen_hier_ar1(fit, [120] * 200, rng)
    acf_h = sig.acf(syn, maxlag=8)
    flat = mirrors.gen_ar1({"process": "ar1", "mu": 1.0, "rho": fit["rho"],
                            "sigma": fit["sigma"]}, [120] * 200, rng)
    acf_f = sig.acf(flat, maxlag=8)
    assert acf_h[5] > 0.4                      # plateau ≈ level share of variance
    assert acf_f[5] < 0.1                      # plain AR(1): rho^6 ≈ 0
    assert abs(acf_h[5] - sig.acf(seqs, maxlag=8)[5]) < 0.1   # round-trip


def _hier_cat_truth():
    # Four doc types, one per dominant symbol (pooled marginal uniform, so
    # every long-lag pooled statistic comes from doc heterogeneity), with a
    # heavy-ish empirical dwell (mean ~3.5, recipe-instruction-like) so short
    # lags carry genuine run structure rather than parity artifacts.
    props = []
    for t in range(4):
        props += [list(np.roll([0.70, 0.10, 0.10, 0.10], t))] * 50
    return {"process": "hier_categorical", "n_symbols": 4, "alpha": 0.85,
            "jump_P": ((np.ones((4, 4)) - np.eye(4)) / 3).tolist(),
            "doc_props": props,
            "dwell": [[1, 1, 2, 2, 3, 4, 6, 9]] * 4}


def test_mirror_hier_categorical_plateau_recovery():
    rng = np.random.default_rng(11)
    true = _hier_cat_truth()
    seqs = mirrors.gen_hier_categorical(true, [120] * 200, rng)
    fit = mirrors.fit_hier_categorical(seqs, n_symbols=4)
    assert fit["alpha"] > 0.5                  # doc tilt recovered as dominant
    # round-trip under the uniform C3/C4 gate rule (±20% rel): the pooled
    # self-match ACF(4) plateau AND pooled MI(2) both survive a fit->generate
    # cycle (the self-consistency deconvolution of doc propensities is what
    # makes this hold — raw doc marginals flatten every round).
    syn = mirrors.gen_hier_categorical(fit, [120] * 200, rng)
    acf_r = sig.selfmatch_acf(seqs, maxlag=8)
    acf_s = sig.selfmatch_acf(syn, maxlag=8)
    assert acf_r[3] > 0.1                      # the plateau exists in the data
    assert abs(acf_s[3] - acf_r[3]) < 0.2 * acf_r[3]
    mi_r = sig.mi_vs_lag(seqs, 8, 4)
    mi_s = sig.mi_vs_lag(syn, 8, 4)
    assert abs(mi_s[1] - mi_r[1]) < max(0.2 * mi_r[1], 0.003)


def test_mirror_semi_markov_cannot_hold_categorical_plateau():
    # The C3 failure mode the extension exists for: a single global dwell+jump
    # process fit to doc-heterogeneous data understates the long-lag pooled
    # statistics — how proof-operation (MI(2) halved) and recipe-instruction
    # (ACF(4) −32%) died. In this toy the flat fit misses ACF(4) by ~-25%,
    # ACF(6) by ~-70%, and MI(4) by ~-40%.
    rng = np.random.default_rng(12)
    seqs = mirrors.gen_hier_categorical(_hier_cat_truth(), [120] * 200, rng)
    flat = mirrors.fit_semi_markov(seqs, n_symbols=4)
    syn_flat = mirrors.gen_semi_markov(flat, [120] * 200, rng)
    acf_r = sig.selfmatch_acf(seqs, maxlag=8)
    acf_f = sig.selfmatch_acf(syn_flat, maxlag=8)
    assert acf_f[3] < acf_r[3] - 0.2 * acf_r[3]   # fails the ±20% rel gate
    assert acf_f[5] < acf_r[5] - 0.2 * acf_r[5]   # plateau collapses by lag 6
    mi_r = sig.mi_vs_lag(seqs, 8, 4)
    mi_f = sig.mi_vs_lag(syn_flat, 8, 4)
    assert mi_f[3] < mi_r[3] - max(0.2 * mi_r[3], 0.003)


def test_mirror_periodic_hawkes_recovery():
    rng = np.random.default_rng(7)
    true = {"process": "periodic_hawkes", "period": 8, "K": 2, "intercept": -2.8,
            "b_cos": 1.2, "b_sin": 0.0, "kernel_w": [1.6, 0.7]}
    seqs = mirrors.gen_periodic_hawkes(true, [160] * 120, rng)
    fit = mirrors.fit_periodic_hawkes(seqs, K=2, max_period=32)
    assert fit["period"] == 8
    assert fit["kernel_w"][0] > 0.8            # self-excitation recovered
    assert fit["b_cos"] > 0.6                  # rhythm recovered
    # round-trip: the hybrid holds BOTH the spectral peak and the burstiness,
    # which neither pure-menu parent can do simultaneously
    syn = mirrors.gen_periodic_hawkes(fit, [s.size for s in seqs], rng)
    assert sig.spec_peak(syn) > 2.0
    assert abs(sig.fano(syn) - sig.fano(seqs)) < 0.25
    pure = mirrors.fit_periodic_rate(seqs, max_period=32)
    syn_pure = mirrors.gen_periodic_rate(pure, [s.size for s in seqs], rng)
    assert sig.fano(syn_pure) < sig.fano(seqs) - 0.25   # the C2 failure mode


def test_labeler_parse_and_agreement():
    assert labeler._parse_labels("[0, 1, 0]", 3, 1).tolist() == [0, 1, 0]
    assert labeler._parse_labels("Here you go: [0,1,1]", 3, 1).tolist() == [0, 1, 1]
    assert labeler._parse_labels("[0, 1]", 3, 1) is None          # wrong length
    assert labeler._parse_labels("[0, 2, 0]", 3, 1) is None       # out of range
    assert labeler._parse_labels("no array here", 3, 1) is None
    a = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    assert labeler.cohen_kappa(a, a, 2) == pytest.approx(1.0)
    assert labeler.noise_floor_from_disagreement(0.0) == 0.0
    assert 0.05 < labeler.noise_floor_from_disagreement(0.1) < 0.06
    xc = labeler.crosscheck_binary([a], [a])
    assert xc["f1"] == pytest.approx(1.0)


def test_meter_hard_cap(tmp_path):
    m = Meter(tmp_path / "spend.json", cap_usd=0.001)
    m.check()                                        # under cap: fine
    m.add("claude-haiku-4-5-20251001", 500_000, 100_000)   # ≈ $1.0
    with pytest.raises(SpendCapExceeded):
        m.check()
    m2 = Meter(tmp_path / "spend.json", cap_usd=0.001)     # persisted across restart
    assert m2.spent == pytest.approx(m.spent)


def test_corpus_splitter():
    text = ("Dr. Smith went home. It was late! Was it? \"Yes.\" She said so.\n"
            "A new paragraph starts here and it keeps going.")
    sents = corpus.split_sentences(text)
    assert "Dr. Smith went home." in sents[0]
    assert len(sents) >= 4
    assert all(len(s) >= 8 for s in sents)
