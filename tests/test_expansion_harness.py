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
