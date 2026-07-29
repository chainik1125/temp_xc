from __future__ import annotations

import numpy as np

from experiments.correlation_audit.corrective import (
    aggregate_gamma,
    article_sufficient_statistics,
    center_blocks,
    contiguous_article_runs,
    corrected_decay_fits,
    hermitian_lag_window_spectrum,
    matrix_norm_curves,
    reconstruct_article_prefixes,
)


def test_reconstruction_joins_only_contiguous_article_blocks() -> None:
    blocks = np.arange(4 * 3, dtype=np.float32).reshape(4, 3, 1)
    articles, ids = reconstruct_article_prefixes(blocks, np.asarray([7, 7, 9, 9]))
    assert ids == [7, 9]
    assert [len(article) for article in articles] == [6, 6]
    assert articles[0][0, 0] == 0
    assert articles[0][-1, 0] == 5


def test_noncontiguous_article_id_is_rejected() -> None:
    try:
        contiguous_article_runs(np.asarray([1, 2, 1]))
    except ValueError as error:
        assert "noncontiguous" in str(error)
    else:
        raise AssertionError("expected noncontiguous article ids to fail")


def test_all_centering_modes_have_their_declared_zero_mean() -> None:
    rng = np.random.default_rng(1)
    blocks = rng.normal(size=(5, 4, 3)).astype(np.float32)
    assert np.allclose(center_blocks(blocks, "global").mean(axis=(0, 1)), 0, atol=1e-6)
    assert np.allclose(center_blocks(blocks, "position").mean(axis=0), 0, atol=1e-6)
    assert np.allclose(center_blocks(blocks, "sequence").mean(axis=1), 0, atol=1e-6)


def test_article_sufficient_statistics_include_cross_block_pairs() -> None:
    # Two original length-3 blocks become one length-6 article. Lag one has five
    # pairs; a within-block-only computation would have only four.
    blocks = np.arange(1, 7, dtype=np.float32).reshape(2, 3, 1)
    articles, _ = reconstruct_article_prefixes(blocks, np.asarray([0, 0]))
    cross, counts = article_sufficient_statistics(articles, max_lag=2)
    assert counts.tolist() == [[6, 5, 4]]
    assert np.isclose(cross[0, 1, 0, 0], sum(a * b for a, b in zip(range(1, 6), range(2, 7))))
    gamma = aggregate_gamma(cross, counts)
    assert np.isclose(gamma[1, 0, 0], cross[0, 1, 0, 0] / 5)


def test_signed_gamma_and_matrix_norms_on_vector_process() -> None:
    article = np.asarray([[1.0, 2.0], [2.0, -1.0], [0.5, 3.0]], dtype=np.float32)
    cross, counts = article_sufficient_statistics([article], max_lag=2)
    gamma = aggregate_gamma(cross, counts)
    expected_plus = article[:-1].T @ article[1:] / 2
    assert np.allclose(gamma[1], expected_plus)
    assert np.allclose(gamma[1].T, article[1:].T @ article[:-1] / 2)
    norms = matrix_norm_curves(gamma)
    assert set(norms) == {"frobenius", "operator", "nuclear"}
    assert np.all(norms["nuclear"] >= norms["operator"])


def test_lag_window_spectrum_is_hermitian() -> None:
    rng = np.random.default_rng(3)
    articles = [rng.normal(size=(40, 3)).astype(np.float32) for _ in range(4)]
    cross, counts = article_sufficient_statistics(articles, max_lag=6)
    frequencies, spectrum, eigenvalues = hermitian_lag_window_spectrum(
        cross, counts, n_frequencies=9
    )
    assert frequencies.shape == (9,)
    assert spectrum.shape == (9, 3, 3)
    assert eigenvalues.shape == (9, 3)
    assert np.allclose(spectrum, spectrum.conj().transpose(0, 2, 1), atol=1e-10)
    # Bartlett smoothing of common-denominator sufficient statistics is PSD up
    # to ordinary floating-point tolerance.
    assert eigenvalues.min() > -1e-8


def test_corrected_aicc_counts_residual_variance() -> None:
    lags = np.arange(2, 30)
    curve = 0.8 * lags ** -0.7 + 0.05
    # JSON round-tripping turns arrays into lists; report generation must accept
    # those persisted values as well as native arrays.
    fits = corrected_decay_fits(lags.tolist(), curve.tolist())
    assert fits
    for fit in fits:
        assert fit["aicc_parameter_count"] == len(fit["params"]) + 1
        assert "legacy_aicc" in fit
