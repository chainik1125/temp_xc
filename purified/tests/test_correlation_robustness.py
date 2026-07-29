"""Numerical contracts for the E0 correlation robustness extension."""

from __future__ import annotations

import numpy as np

from experiments.correlation_audit import extract_legacy
from experiments.correlation_audit.robustness import (
    compare_removal_estimators,
    covariance_sequence,
    direct_psd_audit,
    fixed_directions,
    signed_direction_audit,
    stationarity_audit,
)


def _ar_documents(*, documents: int, positions: int, channels: int, rho: float, seed: int):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(documents, positions, channels))
    values = np.zeros_like(noise)
    values[:, 0] = noise[:, 0]
    innovation_scale = np.sqrt(1 - rho**2)
    for position in range(1, positions):
        values[:, position] = rho * values[:, position - 1] + innovation_scale * noise[:, position]
    return values.astype(np.float32)


def test_signed_covariance_preserves_ar_sign_and_scale():
    z = _ar_documents(documents=500, positions=80, channels=2, rho=0.7, seed=0)
    covariance, _ = covariance_sequence(z, max_lag=5)
    audit = signed_direction_audit(covariance, ["axis"], np.array([[1.0, 0.0]]))
    rho = np.asarray(audit["directions"]["axis"]["rho"])
    assert np.all(rho[1:] > 0)
    assert abs(rho[1] - 0.7) < 0.04
    assert abs(rho[3] - 0.7**3) < 0.04


def test_crossfit_persistent_removal_is_held_out_and_reduces_tail():
    rng = np.random.default_rng(1)
    z = _ar_documents(documents=120, positions=72, channels=6, rho=0.55, seed=2)
    # A document-level offset supplies a truly persistent rank-one sector.
    offset = rng.normal(size=(len(z), 1, 1)).astype(np.float32)
    z[:, :, :1] += 3.0 * offset
    audit = compare_removal_estimators(
        z, max_lag=20, persistent_rank=1, seed=3
    )
    no_tail = np.mean(audit["no_removal"]["curve"][-5:])
    crossfit_tail = np.mean(audit["crossfit_removal"]["curve"][-5:])
    assert len(audit["crossfit_removal"]["fold_curves"]) == 2
    assert crossfit_tail < 0.5 * no_tail


def test_direct_psd_reports_document_bootstrap_interval():
    z = _ar_documents(documents=40, positions=128, channels=2, rho=0.8, seed=4)
    result = direct_psd_audit(
        z,
        ["axis"],
        np.array([[1.0, 0.0]]),
        bootstrap=20,
        seed=5,
    )
    row = result["directions"]["axis"]
    assert len(row["psd"]) == len(result["frequencies"])
    assert row["beta_q025"] <= row["low_frequency_beta"] <= row["beta_q975"]
    assert row["psd"][1] > row["psd"][-1]


def test_position_stationarity_detects_mean_shift():
    z = _ar_documents(documents=80, positions=90, channels=3, rho=0.4, seed=6)
    z[:, 60:, 0] += 4.0
    _, directions = fixed_directions(3, n_pca=2, n_random=1, seed=7)
    result = stationarity_audit(z, directions, max_lag=8)
    assert result["standardized_max_mean_drift"] > 1.0
    assert len(result["bins"]) == 3


def test_legacy_token_builder_preserves_article_groups(monkeypatch):
    class Tokenizer:
        bos_token_id = 99
        eos_token_id = 98

        @staticmethod
        def encode(article):
            token = 1 if article == "first" else 2
            return [token] * 10

    monkeypatch.setattr(
        extract_legacy,
        "iter_wikitext_articles",
        lambda *_args, **_kwargs: iter(("first", "second")),
    )
    tokens, article_ids = extract_legacy.build_token_sequences(
        Tokenizer(),
        dataset_repo="ignored",
        dataset_revision="deadbeef",
        sequence_length=5,
        num_sequences=3,
        min_article_tokens=1,
    )
    assert article_ids.tolist() == [0, 0, 1]
    assert tokens[:, 0].tolist() == [99, 99, 99]
    assert tokens[0, 1:].unique().tolist() == [1]
    assert tokens[2, 1:].unique().tolist() == [2]
