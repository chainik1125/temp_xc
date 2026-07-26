"""Focused contracts for the event-conditioned SWR audit."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from experiments.swr_audit.aggregate import audited_summary
from experiments.swr_audit.dictionary import encode_topk_pool_batch, encode_txc_batch
from experiments.swr_audit.matched_filter import (
    fit_matched_filter,
    phase_scramble,
    task_spectrum,
)
from experiments.swr_audit.report import markdown_table
from experiments.swr_audit.report_swr import markdown_table as swr_markdown_table
from experiments.swr_audit.report_dictionary import markdown_table as dictionary_markdown_table
from experiments.swr_audit.run import (
    FoldPreprocessor,
    MeanPoolBottleneck,
    c7_groups,
    fit_bottleneck,
    score_model,
    trailing_window,
)


def test_c7_group_and_trailing_window_contracts():
    keys = np.array(["q0|0|1", "q0|1|2", "q1|0|1"], dtype=object)
    assert c7_groups(keys).tolist() == ["q0", "q0", "q1"]
    x = np.arange(3 * 6 * 2).reshape(3, 6, 2)
    assert np.array_equal(trailing_window(x, 2), x[:, 4:, :])


def test_c7_fold_reports_physical_artifact_offset():
    rng = np.random.default_rng(11)
    n_groups = 20
    rows_per_group = 8
    groups = np.repeat([f"q{i}" for i in range(n_groups)], rows_per_group)
    y = np.tile([0, 1, 0, 0, 1, 0, 0, 1], n_groups)
    x = rng.normal(size=(len(y), 6, 5)).astype(np.float32)
    x[y == 1, -2, 0] += 2.0
    train_idx = np.flatnonzero(np.isin(groups, [f"q{i}" for i in range(15)]))
    test_idx = np.flatnonzero(~np.isin(groups, [f"q{i}" for i in range(15)]))
    from experiments.swr_audit.run import run_c7_fold

    row = run_c7_fold(
        x,
        y,
        groups,
        train_idx,
        test_idx,
        fold=0,
        window=5,
        normalization="raw",
        pca_dim=3,
        rank=2,
        seed=3,
        device="cpu",
        epochs=8,
        batch_size=64,
        learning_rate=1e-2,
        pca_sample_tokens=1_000,
        artifact_offsets=(-13, -12, -11, -10, -9, -8),
    )
    assert row["window_offsets"] == [-12, -11, -10, -9, -8]
    assert row["best_offset_relative"] in row["window_offsets"]


def test_preprocessor_is_fit_on_train_fold_only():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(64, 3, 10)).astype(np.float32)
    test = (rng.normal(size=(16, 3, 10)) + 20).astype(np.float32)
    pre = FoldPreprocessor("raw", pca_dim=5, seed=0).fit(train)
    z_train = pre.transform(train)
    z_test = pre.transform(test)
    assert z_train.shape == (64, 3, 5)
    assert np.abs(z_train.mean()) < 1e-5
    assert np.abs(z_test.mean()) > 1.0


def test_mean_pool_bottleneck_is_exactly_permutation_invariant():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(20, 6, 5)).astype(np.float32)
    model = MeanPoolBottleneck(d_in=5, hidden=18).eval()
    forward = model(torch.from_numpy(x)).detach().numpy()
    shuffled = model(torch.from_numpy(x[:, [3, 0, 5, 1, 4, 2]])).detach().numpy()
    reversed_ = model(torch.from_numpy(x[:, ::-1].copy())).detach().numpy()
    np.testing.assert_allclose(forward, shuffled, rtol=0, atol=1e-7)
    np.testing.assert_allclose(forward, reversed_, rtol=0, atol=1e-7)


def test_ordered_bottleneck_beats_exact_invariant_null_on_transition_signal():
    rng = np.random.default_rng(4)
    n = 700
    y = rng.integers(0, 2, size=n)
    sign = 2 * y - 1
    phase = rng.normal(size=n)
    x = rng.normal(scale=0.25, size=(n, 2, 4)).astype(np.float32)
    x[:, 0, 0] += phase - 0.5 * sign
    x[:, 1, 0] += phase + 0.5 * sign
    fit, val, test = np.arange(450), np.arange(450, 575), np.arange(575, n)
    common = dict(rank=4, device="cpu", epochs=60, batch_size=128, learning_rate=1e-2)
    ordered = fit_bottleneck(x[fit], y[fit], x[val], y[val], seed=1, **common)
    invariant = fit_bottleneck(
        x[fit],
        y[fit],
        x[val],
        y[val],
        seed=2,
        model_kind="mean_pool",
        hidden=8,
        **common,
    )
    ordered_ap = score_model(ordered, x[test], y[test], device="cpu", batch_size=128)[
        "pr_auc"
    ]
    invariant_ap = score_model(
        invariant, x[test], y[test], device="cpu", batch_size=128
    )["pr_auc"]
    invariant_reversed_ap = score_model(
        invariant, x[test, ::-1].copy(), y[test], device="cpu", batch_size=128
    )["pr_auc"]
    assert ordered_ap > 0.9
    assert ordered_ap - invariant_ap > 0.2
    assert abs(invariant_ap - invariant_reversed_ap) < 0.1


def test_aggregate_recomputes_conservative_swr_from_components():
    row = {
        "fold": 2,
        "window": 6,
        "normalization": "raw",
        "pca_dim_actual": 32,
        "rank": 20,
        "ordered": {"pr_auc": 0.30624},
        "mean_pool_param_matched": {"pr_auc": 0.28912},
        "mean_pool_same_rank": {"pr_auc": 0.31105},
        "best_token": {"pr_auc": 0.25857},
        "order_gap_pr_auc": 0.04860,
        "swr_pr_auc": 0.01712,
    }
    result = audited_summary([row])["summaries"][0]
    assert np.isclose(result["swr_pr_auc_conservative"]["mean"], -0.00481)
    assert result["swr_pr_auc_conservative"]["grouped_fold_t_95"] is None


def test_dictionary_encoders_expose_expected_order_contract():
    x = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.5, -1.0], [1.0, 0.5]]]
    )
    reversed_x = x.flip(1)
    topk_state = {
        "W_enc": torch.eye(2),
        "b_enc": torch.zeros(2),
        "b_dec": torch.zeros(2),
    }
    topk = encode_topk_pool_batch(x, topk_state, k_pos=1)
    topk_reversed = encode_topk_pool_batch(reversed_x, topk_state, k_pos=1)
    np.testing.assert_array_equal(topk.toarray(), topk_reversed.toarray())

    txc_state = {
        "W_enc": torch.tensor([[[1.0], [0.0]], [[-1.0], [0.0]]]),
        "b_enc": torch.zeros(1),
    }
    txc = encode_txc_batch(x, txc_state, k_pos=1)
    txc_reversed = encode_txc_batch(reversed_x, txc_state, k_pos=1)
    assert not np.array_equal(txc.toarray(), txc_reversed.toarray())


def test_matched_filter_recovers_an_ordered_transition():
    rng = np.random.default_rng(8)
    n = 800
    y = rng.integers(0, 2, size=n)
    sign = 2 * y - 1
    level = rng.normal(size=n)
    x = rng.normal(scale=0.35, size=(n, 2, 3)).astype(np.float32)
    x[:, 0, 0] += level - sign
    x[:, 1, 0] += level + sign
    fit, val, test = np.arange(500), np.arange(500, 650), np.arange(650, n)
    ordered = fit_matched_filter(
        x[fit].reshape(len(fit), -1),
        y[fit],
        x[val].reshape(len(val), -1),
        y[val],
    )
    invariant = fit_matched_filter(
        x[fit].mean(axis=1), y[fit], x[val].mean(axis=1), y[val]
    )
    ordered_ap = average_precision_score(
        y[test], ordered.probabilities(x[test].reshape(len(test), -1))
    )
    invariant_ap = average_precision_score(
        y[test], invariant.probabilities(x[test].mean(axis=1))
    )
    assert ordered_ap > 0.95
    assert ordered_ap - invariant_ap > 0.25


def test_phase_scramble_preserves_per_channel_fft_magnitude():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(12, 6, 4)).astype(np.float32)
    scrambled = phase_scramble(x, seed=4)
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(scrambled, axis=1)),
        np.abs(np.fft.rfft(x, axis=1)),
        rtol=1e-5,
        atol=1e-5,
    )
    assert not np.allclose(x, scrambled)


def test_task_spectrum_is_nonnegative_and_normalized():
    rng = np.random.default_rng(10)
    y = np.tile([0, 1], 100)
    z = rng.normal(size=(len(y), 6, 5)).astype(np.float32)
    z[y == 1, -2:, 0] += 0.5
    result = task_spectrum(z, y)
    assert len(result["j_y_fraction"]) == 4
    assert min(result["j_y_fraction"]) >= 0
    assert np.isclose(sum(result["j_y_fraction"]), 1.0)


def test_matched_filter_markdown_table_exposes_gate():
    summary = {
        "window": 2,
        "window_offsets": [-9, -8],
        "ordered_pr_auc": {"mean": 0.3, "fold_values": [0.29, 0.31]},
        "invariant_mean_pr_auc": {"mean": 0.2, "fold_values": [0.19, 0.21]},
        "best_token_pr_auc": {"mean": 0.22, "fold_values": [0.21, 0.23]},
        "g_order_pr_auc": {
            "mean": 0.08,
            "cluster_bootstrap": {"lower_95": 0.04, "upper_95": 0.12},
        },
        "best_offsets": [-9, -8],
        "task_spectrum": {"mean_j_y_fraction": [0.4, 0.6]},
    }
    table = markdown_table({"summaries": [summary]})
    assert "| 2 | -9…-8 |" in table
    assert "+0.080" in table
    assert "[0.040, 0.120]" in table


def test_swr_markdown_table_exposes_conservative_gate():
    stats = {"mean": 0.2, "std_sample": 0.01, "n_folds": 5}
    payload = {
        "summaries": [
            {
                "window": 6,
                "normalization": "raw",
                "ordered_pr_auc": stats,
                "mean_pool_param_matched_pr_auc": stats,
                "mean_pool_same_rank_pr_auc": stats,
                "best_token_pr_auc": stats,
                "swr_pr_auc_conservative": {
                    "mean": -0.012,
                    "n_above_0_02": 1,
                    "n_folds": 5,
                    "grouped_fold_t_95": [-0.03, 0.006],
                },
            }
        ]
    }
    table = swr_markdown_table(payload)
    assert "| 6 | raw |" in table
    assert "-0.012" in table
    assert "[-0.030, +0.006]" in table
    assert "1/5" in table


def test_dictionary_markdown_table_exposes_fixed_probe_order_gap():
    payload = {
        "summaries": [
            {
                "arch": "txc_base",
                "n_features": 8,
                "ordered_mean": 0.28,
                "control_means": {"shuffle": 0.24, "reverse": 0.25, "circular": 0.23},
                "gap_means": {"shuffle": 0.04, "reverse": 0.03, "circular": 0.05},
            }
        ]
    }
    table = dictionary_markdown_table(payload)
    assert "| txc_base | 8 |" in table
    assert "+0.040" in table
