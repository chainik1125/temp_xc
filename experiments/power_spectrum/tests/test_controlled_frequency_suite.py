from __future__ import annotations

import math

import numpy as np

from experiments.power_spectrum.code import run_controlled_frequency_suite as suite
from experiments.power_spectrum.code.controlled_tasks import (
    generate_factorial_hmm_splits,
    generate_shamir_splits,
)
from temp_bench.archs.batchtopk_sae import BatchTopKSAE
from temp_bench.archs.spectral_txc import SpectralTXCBatchTopK


def test_frozen_plan_is_paired_and_below_incremental_and_overall_caps() -> None:
    config = suite.load_config(
        suite.POWER_ROOT / "configs" / "controlled_frequency_suite.json"
    )
    plan = suite.build_plan(config)
    assert plan["evaluation_cells"] == 70
    assert plan["unique_training_runs"] == 62
    assert plan["total_optimizer_steps"] == 324_000
    assert plan["within_cost_plan"]
    assert plan["within_time_plan"]
    assert plan["worst_case_overall_usd"] < plan["overall_cap_usd"]


def test_token_sae_training_is_reused_across_windows_and_support_is_matched() -> None:
    config = suite.load_config(
        suite.POWER_ROOT / "configs" / "controlled_frequency_suite.json"
    )
    cells = suite.enumerate_cells(config)
    sae_h1_seed1 = [
        cell
        for cell in cells
        if cell["model"] == "sae"
        and cell["group"] == "shamir_h1"
        and cell["seed"] == 1
    ]
    assert len(sae_h1_seed1) == 3
    assert len({cell["training_id"] for cell in sae_h1_seed1}) == 1
    assert {cell["k_pos"] for cell in sae_h1_seed1} == {1}

    txc_h2 = next(
        cell
        for cell in cells
        if cell["model"] == "txc" and cell["task"] == "shamir_h2_w6"
    )
    spectral_h2 = next(
        cell
        for cell in cells
        if cell["model"] == "spectral_v1" and cell["task"] == "shamir_h2_w6"
    )
    assert txc_h2["k_pos"] == 6
    assert spectral_h2["k_pos"] == 1


def test_shamir_evaluator_has_exact_symbolic_oracle_on_fresh_episodes() -> None:
    splits = generate_shamir_splits(
        h=1,
        q=5,
        d=8,
        sigma=0.0,
        seq_len=4,
        split_sizes={"train": 128, "probe": 128, "eval": 128},
        split_seeds={"train": 11, "probe": 22, "eval": 33},
        alphabet_seed=0,
    )
    model = BatchTopKSAE(d_in=8, d_sae=16, k_pos=1)
    model.eval()
    metrics = suite.evaluate_shamir(
        model,
        {
            "window": 2,
            "q": 5,
            "h": 1,
        },
        splits["probe"],
        splits["eval"],
        seed=1,
    )
    assert metrics["symbolic_interpolation_oracle"] == 1.0
    assert metrics["identifiable"] is True
    assert math.isfinite(metrics["secret_recovery"])
    assert metrics["l0_per_window"] > 0


def test_hmm_evaluator_reports_band_localization_and_usage() -> None:
    splits = generate_factorial_hmm_splits(
        lambdas=(0.85, 0.4, -0.4, -0.85),
        d=8,
        sigma=0.2,
        seq_len=12,
        split_sizes={"train": 128, "probe": 128, "eval": 128},
        split_seeds={"train": 41, "probe": 42, "eval": 43},
        emission_seed=0,
    )
    model = SpectralTXCBatchTopK(
        d_in=8,
        d_sae=16,
        T=8,
        k_pos=1,
        bands="multiband",
    )
    model.eval()
    metrics = suite.evaluate_hmm(
        model,
        {
            "window": 8,
            "d_in": 8,
        },
        splits["probe"],
        splits["eval"],
        seed=1,
    )
    assert len(metrics["latent_r2_per_source"]) == 4
    assert len(metrics["band_latent_r2_per_source"]) == 4
    assert all(len(row) == model.n_bands for row in metrics["band_latent_r2_per_source"])
    assert 0.0 <= metrics["band_localization_accuracy"] <= 1.0
    assert math.isclose(
        sum(metrics["spectral_usage"]["activation_energy_share"]),
        1.0,
        rel_tol=1e-6,
    )


def test_secret_selectivity_is_cross_split_and_rejects_one_off_fires() -> None:
    probe_secret = np.repeat(np.arange(4), 40)
    eval_secret = np.repeat(np.arange(4), 40)
    probe = np.zeros((160, 5), dtype=np.float32)
    evaluation = np.zeros((160, 5), dtype=np.float32)
    probe[probe_secret == 2, 0] = 1.0
    evaluation[eval_secret == 2, 0] = 1.0
    # This feature would look perfectly selective without a minimum-fire gate.
    probe[0, 1] = 100.0
    evaluation[0, 1] = 100.0

    result = suite._secret_selectivity(
        probe,
        probe_secret,
        evaluation,
        eval_secret,
        q=4,
    )
    assert result["selective_feature_count"] == 1
    assert result["top_q_selectivity"] == 1.0
    assert result["top_q_secret_coverage"] == 0.25
