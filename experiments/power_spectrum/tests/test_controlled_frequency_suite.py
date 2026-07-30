from __future__ import annotations

import math

import numpy as np

from experiments.power_spectrum.code import run_controlled_frequency_suite as suite
from experiments.power_spectrum.code.controlled_tasks import (
    generate_factorial_hmm_splits,
    generate_narrowband_source_splits,
    generate_shamir_splits,
)
from temp_bench.archs.batchtopk_sae import BatchTopKSAE
from temp_bench.archs.spectral_txc import SpectralTXCBatchTopK


def test_frozen_plan_is_paired_and_below_incremental_and_overall_caps() -> None:
    config = suite.load_config(suite.POWER_ROOT / "configs" / "controlled_frequency_suite.json")
    plan = suite.build_plan(config)
    assert plan["evaluation_cells"] == 58
    assert plan["unique_training_runs"] == 54
    assert plan["total_optimizer_steps"] == 276_000
    assert plan["within_cost_plan"]
    assert plan["within_time_plan"]
    assert plan["worst_case_overall_usd"] < plan["overall_cap_usd"]


def test_token_sae_training_is_reused_across_windows_and_support_is_matched() -> None:
    config = suite.load_config(suite.POWER_ROOT / "configs" / "controlled_frequency_suite.json")
    cells = suite.enumerate_cells(config)
    sae_h1_seed1 = [
        cell
        for cell in cells
        if cell["model"] == "sae" and cell["group"] == "shamir_h1" and cell["seed"] == 1
    ]
    assert len(sae_h1_seed1) == 3
    assert len({cell["training_id"] for cell in sae_h1_seed1}) == 1
    assert {cell["k_pos"] for cell in sae_h1_seed1} == {1}

    spectral_h2 = next(
        cell for cell in cells if cell["model"] == "spectral_v1" and cell["task"] == "shamir_h2_w6"
    )
    assert spectral_h2["k_pos"] == 1
    assert not any(
        cell["family"] == "shamir"
        and cell["group"] == "shamir_h2"
        and cell["model"] in {"sae", "txc"}
        for cell in cells
    )


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
    assert math.isfinite(metrics["direct_latent_r2"])


def test_advantage_plan_matches_support_parameters_and_overall_budget() -> None:
    config = suite.load_config(suite.POWER_ROOT / "configs" / "spectral_advantage.json")
    cells = suite.enumerate_cells(config)
    assert len(cells) == 18
    assert suite.build_plan(config)["within_cost_plan"]
    assert suite.build_plan(config)["within_time_plan"]
    assert suite.build_plan(config)["worst_case_overall_usd"] < config["overall_spend"]["cap_usd"]

    for task in config["tasks"]:
        task_cells = [cell for cell in cells if cell["task"] == task["name"] and cell["seed"] == 1]
        counts = {}
        for cell in task_cells:
            cls = suite._import_model(cell["model_spec"])
            model = cls(**suite._model_kwargs(cell))
            counts[cell["model"]] = sum(parameter.numel() for parameter in model.parameters())
            if cell["model"] == "txc":
                assert cell["k_pos"] == cell["window"]
            else:
                assert cell["k_pos"] == 1
        reference = counts["txc"]
        assert abs(counts["spectral_global"] - reference) / reference < 0.002
        assert abs(counts["spectral_adaptive"] - reference) / reference < 0.002

    t16 = next(
        cell
        for cell in cells
        if cell["task"] == "narrowband_offgrid_t16_matched_params"
        and cell["model"] == "spectral_adaptive"
    )
    assert t16["d_sae"] == 255
    assert t16["k_pos"] == 1
    txc_t16 = next(
        cell
        for cell in cells
        if cell["task"] == "narrowband_offgrid_t16_matched_params" and cell["model"] == "txc"
    )
    assert txc_t16["d_sae"] == 64
    assert txc_t16["k_pos"] == 16


def test_task_specific_model_hparams_are_part_of_training_identity() -> None:
    config = suite.load_config(suite.POWER_ROOT / "configs" / "spectral_advantage.json")
    task = {**config["tasks"][0], "hparams_by_model": {"spectral_adaptive": {"x": 3}}}
    model = next(model for model in config["models"] if model["name"] == "spectral_adaptive")
    identity = suite._training_identity(config, task, model, 1, n_steps=4)
    assert identity["hparams"]["x"] == 3
    assert model["hparams"].get("x") is None


def test_routed_screen_is_parameter_matched_and_frequency_order_free() -> None:
    config = suite.load_config(
        suite.POWER_ROOT / "configs" / "spectral_matryoshka_routed.json"
    )
    cells = suite.enumerate_cells(config)
    plan = suite.build_plan(config)
    assert len(cells) == 18
    assert plan["within_cost_plan"]
    assert plan["within_time_plan"]
    assert plan["worst_case_overall_usd"] < config["overall_spend"]["cap_usd"]

    sparse_tasks = [task for task in config["tasks"] if task["family"] == "narrowband_sources"]
    assert [
        [task["frequencies"].count(value) for value in (0.125, 0.25, 0.375)]
        for task in sparse_tasks
    ] == [[8, 8, 8], [3, 6, 15]]
    assert all(task["active_sources_per_episode"] == 4 for task in sparse_tasks)

    for task in config["tasks"]:
        task_cells = [cell for cell in cells if cell["task"] == task["name"]]
        counts = {}
        for cell in task_cells:
            cls = suite._import_model(cell["model_spec"])
            model = cls(**suite._model_kwargs(cell))
            counts[cell["model"]] = sum(parameter.numel() for parameter in model.parameters())
            assert cell["k_pos"] == (cell["window"] if cell["model"] == "txc" else 1)
        reference = counts["txc"]
        for model_name, count in counts.items():
            if model_name != "txc":
                assert abs(count - reference) / reference < 0.006

    routed = next(model for model in config["models"] if model["name"] == "spectral_fourier_routed")
    assert routed["hparams"]["adaptive_frequency_routing_strength"] == 1.0
    assert routed["hparams"]["frequency_matryoshka_alpha"] == 0.0


def test_narrowband_evaluator_reports_direct_recovery_and_learned_weights() -> None:
    from experiments.power_spectrum.code.spectral_txc_v2 import SpectralTXCV2

    splits = generate_narrowband_source_splits(
        frequencies=(0.08, 0.21, 0.34, 0.46),
        d=10,
        sigma=0.2,
        seq_len=16,
        split_sizes={"train": 128, "probe": 128, "eval": 128},
        split_seeds={"train": 51, "probe": 52, "eval": 53},
        emission_seed=0,
        min_frequency_separation=0.06,
    )
    model = SpectralTXCV2(
        d_in=10,
        d_sae=32,
        T=8,
        k_pos=1,
        bands="multiband",
        selection_mode="global",
        adaptive_frequency_alpha=0.1,
        adaptive_frequency_adversary_alpha=0.1,
    )
    model.eval()
    metrics = suite.evaluate_narrowband(
        model,
        {"window": 8, "d_in": 10},
        splits["probe"],
        splits["eval"],
        seed=1,
    )
    assert len(metrics["direct_latent_r2_per_source"]) == 4
    assert len(metrics["band_latent_r2_per_source"]) == 4
    assert math.isfinite(metrics["direct_latent_r2"])
    assert math.isclose(
        sum(metrics["spectral_usage"]["learned_frequency_weight"]),
        1.0,
        rel_tol=1e-6,
    )


def test_sparse_narrowband_evaluator_decomposes_active_and_inactive_error() -> None:
    from experiments.power_spectrum.code.spectral_txc_v2 import SpectralTXCV2

    splits = generate_narrowband_source_splits(
        frequencies=(0.125, 0.125, 0.25, 0.25, 0.375, 0.375),
        d=12,
        sigma=0.1,
        seq_len=8,
        split_sizes={"train": 64, "probe": 64, "eval": 64},
        split_seeds={"train": 61, "probe": 62, "eval": 63},
        emission_seed=0,
        active_sources_per_episode=2,
        allow_repeated_frequencies=True,
    )
    model = SpectralTXCV2(
        d_in=12,
        d_sae=16,
        T=8,
        k_pos=1,
        temporal_basis="fourier",
        selection_mode="global",
    )
    model.eval()
    metrics = suite.evaluate_narrowband(
        model,
        {"window": 8, "d_in": 12},
        splits["probe"],
        splits["eval"],
        seed=1,
    )
    assert math.isfinite(metrics["direct_latent_r2_active"])
    assert metrics["inactive_latent_prediction_energy"] >= 0
    assert metrics["inactive_to_active_energy_ratio"] >= 0
    assert len(metrics["direct_latent_r2_active_per_source"]) == 6


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


def test_seed_compatible_results_copies_only_current_successes(tmp_path) -> None:
    config = suite.load_config(suite.POWER_ROOT / "configs" / "controlled_frequency_suite.json")
    expected = suite.enumerate_cells(config)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    keep = {**expected[0], "status": "ok", "smoke": False}
    skip_failed = {**expected[1], "status": "failed", "smoke": False}
    skip_unexpected = {
        **expected[2],
        "cell_id": "not-in-reduced-plan",
        "status": "ok",
        "smoke": False,
    }
    for row in (keep, skip_failed, skip_unexpected):
        suite._append_jsonl(source / "results.jsonl", row)

    assert (
        suite.seed_compatible_results(
            config,
            source,
            destination,
            smoke=False,
        )
        == 1
    )
    copied = suite.latest_results(destination / "results.jsonl")
    assert set(copied) == {keep["cell_id"]}
