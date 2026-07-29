from __future__ import annotations

from pathlib import Path

from experiments.power_spectrum.code import run_synthetic_benchmark as runner


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "overnight.json"
CONTROL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "matched_control.json"


def test_overnight_plan_is_below_time_and_cost_caps():
    cfg = runner.load_config(CONFIG)
    plan = runner.build_plan(cfg)
    assert plan["within_cost_plan"]
    assert plan["within_time_plan"]
    assert plan["estimated_cost_usd"] < 45
    assert plan["total_optimizer_steps"] > 2_000_000


def test_matched_baselines_and_v2_are_present():
    cfg = runner.load_config(CONFIG)
    models = {model["name"]: model for model in cfg["models"]}
    assert models["txc_pre"]["fairness_role"] == "equal_window_support"
    assert models["txc_post"]["fairness_role"] == "position_mixing_lower_code_support"
    assert any(
        model["class_path"].endswith("spectral_txc_v2:SpectralTXCV2") for model in cfg["models"]
    )


def test_training_identity_continues_across_phases():
    cfg = runner.load_config(CONFIG)
    smoke = {
        (cell["model"], cell["task"], cell["seed"]): cell
        for cell in runner.enumerate_cells(cfg, "smoke")
    }
    gate = {
        (cell["model"], cell["task"], cell["seed"]): cell
        for cell in runner.enumerate_cells(cfg, "gate")
    }
    key = ("txc_pre", "frequency", 42)
    assert smoke[key]["training_id"] == gate[key]["training_id"]
    assert smoke[key]["cell_id"] != gate[key]["cell_id"]
    assert smoke[key]["target_steps"] == 2
    assert gate[key]["target_steps"] == 1200


def test_every_cell_normalizes_batch_tokens():
    cfg = runner.load_config(CONFIG)
    target = cfg["training"]["batch_tokens"]
    for cell in runner.enumerate_cells(cfg, "full"):
        batch_size = target // cell["T"]
        assert batch_size * cell["T"] == target


def test_new_budget_session_marks_stale_running_session_interrupted(tmp_path):
    cfg = runner.load_config(CONFIG)
    runner.BudgetGuard(cfg, tmp_path, "test-hash")
    resumed = runner.BudgetGuard(cfg, tmp_path, "test-hash")
    assert resumed.ledger["sessions"][0]["status"] == "interrupted"
    assert resumed.ledger["sessions"][1]["status"] == "running"
    resumed.finish("complete")


def test_full_band_control_is_bounded_and_uses_fresh_full_seeds():
    cfg = runner.load_config(CONTROL_CONFIG)
    plan = runner.build_plan(cfg)
    assert plan["within_cost_plan"]
    assert plan["within_time_plan"]
    assert plan["estimated_cost_usd"] < 6
    full = runner.enumerate_cells(cfg, "full")
    assert len(full) == 15
    assert {cell["model"] for cell in full} == {"v2_full_global"}
    assert {cell["seed"] for cell in full} == {1, 2, 42}
    assert all(cell["model_spec"]["hparams"]["bands"] == "full" for cell in full)
    gate_training_ids = {
        cell["training_id"] for cell in runner.enumerate_cells(cfg, "gate")
    }
    assert not gate_training_ids.intersection(cell["training_id"] for cell in full)
