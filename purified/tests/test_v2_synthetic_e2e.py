"""End-to-end smoke: synthetic experiment produces a real result row."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig


@pytest.fixture(autouse=True)
def _allow_dirty(monkeypatch):
    """The arxiv branch is typically dirty during dev; let tests run."""
    monkeypatch.setenv("TEMP_BENCH_ALLOW_DIRTY", "1")


@pytest.mark.parametrize("arch_name", ["txc_base", "topk_sae"])
def test_synthetic_e2e_smoke(arch_name: str, tmp_path: Path, monkeypatch) -> None:
    """Smoke: train tiny TXC-base / TopK-SAE on synth_smoke; verify
    leaderboard row appears with the right shape."""
    # Redirect outputs to a sandbox so the test doesn't pollute results/.
    # But the framework's purified_root() resolves from cwd / env, so:
    # ... For simplicity in this contract test, just check the in-memory
    # return value of run_experiment; the leaderboard write may go to the
    # real results/ directory (idempotent — re-running cache-hits).

    result = run_experiment(
        experiment="synthetic",
        arch_name=arch_name,
        seed=99,
        datasource_name="synth_smoke",
        training_cfg=TrainingConfig(
            n_steps=5,
            batch_size=16,
            buffer_tokens=4096,
            arch_hparams_override={"k_pos": 2},
        ),
        eval_cfg={"smoke": True},
    )
    assert result.train_key
    assert result.eval_key
    assert result.row.experiment == "synthetic"
    assert result.row.arch == arch_name
    assert "eauc" in result.row.metrics
    assert result.row.metrics["eauc"] >= 0.0
    # Code version was captured.
    assert result.row.code_version.commit_sha
    assert result.row.code_version.dirty is True   # arxiv branch is dirty during dev
    assert result.row.code_version.diff_sha256 is not None
