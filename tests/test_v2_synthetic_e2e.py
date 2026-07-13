"""End-to-end smoke: synthetic experiment produces a real result row."""

from __future__ import annotations

import pytest

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig


@pytest.fixture(autouse=True)
def _allow_dirty(monkeypatch):
    """The arxiv branch is typically dirty during dev; let tests run."""
    monkeypatch.setenv("TEMP_BENCH_ALLOW_DIRTY", "1")


@pytest.mark.parametrize("arch_name", ["txc_base", "topk_sae"])
def test_synthetic_e2e_smoke(arch_name: str, sandbox_store) -> None:
    """Smoke: train tiny TXC-base / TopK-SAE on synth_smoke; verify a valid
    leaderboard row is produced.

    ``sandbox_store`` redirects the runner's writes to a tmp dir, so this test
    never touches the canonical ``results/leaderboard.jsonl`` (it used to append
    a real row on every run).
    """
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
    # Code version is always stamped; dirty/diff_sha256 must be *internally
    # consistent* whatever the real tree state (do NOT assume a dirty tree — the
    # test no longer dirties it). A recorded tracked-diff implies dirty; the
    # converse need not hold (an untracked-only tree is dirty with no diff).
    cv = result.row.code_version
    assert cv.commit_sha
    if cv.diff_sha256 is not None:
        assert cv.dirty
