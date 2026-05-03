"""Smoke tests for runner idempotency — the heart of the cache contract.

Uses a tiny mock arch + tiny mock data so the test runs in <1s and
doesn't need a GPU.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_temp_bench_root(monkeypatch):
    """Point TEMP_BENCH_ROOT at a clean tmp dir so tests don't pollute
    the real ``purified/results/`` and ``purified/checkpoints/``.

    Copies the configs/ dir over so loaders find ``locked_archs.yaml``
    and ``datasources.yaml``.
    """
    real_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Copy configs (read-only inputs)
        shutil.copytree(real_root / "configs", tmp / "configs")
        # Empty results + checkpoints
        (tmp / "results").mkdir()
        (tmp / "results" / "runs").mkdir()
        (tmp / "results" / "leaderboard.jsonl").touch()
        (tmp / "checkpoints").mkdir()
        (tmp / "checkpoints" / "manifest.jsonl").touch()

        monkeypatch.setenv("TEMP_BENCH_ROOT", str(tmp))
        # Bust the lru_cache on _load_archs_yaml / _load_datasources_yaml
        from temp_bench.config import _load_archs_yaml, _load_datasources_yaml
        _load_archs_yaml.cache_clear()
        _load_datasources_yaml.cache_clear()
        yield tmp


def _mock_train_fn(*, arch_name, arch_hparams, seed, training_cfg, act_cache_key, component):
    """Train returns an empty state dict — enough to exercise save/load paths."""
    import torch
    return {"dummy": torch.zeros(2, 2)}


def _mock_eval_fn(*, model, eval_cfg, component):
    metrics = {"probing_auc": 0.5 + 0.001 * eval_cfg.get("k_feat", 0)}
    return metrics, "probing_auc"


def test_run_cell_is_idempotent(tmp_temp_bench_root):
    """Calling run_cell twice with the same inputs:
    - first call trains + evaluates + appends 1 row
    - second call hits both caches + appends 0 rows
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from temp_bench import cache, runner

    common = dict(
        component="c3",
        arch_name="topk_sae",
        seed=42,
        datasource_name="gemma_2_2b_it_l13_fineweb_24k128",
        training_cfg={"n_steps": 10},  # tiny override to keep it cheap
        eval_cfg={"k_feat": 5, "S": 32},
        eval_protocol_version="0.0.1-test",
        train_fn=_mock_train_fn,
        eval_fn=_mock_eval_fn,
        primary_metric="probing_auc",
        agent="test",
    )

    r1 = runner.run_cell(**common)
    assert r1.cached is False
    rows1 = list(cache.iter_leaderboard())
    assert len(rows1) == 1
    assert rows1[0].eval_key == r1.eval_key

    r2 = runner.run_cell(**common)
    assert r2.cached is True
    assert r2.eval_key == r1.eval_key
    rows2 = list(cache.iter_leaderboard())
    assert len(rows2) == 1, "Second call must not append a duplicate row"


def test_force_train_invalidates_train_cache(tmp_temp_bench_root):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from temp_bench import cache, runner

    common = dict(
        component="c3",
        arch_name="topk_sae",
        seed=42,
        datasource_name="gemma_2_2b_it_l13_fineweb_24k128",
        training_cfg={"n_steps": 10},
        eval_cfg={"k_feat": 5, "S": 32},
        eval_protocol_version="0.0.1-test",
        train_fn=_mock_train_fn,
        eval_fn=_mock_eval_fn,
        primary_metric="probing_auc",
        agent="test",
    )

    r1 = runner.run_cell(**common)
    assert r1.cached is False

    r2 = runner.run_cell(**common, force_eval=True)
    assert r2.cached is False  # forced re-eval
    rows = list(cache.iter_leaderboard())
    # Two leaderboard rows with same eval_key — that's expected when
    # force_eval is used. Caller is responsible for downstream dedup.
    assert len(rows) == 2
    assert all(r.eval_key == r1.eval_key for r in rows)


def test_changing_seed_creates_new_cells(tmp_temp_bench_root):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from temp_bench import cache, runner

    base = dict(
        component="c3",
        arch_name="topk_sae",
        datasource_name="gemma_2_2b_it_l13_fineweb_24k128",
        training_cfg={"n_steps": 10},
        eval_cfg={"k_feat": 5, "S": 32},
        eval_protocol_version="0.0.1-test",
        train_fn=_mock_train_fn,
        eval_fn=_mock_eval_fn,
        primary_metric="probing_auc",
        agent="test",
    )

    r1 = runner.run_cell(seed=1, **base)
    r42 = runner.run_cell(seed=42, **base)
    assert r1.eval_key != r42.eval_key
    assert r1.train_key != r42.train_key
    rows = list(cache.iter_leaderboard())
    assert len(rows) == 2
