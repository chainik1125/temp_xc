"""Contracts for the July 23 backtracking window-size runner."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.backtracking_window_sweep.evaluate import (
    _encode_sae_positional_batch,
    grouped_question_bootstrap,
    summarize_probe,
)
from experiments.backtracking_window_sweep.protocol import (
    FULL_SEEDS,
    FULL_WINDOWS,
    artifact_inventory,
    physical_offsets,
    seed_queue,
    validate_axes,
    window_queue,
)
from experiments.backtracking_window_sweep.train import (
    TrainCellConfig,
    materialize_windows,
    scheduled_window_indices,
    train_dictionary,
)


def test_physical_offset_and_dispatch_contracts():
    assert [physical_offsets(window) for window in FULL_WINDOWS] == [
        (-8,),
        (-9, -8),
        (-10, -9, -8),
        (-11, -10, -9, -8),
        (-12, -11, -10, -9, -8),
        (-13, -12, -11, -10, -9, -8),
    ]
    assert window_queue(FULL_WINDOWS) == (1, 6, 2, 3, 4, 5)
    assert seed_queue(FULL_SEEDS) == (42, 1, 2)
    with pytest.raises(ValueError, match="supports only T=1..6"):
        validate_axes((1, 7), FULL_SEEDS)


def test_counter_schedule_is_reproducible_and_shared():
    kwargs = dict(
        step=17,
        n_sequences=11,
        sequence_length=9,
        window=3,
        batch_size=5,
        schedule_seed=907,
    )
    first = scheduled_window_indices(**kwargs)
    second = scheduled_window_indices(**kwargs)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    changed = scheduled_window_indices(**{**kwargs, "step": 18})
    assert not (
        np.array_equal(first[0], changed[0])
        and np.array_equal(first[1], changed[1])
    )
    t1 = scheduled_window_indices(**{**kwargs, "window": 1})
    t6 = scheduled_window_indices(**{**kwargs, "window": 6})
    np.testing.assert_array_equal(t1[0], t6[0])
    np.testing.assert_array_equal(t1[1], t6[1] + 5)

    cache = np.arange(11 * 9 * 2, dtype=np.float32).reshape(11, 9, 2)
    windows = materialize_windows(
        cache,
        step=17,
        window=3,
        batch_size=5,
        schedule_seed=907,
    )
    seq, start = first
    expected = np.stack(
        [cache[sequence, offset : offset + 3] for sequence, offset in zip(seq, start)]
    )
    np.testing.assert_array_equal(windows, expected)


def test_sae_positional_blocks_retain_order():
    state = {
        "W_enc": torch.eye(2),
        "b_enc": torch.zeros(2),
        "b_dec": torch.zeros(2),
    }
    x = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])
    ordered = _encode_sae_positional_batch(x, state, k_pos=1).toarray()
    reversed_ = _encode_sae_positional_batch(x.flip(1), state, k_pos=1).toarray()
    assert ordered.shape == (1, 4)
    assert not np.array_equal(ordered, reversed_)
    np.testing.assert_array_equal(ordered, [[2.0, 0.0, 0.0, 3.0]])


def test_training_cell_resumes_from_complete_checkpoint(tmp_path):
    cache = np.random.default_rng(0).normal(size=(8, 6, 4)).astype(np.float32)
    cache_path = tmp_path / "acts.npy"
    np.save(cache_path, cache)
    config = TrainCellConfig(
        arch="txc",
        window=2,
        seed=3,
        d_in=4,
        d_sae=8,
        k_pos=2,
        batch_size=4,
        steps=2,
        learning_rate=1e-3,
        warmup_steps=0,
        checkpoint_every=1,
        schedule_seed=91,
        amp=False,
    )
    checkpoint = tmp_path / "checkpoint"
    first = train_dictionary(
        activation_cache=cache_path,
        checkpoint_dir=checkpoint,
        config=config,
        device="cpu",
    )
    second = train_dictionary(
        activation_cache=cache_path,
        checkpoint_dir=checkpoint,
        config=config,
        device="cpu",
    )
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["completed_steps"] == 2
    assert (checkpoint / "model.safetensors").exists()
    assert (checkpoint / "training_state.pt").exists()


def test_smoke_inventory_accepts_small_row_counts(tmp_path):
    artifact = tmp_path / "artifact.npz"
    np.savez(
        artifact,
        X=np.zeros((5, 6, 4096), dtype=np.float16),
        is_bt=np.array([0, 1, 0, 1, 0]),
        keys=np.array(["q0|0|0"] * 5, dtype=object),
    )
    cache = tmp_path / "acts.npy"
    np.save(cache, np.zeros((3, 8, 4096), dtype=np.float16))
    result = artifact_inventory(artifact, cache, strict_full=False)
    assert result["missing"] == []
    assert result["artifact_shape_ok"]
    assert result["activation_cache_shape_ok"]


def test_probe_summary_keeps_each_feature_budget():
    rows = []
    for n_features in (4, 8):
        for fold in (0, 1):
            ordered = 0.3 + 0.01 * fold
            controls = {
                name: {"pr_auc": 0.2, "roc_auc": 0.5, "log_loss": 0.7}
                for name in ("shuffle", "reverse", "circular")
            }
            rows.append(
                {
                    "fold": fold,
                    "n_features": n_features,
                    "ordered": {
                        "pr_auc": ordered,
                        "roc_auc": 0.6,
                        "log_loss": 0.6,
                    },
                    "controls": controls,
                    "fixed_probe_order_gap_pr_auc": {
                        name: ordered - 0.2 for name in controls
                    },
                }
            )
    summary = summarize_probe(rows)
    assert [row["n_features"] for row in summary] == [4, 8]
    assert np.isclose(summary[0]["ordered_pr_auc"]["mean"], 0.305)


def test_question_bootstrap_is_paired_deterministic_and_reported(tmp_path):
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    test_indices = np.arange(len(y), dtype=np.int64)
    predictions = {
        "txc": np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85]),
        "sae_positional": np.array([0.25, 0.75, 0.3, 0.7, 0.35, 0.65]),
        "sae_invariant": np.array([0.4, 0.6, 0.45, 0.55, 0.4, 0.6]),
        "sae_last_token": np.array([0.35, 0.65, 0.4, 0.6, 0.45, 0.55]),
        "residual": np.array([0.05, 0.95, 0.1, 0.9, 0.05, 0.95]),
    }
    probes = {}
    for name in ("txc", "sae_positional", "sae_invariant", "sae_last_token"):
        fold_rows = []
        for fold in range(2):
            path = tmp_path / f"{name}_{fold}.npz"
            payload = {
                "test_indices": test_indices,
                "y": y,
                "groups": groups,
                "ordered": predictions[name].astype(np.float32),
                "control_shuffle": predictions["sae_invariant"].astype(np.float32),
                "control_reverse": predictions["sae_invariant"].astype(np.float32),
                "control_circular": predictions["sae_invariant"].astype(np.float32),
            }
            np.savez_compressed(path, **payload)
            fold_rows.append({"fold": fold, "prediction_path": str(path)})
        probes[name] = [{"n_features": 8, "folds": fold_rows}]
    residual_folds = []
    for fold in range(2):
        path = tmp_path / f"residual_{fold}.npz"
        np.savez_compressed(
            path,
            test_indices=test_indices,
            y=y,
            groups=groups,
            ordered=predictions["residual"].astype(np.float32),
        )
        residual_folds.append({"fold": fold, "prediction_path": str(path)})
    residual = {"folds": residual_folds}

    first = grouped_question_bootstrap(probes, residual, repeats=100, seed=7)
    second = grouped_question_bootstrap(probes, residual, repeats=100, seed=7)
    assert first == second
    comparisons = first["comparisons"]
    assert "txc_minus_sae_positional" in comparisons
    assert "txc_minus_strongest_learned_control" in comparisons
    assert comparisons["txc_minus_sae_positional"]["repeats"] == 100
