"""Contracts for the isolated common-cohort T16 backtracking extension."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy import sparse

from experiments.backtracking_window_sweep.evaluate import sparse_effective_l0
from experiments.backtracking_window_sweep.plot_publication import render
from experiments.backtracking_window_sweep.protocol_t16 import (
    ARTIFACT_OFFSETS,
    FULL_WINDOWS,
    artifact_inventory,
    assert_inventory,
    cohort_sha256,
    physical_offsets,
    validate_axes,
    window_queue,
)
from experiments.backtracking_window_sweep.train import (
    TrainCellConfig,
    run_memory_smoke,
    scheduled_window_indices,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _small_t16_artifacts(tmp_path):
    reference = tmp_path / "sentence_acts_L10.npz"
    reference_x = np.zeros((5, 6, 4096), dtype=np.float16)
    reference_labels = np.asarray([0, 1, 0, 1, 0], dtype=np.uint8)
    reference_keys = np.asarray(
        [f"q{index}|0|0" for index in range(5)], dtype=object
    )
    np.savez(
        reference,
        X=reference_x,
        is_bt=reference_labels,
        keys=reference_keys,
    )

    artifact = tmp_path / "sentence_acts_L10_T16.npz"
    positions = np.asarray([1, 3])
    x = np.zeros((2, 16, 4096), dtype=np.float16)
    x[:, -6:] = reference_x[positions]
    np.savez(
        artifact,
        X=x,
        is_bt=reference_labels[positions],
        keys=reference_keys[positions],
        offsets=np.asarray(ARTIFACT_OFFSETS),
    )
    manifest = tmp_path / "sentence_acts_L10_T16.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol_version": "test-builder",
                "output": {
                    "shape": list(x.shape),
                    "sha256": _sha256(artifact),
                    "offsets": list(ARTIFACT_OFFSETS),
                },
                "official_artifact": {"sha256": _sha256(reference)},
                "exact_key_order": True,
                "trailing_six": {
                    "offsets": list(range(-13, -7)),
                    "exact_equal": True,
                    "max_abs": 0.0,
                    "mismatched_values": 0,
                },
            }
        )
    )
    cache = tmp_path / "resid_post_L10.npy"
    np.save(cache, np.zeros((3, 16, 4096), dtype=np.float16))
    return artifact, manifest, reference, cache


def test_t16_grid_offsets_and_nested_schedule():
    assert FULL_WINDOWS == (1, 2, 4, 6, 8, 10, 12, 14, 16)
    assert physical_offsets(1) == (-8,)
    assert physical_offsets(16) == tuple(range(-23, -7))
    assert window_queue(FULL_WINDOWS) == (1, 16, 6, 2, 4, 8, 10, 12, 14)
    validate_axes(FULL_WINDOWS, (1, 2, 42))

    common = dict(
        step=7,
        n_sequences=9,
        sequence_length=32,
        batch_size=5,
        schedule_seed=101,
        max_window=16,
    )
    t1 = scheduled_window_indices(window=1, **common)
    t16 = scheduled_window_indices(window=16, **common)
    np.testing.assert_array_equal(t1[0], t16[0])
    np.testing.assert_array_equal(t1[1], t16[1] + 15)


def test_t16_inventory_locks_common_subset_and_manifest(tmp_path):
    artifact, manifest, reference, cache = _small_t16_artifacts(tmp_path)
    inventory = artifact_inventory(
        artifact,
        manifest,
        reference,
        cache,
        strict_full=False,
    )
    assert inventory["common_cohort_rows"] == 2
    assert inventory["official_key_subset_ok"]
    assert inventory["labels_match_official"]
    assert inventory["exact_key_order"]
    assert inventory["manifest_trailing_six_ok"]
    assert_inventory(inventory, strict_full=False)


def test_t16_inventory_rejects_reordered_common_cohort(tmp_path):
    artifact, manifest, reference, cache = _small_t16_artifacts(tmp_path)
    with np.load(artifact, allow_pickle=True) as payload:
        x = payload["X"][::-1]
        labels = payload["is_bt"][::-1]
        keys = payload["keys"][::-1]
        offsets = payload["offsets"]
    np.savez(
        artifact,
        X=x,
        is_bt=labels,
        keys=keys,
        offsets=offsets,
    )
    manifest_payload = json.loads(manifest.read_text())
    manifest_payload["output"]["sha256"] = _sha256(artifact)
    manifest.write_text(json.dumps(manifest_payload))
    inventory = artifact_inventory(
        artifact,
        manifest,
        reference,
        cache,
        strict_full=False,
    )
    assert not inventory["exact_key_order"]
    with np.testing.assert_raises_regex(
        ValueError, "T16 artifact provenance mismatch"
    ):
        assert_inventory(inventory, strict_full=False)


def test_teacher_force_manifest_records_fail_closed_provenance(tmp_path):
    artifact, manifest, reference, cache = _small_t16_artifacts(tmp_path)
    with np.load(artifact, allow_pickle=True) as payload:
        x_shape = list(payload["X"].shape)
        x_dtype = str(payload["X"].dtype)
        keys = payload["keys"].astype(str)
        labels = payload["is_bt"].astype(np.uint8)
    common_hash = cohort_sha256(keys, labels)
    teacher_manifest = {
        "protocol_version": "ward-c7-wide-teacher-force.v1",
        "status": "complete",
        "offsets": list(ARTIFACT_OFFSETS),
        "source": {
            "path": "pinned/eval_traces.jsonl",
            "sha256": "a" * 64,
            "verified_sha256": "a" * 64,
            "commit": "b" * 40,
            "field": "full_response",
        },
        "model": {
            "id": "NousResearch/Meta-Llama-3.1-8B",
            "revision": "c" * 40,
        },
        "tokenizer": {
            "id": "NousResearch/Meta-Llama-3.1-8B",
            "revision": "c" * 40,
        },
        "activation": {"layer": 10, "component": "resid_post"},
        "official_artifact": {"sha256": _sha256(reference)},
        "common_cohort": {
            "rows": len(keys),
            "sha256": common_hash,
            "exact_key_order": True,
        },
        "trailing_six": {
            "comparison": "exact_keyed_join",
            "offsets": list(range(-13, -7)),
            "matched_keys": len(keys),
            "exact_equal": True,
            "max_abs": 0.0,
            "mismatched_values": 0,
        },
        "validation": {"wide_rows": len(keys)},
        "output": {
            "shape": x_shape,
            "dtype": x_dtype,
            "sha256": _sha256(artifact),
            "exact_key_order": True,
        },
    }
    manifest.write_text(json.dumps(teacher_manifest))
    inventory = artifact_inventory(
        artifact,
        manifest,
        reference,
        cache,
        strict_full=False,
    )
    teacher_checks = {
        key: value
        for key, value in inventory.items()
        if key.startswith("teacher_")
    }
    assert teacher_checks
    assert all(teacher_checks.values())
    assert inventory["manifest_trailing_six_ok"]
    assert inventory["manifest_validation_counts_ok"]

    teacher_manifest["source"]["field"] = "thinking"
    manifest.write_text(json.dumps(teacher_manifest))
    invalid = artifact_inventory(
        artifact,
        manifest,
        reference,
        cache,
        strict_full=False,
    )
    assert not invalid["teacher_source_field_ok"]


def test_effective_l0_reports_topk_relu_underfill():
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [1.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )
    summary = sparse_effective_l0(matrix, nominal_l0=2)
    assert summary["nominal_l0"] == 2
    assert summary["effective_l0_mean"] == 1.0
    assert summary["fill_fraction_mean"] == 0.5
    assert np.isclose(summary["underfilled_row_fraction"], 2 / 3)
    assert np.isclose(summary["zero_row_fraction"], 1 / 3)


def test_t16_memory_smoke_uses_real_window_without_checkpoint(tmp_path):
    cache = tmp_path / "acts.npy"
    np.save(
        cache,
        np.random.default_rng(0).normal(size=(6, 20, 4)).astype(np.float32),
    )
    config = TrainCellConfig(
        arch="txc",
        window=16,
        seed=42,
        d_in=4,
        d_sae=16,
        k_pos=1,
        batch_size=2,
        steps=1,
        learning_rate=1e-3,
        warmup_steps=0,
        checkpoint_every=1,
        schedule_seed=17,
        amp=False,
        schedule_max_window=16,
        record_effective_l0=True,
    )
    result = run_memory_smoke(
        activation_cache=cache,
        config=config,
        device="cpu",
    )
    assert result["status"] == "complete"
    assert result["window"] == 16
    assert result["checkpoint_written"] is False
    assert 0 <= result["effective_l0"] <= result["nominal_l0"]


def _probe(ordered: float, shuffled: float) -> dict:
    return {
        "n_features": 32,
        "ordered_pr_auc": {"mean": ordered},
        "control_pr_auc": {
            "shuffle": {"mean": shuffled},
            "reverse": {"mean": shuffled - 0.005},
            "circular": {"mean": shuffled - 0.01},
        },
    }


def test_publication_plot_accepts_arbitrary_window_grid(tmp_path):
    root = tmp_path / "results"
    windows = (1, 8, 16)
    seeds = (1, 2)
    for window in windows:
        for seed in seeds:
            ordered = 0.20 + 0.002 * window + 0.001 * seed
            shuffled = 0.20 + 0.001 * window
            payload = {
                "status": "complete",
                "window": window,
                "seed": seed,
                "probes": {
                    "txc": [_probe(ordered, shuffled)],
                    "sae_positional": [_probe(ordered - 0.01, shuffled)],
                    "sae_invariant": [_probe(shuffled, shuffled)],
                    "sae_last_token": [_probe(0.20, 0.20)],
                },
            }
            path = root / "cells" / f"T{window}_seed{seed}" / "result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload))
    output = tmp_path / "publication"
    result = render(
        root,
        output,
        allow_partial=False,
        windows=windows,
        seeds=seeds,
    )
    assert result["n_cells"] == len(windows) * len(seeds)
    assert result["endpoint_windows"] == [1, 16]
    assert (output / "txc_window_length.png").exists()
    assert (output / "txc_ordered_minus_shuffled.png").exists()
    assert (output / "txc_order_sensitivity.png").exists()
    markdown = (output / "reviewer_figures.md").read_text()
    assert "order-invariant/DC-like component" in markdown
