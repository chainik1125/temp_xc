"""Contracts for the isolated common-cohort T16 backtracking extension."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
from scipy import sparse

from experiments.backtracking_window_sweep import protocol_t16
from experiments.backtracking_window_sweep.evaluate import sparse_effective_l0
from experiments.backtracking_window_sweep.plot_publication import render
from experiments.backtracking_window_sweep.protocol_t16 import (
    ARTIFACT_OFFSETS,
    FULL_WINDOWS,
    artifact_inventory,
    assert_inventory,
    cohort_sha256,
    physical_offsets,
    profile,
    validate_axes,
    window_queue,
)
from experiments.backtracking_window_sweep.run_t16 import (
    _cell_run_contract,
    _cell_shard,
    _fingerprinted_completed_checks,
    _legacy_completed_checks,
    _run_fingerprint,
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
    positions = np.asarray([0, 1])
    x = np.zeros((2, 16, 4096), dtype=np.float32)
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


def test_one_seed_cells_are_distributed_across_all_gpu_shards():
    shards = [
        _cell_shard(
            FULL_WINDOWS,
            (42,),
            num_shards=3,
            shard_index=shard_index,
        )
        for shard_index in range(3)
    ]
    assert all(len(shard) == 3 for shard in shards)
    assert sorted(cell for shard in shards for cell in shard) == sorted(
        (42, window) for window in FULL_WINDOWS
    )
    assert not set(shards[0]) & set(shards[1])
    assert not set(shards[0]) & set(shards[2])
    assert not set(shards[1]) & set(shards[2])


def test_t16_tmux_launcher_forwards_pinned_runtime_paths():
    launcher = (
        __import__("pathlib").Path(__file__).parents[1]
        / "experiments/backtracking_window_sweep/launch_t16_tmux.sh"
    ).read_text()
    for variable in (
        "TXC_RUNPOD_PYTHON",
        "BACKTRACKING_T16_ARTIFACT",
        "BACKTRACKING_T16_MANIFEST",
        "BACKTRACKING_T16_REFERENCE",
        "BACKTRACKING_ACTIVATION_CACHE",
        "BACKTRACKING_T16_RESULT_ROOT",
        "BACKTRACKING_T16_CHECKPOINT_ROOT",
        "BACKTRACKING_T16_WINDOWS",
        "BACKTRACKING_T16_SEEDS",
    ):
        assert f'"{variable}=' in launcher


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


def test_full_inventory_pins_activation_cache_bytes_shape_and_dtype(
    tmp_path, monkeypatch
):
    artifact, manifest, reference, cache = _small_t16_artifacts(tmp_path)
    cache_digest = _sha256(cache)
    monkeypatch.setattr(protocol_t16, "CACHE_SHAPE", (3, 16, 4096))
    monkeypatch.setattr(protocol_t16, "CACHE_SHA256", cache_digest)
    inventory = artifact_inventory(
        artifact,
        manifest,
        reference,
        cache,
        strict_full=True,
    )
    assert inventory["activation_cache_shape_ok"]
    assert inventory["activation_cache_dtype_ok"]
    assert inventory["activation_cache_sha256_ok"]
    assert inventory["activation_cache_contract_ok"]

    np.save(cache, np.zeros((3, 16, 4096), dtype=np.float32))
    invalid = artifact_inventory(
        artifact,
        manifest,
        reference,
        cache,
        strict_full=True,
    )
    assert not invalid["activation_cache_dtype_ok"]
    assert not invalid["activation_cache_sha256_ok"]
    assert not invalid["activation_cache_contract_ok"]


def test_cell_fingerprint_covers_training_eval_and_cache_identity():
    configured = profile("smoke")
    inventory = {
        "activation_cache_sha256": "a" * 64,
        "activation_cache_shape": [3, 16, 4096],
        "activation_cache_dtype": "float16",
        "artifact_sha256": "b" * 64,
        "common_cohort_sha256": "c" * 64,
        "common_cohort_rows": 2,
    }
    contract = _cell_run_contract(
        configured,
        window=16,
        seed=42,
        encode_batch_size=32,
        pca_dim=16,
        inventory=inventory,
    )
    fingerprint = _run_fingerprint(contract)
    completed = {
        "run_contract": contract,
        "run_fingerprint": fingerprint,
    }
    assert all(
        _fingerprinted_completed_checks(
            completed,
            contract=contract,
            fingerprint=fingerprint,
        ).values()
    )
    redispatched = _cell_run_contract(
        replace(configured, windows=(16,), seeds=(42,)),
        window=16,
        seed=42,
        encode_batch_size=32,
        pca_dim=16,
        inventory=inventory,
    )
    assert redispatched == contract
    assert _run_fingerprint(redispatched) == fingerprint

    variants = [
        _cell_run_contract(
            replace(configured, steps=configured.steps + 1),
            window=16,
            seed=42,
            encode_batch_size=32,
            pca_dim=16,
            inventory=inventory,
        ),
        _cell_run_contract(
            replace(configured, d_sae=configured.d_sae + 1),
            window=16,
            seed=42,
            encode_batch_size=32,
            pca_dim=16,
            inventory=inventory,
        ),
        _cell_run_contract(
            replace(configured, k_pos=configured.k_pos + 1),
            window=16,
            seed=42,
            encode_batch_size=32,
            pca_dim=16,
            inventory=inventory,
        ),
        _cell_run_contract(
            replace(configured, batch_size=configured.batch_size + 1),
            window=16,
            seed=42,
            encode_batch_size=32,
            pca_dim=16,
            inventory=inventory,
        ),
        _cell_run_contract(
            replace(configured, folds=configured.folds + 1),
            window=16,
            seed=42,
            encode_batch_size=32,
            pca_dim=16,
            inventory=inventory,
        ),
        _cell_run_contract(
            configured,
            window=16,
            seed=42,
            encode_batch_size=32,
            pca_dim=16,
            inventory={**inventory, "activation_cache_sha256": "b" * 64},
        ),
    ]
    for variant in variants:
        assert _run_fingerprint(variant) != fingerprint
        checks = _fingerprinted_completed_checks(
            completed,
            contract=variant,
            fingerprint=_run_fingerprint(variant),
        )
        assert not all(checks.values())


def test_legacy_cell_migration_checks_checkpoint_and_eval_profile(tmp_path):
    configured = profile("full")
    inventory = {
        "activation_cache_sha256": "a" * 64,
        "activation_cache_shape": [4044, 128, 4096],
        "activation_cache_dtype": "float16",
        "activation_cache_contract_ok": True,
        "artifact_sha256": "b" * 64,
        "common_cohort_sha256": "c" * 64,
        "common_cohort_rows": 20_335,
    }
    contract = _cell_run_contract(
        configured,
        window=16,
        seed=42,
        encode_batch_size=32,
        pca_dim=32,
        inventory=inventory,
    )
    checkpoint_cell = tmp_path / "T16_seed42"
    checkpoint_hashes = {}
    for arch in ("txc", "sae"):
        directory = checkpoint_cell / arch
        directory.mkdir(parents=True)
        config = {
            **contract["training"][arch],
            "exposure_contract": "legacy informational field",
        }
        (directory / "config.json").write_text(json.dumps(config))
        model = directory / "model.safetensors"
        model.write_bytes(f"{arch}-weights".encode())
        checkpoint_hashes[arch] = _sha256(model)
    completed = {
        "artifact_sha256": "b" * 64,
        "cohort_sha256": "c" * 64,
        "window": 16,
        "seed": 42,
        "folds": configured.folds,
        "s_grid": list(configured.s_grid),
        "n_rows": inventory["common_cohort_rows"],
        "grouped_question_bootstrap": {
            "repeats_requested": configured.bootstrap_repeats
        },
        "code_fingerprint": {
            "artifact_sha256": "b" * 64,
            "cohort_sha256": "c" * 64,
            "window": 16,
            "window_offsets": list(ARTIFACT_OFFSETS),
            "seed": 42,
            "txc_checkpoint_sha256": checkpoint_hashes["txc"],
            "sae_checkpoint_sha256": checkpoint_hashes["sae"],
        },
    }
    checks = _legacy_completed_checks(
        completed,
        contract=contract,
        checkpoint_cell=checkpoint_cell,
        inventory=inventory,
        window=16,
        seed=42,
    )
    assert all(checks.values())

    changed = _cell_run_contract(
        replace(configured, steps=configured.steps + 1),
        window=16,
        seed=42,
        encode_batch_size=32,
        pca_dim=32,
        inventory=inventory,
    )
    stale = _legacy_completed_checks(
        completed,
        contract=changed,
        checkpoint_cell=checkpoint_cell,
        inventory=inventory,
        window=16,
        seed=42,
    )
    assert not stale["txc_training_config"]
    assert not stale["sae_training_config"]


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
    retained_cohort = {
        "rows": 2,
        "sha256": common_hash,
        "category_sha256": "d" * 64,
        "class_counts": {"0": 1, "1": 1},
        "category_counts": {"arithmetic": 2},
        "category_class_counts": {
            "arithmetic": {"0": 1, "1": 1}
        },
    }
    official_cohort = {
        "rows": 5,
        "sha256": "e" * 64,
        "category_sha256": "f" * 64,
        "class_counts": {"0": 3, "1": 2},
        "category_counts": {"arithmetic": 5},
        "category_class_counts": {
            "arithmetic": {"0": 3, "1": 2}
        },
    }
    coverage = {
        name: {
            "cohort": name,
            "passed": True,
            "missing_classes": [],
            "missing_categories": [],
            "missing_category_class_cells": [],
        }
        for name in (
            "source_eligible",
            "exact_tail_retained",
            "wide_t16",
        )
    }
    teacher_manifest = {
        "schema_version": "ward-c7-wide-teacher-force-manifest.v4",
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
            **retained_cohort,
            "exact_key_order": True,
            "selection": "source-aligned-and-bit-exact-tail-and-t16.v1",
        },
        "trailing_six": {
            "comparison": "exact_keyed_join",
            "comparator": "float32-uint32-bit-exact.v1",
            "offsets": list(range(-13, -7)),
            "matched_keys": len(keys),
            "exact_equal": True,
            "max_abs": 0.0,
            "mismatched_values": 0,
        },
        "validation": {
            "official_rows": 5,
            "official_cohort": official_cohort,
            "source_excluded_rows": 3,
            "source_eligible_cohort": retained_cohort,
            "exact_tail_retained_cohort": retained_cohort,
            "wide_t16_cohort": retained_cohort,
            "extraction_exclusion_rows": 0,
            "wide_rows_dropped_for_missing_early_offsets": 0,
            "wide_rows": 2,
            "complete_trace_shards": 1,
            "trace_count": 1,
            "coverage": coverage,
            "source_exclusions_sha256": "1" * 64,
            "extraction_exclusions_sha256": "2" * 64,
            "combined_exclusions_sha256": "3" * 64,
            "trace_summary": [
                {
                    "trace_idx": 0,
                    "category": "arithmetic",
                    "source_eligible_rows": 2,
                    "exact_tail_rows": 2,
                    "wide_rows": 2,
                    "source_class_counts": {"0": 1, "1": 1},
                    "exact_tail_class_counts": {"0": 1, "1": 1},
                    "wide_class_counts": {"0": 1, "1": 1},
                    "trace_exclusion_reason": None,
                    "exclusion_reason_counts": {},
                }
            ],
        },
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

    teacher_manifest["source"]["field"] = "full_response"
    teacher_manifest["common_cohort"]["selection"] = "legacy-selection"
    manifest.write_text(json.dumps(teacher_manifest))
    invalid = artifact_inventory(
        artifact,
        manifest,
        reference,
        cache,
        strict_full=False,
    )
    assert not invalid["teacher_exact_subset_policy_ok"]
    assert not invalid["teacher_cohort_metadata_ok"]

    teacher_manifest["common_cohort"]["selection"] = (
        "source-aligned-and-bit-exact-tail-and-t16.v1"
    )
    teacher_manifest["validation"]["extraction_exclusion_rows"] = 1
    manifest.write_text(json.dumps(teacher_manifest))
    invalid = artifact_inventory(
        artifact,
        manifest,
        reference,
        cache,
        strict_full=False,
    )
    assert not invalid["teacher_cohort_accounting_ok"]

    teacher_manifest["validation"]["extraction_exclusion_rows"] = 0
    teacher_manifest["validation"]["coverage"]["wide_t16"]["passed"] = False
    manifest.write_text(json.dumps(teacher_manifest))
    invalid = artifact_inventory(
        artifact,
        manifest,
        reference,
        cache,
        strict_full=False,
    )
    assert not invalid["teacher_coverage_gates_ok"]


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


def test_topk_then_relu_matches_topk_of_relu_forward_and_gradient():
    torch = __import__("torch")
    generator = torch.Generator().manual_seed(17)
    pre = torch.randn(11, 37, generator=generator, requires_grad=True)
    weights = torch.randn(11, 37, generator=generator)
    k = 13

    values, indices = pre.topk(k, dim=-1)
    topk_then_relu = torch.zeros_like(pre).scatter(
        -1, indices, torch.relu(values)
    )

    reference_pre = pre.detach().clone().requires_grad_(True)
    positive = torch.relu(reference_pre)
    values, indices = positive.topk(k, dim=-1)
    relu_then_topk = torch.zeros_like(reference_pre).scatter(
        -1, indices, values
    )

    torch.testing.assert_close(topk_then_relu, relu_then_topk, rtol=0, atol=0)
    (topk_then_relu * weights).sum().backward()
    (relu_then_topk * weights).sum().backward()
    torch.testing.assert_close(pre.grad, reference_pre.grad, rtol=0, atol=0)


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
    assert result["paper_t1_s32_baselines"] == {
        "SAE": 0.229,
        "T-SAE": 0.245,
    }
    assert (output / "txc_window_length.png").exists()
    assert (output / "txc_ordered_minus_shuffled.png").exists()
    assert (output / "txc_order_sensitivity.png").exists()
    markdown = (output / "reviewer_figures.md").read_text()
    assert "order-invariant/DC-like component" in markdown
    assert "submitted-paper S=32 baselines" in markdown
