from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from experiments.gum_pronoun_distance.cohort import (
    _balanced_rows,
    _document_text_and_spans,
    _iter_direct_edges,
    sha256_file,
)
from experiments.gum_pronoun_distance.evaluate_frozen import (
    METHODS,
    SAE_METHODS,
    _equal_group_summary,
    _evaluate_subset,
    _gate,
    _history_permutation,
    _load_cached_code,
    _write_cached_code,
    controlled_windows,
)
from experiments.gum_pronoun_distance.extract_activations import (
    HIDDEN_SIZE,
    _atomic_safetensors,
    _shard_tensors,
    _validate_shard,
)


def test_canonical_document_text_obeys_ud_spaceafter() -> None:
    rows = [
        ["1", "Hello", "_", "INTJ", "_", "_", "0", "root", "_", "SpaceAfter=No"],
        ["2", ",", "_", "PUNCT", "_", "_", "1", "punct", "_", "_"],
        ["3", "world", "_", "NOUN", "_", "_", "1", "vocative", "_", "SpaceAfter=No"],
        ["4", "!", "_", "PUNCT", "_", "_", "1", "punct", "_", "_"],
    ]
    text, spans = _document_text_and_spans(rows)
    assert text == "Hello, world!"
    assert spans == [(0, 5), (5, 6), (7, 12), (12, 13)]


def test_direct_relation_orientation_uses_current_as_source() -> None:
    rows = [
        ["1-1", "0-4", "John", "person[1]", "new[1]", "_", "_", "_", "ana", "1-3[2_1]"],
        ["1-2", "5-8", "saw", "_", "_", "_", "_", "_", "_", "_"],
        ["1-3", "9-11", "him", "person[2]", "giv[2]", "_", "_", "_", "_", "_"],
    ]
    edges = list(_iter_direct_edges(rows, {"1": [0], "2": [2]}))
    assert len(edges) == 1
    assert edges[0]["source_entity_id"] == "1"
    assert edges[0]["target_entity_id"] == "2"
    assert edges[0]["target_index"] == 2


def test_balanced_selection_groups_first_seen_pronoun_then_label() -> None:
    events = [
        {"pronoun": "she", "distance": 3, "id": "s3a"},
        {"pronoun": "he", "distance": 2, "id": "h2"},
        {"pronoun": "she", "distance": 2, "id": "s2a"},
        {"pronoun": "she", "distance": 4, "id": "s4a"},
        {"pronoun": "she", "distance": 2, "id": "s2b"},
        {"pronoun": "she", "distance": 3, "id": "s3b"},
        {"pronoun": "he", "distance": 3, "id": "h3"},
        {"pronoun": "he", "distance": 4, "id": "h4"},
    ]
    selected = _balanced_rows(events)
    assert [event["id"] for event in selected] == [
        "s2a",
        "s3a",
        "s4a",
        "h2",
        "h3",
        "h4",
    ]


def test_history_controls_are_deterministic_and_fix_endpoint() -> None:
    windows = np.arange(4 * 5 * 2).reshape(4, 5, 2)
    hashes = [f"{index:064x}" for index in range(4)]
    shuffled = controlled_windows(windows, hashes, mode="shuffle", seed=17)
    repeated = controlled_windows(windows, hashes, mode="shuffle", seed=17)
    reversed_windows = controlled_windows(windows, hashes, mode="reverse", seed=17)
    assert np.array_equal(shuffled, repeated)
    assert np.array_equal(shuffled[:, -1], windows[:, -1])
    assert np.array_equal(reversed_windows[:, -1], windows[:, -1])
    assert np.array_equal(reversed_windows[:, :-1], windows[:, 3::-1])
    for event_hash in hashes:
        permutation = _history_permutation(event_hash, 17)
        assert not np.array_equal(permutation, np.arange(4))


def test_equal_document_bootstrap_does_not_upweight_prolific_document() -> None:
    target = np.asarray([2, 2, 2, 2, 2])
    groups = np.asarray(["many", "many", "many", "many", "one"])
    labels = (2, 3, 4)
    ordered = np.asarray(
        [
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.2, 0.4, 0.4],
        ]
    )
    competitor = np.asarray([[0.5, 0.25, 0.25]] * 5)
    probabilities = {
        name: ordered.copy() if name == "txc_ordered" else competitor.copy()
        for name in METHODS
    }
    summary = _equal_group_summary(
        probabilities,
        target,
        groups,
        labels,
        draws=100,
        seed=3,
    )
    expected = (-np.log(0.9) - np.log(0.2)) / 2
    assert summary["method_equal_document_log_loss"]["txc_ordered"] == pytest.approx(
        expected
    )


def test_preregistered_gate_requires_margin_and_positive_lower_bound() -> None:
    good = {
        "equal_document_mean_log_loss_difference": 0.03,
        "ci95_lower": 0.01,
    }
    weak = {
        "equal_document_mean_log_loss_difference": 0.01,
        "ci95_lower": 0.005,
    }
    summary = {
        "contrasts": {
            "txc_fixed_shuffle_history_minus_txc_ordered": good,
            "txc_fixed_reverse_history_minus_txc_ordered": good,
        },
        "strongest_sae_minus_txc_ordered": good,
    }
    assert _gate(summary, 0.02)["passed"]
    summary["strongest_sae_minus_txc_ordered"] = weak
    assert not _gate(summary, 0.02)["passed"]


def test_fixed_txc_controls_do_not_fit_separate_probes(monkeypatch) -> None:
    rows = 45
    labels = np.tile(np.asarray([2, 3, 4]), rows // 3)
    documents = np.repeat([f"doc-{index:02d}" for index in range(15)], 3)
    frame = pd.DataFrame({"distance": labels, "document": documents})
    base = sparse.csr_matrix(
        np.column_stack(
            [
                np.arange(rows) % 3,
                np.arange(rows) % 5,
                np.ones(rows),
            ]
        )
    )
    matrices = {name: base.copy() for name in METHODS}
    fit_calls = []

    def fake_fit(matrix, target, train, selected, **_kwargs):
        fit_calls.append((matrix.shape, tuple(selected)))
        return object(), object()

    def fake_predict(_matrix, indices, _selected, _scaler, _classifier, labels):
        return np.full((len(indices), len(labels)), 1 / len(labels), dtype=np.float32)

    monkeypatch.setattr(
        "experiments.gum_pronoun_distance.evaluate_frozen._fit_probe",
        fake_fit,
    )
    monkeypatch.setattr(
        "experiments.gum_pronoun_distance.evaluate_frozen._predict_probe",
        fake_predict,
    )
    _evaluate_subset(
        matrices,
        frame,
        np.arange(rows),
        subset_name="test",
        budgets=(2,),
        primary_budget=2,
        folds=5,
        c_value=1.0,
        max_iter=100,
        bootstrap_draws=20,
        seed=4,
        gate_margin=0.02,
    )
    assert len(fit_calls) == 5 * (1 + len(SAE_METHODS))


def test_activation_shard_round_trip_validates_exact_rows(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "event_hash": ["0" * 64, "1" * 64],
            "window_hash": ["2" * 64, "3" * 64],
            "window_token_ids": [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)],
            "distance": [2, 4],
            "balanced_sensitivity": [True, False],
        }
    )
    activations = torch.zeros((2, 5, HIDDEN_SIZE), dtype=torch.float16)
    tensors = _shard_tensors(frame, activations, start=0)
    shard = tmp_path / "shard.safetensors"
    sidecar = tmp_path / "shard.json"
    _atomic_safetensors(tensors, shard)
    payload = {
        "name": "shard",
        "start": 0,
        "stop": 2,
        "rows": 2,
        "sha256": sha256_file(shard),
        "size_bytes": shard.stat().st_size,
        "request_sha256": "request",
        "runtime_sha256": "runtime",
    }
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    _validate_shard(
        shard,
        sidecar,
        frame,
        start=0,
        stop=2,
        request_sha256="request",
        runtime_sha256="runtime",
    )
    payload["request_sha256"] = "tampered"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        _validate_shard(
            shard,
            sidecar,
            frame,
            start=0,
            stop=2,
            request_sha256="request",
            runtime_sha256="runtime",
        )


def test_sparse_code_sidecar_binds_file_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "ordered.npz"
    matrix = sparse.csr_matrix(np.asarray([[0.0, 2.0], [1.0, 0.0]]))
    _write_cached_code(
        matrix,
        path,
        method="txc_ordered",
        metadata_sha256="metadata",
    )
    observed = _load_cached_code(
        path,
        method="txc_ordered",
        rows=2,
        columns=2,
        metadata_sha256="metadata",
    )
    assert np.array_equal(observed.toarray(), matrix.toarray())
    sidecar = path.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["metadata_sha256"] = "stale"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        _load_cached_code(
            path,
            method="txc_ordered",
            rows=2,
            columns=2,
            metadata_sha256="metadata",
        )
