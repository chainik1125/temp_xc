from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from experiments.writing_revision_destination import frozen_dictionary as frozen
from experiments.writing_revision_destination.evaluate_activations import (
    ActivationDataset,
)


def _exact_dataset() -> ActivationDataset:
    label_counts = frozen.EXPECTED_DATASET["label_counts"]
    target = np.concatenate(
        [
            np.full(count, label, dtype=np.int16)
            for label, count in label_counts.items()
        ]
    )
    rows = len(target)
    activations = np.broadcast_to(
        np.zeros((1, 1, 1), dtype=np.float16),
        (
            rows,
            frozen.EXPECTED_DATASET["cache_window_tokens"],
            frozen.EXPECTED_DATASET["hidden_size"],
        ),
    )
    groups = np.asarray(
        [f"writer-{index % frozen.EXPECTED_DATASET['writers']}" for index in range(rows)]
    )
    event_hashes = np.asarray([f"event-{index}" for index in range(rows)])
    provenance = {
        key: value
        for key, value in frozen.EXPECTED_DATASET.items()
        if key
        in {
            "cohort_sha256",
            "cohort_manifest_sha256",
            "request_sha256",
            "runtime_sha256",
            "complete_sha256",
            "model",
            "model_revision_observed",
            "layer",
            "hook_semantics",
        }
    }
    return ActivationDataset(
        activations=activations,
        target=target,
        groups=groups,
        event_hashes=event_hashes,
        target_name="capped_token_label",
        provenance=provenance,
    )


def _synthetic_codes() -> tuple[
    dict[str, sparse.csr_matrix],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    generator = np.random.default_rng(7)
    writer_labels = np.repeat(np.arange(3), 6)
    groups = np.repeat(
        np.asarray([f"writer-{index}" for index in range(len(writer_labels))]),
        2,
    )
    target = np.repeat(writer_labels, 2)
    rows = len(target)
    base = generator.normal(scale=0.08, size=(rows, 12))
    base[np.arange(rows), target] += 2.0
    matrices = {
        "txc_ordered": sparse.csr_matrix(base),
        "txc_fixed_shuffle": sparse.csr_matrix(np.roll(base, 1, axis=1)),
        "txc_fixed_reverse": sparse.csr_matrix(base[:, ::-1]),
        "sae_positional": sparse.csr_matrix(
            base + generator.normal(scale=0.15, size=base.shape)
        ),
        "sae_invariant": sparse.csr_matrix(
            base + generator.normal(scale=0.25, size=base.shape)
        ),
        "sae_last_token": sparse.csr_matrix(
            base + generator.normal(scale=0.35, size=base.shape)
        ),
    }
    event_hashes = np.asarray([f"event-{index}" for index in range(rows)])
    return matrices, target, groups, event_hashes


def _probabilities_from_true_loss(
    losses: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    true_probability = np.exp(-losses)
    probabilities = np.empty((len(losses), 2), dtype=np.float64)
    probabilities[:, 0] = np.where(
        target == 0, true_probability, 1.0 - true_probability
    )
    probabilities[:, 1] = 1.0 - probabilities[:, 0]
    return probabilities


def test_exact_deletion_dataset_contract_accepts_only_frozen_artifact() -> None:
    dataset = _exact_dataset()
    frozen.validate_deletion_dataset(dataset)

    drifted = ActivationDataset(
        activations=dataset.activations,
        target=dataset.target,
        groups=dataset.groups,
        event_hashes=dataset.event_hashes,
        target_name=dataset.target_name,
        provenance={**dataset.provenance, "layer": 9},
    )
    with pytest.raises(ValueError, match="activation cohort drifted"):
        frozen.validate_deletion_dataset(drifted)


def test_multiclass_anova_ranking_is_train_local_and_deterministic() -> None:
    target = np.repeat(np.arange(3), 4)
    values = np.zeros((len(target), 5), dtype=np.float64)
    values[:, 0] = target
    values[:, 1] = 1.0
    values[:, 2] = np.tile([0.0, 1.0], len(target) // 2)

    first = frozen.multiclass_sparse_anova_ranking(
        sparse.csr_matrix(values), target
    )
    second = frozen.multiclass_sparse_anova_ranking(
        sparse.csr_matrix(values), target
    )

    assert first[0] == 0
    assert np.array_equal(first, second)


def test_writer_splits_have_no_overlap_and_preserve_train_labels() -> None:
    _, target, groups, _ = _synthetic_codes()
    splits = frozen.writer_grouped_splits(target, groups, folds=3, seed=11)

    assert len(splits) == 3
    for train, test in splits:
        assert not set(groups[train]).intersection(groups[test])
        assert set(np.unique(target[train])) == {0, 1, 2}


def test_fixed_txc_controls_reuse_one_ordered_probe_per_fold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    matrices, target, groups, event_hashes = _synthetic_codes()
    original = frozen._fit_probe
    calls = []

    def recording_fit(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(frozen, "_fit_probe", recording_fit)
    result, predictions = frozen.evaluate_code_matrices(
        matrices,
        target,
        groups,
        event_hashes,
        budgets=(3,),
        primary_budget=3,
        folds=3,
        c_value=1.0,
        max_iter=500,
        bootstrap_draws=40,
        seed=17,
        gate_margin=0.02,
    )

    assert len(calls) == 3 * 4
    assert result["outer_folds_effective"] == 3
    for name in frozen.METHODS:
        probability = predictions[f"S3__{name}"]
        assert np.isfinite(probability).all()
        assert np.allclose(probability.sum(axis=1), 1.0)
    envelope = {
        "protocol_version": frozen.PROTOCOL_VERSION,
        "primary_budget": 3,
        "evaluation": result,
    }
    assert "Primary S=3 gate" in frozen._render_markdown(envelope)
    frozen._render_plot(envelope, tmp_path)
    assert (tmp_path / "frozen_dictionary.png").is_file()
    assert (tmp_path / "frozen_dictionary.pdf").is_file()


def test_equal_writer_bootstrap_does_not_upweight_prolific_writer() -> None:
    groups = np.asarray(["prolific"] * 100 + ["singleton"])
    target = np.zeros(len(groups), dtype=np.int8)
    txc_loss = np.asarray([0.1] * 100 + [0.9])
    probabilities = {
        "txc_ordered": _probabilities_from_true_loss(txc_loss, target),
    }
    for index, name in enumerate(frozen.METHODS[1:], start=1):
        probabilities[name] = _probabilities_from_true_loss(
            txc_loss + 0.05 * index,
            target,
        )

    summary = frozen.summarize_equal_writer(
        probabilities,
        target,
        groups,
        (0, 1),
        draws=200,
        seed=3,
    )

    assert summary["method_equal_writer_log_loss"]["txc_ordered"] == pytest.approx(
        0.5
    )
    shuffle = summary["contrasts"]["txc_fixed_shuffle_minus_txc_ordered"]
    assert shuffle["equal_writer_mean_log_loss_difference"] == pytest.approx(0.05)
    assert shuffle["writers_total"] == 2


def test_primary_gate_requires_margin_and_positive_lower_bound() -> None:
    passing = {
        "contrasts": {
            "txc_fixed_shuffle_minus_txc_ordered": {
                "equal_writer_mean_log_loss_difference": 0.03,
                "ci95_lower": 0.01,
            },
            "txc_fixed_reverse_minus_txc_ordered": {
                "equal_writer_mean_log_loss_difference": 0.04,
                "ci95_lower": 0.02,
            },
        },
        "strongest_sae_minus_txc_ordered": {
            "equal_writer_mean_log_loss_difference": 0.02,
            "ci95_lower": 0.001,
        },
    }
    assert frozen.primary_gate(passing, margin=0.02)["passed"] is True

    failing = json.loads(json.dumps(passing))
    failing["contrasts"]["txc_fixed_reverse_minus_txc_ordered"][
        "ci95_lower"
    ] = -0.001
    assert frozen.primary_gate(failing, margin=0.02)["passed"] is False


def test_code_cache_fingerprint_includes_condition_seed_and_refuses_drift(
    tmp_path,
) -> None:
    dataset = _exact_dataset()
    first = frozen._code_fingerprint(dataset, {}, seed=1)
    second = frozen._code_fingerprint(dataset, {}, seed=2)
    assert first != second

    code_dir = tmp_path / "codes"
    code_dir.mkdir()
    (code_dir / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="code cache provenance mismatch"):
        frozen.encode_code_matrices(
            dataset,
            checkpoint_root=tmp_path / "checkpoints",
            code_dir=code_dir,
            checkpoint_records={},
            batch_size=1,
            device="cpu",
            seed=1,
        )
