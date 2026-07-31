"""Step 7: predict future Assistant-Axis drift from local and temporal codes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, r2_score

from experiments.power_spectrum.persona_drift_txc.protocol import (
    EXPERIMENT_ROOT,
    ProbeIndex,
    build_probe_indices,
    config_digest,
    file_sha256,
    future_targets,
    iter_jsonl,
    load_config,
    stack_current,
    stack_user_embeddings,
    stack_windows,
    write_json,
)


def _standardize(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = train.astype(np.float64, copy=False)
    validation = validation.astype(np.float64, copy=False)
    test = test.astype(np.float64, copy=False)
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    std[std < 1e-6] = 1.0
    return (
        (train - mean) / std,
        (validation - mean) / std,
        (test - mean) / std,
    )


def _fit_dual_ridge_multi_target(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_y.ndim == 1:
        train_y = train_y[:, None]
        validation_y = validation_y[:, None]
    train_x, validation_x, test_x = _standardize(
        train_x,
        validation_x,
        test_x,
    )
    target_mean = train_y.mean(axis=0, dtype=np.float64)
    centered_y = train_y.astype(np.float64) - target_mean[None, :]
    gram = np.asarray(train_x @ train_x.T, dtype=np.float64)
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_target = eigenvectors.T @ centered_y
    validation_cross = np.asarray(validation_x @ train_x.T, dtype=np.float64)
    test_cross = np.asarray(test_x @ train_x.T, dtype=np.float64)

    n_targets = train_y.shape[1]
    best_alpha = np.full(n_targets, float(alphas[0]), dtype=np.float64)
    best_validation_r2 = np.full(n_targets, -float("inf"), dtype=np.float64)
    best_coefficients = np.zeros((len(train_x), n_targets), dtype=np.float64)
    for alpha in alphas:
        coefficients = eigenvectors @ (
            projected_target / np.maximum(eigenvalues[:, None] + float(alpha), 1e-12)
        )
        prediction = validation_cross @ coefficients + target_mean[None, :]
        for target_index in range(n_targets):
            score = float(r2_score(validation_y[:, target_index], prediction[:, target_index]))
            if score > best_validation_r2[target_index]:
                best_validation_r2[target_index] = score
                best_alpha[target_index] = float(alpha)
                best_coefficients[:, target_index] = coefficients[:, target_index]
    test_prediction = test_cross @ best_coefficients + target_mean[None, :]
    return (
        test_prediction.astype(np.float32),
        best_alpha,
        best_validation_r2,
    )


def _fit_dual_ridge(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, float, float]:
    test_prediction, best_alpha, best_validation_r2 = _fit_dual_ridge_multi_target(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        test_x=test_x,
        alphas=alphas,
    )
    return (
        test_prediction[:, 0],
        float(best_alpha[0]),
        float(best_validation_r2[0]),
    )


def _metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    breach: np.ndarray,
    *,
    compute_breach_metrics: bool,
) -> dict[str, float]:
    result = {
        "r2": float(r2_score(target, prediction)),
        "rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
    }
    if compute_breach_metrics and len(np.unique(breach)) == 2:
        result["breach_auprc"] = float(average_precision_score(breach, -prediction))
        result["breach_prevalence"] = float(breach.mean())
    elif compute_breach_metrics:
        result["breach_auprc"] = float("nan")
        result["breach_prevalence"] = float(breach.mean())
    else:
        result["breach_auprc"] = float("nan")
        result["breach_prevalence"] = float("nan")
    return result


def _bootstrap_delta(
    *,
    target: np.ndarray,
    breach: np.ndarray,
    local_prediction: np.ndarray,
    temporal_prediction: np.ndarray,
    conversation_ids: np.ndarray,
    domains: np.ndarray,
    repetitions: int,
    seed: int,
    compute_auprc: bool,
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    groups_by_domain = {
        domain: np.unique(conversation_ids[domains == domain]) for domain in np.unique(domains)
    }
    r2_deltas: list[float] = []
    auprc_deltas: list[float] = []
    for _ in range(repetitions):
        sampled = np.concatenate(
            [
                generator.choice(groups, size=len(groups), replace=True)
                for groups in groups_by_domain.values()
            ]
        )
        indices = np.concatenate([np.flatnonzero(conversation_ids == group) for group in sampled])
        if np.var(target[indices]) <= 1e-12:
            continue
        r2_deltas.append(
            float(
                r2_score(target[indices], temporal_prediction[indices])
                - r2_score(target[indices], local_prediction[indices])
            )
        )
        if compute_auprc and len(np.unique(breach[indices])) == 2:
            auprc_deltas.append(
                float(
                    average_precision_score(breach[indices], -temporal_prediction[indices])
                    - average_precision_score(breach[indices], -local_prediction[indices])
                )
            )
    return {
        "delta_r2_mean": float(np.mean(r2_deltas)),
        "delta_r2_ci_low": float(np.quantile(r2_deltas, 0.025)),
        "delta_r2_ci_high": float(np.quantile(r2_deltas, 0.975)),
        "valid_r2_bootstrap_repetitions": len(r2_deltas),
        "delta_auprc_mean": (float(np.mean(auprc_deltas)) if auprc_deltas else float("nan")),
        "delta_auprc_ci_low": (
            float(np.quantile(auprc_deltas, 0.025)) if auprc_deltas else float("nan")
        ),
        "delta_auprc_ci_high": (
            float(np.quantile(auprc_deltas, 0.975)) if auprc_deltas else float("nan")
        ),
        "valid_auprc_bootstrap_repetitions": len(auprc_deltas),
    }


def _select_rows(array: np.ndarray, splits: np.ndarray, split: str) -> np.ndarray:
    return array[splits == split]


def _load_codes(
    representation_root: Path,
    name: str,
    conversation_ids: list[str],
) -> torch.Tensor:
    payload = torch.load(
        representation_root / name / "codes.pt",
        map_location="cpu",
        weights_only=False,
    )
    if list(payload["conversation_ids"]) != conversation_ids:
        raise ValueError(f"{name}: code and activation conversation ordering differs")
    return payload["codes"]


def _code_at_endpoint(
    codes: torch.Tensor,
    rows: list[ProbeIndex],
    *,
    code_window: int,
) -> torch.Tensor:
    values = []
    for row in rows:
        code_index = row.turn if code_window == 1 else row.turn - code_window + 1
        values.append(codes[row.conversation_index, code_index])
    return torch.stack(values)


def _embedding_history(
    embeddings: torch.Tensor,
    rows: list[ProbeIndex],
) -> torch.Tensor:
    return torch.stack(
        [
            embeddings[
                row.conversation_index,
                row.turn - row.window + 1 : row.turn + 1,
            ].flatten()
            for row in rows
        ]
    )


def _future_embedding_oracle(
    embeddings: torch.Tensor,
    rows: list[ProbeIndex],
) -> torch.Tensor:
    return torch.stack(
        [
            embeddings[
                row.conversation_index,
                row.turn + 1 : row.turn + row.horizon + 1,
            ].flatten()
            for row in rows
        ]
    )


def _feature_sets(
    *,
    current_axis: np.ndarray,
    position_context: np.ndarray,
    user: np.ndarray,
    raw_current: np.ndarray,
    raw_history: np.ndarray,
    sae: np.ndarray,
    tsae: np.ndarray,
    txc: np.ndarray,
    user_history: np.ndarray | None = None,
    future_user_oracle: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    axis = current_axis[:, None]
    result = {
        "axis_only": axis,
        "axis_position": np.concatenate((axis, position_context), axis=1),
        "user_axis": np.concatenate((axis, position_context, user), axis=1),
        "raw_local": np.concatenate((axis, position_context, user, raw_current), axis=1),
        "raw_history": np.concatenate((axis, position_context, user, raw_history), axis=1),
        "sae": np.concatenate((axis, position_context, user, sae), axis=1),
        "tsae": np.concatenate((axis, position_context, user, tsae), axis=1),
        "txc": np.concatenate((axis, position_context, user, txc), axis=1),
        "sae_plus_txc": np.concatenate((axis, position_context, user, sae, txc), axis=1),
        "tsae_plus_txc": np.concatenate((axis, position_context, user, tsae, txc), axis=1),
    }
    if user_history is not None and future_user_oracle is not None:
        result.update(
            {
                "past_user_axis": np.concatenate((axis, position_context, user_history), axis=1),
                "future_user_oracle": np.concatenate(
                    (
                        axis,
                        position_context,
                        user_history,
                        future_user_oracle,
                    ),
                    axis=1,
                ),
                "raw_local_user_history": np.concatenate(
                    (axis, position_context, user_history, raw_current),
                    axis=1,
                ),
                "raw_history_user_history": np.concatenate(
                    (axis, position_context, user_history, raw_history),
                    axis=1,
                ),
                "sae_user_history": np.concatenate(
                    (axis, position_context, user_history, sae),
                    axis=1,
                ),
                "sae_plus_txc_user_history": np.concatenate(
                    (axis, position_context, user_history, sae, txc),
                    axis=1,
                ),
            }
        )
    return result


def run_probes(
    *,
    activation_path: Path,
    metadata_path: Path,
    embedding_path: Path,
    representation_root: Path,
    output_root: Path,
) -> None:
    config = load_config()
    activation_payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    activations = activation_payload["activations"]
    axis_scores = activation_payload["axis_scores"]
    conversation_ids = list(activation_payload["conversation_ids"])
    metadata = list(iter_jsonl(metadata_path))
    if conversation_ids != [record["conversation_id"] for record in metadata]:
        raise ValueError("metadata and activation ordering differs")
    embedding_payload = torch.load(embedding_path, map_location="cpu", weights_only=False)
    if list(embedding_payload["conversation_ids"]) != conversation_ids:
        raise ValueError("embedding and activation ordering differs")
    embeddings = embedding_payload["embeddings"]

    normalization_payload = torch.load(
        representation_root / "sae" / "normalization.pt",
        map_location="cpu",
        weights_only=False,
    )
    normalized = (activations.float() - normalization_payload["mean"]) / float(
        normalization_payload["scalar_rms"]
    )
    sae_codes = _load_codes(representation_root, "sae", conversation_ids)
    tsae_codes = _load_codes(representation_root, "tsae", conversation_ids)

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    primary_window = int(config["probe"]["primary_window"])
    primary_horizon = int(config["probe"]["primary_horizon"])
    for window in (4, 8):
        txc_name = f"txc_w{window}"
        txc_codes = _load_codes(representation_root, txc_name, conversation_ids)
        for horizon in config["probe"]["horizons"]:
            rows = build_probe_indices(
                metadata,
                turns_per_conversation=[activations.shape[1]] * len(metadata),
                window=window,
                horizon=int(horizon),
            )
            targets = future_targets(
                axis_scores,
                rows,
                safe_threshold=float(config["safe_threshold"]),
            )
            splits = np.asarray([row.split for row in rows])
            row_conversation_ids = np.asarray([row.conversation_id for row in rows])
            row_domains = np.asarray([row.domain for row in rows])
            current = stack_current(normalized, rows).float().numpy()
            history = stack_windows(normalized, rows).flatten(1).float().numpy()
            user = stack_user_embeddings(embeddings, rows).float().numpy()
            sae = _code_at_endpoint(sae_codes, rows, code_window=1).float().numpy()
            tsae = _code_at_endpoint(tsae_codes, rows, code_window=1).float().numpy()
            txc = _code_at_endpoint(txc_codes, rows, code_window=window).float().numpy()
            domain_names = list(config["domains"])
            position_context = np.asarray(
                [
                    [
                        row.turn / max(int(config["turns_per_conversation"]) - 1, 1),
                        *[float(row.domain == domain) for domain in domain_names[:-1]],
                    ]
                    for row in rows
                ],
                dtype=np.float32,
            )
            is_primary_cell = window == primary_window and int(horizon) == primary_horizon
            features = _feature_sets(
                current_axis=targets["current"],
                position_context=position_context,
                user=user,
                raw_current=current,
                raw_history=history,
                sae=sae,
                tsae=tsae,
                txc=txc,
                user_history=(
                    _embedding_history(embeddings, rows).float().numpy()
                    if is_primary_cell
                    else None
                ),
                future_user_oracle=(
                    _future_embedding_oracle(embeddings, rows).float().numpy()
                    if is_primary_cell
                    else None
                ),
            )

            target_names = ("future_min", "future_delta")
            prediction_maps: dict[str, dict[str, np.ndarray]] = {
                target_name: {} for target_name in target_names
            }
            train_targets = np.column_stack(
                [
                    _select_rows(targets[target_name], splits, "train")
                    for target_name in target_names
                ]
            )
            validation_targets = np.column_stack(
                [
                    _select_rows(targets[target_name], splits, "validation")
                    for target_name in target_names
                ]
            )
            for model_name, design in features.items():
                predictions, alphas, validation_scores = _fit_dual_ridge_multi_target(
                    train_x=_select_rows(design, splits, "train"),
                    train_y=train_targets,
                    validation_x=_select_rows(design, splits, "validation"),
                    validation_y=validation_targets,
                    test_x=_select_rows(design, splits, "test"),
                    alphas=[float(value) for value in config["probe"]["ridge_alphas"]],
                )
                for target_index, target_name in enumerate(target_names):
                    target = targets[target_name]
                    prediction = predictions[:, target_index]
                    test_target = _select_rows(target, splits, "test")
                    test_breach = _select_rows(targets["future_breach"], splits, "test")
                    metric = _metrics(
                        test_target,
                        prediction,
                        test_breach,
                        compute_breach_metrics=target_name == "future_min",
                    )
                    summary_rows.append(
                        {
                            "window": window,
                            "horizon": int(horizon),
                            "target": target_name,
                            "model": model_name,
                            "alpha": float(alphas[target_index]),
                            "validation_r2": float(validation_scores[target_index]),
                            "n_train": int((splits == "train").sum()),
                            "n_validation": int((splits == "validation").sum()),
                            "n_test": int((splits == "test").sum()),
                            **metric,
                        }
                    )
                    prediction_maps[target_name][model_name] = prediction
                    for row, true_value, predicted_value, breach in zip(
                        [row for row in rows if row.split == "test"],
                        test_target,
                        prediction,
                        test_breach,
                        strict=True,
                    ):
                        prediction_rows.append(
                            {
                                **asdict(row),
                                "target": target_name,
                                "model": model_name,
                                "true_value": float(true_value),
                                "prediction": float(predicted_value),
                                "future_breach": int(breach),
                            }
                        )

            for target_name in target_names:
                target = targets[target_name]
                prediction_map = prediction_maps[target_name]
                test_target = _select_rows(target, splits, "test")
                test_breach = _select_rows(targets["future_breach"], splits, "test")
                test_conversations = _select_rows(row_conversation_ids, splits, "test")
                test_domains = _select_rows(row_domains, splits, "test")
                comparisons = [
                    ("raw_local", "raw_history"),
                    ("sae", "sae_plus_txc"),
                    ("tsae", "tsae_plus_txc"),
                    ("sae", "txc"),
                ]
                if is_primary_cell:
                    comparisons.extend(
                        [
                            (
                                "raw_local_user_history",
                                "raw_history_user_history",
                            ),
                            (
                                "sae_user_history",
                                "sae_plus_txc_user_history",
                            ),
                        ]
                    )
                for local_name, temporal_name in comparisons:
                    bootstrap_rows.append(
                        {
                            "window": window,
                            "horizon": int(horizon),
                            "target": target_name,
                            "local_model": local_name,
                            "temporal_model": temporal_name,
                            **_bootstrap_delta(
                                target=test_target,
                                breach=test_breach,
                                local_prediction=prediction_map[local_name],
                                temporal_prediction=prediction_map[temporal_name],
                                conversation_ids=test_conversations,
                                domains=test_domains,
                                repetitions=int(config["probe"]["bootstrap_repetitions"]),
                                seed=(
                                    int(config["probe"]["seed"])
                                    + 100 * window
                                    + 10 * int(horizon)
                                    + len(bootstrap_rows)
                                ),
                                compute_auprc=target_name == "future_min",
                            ),
                        }
                    )

    primary_gate_rows = [
        row
        for row in bootstrap_rows
        if row["target"] == "future_min"
        and int(row["window"]) == primary_window
        and int(row["horizon"]) == primary_horizon
        and (
            (row["local_model"], row["temporal_model"])
            in {
                ("raw_local", "raw_history"),
                ("sae", "sae_plus_txc"),
                ("raw_local_user_history", "raw_history_user_history"),
                ("sae_user_history", "sae_plus_txc_user_history"),
            }
        )
    ]
    gate = {
        "primary_window": primary_window,
        "primary_horizon": primary_horizon,
        "primary_target": "future_min",
        "criterion": "conversation-bootstrap delta R2 95% CI lower bound > 0",
        "conditional_on_single_generation_and_representation_seed": True,
        "comparisons": [
            {
                **row,
                "supported": bool(row["delta_r2_ci_low"] > 0),
            }
            for row in primary_gate_rows
        ],
    }
    gate["initial_decision_passed"] = any(
        comparison["supported"]
        for comparison in gate["comparisons"]
        if (
            comparison["local_model"],
            comparison["temporal_model"],
        )
        in {
            ("raw_local", "raw_history"),
            ("sae", "sae_plus_txc"),
        }
    )

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "probe_summary.csv", summary_rows)
    _write_csv(output_root / "probe_predictions.csv", prediction_rows)
    _write_csv(output_root / "probe_bootstrap_deltas.csv", bootstrap_rows)
    write_json(
        output_root / "probe_summary.json",
        {
            "primary_target": "future_min",
            "primary_window": primary_window,
            "primary_horizon": primary_horizon,
            "safe_threshold": config["safe_threshold"],
            "config_sha256": config_digest(config),
            "input_sha256": {
                "activations": file_sha256(activation_path),
                "metadata": file_sha256(metadata_path),
                "embeddings": file_sha256(embedding_path),
                **{
                    f"{name}_codes": file_sha256(representation_root / name / "codes.pt")
                    for name in ("sae", "tsae", "txc_w4", "txc_w8")
                },
            },
            "gate": gate,
            "summary": summary_rows,
            "bootstrap_deltas": bootstrap_rows,
        },
    )
    write_json(output_root / "step7_gate.json", gate)
    plot_probe_results(summary_rows, bootstrap_rows, output_root)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_probe_results(
    summary_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    models = ("user_axis", "raw_local", "raw_history", "sae", "txc", "sae_plus_txc")
    labels = {
        "user_axis": "Axis + position + user",
        "raw_local": "Raw local",
        "raw_history": "Raw history",
        "sae": "SAE",
        "txc": "TXC",
        "sae_plus_txc": "SAE + TXC",
    }
    colors = {
        "user_axis": "#999999",
        "raw_local": "#56B4E9",
        "raw_history": "#0072B2",
        "sae": "#009E73",
        "txc": "#D55E00",
        "sae_plus_txc": "#CC79A7",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    primary = [row for row in summary_rows if row["target"] == "future_min" and row["window"] == 8]
    horizons = sorted({int(row["horizon"]) for row in primary})
    width = 0.12
    positions = np.arange(len(horizons))
    for offset, model in enumerate(models):
        values = [
            next(
                row["r2"]
                for row in primary
                if row["model"] == model and int(row["horizon"]) == horizon
            )
            for horizon in horizons
        ]
        axes[0].bar(
            positions + (offset - (len(models) - 1) / 2) * width,
            values,
            width=width,
            label=labels[model],
            color=colors[model],
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(positions, [str(value) for value in horizons])
    axes[0].set_xlabel("Future horizon (turns), W=8")
    axes[0].set_ylabel(r"Test $R^2$ for future minimum axis")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].grid(axis="y", alpha=0.2)

    for local_name, temporal_name, label, color in (
        ("raw_local", "raw_history", "Raw history − raw local", "#0072B2"),
        ("sae", "sae_plus_txc", "SAE + TXC − SAE", "#CC79A7"),
    ):
        deltas = [
            row
            for row in bootstrap_rows
            if row["target"] == "future_min"
            and row["window"] == 8
            and row["local_model"] == local_name
            and row["temporal_model"] == temporal_name
        ]
        means = np.asarray([row["delta_r2_mean"] for row in deltas])
        low = np.asarray([row["delta_r2_ci_low"] for row in deltas])
        high = np.asarray([row["delta_r2_ci_high"] for row in deltas])
        axes[1].errorbar(
            [row["horizon"] for row in deltas],
            means,
            yerr=np.vstack((means - low, high - means)),
            marker="o",
            color=color,
            label=label,
            capsize=3,
        )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Future horizon (turns), W=8")
    axes[1].set_ylabel(r"Temporal gain, $\Delta R^2$")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_root / "future_drift_probe.png", dpi=220)
    figure.savefig(output_root / "future_drift_probe.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activations",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts" / "activations" / "turn_activations.pt",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts" / "activations" / "metadata.jsonl",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts" / "user_embeddings.pt",
    )
    parser.add_argument(
        "--representations",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts" / "representations",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_probes(
        activation_path=args.activations,
        metadata_path=args.metadata,
        embedding_path=args.embeddings,
        representation_root=args.representations,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
