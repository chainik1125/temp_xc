"""Grouped covariance-whitened raw-activation gate for C7 backtracking.

This is the theory-native companion to :mod:`experiments.swr_audit.run`.  It
fits a shrinkage Gaussian matched filter to an ordered event-aligned window
and compares it with the best validation-selected single offset and an exact
permutation-invariant mean of the same offsets.  Every preprocessing and
model-fit operation is contained in the outer training groups.

The output is a compact JSON document with fold scores, a prompt-clustered
bootstrap interval for ``G_order``, the event-triggered mean waveform, and a
short-window estimate of the task-conditioned spectral discriminability.
The spectral estimate is diagnostic at T<=6, not evidence of a continuum
power law or a well-resolved frequency band.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from experiments.swr_audit.run import (
    FoldPreprocessor,
    _shuffle_rows,
    c7_groups,
    trailing_window,
)


PROTOCOL_VERSION = "0.1.0"


@dataclass
class MatchedFilter:
    weight: np.ndarray
    intercept: float
    calibrator: LogisticRegression | None
    d_squared: float

    def logits(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x @ self.weight + self.intercept, dtype=np.float64)

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        logits = self.logits(x)
        if self.calibrator is None:
            return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return self.calibrator.predict_proba(logits[:, None])[:, 1]


def fit_matched_filter(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> MatchedFilter:
    """Fit shrinkage LDA on ``fit`` and a monotone calibrator on ``val``."""
    if x_fit.ndim != 2:
        raise ValueError(f"expected a matrix, got {x_fit.shape}")
    if set(np.unique(y_fit)) != {0, 1}:
        raise ValueError("matched-filter fit requires both classes")
    mu0 = x_fit[y_fit == 0].mean(axis=0, dtype=np.float64)
    mu1 = x_fit[y_fit == 1].mean(axis=0, dtype=np.float64)
    residual = np.asarray(x_fit, dtype=np.float64).copy()
    residual[y_fit == 0] -= mu0
    residual[y_fit == 1] -= mu1
    covariance = LedoitWolf(assume_centered=True).fit(residual).covariance_
    delta = mu1 - mu0
    weight = np.linalg.solve(covariance, delta)
    prior = np.clip(float(y_fit.mean()), 1e-6, 1.0 - 1e-6)
    intercept = float(-0.5 * (mu1 + mu0) @ weight + math.log(prior / (1 - prior)))
    val_logits = np.asarray(x_val @ weight + intercept, dtype=np.float64)
    calibrator: LogisticRegression | None = None
    if len(np.unique(y_val)) == 2:
        calibrator = LogisticRegression(C=1e3, solver="lbfgs").fit(
            val_logits[:, None], y_val
        )
    return MatchedFilter(
        weight=np.asarray(weight, dtype=np.float64),
        intercept=intercept,
        calibrator=calibrator,
        d_squared=float(delta @ weight),
    )


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
    }


def _inner_split(
    train_idx: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    for attempt in range(100):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=seed + attempt
        )
        fit_rel, val_rel = next(
            splitter.split(train_idx, y[train_idx], groups=groups[train_idx])
        )
        fit_idx, val_idx = train_idx[fit_rel], train_idx[val_rel]
        if len(np.unique(y[fit_idx])) == 2 and len(np.unique(y[val_idx])) == 2:
            return fit_idx, val_idx
    raise RuntimeError("could not construct a grouped inner split with both classes")


def phase_scramble(x: np.ndarray, seed: int) -> np.ndarray:
    """Randomize non-DC phase while preserving each channel's FFT magnitude."""
    if x.shape[1] <= 2:
        return x.copy()
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(x, axis=1)
    stop = spectrum.shape[1] - 1 if x.shape[1] % 2 == 0 else spectrum.shape[1]
    phase = rng.uniform(-np.pi, np.pi, size=(x.shape[0], stop - 1, x.shape[2]))
    spectrum[:, 1:stop] = np.abs(spectrum[:, 1:stop]) * np.exp(1j * phase)
    return np.fft.irfft(spectrum, n=x.shape[1], axis=1).astype(np.float32)


def task_spectrum(
    z_fit: np.ndarray, y_fit: np.ndarray, ridge_fraction: float = 0.1
) -> dict:
    """Estimate short-window ``Delta_hat^* S^-1 Delta_hat`` on fit data."""
    mu0 = z_fit[y_fit == 0].mean(axis=0, dtype=np.float64)
    mu1 = z_fit[y_fit == 1].mean(axis=0, dtype=np.float64)
    delta = mu1 - mu0
    residual = np.asarray(z_fit, dtype=np.float64).copy()
    residual[y_fit == 0] -= mu0
    residual[y_fit == 1] -= mu1
    scale = math.sqrt(z_fit.shape[1])
    residual_fft = np.fft.rfft(residual, axis=1) / scale
    delta_fft = np.fft.rfft(delta, axis=0) / scale
    j_values = []
    for frequency_index in range(delta_fft.shape[0]):
        values = residual_fft[:, frequency_index]
        covariance = values.conj().T @ values / max(len(values) - 1, 1)
        diagonal_scale = max(float(np.trace(covariance).real / len(covariance)), 1e-8)
        regularized = covariance + ridge_fraction * diagonal_scale * np.eye(
            len(covariance)
        )
        component = delta_fft[frequency_index]
        value = float(np.real(component.conj() @ np.linalg.solve(regularized, component)))
        is_nyquist = z_fit.shape[1] % 2 == 0 and frequency_index == len(delta_fft) - 1
        multiplier = 1.0 if frequency_index == 0 or is_nyquist else 2.0
        j_values.append(max(multiplier * value, 0.0))
    total = max(float(np.sum(j_values)), 1e-12)
    return {
        "frequency_cycles_per_window_step": np.fft.rfftfreq(z_fit.shape[1]).tolist(),
        "j_y": j_values,
        "j_y_fraction": (np.asarray(j_values) / total).tolist(),
        "event_waveform_l2": np.linalg.norm(delta, axis=1).tolist(),
    }


def run_fold(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    fold: int,
    window: int,
    artifact_offsets: tuple[int, ...],
    normalization: str,
    pca_dim: int,
    pca_sample_tokens: int,
    seed: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    x_window = trailing_window(x, window)
    offsets = artifact_offsets[-window:]
    preprocessor = FoldPreprocessor(
        normalization, pca_dim, seed + fold, pca_sample_tokens
    ).fit(x_window[train_idx])
    z = preprocessor.transform(x_window)
    fit_idx, val_idx = _inner_split(train_idx, y, groups, seed + fold)

    ordered = fit_matched_filter(
        z[fit_idx].reshape(len(fit_idx), -1),
        y[fit_idx],
        z[val_idx].reshape(len(val_idx), -1),
        y[val_idx],
    )
    invariant = fit_matched_filter(
        z[fit_idx].mean(axis=1),
        y[fit_idx],
        z[val_idx].mean(axis=1),
        y[val_idx],
    )
    best_offset = 0
    best_offset_val_ap = -math.inf
    best_token: MatchedFilter | None = None
    for offset_index in range(window):
        candidate = fit_matched_filter(
            z[fit_idx, offset_index],
            y[fit_idx],
            z[val_idx, offset_index],
            y[val_idx],
        )
        val_ap = average_precision_score(
            y[val_idx], candidate.probabilities(z[val_idx, offset_index])
        )
        if val_ap > best_offset_val_ap:
            best_offset_val_ap = float(val_ap)
            best_offset = offset_index
            best_token = candidate
    assert best_token is not None

    ordered_test = z[test_idx].reshape(len(test_idx), -1)
    probabilities = {
        "ordered": ordered.probabilities(ordered_test),
        "invariant_mean": invariant.probabilities(z[test_idx].mean(axis=1)),
        "best_token": best_token.probabilities(z[test_idx, best_offset]),
    }
    controls = {}
    for control_index, mode in enumerate(("shuffle", "reverse", "circular")):
        controlled = _shuffle_rows(
            z[test_idx], seed + 10_000 * (fold + 1) + control_index, mode
        )
        controls[mode] = ordered.probabilities(controlled.reshape(len(test_idx), -1))
    controlled = phase_scramble(z[test_idx], seed + 20_000 * (fold + 1))
    controls["phase_scramble"] = ordered.probabilities(
        controlled.reshape(len(test_idx), -1)
    )

    scores = {name: _metrics(y[test_idx], value) for name, value in probabilities.items()}
    control_scores = {name: _metrics(y[test_idx], value) for name, value in controls.items()}
    baseline_ap = max(scores["invariant_mean"]["pr_auc"], scores["best_token"]["pr_auc"])
    spectral = task_spectrum(z[fit_idx], y[fit_idx])
    row = {
        "fold": fold,
        "window": window,
        "window_offsets": list(offsets),
        "normalization": normalization,
        "pca_dim_actual": int(z.shape[-1]),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_test_groups": int(len(np.unique(groups[test_idx]))),
        "test_positive_rate": float(y[test_idx].mean()),
        "best_offset": int(offsets[best_offset]),
        "best_offset_validation_pr_auc": best_offset_val_ap,
        "ordered": scores["ordered"],
        "invariant_mean": scores["invariant_mean"],
        "best_token": scores["best_token"],
        "controls": control_scores,
        "g_order_pr_auc": float(scores["ordered"]["pr_auc"] - baseline_ap),
        "order_gap_pr_auc": {
            name: float(scores["ordered"]["pr_auc"] - score["pr_auc"])
            for name, score in control_scores.items()
        },
        "matched_d_squared": ordered.d_squared,
        "task_spectrum": spectral,
    }
    predictions = {
        **probabilities,
        **{f"control_{name}": value for name, value in controls.items()},
        "y": y[test_idx],
        "groups": groups[test_idx],
    }
    return row, predictions


def _bootstrap_mean_g_order(
    folds: list[dict[str, np.ndarray]], repeats: int, seed: int
) -> dict:
    if repeats < 1:
        return {"repeats": 0, "lower_95": None, "upper_95": None}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        fold_gaps = []
        for fold in folds:
            unique_groups = np.unique(fold["groups"])
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            indices = np.concatenate(
                [np.flatnonzero(fold["groups"] == group) for group in sampled]
            )
            y_boot = fold["y"][indices]
            if len(np.unique(y_boot)) < 2:
                continue
            ordered_ap = average_precision_score(y_boot, fold["ordered"][indices])
            baseline_ap = max(
                average_precision_score(y_boot, fold["invariant_mean"][indices]),
                average_precision_score(y_boot, fold["best_token"][indices]),
            )
            fold_gaps.append(float(ordered_ap - baseline_ap))
        if fold_gaps:
            values.append(float(np.mean(fold_gaps)))
    if not values:
        raise RuntimeError("all cluster bootstrap replicates were single-class")
    array = np.asarray(values)
    return {
        "repeats": int(len(array)),
        "lower_95": float(np.quantile(array, 0.025)),
        "median": float(np.median(array)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def _mean_metric(rows: list[dict], model: str, metric: str) -> dict:
    values = np.asarray([row[model][metric] for row in rows], dtype=np.float64)
    return {
        "fold_values": values.tolist(),
        "mean": float(values.mean()),
        "std_sample": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def summarize_configuration(
    rows: list[dict], predictions: list[dict[str, np.ndarray]], repeats: int, seed: int
) -> dict:
    g_order = np.asarray([row["g_order_pr_auc"] for row in rows])
    frequencies = rows[0]["task_spectrum"]["frequency_cycles_per_window_step"]
    j_fractions = np.asarray(
        [row["task_spectrum"]["j_y_fraction"] for row in rows], dtype=np.float64
    )
    waveform = np.asarray(
        [row["task_spectrum"]["event_waveform_l2"] for row in rows], dtype=np.float64
    )
    return {
        "window": rows[0]["window"],
        "window_offsets": rows[0]["window_offsets"],
        "normalization": rows[0]["normalization"],
        "folds": rows,
        "ordered_pr_auc": _mean_metric(rows, "ordered", "pr_auc"),
        "invariant_mean_pr_auc": _mean_metric(rows, "invariant_mean", "pr_auc"),
        "best_token_pr_auc": _mean_metric(rows, "best_token", "pr_auc"),
        "g_order_pr_auc": {
            "fold_values": g_order.tolist(),
            "mean": float(g_order.mean()),
            "std_sample": float(g_order.std(ddof=1)) if len(g_order) > 1 else 0.0,
            "cluster_bootstrap": _bootstrap_mean_g_order(predictions, repeats, seed),
        },
        "best_offsets": [row["best_offset"] for row in rows],
        "task_spectrum": {
            "frequency_cycles_per_window_step": frequencies,
            "mean_j_y_fraction": j_fractions.mean(axis=0).tolist(),
            "std_j_y_fraction": j_fractions.std(axis=0, ddof=1).tolist(),
            "mean_event_waveform_l2": waveform.mean(axis=0).tolist(),
            "std_event_waveform_l2": waveform.std(axis=0, ddof=1).tolist(),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", default="1,3,5,6")
    parser.add_argument("--artifact-offsets", default="-13,-12,-11,-10,-9,-8")
    parser.add_argument("--normalizations", default="raw")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--pca-sample-tokens", type=int, default=50_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=1_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with np.load(args.artifact, allow_pickle=True) as payload:
        x = payload["X"].astype(np.float32, copy=False)
        y = payload["is_bt"].astype(np.int64, copy=False)
        groups = c7_groups(payload["keys"])
    offsets = _csv_ints(args.artifact_offsets)
    if len(offsets) != x.shape[1]:
        raise ValueError(
            f"--artifact-offsets has {len(offsets)} entries but X has T={x.shape[1]}"
        )
    if args.max_rows is not None and args.max_rows < len(x):
        rng = np.random.default_rng(args.seed)
        keep = []
        for group in rng.permutation(np.unique(groups)):
            keep.extend(np.flatnonzero(groups == group).tolist())
            if len(keep) >= args.max_rows:
                break
        keep_array = np.asarray(sorted(keep), dtype=np.int64)
        x, y, groups = x[keep_array], y[keep_array], groups[keep_array]

    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    splits = list(splitter.split(x, y, groups))
    summaries = []
    for window in _csv_ints(args.windows):
        for normalization in args.normalizations.split(","):
            rows = []
            predictions = []
            for fold, (train_idx, test_idx) in enumerate(splits):
                row, fold_predictions = run_fold(
                    x,
                    y,
                    groups,
                    train_idx,
                    test_idx,
                    fold=fold,
                    window=window,
                    artifact_offsets=offsets,
                    normalization=normalization,
                    pca_dim=args.pca_dim,
                    pca_sample_tokens=args.pca_sample_tokens,
                    seed=args.seed,
                )
                rows.append(row)
                predictions.append(fold_predictions)
                print(json.dumps(row, sort_keys=True), flush=True)
            summaries.append(
                summarize_configuration(
                    rows,
                    predictions,
                    args.bootstrap_repeats,
                    args.seed + 100_000 * window,
                )
            )

    result = {
        "schema_version": "1.0.0",
        "protocol_version": PROTOCOL_VERSION,
        "interpretation": (
            "supervised covariance-whitened raw-activation upper bound; positive "
            "G_order is necessary but not sufficient for unsupervised TXC recovery"
        ),
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": _sha256(args.artifact),
        "artifact_offsets": list(offsets),
        "n_rows": int(len(x)),
        "n_groups": int(len(np.unique(groups))),
        "positive_rate": float(y.mean()),
        "unavailable_from_artifact": [
            "lexical or sentence-length residualization",
            "onset jitter beyond the six cached offsets",
            "strictly wider pre-onset windows",
        ],
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "summaries": len(summaries)}))


if __name__ == "__main__":
    main()
