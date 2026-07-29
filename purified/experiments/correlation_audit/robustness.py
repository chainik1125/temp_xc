"""E0 robustness audit for the residual-stream power-law observation.

This module intentionally keeps three estimators separate:

``legacy_same_data``
    Reproduces the tensor-network-futurelens Exp 16D construction: PCA-64
    whitening, a second C(0) whitening, persistent directions estimated from
    the same long-lag covariance matrices, Frobenius norm, and lag-1
    normalization.

``no_removal``
    Applies the same whitening but leaves the persistent sector intact.

``crossfit_removal``
    Estimates whitening and persistent directions on one document split and
    evaluates the scalar norm curve on the held-out split, then swaps folds.

The audit also measures signed scalar autocovariances along directions fixed
without looking at lagged outcomes, computes a PSD directly with Welch's
method, and checks absolute-position stationarity.  A matrix norm is useful as
a dependence diagnostic, but it is not a covariance sequence and is never
Fourier transformed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import welch

from experiments.correlation_audit.run import fit_decay_models


@dataclass(frozen=True)
class Projection:
    """A fixed affine PCA-whitening map."""

    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray


def _safe_invsqrt(matrix: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, eps) ** -0.5) @ vectors.T


def fit_legacy_projection(
    activations: np.ndarray,
    *,
    dimension: int = 64,
    fit_tokens: int = 60_000,
    fit_documents: int = 1_000,
    seed: int = 0,
) -> Projection:
    """Fit the Exp 16D PCA-whitening map on the first cache shard.

    The original experiment sampled at most 60k positions from its first
    1,000-document shard, then used a full SVD.  Keeping that choice explicit
    is important because fitting the projection on every document is a subtle
    change to the legacy estimator.
    """
    import torch

    source = np.asarray(
        activations[: min(fit_documents, len(activations))], dtype=np.float32
    ).reshape(-1, activations.shape[-1])
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(source), generator=generator)[: min(fit_tokens, len(source))]
    sample = torch.from_numpy(source)[order].to(torch.float64)
    mean = sample.mean(dim=0)
    centered = sample - mean
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:dimension]
    eigenvalues = singular[:dimension].square() / max(len(sample) - 1, 1)
    scale = (eigenvalues + 1e-5).rsqrt()
    return Projection(
        mean=mean.cpu().numpy(),
        components=components.cpu().numpy(),
        scale=scale.cpu().numpy(),
    )


def apply_projection(
    activations: np.ndarray,
    projection: Projection,
    *,
    batch_size: int = 16_384,
    device: str = "cuda",
) -> np.ndarray:
    """Project a cache without materializing a float64 activation copy."""
    import torch

    flat = activations.reshape(-1, activations.shape[-1])
    out = np.empty((len(flat), len(projection.scale)), dtype=np.float32)
    mean = torch.from_numpy(projection.mean.astype(np.float32)).to(device)
    components = torch.from_numpy(projection.components.astype(np.float32)).to(device)
    scale = torch.from_numpy(projection.scale.astype(np.float32)).to(device)
    for start in range(0, len(flat), batch_size):
        batch = torch.from_numpy(np.asarray(flat[start : start + batch_size], dtype=np.float32))
        batch = batch.to(device)
        out[start : start + len(batch)] = (
            ((batch - mean) @ components.T) * scale
        ).cpu().numpy()
    return out.reshape(*activations.shape[:2], -1)


def covariance_sequence(
    z: np.ndarray,
    *,
    max_lag: int,
    documents: np.ndarray | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute legacy globally-centered C(lag), including lag zero."""
    if z.ndim != 3:
        raise ValueError(f"expected (documents, positions, channels), got {z.shape}")
    docs = np.arange(len(z)) if documents is None else np.asarray(documents)
    max_lag = min(max_lag, z.shape[1] - 1)
    if device == "cpu":
        selected = np.asarray(z[docs], dtype=np.float64)
        mean = selected.mean(axis=(0, 1))
        covariances = np.empty((max_lag + 1, selected.shape[-1], selected.shape[-1]))
        counts = np.empty(max_lag + 1, dtype=np.int64)
        for lag in range(max_lag + 1):
            left = selected if lag == 0 else selected[:, :-lag]
            right = selected if lag == 0 else selected[:, lag:]
            left = left.reshape(-1, selected.shape[-1])
            right = right.reshape(-1, selected.shape[-1])
            covariances[lag] = left.T @ right / len(left) - np.outer(mean, mean)
            counts[lag] = len(left)
        return covariances, counts

    import torch

    selected = torch.from_numpy(np.asarray(z[docs], dtype=np.float32)).to(device)
    mean = selected.mean(dim=(0, 1))
    outer = torch.outer(mean, mean)
    covariances = []
    counts = []
    for lag in range(max_lag + 1):
        left = selected if lag == 0 else selected[:, :-lag]
        right = selected if lag == 0 else selected[:, lag:]
        left = left.reshape(-1, selected.shape[-1])
        right = right.reshape(-1, selected.shape[-1])
        covariances.append(left.T @ right / len(left) - outer)
        counts.append(len(left))
    return torch.stack(covariances).cpu().numpy(), np.asarray(counts)


def whiten_covariances(
    reference: np.ndarray, measured: np.ndarray | None = None
) -> np.ndarray:
    """Whiten ``measured`` using C(0) from the reference split."""
    whitener = _safe_invsqrt(reference[0])
    target = reference if measured is None else measured
    return np.einsum("ij,ljk,km->lim", whitener, target, whitener)


def persistent_basis(
    whitened_covariances: np.ndarray,
    *,
    rank: int,
    tail_start: int | None = None,
) -> np.ndarray:
    """Legacy long-lag eigenspace, ranked by absolute eigenvalue."""
    if rank == 0:
        return np.empty((whitened_covariances.shape[-1], 0))
    start = len(whitened_covariances) // 2 if tail_start is None else tail_start
    tail = whitened_covariances[start:]
    symmetric = (0.5 * (tail + tail.transpose(0, 2, 1))).mean(axis=0)
    values, vectors = np.linalg.eigh(symmetric)
    return vectors[:, np.argsort(-np.abs(values))[:rank]]


def norm_curve(
    whitened_covariances: np.ndarray,
    basis: np.ndarray | None = None,
    *,
    normalize_lag: int = 1,
) -> np.ndarray:
    """Frobenius norm after optional two-sided subspace removal."""
    if basis is None or basis.shape[1] == 0:
        residual = whitened_covariances
    else:
        projector = np.eye(whitened_covariances.shape[-1]) - basis @ basis.T
        residual = np.einsum("ij,ljk,km->lim", projector, whitened_covariances, projector)
    curve = np.linalg.norm(residual, axis=(1, 2))
    return curve / max(curve[normalize_lag], 1e-12)


def legacy_two_model_fits(curve: np.ndarray, *, dmin: int = 2) -> dict:
    """Reproduce Exp 16D's two straight-line fits exactly."""
    distances = np.arange(dmin, len(curve), dtype=np.float64)
    response = np.log(np.maximum(curve[dmin:], 1e-12))
    rows = {}
    for name, predictor in (
        ("exp", distances),
        ("power", np.log(distances)),
    ):
        design = np.stack([predictor, np.ones_like(predictor)], axis=1)
        coefficient, *_ = np.linalg.lstsq(design, response, rcond=None)
        prediction = design @ coefficient
        residual = float(np.sum((response - prediction) ** 2))
        total = float(np.sum((response - response.mean()) ** 2))
        rows[name] = {
            "slope": float(coefficient[0]),
            "r2": float(1 - residual / max(total, 1e-12)),
            "aic": float(len(response) * np.log(max(residual / len(response), 1e-12)) + 4),
            "parameter": float(
                -1 / coefficient[0]
                if name == "exp" and coefficient[0] < 0
                else -coefficient[0]
            ),
        }
    rows["winner"] = min(("exp", "power"), key=lambda name: rows[name]["aic"])
    return rows


def compare_removal_estimators(
    z: np.ndarray,
    *,
    max_lag: int,
    persistent_rank: int,
    seed: int,
    group_ids: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """Compute no-removal, same-data legacy, and two-fold cross-fitted curves."""
    all_cov, counts = covariance_sequence(z, max_lag=max_lag, device=device)
    all_white = whiten_covariances(all_cov)
    no_removal = norm_curve(all_white)

    groups = np.arange(len(z)) if group_ids is None else np.asarray(group_ids)
    if len(groups) != len(z):
        raise ValueError("group_ids must have one entry per activation sequence")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("cross-fitting requires at least two distinct document groups")
    shuffled_groups = np.random.default_rng(seed).permutation(unique_groups)
    group_halves = np.array_split(shuffled_groups, 2)
    halves = [np.flatnonzero(np.isin(groups, group_half)) for group_half in group_halves]
    fold_covariances = []
    for train, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        train_cov, _ = covariance_sequence(
            z, max_lag=max_lag, documents=train, device=device
        )
        test_cov, _ = covariance_sequence(
            z, max_lag=max_lag, documents=test, device=device
        )
        train_white = whiten_covariances(train_cov)
        test_white = whiten_covariances(train_cov, test_cov)
        fold_covariances.append((train_white, test_white))

    lags = np.arange(2, max_lag + 1)

    def summarize(curve: np.ndarray) -> dict:
        fits = fit_decay_models(lags, curve[2:])
        return {
            "curve": curve.tolist(),
            "fits": fits,
            "legacy_two_model_fits": legacy_two_model_fits(curve),
            "aicc_winner": fits[0]["model"] if fits else None,
            "tail_rmse_winner": min(
                (row for row in fits if row["tail_log_rmse"] is not None),
                key=lambda row: row["tail_log_rmse"],
                default={},
            ).get("model"),
        }

    def at_rank(rank: int) -> dict:
        legacy_basis = persistent_basis(all_white, rank=rank)
        legacy = norm_curve(all_white, legacy_basis)
        fold_curves = []
        for train_white, test_white in fold_covariances:
            basis = persistent_basis(train_white, rank=rank)
            fold_curves.append(norm_curve(test_white, basis))
        crossfit_folds = np.stack(fold_curves)
        crossfit = crossfit_folds.mean(axis=0)
        return {
            "legacy_same_data": summarize(legacy),
            "crossfit_removal": {
                **summarize(crossfit),
                "fold_curves": crossfit_folds.tolist(),
                "fold_max_abs_difference": float(
                    np.max(np.abs(crossfit_folds[0] - crossfit_folds[1]))
                ),
                "fold_ranks": [rank, rank],
            },
        }

    primary = at_rank(persistent_rank)
    sensitivity = {
        str(rank): at_rank(rank)
        for rank in sorted({4, persistent_rank, 16})
        if rank < z.shape[-1]
    }
    return {
        "lags": np.arange(max_lag + 1).tolist(),
        "counts": counts.tolist(),
        "no_removal": summarize(no_removal),
        **primary,
        "crossfit_grouping": "article" if group_ids is not None else "sequence",
        "crossfit_group_count": int(len(unique_groups)),
        "persistent_rank_sensitivity": sensitivity,
    }


def fixed_directions(dimension: int, *, n_pca: int = 4, n_random: int = 4, seed: int = 0):
    """Directions fixed without consulting lagged covariance outcomes."""
    names: list[str] = []
    vectors: list[np.ndarray] = []
    for index in range(min(n_pca, dimension)):
        vector = np.zeros(dimension)
        vector[index] = 1.0
        names.append(f"pca_{index}")
        vectors.append(vector)
    random = np.random.default_rng(seed).standard_normal((dimension, n_random))
    q, _ = np.linalg.qr(random)
    for index in range(n_random):
        names.append(f"random_{index}")
        vectors.append(q[:, index])
    return names, np.stack(vectors)


def signed_direction_audit(
    covariances: np.ndarray,
    names: list[str],
    directions: np.ndarray,
) -> dict:
    """Signed normalized autocovariances and positive-curve model fits."""
    rows = {}
    lags = np.arange(2, len(covariances))
    for name, direction in zip(names, directions):
        values = np.einsum("i,lij,j->l", direction, covariances, direction)
        rho = values / max(values[0], 1e-12)
        selected = rho[2:]
        positive_fraction = float(np.mean(selected > 0))
        fits = fit_decay_models(lags, selected) if positive_fraction >= 0.9 else []
        rows[name] = {
            "rho": rho.tolist(),
            "positive_fraction_lags_2_plus": positive_fraction,
            "sign_changes": int(np.sum(np.signbit(selected[1:]) != np.signbit(selected[:-1]))),
            "fits": fits,
            "aicc_winner": fits[0]["model"] if fits else None,
        }
    return {"lags": np.arange(len(covariances)).tolist(), "directions": rows}


def direct_psd_audit(
    z: np.ndarray,
    names: list[str],
    directions: np.ndarray,
    *,
    nperseg: int = 128,
    bootstrap: int = 200,
    seed: int = 0,
    group_ids: np.ndarray | None = None,
) -> dict:
    """Direct Welch PSD with document-bootstrap low-frequency slope uncertainty."""
    projected = np.einsum("ntp,kp->ntk", z.astype(np.float64, copy=False), directions)
    nperseg = min(nperseg, projected.shape[1])
    frequencies, per_doc = welch(
        projected,
        fs=1.0,
        axis=1,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    # scipy returns (document, direction, frequency) for axis=1.
    if per_doc.shape[-1] != len(frequencies):
        per_doc = np.moveaxis(per_doc, 1, -1)
    mean_psd = per_doc.mean(axis=0)
    low = np.arange(1, min(13, len(frequencies)))

    def beta(values: np.ndarray) -> float:
        slope, _ = np.polyfit(
            np.log(frequencies[low]), np.log(np.maximum(values[low], 1e-30)), 1
        )
        return float(-slope)

    rng = np.random.default_rng(seed)
    groups = np.arange(len(z)) if group_ids is None else np.asarray(group_ids)
    unique_groups = np.unique(groups)
    rows = {}
    for index, name in enumerate(names):
        boot = []
        for _ in range(bootstrap):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            sample = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
            boot.append(beta(per_doc[sample, index].mean(axis=0)))
        rows[name] = {
            "psd": mean_psd[index].tolist(),
            "low_frequency_beta": beta(mean_psd[index]),
            "beta_q025": float(np.quantile(boot, 0.025)),
            "beta_q975": float(np.quantile(boot, 0.975)),
        }
    return {
        "method": "welch_by_document",
        "bootstrap_unit": "article" if group_ids is not None else "sequence",
        "bootstrap_group_count": int(len(unique_groups)),
        "nperseg": nperseg,
        "frequencies": frequencies.tolist(),
        "directions": rows,
    }


def stationarity_audit(
    z: np.ndarray,
    directions: np.ndarray,
    *,
    max_lag: int = 16,
    device: str = "cpu",
) -> dict:
    """Compare means, zero-lag covariance, and signed curves by position third."""
    edges = np.linspace(0, z.shape[1], 4, dtype=int)
    overall_cov, _ = covariance_sequence(z, max_lag=0, device=device)
    scale = np.sqrt(max(float(np.trace(overall_cov[0])), 1e-12))
    bins = {}
    means = []
    zero_covariances = []
    signed_curves = []
    for index in range(3):
        start, end = int(edges[index]), int(edges[index + 1])
        section = z[:, start:end]
        covariance, _ = covariance_sequence(
            section, max_lag=min(max_lag, section.shape[1] - 1), device=device
        )
        means.append(section.mean(axis=(0, 1)))
        zero_covariances.append(covariance[0])
        variances = np.einsum("ki,ij,kj->k", directions, covariance[0], directions)
        rho = np.einsum("ki,lij,kj->kl", directions, covariance, directions)
        rho = rho / np.maximum(variances[:, None], 1e-12)
        median_rho = np.median(rho, axis=0)
        signed_curves.append(median_rho)
        bins[f"position_third_{index}"] = {
            "start": start,
            "end": end,
            "mean_norm": float(np.linalg.norm(means[-1])),
            "covariance_trace": float(np.trace(covariance[0])),
            "median_signed_rho": median_rho.tolist(),
        }
    mean_drift = max(
        np.linalg.norm(left - right) / scale
        for left in means
        for right in means
    )
    covariance_drift = max(
        np.linalg.norm(left - right) / max(np.linalg.norm(overall_cov[0]), 1e-12)
        for left in zero_covariances
        for right in zero_covariances
    )
    stack = np.stack(signed_curves)
    curve_spread = np.max(stack, axis=0) - np.min(stack, axis=0)
    return {
        "bins": bins,
        "standardized_max_mean_drift": float(mean_drift),
        "relative_max_covariance_drift": float(covariance_drift),
        "max_signed_curve_spread_lags_1_plus": float(np.max(curve_spread[:,][1:])),
    }


def _sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def load_futurelens_cache(
    cache_dir: Path, layer: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one layer from the public tensor-network-futurelens shard format."""
    import torch

    paths = sorted(cache_dir.glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no shard_*.pt files in {cache_dir}")
    chunks = []
    article_ids = []
    metadata = None
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if layer not in shard["residuals"]:
            raise KeyError(f"layer {layer} absent from {path}")
        # Exp 16D discarded BOS before every correlation measurement.
        chunks.append(shard["residuals"][layer][:, 1:, :].numpy())
        article_ids.append(shard["article_ids"].numpy())
        metadata = shard["meta"]
    activations = np.concatenate(chunks, axis=0)
    groups = np.concatenate(article_ids)
    return activations, groups, {
        "cache_dir": str(cache_dir),
        "shards": [path.name for path in paths],
        "cache_sha256": _sha256_files(paths),
        "article_count": int(len(np.unique(groups))),
        "cache_metadata": metadata,
    }


def plot_audit(result: dict, output: Path) -> None:
    """Write one four-panel figure containing every claim-bearing diagnostic."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    removal = result["removal_estimators"]
    lag = np.asarray(removal["lags"])
    for key, label in (
        ("no_removal", "no removal"),
        ("legacy_same_data", "legacy same-data removal"),
        ("crossfit_removal", "cross-fitted removal"),
    ):
        axes[0, 0].loglog(lag[1:], np.asarray(removal[key]["curve"])[1:], label=label)
    axes[0, 0].set(title="Matrix-norm dependence diagnostic", xlabel="lag", ylabel="normalized Frobenius norm")
    axes[0, 0].legend(fontsize=8)

    signed = result["signed_covariance"]
    signed_lag = np.asarray(signed["lags"])
    for name, row in signed["directions"].items():
        axes[0, 1].plot(signed_lag[1:], np.asarray(row["rho"])[1:], alpha=0.75, label=name)
    axes[0, 1].axhline(0, color="black", linewidth=0.7)
    axes[0, 1].set(title="Signed fixed-direction covariance", xlabel="lag", ylabel=r"$\rho_u(\ell)$")
    axes[0, 1].legend(fontsize=7, ncol=2)

    psd = result["direct_psd"]
    frequencies = np.asarray(psd["frequencies"])
    for name, row in psd["directions"].items():
        axes[1, 0].loglog(frequencies[1:], np.asarray(row["psd"])[1:], alpha=0.75, label=name)
    axes[1, 0].set(title="Direct Welch PSD", xlabel="cycles/token", ylabel="spectral density")
    axes[1, 0].legend(fontsize=7, ncol=2)

    for name, row in result["stationarity"]["bins"].items():
        curve = np.asarray(row["median_signed_rho"])
        axes[1, 1].plot(np.arange(1, len(curve)), curve[1:], label=name.replace("position_", ""))
    axes[1, 1].axhline(0, color="black", linewidth=0.7)
    axes[1, 1].set(title="Position-stationarity check", xlabel="lag", ylabel="median signed correlation")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle(f"E0 correlation robustness audit: layer {result['provenance']['layer']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_fit_table(result: dict, output: Path) -> None:
    """Write a compact table for direct inclusion in the live research log."""
    rows = []
    for estimator in ("no_removal", "legacy_same_data", "crossfit_removal"):
        data = result["removal_estimators"][estimator]
        winner = data["fits"][0] if data["fits"] else {}
        power = next((row for row in data["fits"] if row["model"] == "power"), {})
        rows.append(
            {
                "layer": result["provenance"]["layer"],
                "estimator": estimator,
                "aicc_winner": data["aicc_winner"],
                "tail_rmse_winner": data["tail_rmse_winner"],
                "winner_aicc": winner.get("aicc"),
                "pure_power_alpha": power.get("named_params", {}).get("alpha"),
                "pure_power_aicc": power.get("aicc"),
                "pure_power_tail_log_rmse": power.get("tail_log_rmse"),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_spectral_table(result: dict, output: Path) -> None:
    """Pair signed-covariance exponents with independently measured PSD slopes."""
    rows = []
    signed = result["signed_covariance"]["directions"]
    psd = result["direct_psd"]["directions"]
    for name, signed_row in signed.items():
        power = next(
            (row for row in signed_row["fits"] if row["model"] == "power"), None
        )
        alpha = power["named_params"]["alpha"] if power is not None else None
        beta = psd[name]["low_frequency_beta"]
        rows.append(
            {
                "layer": result["provenance"]["layer"],
                "direction": name,
                "signed_positive_fraction": signed_row["positive_fraction_lags_2_plus"],
                "signed_aicc_winner": signed_row["aicc_winner"],
                "signed_power_alpha": alpha,
                "direct_psd_beta": beta,
                "direct_psd_beta_q025": psd[name]["beta_q025"],
                "direct_psd_beta_q975": psd[name]["beta_q975"],
                "tauberian_beta_if_valid": 1 - alpha if alpha is not None else None,
                "beta_minus_tauberian": beta - (1 - alpha) if alpha is not None else None,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--fit-tokens", type=int, default=60_000)
    parser.add_argument("--fit-documents", type=int, default=1_000)
    parser.add_argument("--max-lag", type=int, default=48)
    parser.add_argument("--persistent-rank", type=int, default=8)
    parser.add_argument("--psd-bootstrap", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    activations, article_ids, cache_provenance = load_futurelens_cache(
        args.cache_dir, args.layer
    )
    projection = fit_legacy_projection(
        activations,
        dimension=args.projection_dim,
        fit_tokens=args.fit_tokens,
        fit_documents=args.fit_documents,
        seed=args.seed,
    )
    z = apply_projection(activations, projection, device=args.device)
    covariance, _ = covariance_sequence(
        z, max_lag=args.max_lag, device=args.device
    )
    names, directions = fixed_directions(args.projection_dim, seed=args.seed)
    result = {
        "provenance": {
            **cache_provenance,
            "layer": args.layer,
            "activation_shape_after_bos_removal": list(activations.shape),
            "projection": "legacy_first_shard_full_svd_pca_whitening",
            "projection_dim": args.projection_dim,
            "fit_tokens": min(args.fit_tokens, args.fit_documents * activations.shape[1]),
            "fit_documents": min(args.fit_documents, len(activations)),
            "max_lag": args.max_lag,
            "persistent_rank": args.persistent_rank,
            "seed": args.seed,
        },
        "removal_estimators": compare_removal_estimators(
            z,
            max_lag=args.max_lag,
            persistent_rank=args.persistent_rank,
            seed=args.seed,
            group_ids=article_ids,
            device=args.device,
        ),
        "signed_covariance": signed_direction_audit(covariance, names, directions),
        "direct_psd": direct_psd_audit(
            z,
            names,
            directions,
            bootstrap=args.psd_bootstrap,
            seed=args.seed,
            group_ids=article_ids,
        ),
        "stationarity": stationarity_audit(z, directions, device=args.device),
    }
    winners = [
        row["aicc_winner"]
        for row in result["signed_covariance"]["directions"].values()
        if row["aicc_winner"] is not None
    ]
    result["summary"] = {
        "legacy_aicc_winner": result["removal_estimators"]["legacy_same_data"]["aicc_winner"],
        "crossfit_aicc_winner": result["removal_estimators"]["crossfit_removal"]["aicc_winner"],
        "signed_direction_aicc_winners": dict(Counter(winners)),
        "signed_directions_fit": len(winners),
        "signed_directions_total": len(names),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"layer_{args.layer}"
    json_path = args.output_dir / f"{stem}.json"
    figure_path = args.output_dir / f"{stem}.png"
    table_path = args.output_dir / f"{stem}_fits.csv"
    spectral_table_path = args.output_dir / f"{stem}_spectral.csv"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    plot_audit(result, figure_path)
    write_fit_table(result, table_path)
    write_spectral_table(result, spectral_table_path)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "figure": str(figure_path),
                "table": str(table_path),
                "spectral_table": str(spectral_table_path),
                "summary": result["summary"],
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    main()
