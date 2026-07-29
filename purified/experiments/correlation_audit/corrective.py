"""Corrective E0 article-prefix covariance and matrix-spectrum audit.

The public cache stores consecutive 255-token blocks from each WikiText article.
This audit concatenates contiguous blocks sharing an article id so lag pairs may
cross cache-block boundaries, then resamples article ids for uncertainty.  The
result is deliberately called an *article-prefix reconstruction*: each cached
block was evaluated with a fresh BOS/context/position reset and article
remainders were discarded, so it is not an uninterrupted model trajectory.

For each centering mode we estimate signed matrix covariances Gamma(+/- k),
three matrix-norm curves, a Hermitian Bartlett lag-window cross-spectrum, and
article-bootstrap stability of positive Frobenius-norm decay-law fits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.correlation_audit.robustness import (
    apply_projection,
    fit_legacy_projection,
    fixed_directions,
    load_futurelens_cache,
)
from experiments.correlation_audit.run import fit_decay_models


PROTOCOL_VERSION = "0.1.0"
CENTERING_MODES = ("global", "position", "sequence")
LIMITATION = (
    "cached article-prefix reconstruction: source tokens are contiguous across "
    "joined 255-token blocks, but activations were produced with a fresh BOS, "
    "context, and positional reset per block; discarded article remainders cannot "
    "be recovered"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contiguous_article_runs(article_ids: np.ndarray) -> list[np.ndarray]:
    """Return input-order block indices, rejecting noncontiguous repeated ids."""
    ids = np.asarray(article_ids)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("article_ids must be a nonempty vector")
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    runs = [np.arange(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    run_ids = [ids[index[0]].item() for index in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("an article id occurs in multiple noncontiguous runs")
    return runs


def center_blocks(blocks: np.ndarray, mode: str) -> np.ndarray:
    """Center projected blocks globally, by block position, or within sequence."""
    blocks = np.asarray(blocks, dtype=np.float32)
    if blocks.ndim != 3:
        raise ValueError(f"expected blocks=(N,T,D), got {blocks.shape}")
    if mode == "global":
        mean = blocks.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        return blocks - mean
    if mode == "position":
        mean = blocks.mean(axis=0, dtype=np.float64).astype(np.float32)
        return blocks - mean[None]
    if mode == "sequence":
        mean = blocks.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
        return blocks - mean
    raise ValueError(f"unknown centering mode {mode!r}")


def reconstruct_article_prefixes(
    centered_blocks: np.ndarray, article_ids: np.ndarray
) -> tuple[list[np.ndarray], list[int]]:
    """Join consecutive cached blocks while preserving article and token order."""
    runs = contiguous_article_runs(article_ids)
    articles = [np.concatenate(centered_blocks[index], axis=0) for index in runs]
    ids = [int(np.asarray(article_ids)[index[0]]) for index in runs]
    return articles, ids


def article_sufficient_statistics(
    articles: list[np.ndarray],
    *,
    max_lag: int,
    device: str = "cpu",
    batch_articles: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-article cross-products and pair counts for lags 0..K.

    Padding is zero, so batched matrix products include no cross-article pairs.
    The returned float32 tensor is the expensive sufficient statistic reused by
    every bootstrap draw; aggregation and division use float64.
    """
    if not articles:
        raise ValueError("at least one article is required")
    dimension = articles[0].shape[1]
    if any(article.ndim != 2 or article.shape[1] != dimension for article in articles):
        raise ValueError("all articles must have shape (tokens, shared_dimension)")
    max_lag = min(max_lag, max(len(article) for article in articles) - 1)
    cross = np.zeros(
        (len(articles), max_lag + 1, dimension, dimension), dtype=np.float32
    )
    counts = np.zeros((len(articles), max_lag + 1), dtype=np.int64)

    import torch

    torch_device = torch.device(device)
    order = np.argsort([len(article) for article in articles])
    for start in range(0, len(order), batch_articles):
        indices = order[start : start + batch_articles]
        lengths = np.asarray([len(articles[index]) for index in indices], dtype=np.int64)
        longest = int(lengths.max())
        padded = np.zeros((len(indices), longest, dimension), dtype=np.float32)
        for row, index in enumerate(indices):
            padded[row, : lengths[row]] = articles[index]
        values = torch.from_numpy(padded).to(torch_device)
        for lag in range(max_lag + 1):
            available = np.maximum(lengths - lag, 0)
            counts[indices, lag] = available
            if not np.any(available):
                continue
            if lag == 0:
                left, right = values, values
            else:
                left, right = values[:, :-lag], values[:, lag:]
            # Right-side padding is zero exactly where a row has no valid pair.
            products = torch.bmm(left.transpose(1, 2), right)
            cross[indices, lag] = products.float().cpu().numpy()
        del values
    return cross, counts


def aggregate_gamma(cross: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Pair-weighted signed Gamma(+k) from per-article sufficient statistics."""
    total_counts = counts.sum(axis=0, dtype=np.int64)
    if np.any(total_counts <= 0):
        raise ValueError("one or more requested lags have no valid pairs")
    total_cross = cross.sum(axis=0, dtype=np.float64)
    return total_cross / total_counts[:, None, None]


def matrix_norm_curves(gamma: np.ndarray) -> dict[str, np.ndarray]:
    """Return signed-matrix Frobenius, operator, and nuclear norm curves."""
    singular = np.linalg.svd(gamma, compute_uv=False)
    return {
        "frobenius": np.linalg.norm(gamma, axis=(1, 2)),
        "operator": singular[:, 0],
        "nuclear": singular.sum(axis=1),
    }


def signed_direction_curves(
    gamma: np.ndarray, *, seed: int
) -> dict[str, list[float]]:
    """Signed normalized autocovariances along fixed non-lag-selected directions."""
    names, directions = fixed_directions(gamma.shape[-1], seed=seed)
    rows = {}
    for name, direction in zip(names, directions):
        values = np.einsum("i,kij,j->k", direction, gamma, direction)
        rows[name] = (values / max(float(values[0]), 1e-12)).tolist()
    return rows


def hermitian_lag_window_spectrum(
    cross: np.ndarray,
    counts: np.ndarray,
    *,
    n_frequencies: int = 65,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bartlett lag-window matrix spectrum using common-denominator covariances.

    The common lag-zero denominator and Bartlett taper yield the finite-sample
    cross-periodogram smoothing analogue.  Both +/- lags enter explicitly and
    the result is symmetrized to remove floating-point anti-Hermitian residue.
    """
    total_cross = cross.sum(axis=0, dtype=np.float64)
    denominator = float(counts[:, 0].sum())
    biased = total_cross / max(denominator, 1.0)
    max_lag = len(biased) - 1
    frequencies = np.linspace(0.0, math.pi, n_frequencies)
    spectra = np.empty(
        (n_frequencies, biased.shape[-1], biased.shape[-1]), dtype=np.complex128
    )
    for frequency_index, omega in enumerate(frequencies):
        value = biased[0].astype(np.complex128)
        for lag in range(1, max_lag + 1):
            weight = 1.0 - lag / (max_lag + 1.0)
            phase = np.exp(-1j * omega * lag)
            value += weight * (biased[lag] * phase + biased[lag].T * phase.conjugate())
        spectra[frequency_index] = 0.5 * (value + value.conj().T) / (2 * math.pi)
    eigenvalues = np.linalg.eigvalsh(spectra)
    return frequencies, spectra, eigenvalues


def corrected_decay_fits(lags: np.ndarray, curve: np.ndarray) -> list[dict]:
    """Refit/rank decay laws with residual variance included in AICc's k."""
    lags = np.asarray(lags, dtype=np.float64)
    curve = np.asarray(curve, dtype=np.float64)
    rows = fit_decay_models(lags, curve)
    n = int(np.sum(np.isfinite(curve) & (curve > 0) & (lags > 0)))
    corrected = []
    for row in rows:
        copy = dict(row)
        curve_parameters = len(row["params"])
        k = curve_parameters + 1  # Gaussian log-residual variance.
        rss = max(float(row["log_rmse"]) ** 2 * n, 1e-30)
        aic = n * math.log(rss / n) + 2 * k
        aicc = (
            aic + 2 * k * (k + 1) / (n - k - 1)
            if n > k + 1
            else math.inf
        )
        copy.update(
            {
                "legacy_aicc": row["aicc"],
                "aic": float(aic),
                "aicc": float(aicc),
                "aicc_parameter_count": k,
            }
        )
        corrected.append(copy)
    corrected.sort(key=lambda row: row["aicc"])
    return corrected


def bootstrap_decay_stability(
    cross: np.ndarray,
    counts: np.ndarray,
    *,
    repeats: int,
    seed: int,
    device: str = "cpu",
    draw_batch: int = 20,
) -> tuple[dict, np.ndarray]:
    """Article-bootstrap Frobenius decay fits from cached sufficient statistics."""
    if repeats < 1:
        return {"repeats": 0}, np.empty((0, cross.shape[1]))
    rng = np.random.default_rng(seed)
    n_articles, n_lags, dimension, _ = cross.shape
    weights = rng.multinomial(
        n_articles, np.full(n_articles, 1.0 / n_articles), size=repeats
    ).astype(np.float32)

    import torch

    torch_device = torch.device(device)
    cross_flat = torch.from_numpy(cross.reshape(n_articles, -1)).to(torch_device)
    count_tensor = torch.from_numpy(counts.astype(np.float32)).to(torch_device)
    curves = []
    for start in range(0, repeats, draw_batch):
        weight = torch.from_numpy(weights[start : start + draw_batch]).to(torch_device)
        numerator = (weight @ cross_flat).reshape(-1, n_lags, dimension, dimension)
        denominator = (weight @ count_tensor).clamp_min(1.0)
        gamma = numerator / denominator[:, :, None, None]
        curve = torch.linalg.matrix_norm(gamma, ord="fro", dim=(-2, -1))
        curve = curve / curve[:, 1:2].clamp_min(1e-12)
        curves.append(curve.cpu().numpy())
    curve_array = np.concatenate(curves, axis=0).astype(np.float64)
    del cross_flat, count_tensor

    lags = np.arange(2, n_lags)
    winners: list[str] = []
    alphas: list[float] = []
    for curve in curve_array:
        fits = corrected_decay_fits(lags, curve[2:])
        if not fits:
            continue
        winners.append(fits[0]["model"])
        power = next((row for row in fits if row["model"] == "power"), None)
        if power is not None:
            alphas.append(float(power["named_params"]["alpha"]))
    winner_counts = Counter(winners)
    alpha = np.asarray(alphas, dtype=np.float64)
    return (
        {
            "unit": "source_article_id",
            "estimand_weighting": "token-pair weighted within each resampled article set",
            "repeats": int(len(curve_array)),
            "winner_counts": dict(winner_counts),
            "winner_fractions": {
                name: count / max(len(winners), 1) for name, count in winner_counts.items()
            },
            "pure_power_alpha": {
                "median": float(np.median(alpha)) if len(alpha) else None,
                "q025": float(np.quantile(alpha, 0.025)) if len(alpha) else None,
                "q975": float(np.quantile(alpha, 0.975)) if len(alpha) else None,
            },
            "frobenius_curve_q025": np.quantile(curve_array, 0.025, axis=0).tolist(),
            "frobenius_curve_q975": np.quantile(curve_array, 0.975, axis=0).tolist(),
        },
        curve_array,
    )


def write_curve_table(layer: int, modes: dict, output: Path) -> None:
    rows = []
    for mode, result in modes.items():
        for lag in range(len(result["norms"]["frobenius"])):
            rows.append(
                {
                    "layer": layer,
                    "centering": mode,
                    "lag": lag,
                    "pair_count": result["pair_counts"][lag],
                    "frobenius": result["norms"]["frobenius"][lag],
                    "frobenius_normalized": result["norms"]["frobenius_normalized"][lag],
                    "operator": result["norms"]["operator"][lag],
                    "nuclear": result["norms"]["nuclear"][lag],
                    "frobenius_boot_q025": result["bootstrap"]["frobenius_curve_q025"][lag],
                    "frobenius_boot_q975": result["bootstrap"]["frobenius_curve_q975"][lag],
                }
            )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_layer(layer: int, modes: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"global": "#5E81AC", "position": "#A3BE8C", "sequence": "#D08770"}
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    for mode, result in modes.items():
        lag = np.arange(len(result["norms"]["frobenius"]))
        curve = np.asarray(result["norms"]["frobenius_normalized"])
        lower = np.asarray(result["bootstrap"]["frobenius_curve_q025"])
        upper = np.asarray(result["bootstrap"]["frobenius_curve_q975"])
        axes[0].loglog(lag[1:], curve[1:], label=mode, color=colors[mode])
        axes[0].fill_between(lag[1:], lower[1:], upper[1:], color=colors[mode], alpha=0.15)
        spectrum = result["spectrum_summary"]
        axes[2].plot(
            np.asarray(spectrum["frequency_radians"]) / (2 * math.pi),
            spectrum["trace"],
            label=mode,
            color=colors[mode],
        )
    axes[0].set(xlabel="lag", ylabel="normalized Frobenius norm", title="Article-prefix covariance")
    axes[0].legend(frameon=False)

    model_names = sorted(
        {name for result in modes.values() for name in result["bootstrap"]["winner_fractions"]}
    )
    x = np.arange(len(model_names))
    width = 0.24
    for index, (mode, result) in enumerate(modes.items()):
        values = [result["bootstrap"]["winner_fractions"].get(name, 0.0) for name in model_names]
        axes[1].bar(x + (index - 1) * width, values, width, label=mode, color=colors[mode])
    axes[1].set_xticks(x, [name.replace("_", "\n") for name in model_names])
    axes[1].set(ylabel="article-bootstrap winner fraction", title="Decay-family stability")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].set(
        xlabel="cycles / token",
        ylabel="trace matrix spectrum",
        title="Hermitian Bartlett cross-spectrum",
    )
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(f"E0 corrective article-prefix audit: GPT-2 layer {layer}")
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _jsonable_mode_result(
    *,
    gamma: np.ndarray,
    counts: np.ndarray,
    norms: dict[str, np.ndarray],
    directions: dict[str, list[float]],
    frequencies: np.ndarray,
    spectra: np.ndarray,
    eigenvalues: np.ndarray,
    bootstrap: dict,
) -> dict:
    normalized = {
        name: (curve / max(float(curve[1]), 1e-12)).tolist()
        for name, curve in norms.items()
    }
    full_fits = corrected_decay_fits(np.arange(2, len(gamma)), normalized["frobenius"][2:])
    antihermitian = np.max(np.abs(spectra - spectra.conj().transpose(0, 2, 1)))
    return {
        "pair_counts": counts.sum(axis=0).tolist(),
        "norms": {
            **{name: curve.tolist() for name, curve in norms.items()},
            **{f"{name}_normalized": values for name, values in normalized.items()},
        },
        "signed_fixed_direction_rho": directions,
        "frobenius_decay_fits": full_fits,
        "frobenius_aicc_winner": full_fits[0]["model"] if full_fits else None,
        "bootstrap": bootstrap,
        "spectrum_summary": {
            "method": "common-denominator Bartlett lag-window matrix cross-spectrum",
            "frequency_radians": frequencies.tolist(),
            "trace": np.trace(spectra, axis1=1, axis2=2).real.tolist(),
            "minimum_eigenvalue": float(eigenvalues.min()),
            "maximum_eigenvalue": float(eigenvalues.max()),
            "maximum_antihermitian_residual": float(antihermitian),
        },
        "gamma_minus_definition": "Gamma(-k) = Gamma(+k)^T on the identical pair set",
    }


def run_layer(args: argparse.Namespace, layer: int) -> dict:
    activations, article_ids, cache_provenance = load_futurelens_cache(args.cache_dir, layer)
    projection = fit_legacy_projection(
        activations,
        dimension=args.projection_dim,
        fit_tokens=args.fit_tokens,
        fit_documents=args.fit_blocks,
        seed=args.seed,
    )
    projected = apply_projection(activations, projection, device=args.device)
    modes = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode_index, mode in enumerate(CENTERING_MODES):
        print(f"layer={layer} centering={mode}: reconstructing articles", flush=True)
        centered = center_blocks(projected, mode)
        articles, reconstructed_ids = reconstruct_article_prefixes(centered, article_ids)
        cross, counts = article_sufficient_statistics(
            articles,
            max_lag=args.max_lag,
            device=args.device,
            batch_articles=args.batch_articles,
        )
        gamma = aggregate_gamma(cross, counts)
        gamma_minus = gamma.transpose(0, 2, 1)
        norms = matrix_norm_curves(gamma)
        directions = signed_direction_curves(gamma, seed=args.seed)
        frequencies, spectra, eigenvalues = hermitian_lag_window_spectrum(
            cross, counts, n_frequencies=args.spectrum_frequencies
        )
        bootstrap, bootstrap_curves = bootstrap_decay_stability(
            cross,
            counts,
            repeats=args.bootstrap,
            seed=args.seed + 10_000 * layer + mode_index,
            device=args.device,
            draw_batch=args.bootstrap_batch,
        )
        np.savez_compressed(
            args.output_dir / f"layer_{layer}_{mode}_matrices.npz",
            gamma_plus=gamma,
            gamma_minus=gamma_minus,
            frequency_radians=frequencies,
            spectrum_real=spectra.real,
            spectrum_imag=spectra.imag,
            spectrum_eigenvalues=eigenvalues,
            bootstrap_frobenius_curves=bootstrap_curves,
            article_ids=np.asarray(reconstructed_ids),
            pair_counts=counts.sum(axis=0),
        )
        modes[mode] = _jsonable_mode_result(
            gamma=gamma,
            counts=counts,
            norms=norms,
            directions=directions,
            frequencies=frequencies,
            spectra=spectra,
            eigenvalues=eigenvalues,
            bootstrap=bootstrap,
        )
        del centered, articles, cross, counts, gamma, gamma_minus, spectra, eigenvalues

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "estimand": (
            "token-pair-weighted signed covariance of legacy-PCA64 GPT-2 residuals "
            "within cached article-prefix reconstructions; article ids are the "
            "bootstrap unit"
        ),
        "unavoidable_limitation": LIMITATION,
        "provenance": {
            **cache_provenance,
            "layer": layer,
            "activation_shape_after_bos_removal": list(activations.shape),
            "projection": "frozen legacy first-1000-block PCA whitening",
            "projection_dim": args.projection_dim,
            "fit_tokens": min(args.fit_tokens, args.fit_blocks * activations.shape[1]),
            "fit_blocks": min(args.fit_blocks, len(activations)),
            "max_lag": args.max_lag,
            "article_count": int(len(np.unique(article_ids))),
            "centering_modes": list(CENTERING_MODES),
            "bootstrap_repeats": args.bootstrap,
            "corrective_source_sha256": sha256_file(Path(__file__)),
            "seed": args.seed,
        },
        "modes": modes,
    }
    (args.output_dir / f"layer_{layer}_corrective.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_curve_table(layer, modes, args.output_dir / f"layer_{layer}_corrective_curves.csv")
    plot_layer(layer, modes, args.output_dir / f"layer_{layer}_corrective.png")
    return result


def write_summary(results: list[dict], output: Path) -> None:
    lines = [
        "# E0 corrective article-prefix audit",
        "",
        f"**Estimand.** {results[0]['estimand']}",
        "",
        f"**Unavoidable limitation.** {results[0]['unavoidable_limitation']}",
        "",
        "| Layer | Centering | AICc winner | Bootstrap modal winner | Modal fraction | Pure-power alpha 95% article interval | Min spectral eigenvalue |",
        "|---:|---|---|---|---:|---|---:|",
    ]
    for result in results:
        layer = result["provenance"]["layer"]
        for mode, row in result["modes"].items():
            fractions = row["bootstrap"]["winner_fractions"]
            modal = max(fractions, key=fractions.get) if fractions else "none"
            alpha = row["bootstrap"]["pure_power_alpha"]
            lines.append(
                f"| {layer} | {mode} | {row['frobenius_aicc_winner']} | {modal} | "
                f"{fractions.get(modal, 0):.3f} | [{alpha['q025']:.3f}, {alpha['q975']:.3f}] | "
                f"{row['spectrum_summary']['minimum_eigenvalue']:.3e} |"
            )
    lines.extend(["", "## Diagnostics", ""])
    for result in results:
        layer = result["provenance"]["layer"]
        lines.append(f"![Layer {layer} corrective diagnostics](layer_{layer}_corrective.png)")
        lines.append("")
    lines.append(
        "AICc here uses a Gaussian log-residual working model and counts residual "
        "variance as a fitted parameter. Article bootstrapping measures corpus "
        "sampling stability, but it does not make lag residuals independent."
    )
    output.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--fit-tokens", type=int, default=60_000)
    parser.add_argument("--fit-blocks", type=int, default=1_000)
    parser.add_argument("--max-lag", type=int, default=48)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--bootstrap-batch", type=int, default=20)
    parser.add_argument("--batch-articles", type=int, default=16)
    parser.add_argument("--spectrum-frequencies", type=int, default=65)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> list[dict]:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_layer(args, layer) for layer in args.layers]
    write_summary(results, args.output_dir / "summary.md")
    print(
        json.dumps(
            {
                "status": "ok",
                "layers": args.layers,
                "output_dir": str(args.output_dir),
                "limitation": LIMITATION,
            },
            indent=2,
        ),
        flush=True,
    )
    return results


if __name__ == "__main__":
    main()
