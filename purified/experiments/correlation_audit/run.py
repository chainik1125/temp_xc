"""Audit whether residual-stream lag correlations support a power-law claim.

This is deliberately a diagnostic rather than a TempBench leaderboard cell: it
does not train an architecture, and raw/cache paths are supplied explicitly so
large artifacts and results can remain outside Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


MODEL_SPECS = {
    "power": (2, lambda p, d: np.exp(p[0]) * d ** (-p[1])),
    "exp": (2, lambda p, d: np.exp(p[0] - p[1] * d)),
    "power_floor": (3, lambda p, d: np.exp(p[0]) * d ** (-p[1]) + np.exp(p[2])),
    "exp_floor": (3, lambda p, d: np.exp(p[0] - p[1] * d) + np.exp(p[2])),
    "stretched_floor": (
        4,
        lambda p, d: np.exp(p[0] - (d / np.exp(p[1])) ** p[2]) + np.exp(p[3]),
    ),
    "power_cutoff_floor": (
        4,
        lambda p, d: np.exp(p[0]) * d ** (-p[1]) * np.exp(-d / np.exp(p[2]))
        + np.exp(p[3]),
    ),
}


def _named_params(name: str, params: np.ndarray) -> dict[str, float]:
    """Translate optimizer coordinates into scientifically readable parameters."""
    named = {"amplitude": float(np.exp(params[0]))}
    if name.startswith("power"):
        named["alpha"] = float(params[1])
    elif name == "exp" or name == "exp_floor":
        named["rate"] = float(params[1])
        named["xi"] = float(1.0 / max(params[1], 1e-12))
    if name == "power_floor" or name == "exp_floor":
        named["floor"] = float(np.exp(params[2]))
    elif name == "stretched_floor":
        named.update(
            {
                "xi": float(np.exp(params[1])),
                "beta": float(params[2]),
                "floor": float(np.exp(params[3])),
            }
        )
    elif name == "power_cutoff_floor":
        named.update({"xi": float(np.exp(params[2])), "floor": float(np.exp(params[3]))})
    return named


def _model_init(
    name: str, d: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_amp = float(np.log(max(y[0], 1e-12)))
    log_floor = float(np.log(max(y[-1] * 0.5, 1e-12)))
    if name == "power":
        return np.array([log_amp, 0.5]), np.array([-40.0, 0.0]), np.array([40.0, 8.0])
    if name == "exp":
        return np.array([log_amp, 0.05]), np.array([-40.0, 0.0]), np.array([40.0, 4.0])
    if name == "power_floor":
        return (
            np.array([log_amp, 0.5, log_floor]),
            np.array([-40.0, 0.0, -40.0]),
            np.array([40.0, 8.0, 40.0]),
        )
    if name == "exp_floor":
        return (
            np.array([log_amp, 0.05, log_floor]),
            np.array([-40.0, 0.0, -40.0]),
            np.array([40.0, 4.0, 40.0]),
        )
    if name == "stretched_floor":
        return (
            np.array([log_amp, np.log(np.median(d)), 0.7, log_floor]),
            np.array([-40.0, -5.0, 0.05, -40.0]),
            np.array([40.0, 12.0, 4.0, 40.0]),
        )
    return (
        np.array([log_amp, 0.5, np.log(max(d[-1], 2.0)), log_floor]),
        np.array([-40.0, 0.0, -5.0, -40.0]),
        np.array([40.0, 8.0, 12.0, 40.0]),
    )


def fit_decay_models(lags: np.ndarray, curve: np.ndarray) -> list[dict]:
    """Fit competing positive decay laws in log space and rank by AICc."""
    keep = np.isfinite(curve) & (curve > 0) & (lags > 0)
    d = np.asarray(lags[keep], dtype=np.float64)
    y = np.asarray(curve[keep], dtype=np.float64)
    if len(d) < 5:
        return []
    rows = []
    for name, (n_params, fn) in MODEL_SPECS.items():
        p0, lo, hi = _model_init(name, d, y)
        fit = least_squares(
            lambda p: np.log(np.maximum(fn(p, d), 1e-30)) - np.log(y),
            p0,
            bounds=(lo, hi),
            max_nfev=20_000,
        )
        residual = np.log(np.maximum(fn(fit.x, d), 1e-30)) - np.log(y)
        rss = max(float(residual @ residual), 1e-30)
        n = len(d)
        aic = n * np.log(rss / n) + 2 * n_params
        if n > n_params + 1:
            aicc = aic + 2 * n_params * (n_params + 1) / (n - n_params - 1)
        else:
            aicc = np.inf
        split = max(n_params + 2, int(np.ceil(0.67 * n)))
        train_fit = least_squares(
            lambda p: np.log(np.maximum(fn(p, d[:split]), 1e-30)) - np.log(y[:split]),
            fit.x,
            bounds=(lo, hi),
            max_nfev=20_000,
        )
        holdout = np.log(np.maximum(fn(train_fit.x, d[split:]), 1e-30)) - np.log(y[split:])
        rows.append(
            {
                "model": name,
                "aicc": float(aicc),
                "log_rmse": float(np.sqrt(rss / n)),
                "tail_log_rmse": float(np.sqrt(np.mean(holdout**2))) if holdout.size else None,
                "params": fit.x.tolist(),
                "named_params": _named_params(name, fit.x),
            }
        )
    rows.sort(key=lambda row: row["aicc"])
    return rows


def center_projected(z: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """Center projected activations while respecting padding masks."""
    out = np.asarray(z, dtype=np.float32).copy()
    if mode == "global":
        out -= out[mask].mean(axis=0)
    elif mode == "position":
        for pos in range(out.shape[1]):
            valid = mask[:, pos]
            if valid.any():
                out[:, pos] -= out[valid, pos].mean(axis=0)
    elif mode == "sequence":
        denom = np.maximum(mask.sum(axis=1, keepdims=True), 1)
        means = (out * mask[..., None]).sum(axis=1) / denom
        out -= means[:, None, :]
    else:
        raise ValueError(f"unknown centering mode: {mode}")
    return out


def lag_covariance(
    z: np.ndarray,
    mask: np.ndarray,
    lag: int,
    docs: np.ndarray,
    *,
    endpoint_centering: bool,
) -> tuple[np.ndarray, int]:
    pair_mask = mask[docs, :-lag] & mask[docs, lag:]
    left = z[docs, :-lag][pair_mask]
    right = z[docs, lag:][pair_mask]
    if len(left) == 0:
        return np.full((z.shape[-1], z.shape[-1]), np.nan), 0
    if endpoint_centering:
        left = left - left.mean(axis=0)
        right = right - right.mean(axis=0)
    return left.T @ right / len(left), len(left)


def _document_block_stats(
    z: np.ndarray,
    mask: np.ndarray,
    lags: np.ndarray,
    *,
    n_blocks: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sufficient statistics for a clustered document bootstrap.

    Blocks are random partitions of documents. Resampling blocks preserves
    complete sequences and lets each bootstrap reproduce endpoint centering
    without materializing a document-by-lag-by-P-by-P tensor.
    """
    docs = np.random.default_rng(seed).permutation(z.shape[0])
    blocks = np.array_split(docs, min(n_blocks, len(docs)))
    shape = (len(blocks), len(lags))
    counts = np.zeros(shape, dtype=np.int64)
    sum_left = np.zeros((*shape, z.shape[-1]), dtype=np.float64)
    sum_right = np.zeros_like(sum_left)
    cross = np.zeros((*shape, z.shape[-1], z.shape[-1]), dtype=np.float64)
    for block_i, block in enumerate(blocks):
        for lag_i, lag in enumerate(lags):
            pair_mask = mask[block, :-lag] & mask[block, lag:]
            left = z[block, :-lag][pair_mask].astype(np.float64, copy=False)
            right = z[block, lag:][pair_mask].astype(np.float64, copy=False)
            counts[block_i, lag_i] = len(left)
            if len(left):
                sum_left[block_i, lag_i] = left.sum(axis=0)
                sum_right[block_i, lag_i] = right.sum(axis=0)
                cross[block_i, lag_i] = left.T @ right
    return counts, sum_left, sum_right, cross


def _covariance_from_blocks(
    indices: np.ndarray,
    lag_i: int,
    stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    endpoint_centering: bool,
) -> np.ndarray:
    counts, sum_left, sum_right, cross = stats
    count = int(counts[indices, lag_i].sum())
    if count == 0:
        return np.full(cross.shape[-2:], np.nan)
    left = sum_left[indices, lag_i].sum(axis=0)
    right = sum_right[indices, lag_i].sum(axis=0)
    total_cross = cross[indices, lag_i].sum(axis=0)
    if endpoint_centering:
        total_cross -= np.outer(left, right) / count
    return total_cross / count


def _bootstrap_half_pools(n_blocks: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed disjoint block pools for split-half noise debiasing."""
    if n_blocks < 2:
        raise ValueError("document bootstrap needs at least two blocks")
    blocks = np.random.default_rng(seed).permutation(n_blocks)
    first, second = np.array_split(blocks, 2)
    return first, second


def document_bootstrap(
    z: np.ndarray,
    mask: np.ndarray,
    lags: np.ndarray,
    *,
    n_bootstrap: int,
    n_blocks: int,
    seed: int,
    endpoint_centering: bool,
) -> dict:
    """Estimate fit uncertainty by resampling randomized document blocks."""
    if n_bootstrap <= 0:
        return {}
    stats = _document_block_stats(z, mask, lags, n_blocks=n_blocks, seed=seed)
    n_blocks = stats[0].shape[0]
    rng = np.random.default_rng(seed + 1)
    first_pool, second_pool = _bootstrap_half_pools(n_blocks, seed=seed + 2)
    best_models: list[str] = []
    power_alphas: list[float] = []
    curves = []
    for _ in range(n_bootstrap):
        first = rng.choice(first_pool, size=len(first_pool), replace=True)
        second = rng.choice(second_pool, size=len(second_pool), replace=True)
        curve = []
        for lag_i in range(len(lags)):
            left = _covariance_from_blocks(
                first, lag_i, stats, endpoint_centering=endpoint_centering
            )
            right = _covariance_from_blocks(
                second, lag_i, stats, endpoint_centering=endpoint_centering
            )
            curve.append(float(np.sqrt(max(float(np.sum(left * right)), 0.0))))
        curve_array = np.asarray(curve)
        curves.append(curve_array)
        fits = fit_decay_models(lags, curve_array)
        if fits:
            best_models.append(fits[0]["model"])
            power = next((row for row in fits if row["model"] == "power"), None)
            if power is not None:
                power_alphas.append(power["named_params"]["alpha"])
    unique, model_counts = np.unique(best_models, return_counts=True)
    alpha = np.asarray(power_alphas)
    curve_stack = np.asarray(curves)
    return {
        "method": "randomized_document_block_bootstrap",
        "n_bootstrap": n_bootstrap,
        "n_blocks": n_blocks,
        "endpoint_centering": endpoint_centering,
        "best_model_fraction": {
            str(name): float(count / max(len(best_models), 1))
            for name, count in zip(unique, model_counts)
        },
        "pure_power_alpha": {
            "median": float(np.median(alpha)) if len(alpha) else None,
            "q025": float(np.quantile(alpha, 0.025)) if len(alpha) else None,
            "q975": float(np.quantile(alpha, 0.975)) if len(alpha) else None,
        },
        "debiased_fro_q025": np.quantile(curve_stack, 0.025, axis=0).tolist(),
        "debiased_fro_q975": np.quantile(curve_stack, 0.975, axis=0).tolist(),
    }


def analyze_projected(
    z: np.ndarray,
    mask: np.ndarray,
    *,
    max_lag: int,
    center_modes: tuple[str, ...] = ("global", "position", "sequence"),
    seed: int = 0,
    n_bootstrap: int = 0,
    bootstrap_blocks: int = 32,
    endpoint_centering: bool = True,
) -> dict:
    """Compute raw and split-half-debiased matrix correlation curves."""
    if z.ndim != 3 or mask.shape != z.shape[:2]:
        raise ValueError(f"expected z=(N,S,P), mask=(N,S); got {z.shape}, {mask.shape}")
    max_lag = min(max_lag, z.shape[1] - 1)
    rng = np.random.default_rng(seed)
    docs = rng.permutation(z.shape[0])
    halves = np.array_split(docs, 2)
    lags = np.arange(1, max_lag + 1)
    result = {"lags": lags.tolist(), "centering": {}}
    for mode in center_modes:
        centered = center_projected(z, mask, mode)
        rows = []
        for lag in lags:
            full, n_pairs = lag_covariance(
                centered, mask, int(lag), docs, endpoint_centering=endpoint_centering
            )
            first, _ = lag_covariance(
                centered, mask, int(lag), halves[0], endpoint_centering=endpoint_centering
            )
            second, _ = lag_covariance(
                centered, mask, int(lag), halves[1], endpoint_centering=endpoint_centering
            )
            cross = float(np.sum(first * second))
            singular = np.linalg.svd(full, compute_uv=False)
            rows.append(
                {
                    "lag": int(lag),
                    "n_pairs": n_pairs,
                    "fro": float(np.linalg.norm(full)),
                    "debiased_fro": float(np.sqrt(max(cross, 0.0))),
                    "operator": float(singular[0]),
                    "nuclear": float(singular.sum()),
                }
            )
        curves = {
            key: np.array([row[key] for row in rows])
            for key in ("fro", "debiased_fro", "operator", "nuclear")
        }
        result["centering"][mode] = {
            "curve": rows,
            "fits": {key: fit_decay_models(lags, value) for key, value in curves.items()},
            "bootstrap": document_bootstrap(
                centered,
                mask,
                lags,
                n_bootstrap=n_bootstrap,
                n_blocks=bootstrap_blocks,
                seed=seed,
                endpoint_centering=endpoint_centering,
            ),
        }
    return result


def _fit_projection(
    acts: np.ndarray,
    mask: np.ndarray,
    *,
    p: int,
    n_fit: int,
    seed: int,
    method: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    flat = acts.reshape(-1, acts.shape[-1])
    valid = np.flatnonzero(mask.reshape(-1))
    chosen = np.sort(rng.choice(valid, size=min(n_fit, len(valid)), replace=False))
    sample = np.asarray(flat[chosen], dtype=np.float32)
    mean = sample.mean(axis=0)
    sample -= mean
    if method == "random":
        q, _ = np.linalg.qr(rng.standard_normal((acts.shape[-1], p)).astype(np.float32))
        scale = None
        return mean, q[:, :p], scale

    import torch

    x = torch.from_numpy(sample).to(device)
    _, singular, components = torch.pca_lowrank(x, q=p, center=False, niter=4)
    scale = (singular / np.sqrt(max(len(sample) - 1, 1))).clamp_min(1e-6)
    return mean, components.cpu().numpy(), scale.cpu().numpy()


def project_activations(
    acts: np.ndarray,
    mask: np.ndarray,
    *,
    p: int,
    n_fit: int,
    seed: int,
    method: str,
    device: str,
    whiten: bool,
    batch_size: int,
) -> np.ndarray:
    """Fit a fixed projection on sampled valid tokens and project the full cache."""
    mean, components, scale = _fit_projection(
        acts, mask, p=p, n_fit=n_fit, seed=seed, method=method, device=device
    )
    flat = acts.reshape(-1, acts.shape[-1])
    out = np.empty((len(flat), p), dtype=np.float32)
    if method == "pca":
        import torch

        comp_t = torch.from_numpy(components).to(device)
        mean_t = torch.from_numpy(mean).to(device)
        scale_t = torch.from_numpy(scale).to(device) if whiten else None
        for start in range(0, len(flat), batch_size):
            x = torch.from_numpy(
                np.asarray(flat[start : start + batch_size], dtype=np.float32)
            ).to(device)
            projected = (x - mean_t) @ comp_t
            if scale_t is not None:
                projected = projected / scale_t
            out[start : start + len(x)] = projected.cpu().numpy()
    else:
        for start in range(0, len(flat), batch_size):
            x = np.asarray(flat[start : start + batch_size], dtype=np.float32)
            out[start : start + len(x)] = (x - mean) @ components
    return out.reshape(*acts.shape[:2], p)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--token-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-revision")
    parser.add_argument("--activation-sha256")
    parser.add_argument("--token-ids-sha256")
    parser.add_argument("--pad-token-id", type=int, action="append", default=[])
    parser.add_argument("--projection", choices=("pca", "random"), default="pca")
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--fit-tokens", type=int, default=100_000)
    parser.add_argument("--max-lag", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--bootstrap-blocks", type=int, default=32)
    parser.add_argument(
        "--lag-centering",
        choices=("endpoint", "precentered"),
        default="endpoint",
        help="Endpoint means per lag, or only the selected preprocessing center.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    acts = np.load(args.activations, mmap_mode="r")
    token_ids = np.load(args.token_ids, mmap_mode="r")
    if acts.ndim != 3 or token_ids.shape != acts.shape[:2]:
        raise ValueError(f"incompatible activation/token shapes: {acts.shape}, {token_ids.shape}")
    mask = np.ones(token_ids.shape, dtype=bool)
    for token_id in args.pad_token_id:
        mask &= token_ids != token_id
    projected = project_activations(
        acts,
        mask,
        p=args.projection_dim,
        n_fit=args.fit_tokens,
        seed=args.seed,
        method=args.projection,
        device=args.device,
        whiten=args.whiten,
        batch_size=args.batch_size,
    )
    result = analyze_projected(
        projected,
        mask,
        max_lag=args.max_lag,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
        bootstrap_blocks=args.bootstrap_blocks,
        endpoint_centering=args.lag_centering == "endpoint",
    )
    result["provenance"] = {
        "activations": str(args.activations),
        "token_ids": str(args.token_ids),
        "hf_revision": args.hf_revision,
        "activation_sha256": args.activation_sha256,
        "token_ids_sha256": args.token_ids_sha256,
        "shape": list(acts.shape),
        "valid_tokens": int(mask.sum()),
        "projection": args.projection,
        "projection_dim": args.projection_dim,
        "fit_tokens": min(args.fit_tokens, int(mask.sum())),
        "whiten": args.whiten,
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "bootstrap_blocks": args.bootstrap_blocks,
        "persistent_subspace_removal": False,
        "lag_centering": args.lag_centering,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "provenance": result["provenance"]}, indent=2))
    return result


if __name__ == "__main__":
    main()
