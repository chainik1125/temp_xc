"""Replay the paper Denoising cell and measure how Spectral v1 uses DC versus AC.

This module intentionally lives inside the isolated power-spectrum experiment.
It reuses the pinned paper data and training recipe, but adds diagnostics that
are specific to a native ``T=2, k_pos=20`` multiband Spectral-v1 model:

- per-feature firing and activation magnitudes on native sliding windows;
- static and realized per-band allocation;
- actual decoded DC/AC energy, excluding the decoder bias;
- the decoder-bias spectrum as a separate quantity;
- activation-weighted decoder-coefficient energy (an additive proxy);
- hidden-state Ridge probes using full, DC-only, and AC-only codes; and
- ground-truth DCT power in activations, observed support, and hidden support.

The CLI trains independent model seeds against the same paper training data and
uses a separately generated evaluation dataset. It does not launch or configure
remote compute.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from experiments.power_spectrum.code import run_paper_synthetic_v1 as paper_runner
from temp_bench.archs.spectral_txc import _dct_basis


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "denoising_frequency_usage.json"
)


class _NoOpBudget:
    """Minimal ``train_cell`` budget interface for a local, caller-controlled run."""

    def check(self) -> None:
        return None


def paper_training_config(
    *,
    n_steps: int = 30_000,
    batch_size: int = 1024,
    learning_rate: float = 3e-4,
    precision: str = "bf16",
) -> dict[str, Any]:
    """Return the subset of the paper config consumed by ``train_cell``."""
    return {
        "model": {
            "name": "spectral_v1",
            "class_path": "temp_bench.archs.spectral_txc:SpectralTXCBatchTopK",
            "bands": "multiband",
            "auxk_alpha": 1.0 / 32.0,
        },
        "training": {
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "warmup_steps": min(1000, int(n_steps)),
            "gradient_clip_norm": 1.0,
            "precision": precision,
            "budget_check_every_steps": 250,
        },
    }


def paper_cell(*, seed: int, n_steps: int) -> dict[str, Any]:
    """The dense-code Denoising replay cell requested for this diagnostic."""
    return {
        "task": "denoising",
        "seed": int(seed),
        "T": 2,
        "k_pos": 20,
        "d_sae": 40,
        "n_steps": int(n_steps),
    }


def native_windows(values: torch.Tensor, *, T: int = 2) -> torch.Tensor:
    """Return all sliding native windows as ``(n_windows, T, d)``.

    ``values`` must be time-major ``(n_sequences, sequence_length, d)``.
    Windows from different sequences are never joined.
    """
    if values.ndim != 3:
        raise ValueError(f"expected (n_sequences, sequence_length, d), got {values.shape}")
    if T < 1 or T > values.shape[1]:
        raise ValueError(f"invalid T={T} for sequence length {values.shape[1]}")
    unfolded = values.unfold(1, T, 1)  # (n_sequences, n_starts, d, T)
    return unfolded.permute(0, 1, 3, 2).reshape(-1, T, values.shape[-1])


@torch.no_grad()
def encode_native_windows(
    model: Any,
    eval_x: torch.Tensor,
    *,
    T: int = 2,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Encode every native sliding window and return float32 CPU codes."""
    windows = native_windows(eval_x, T=T)
    device = next(model.parameters()).device
    was_training = bool(model.training)
    model.eval()
    chunks: list[torch.Tensor] = []
    try:
        for start in range(0, windows.shape[0], batch_size):
            batch = windows[start : start + batch_size].to(device, dtype=torch.float32)
            code = model.encode(batch)
            if code.ndim == 3:
                if code.shape[1] != 1:
                    raise ValueError(f"expected a shared window code, got {code.shape}")
                code = code[:, 0]
            chunks.append(code.detach().float().cpu())
    finally:
        model.train(was_training)
    return torch.cat(chunks, dim=0)


def per_feature_usage(codes: torch.Tensor) -> dict[str, list[float]]:
    """Per-feature native-window firing, mean L1, and RMS L2 activation."""
    if codes.ndim != 2:
        raise ValueError(f"expected two-dimensional codes, got {codes.shape}")
    values = codes.detach().float().cpu()
    return {
        "fire_rate": (values != 0).float().mean(dim=0).tolist(),
        "l1_mean": values.abs().mean(dim=0).tolist(),
        "l2_rms": values.square().mean(dim=0).sqrt().tolist(),
    }


def frequency_feature_masks(model: Any) -> dict[str, torch.Tensor]:
    """Return exclusive full/DC/AC feature masks from the model's band slices.

    A mixed-frequency band is deliberately excluded from both the DC-only and
    AC-only masks. The requested T=2 multiband model has no mixed band.
    """
    d_sae = int(model._d_sae)
    masks = {
        "full": torch.ones(d_sae, dtype=torch.bool),
        "dc": torch.zeros(d_sae, dtype=torch.bool),
        "ac": torch.zeros(d_sae, dtype=torch.bool),
        "mixed": torch.zeros(d_sae, dtype=torch.bool),
    }
    for band, (start, end) in zip(model.bands, model.band_slices, strict=True):
        if all(int(frequency) == 0 for frequency in band):
            masks["dc"][start:end] = True
        elif all(int(frequency) != 0 for frequency in band):
            masks["ac"][start:end] = True
        else:
            masks["mixed"][start:end] = True
    return masks


@torch.no_grad()
def decoded_reconstruction_energy(
    model: Any,
    codes: torch.Tensor,
    *,
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Measure actual decoded frequency energy with ``b_dec`` excluded.

    Energies are means of the squared Frobenius norm per native window. Unlike
    the coefficient proxy, this includes cross-feature interference within a
    band. Disjoint DCT bands are orthogonal, so band energies should add to the
    full decoded energy up to floating-point error.
    """
    if codes.ndim != 2 or codes.shape[1] != int(model._d_sae):
        raise ValueError("codes do not match the model dictionary width")
    device = next(model.parameters()).device
    kernels = [model._dec_kernel(b).detach() for b in range(model.n_bands)]
    basis = _dct_basis(int(model._T)).to(device=device, dtype=torch.float32)
    band_sums = torch.zeros(model.n_bands, dtype=torch.float64)
    frequency_sums = torch.zeros(int(model._T), dtype=torch.float64)
    total_sum = 0.0
    n_rows = 0
    for row_start in range(0, codes.shape[0], batch_size):
        code = codes[row_start : row_start + batch_size].to(device, dtype=torch.float32)
        band_reconstructions: list[torch.Tensor] = []
        for band_index, ((start, end), kernel) in enumerate(
            zip(model.band_slices, kernels, strict=True)
        ):
            reconstruction = torch.einsum("bh,htd->btd", code[:, start:end], kernel)
            band_reconstructions.append(reconstruction)
            band_sums[band_index] += reconstruction.square().sum().double().cpu()
        full = torch.stack(band_reconstructions, dim=0).sum(dim=0)
        coefficients = torch.einsum("ft,btd->bfd", basis, full)
        frequency_sums += coefficients.square().sum(dim=(0, 2)).double().cpu()
        total_sum += float(full.square().sum().item())
        n_rows += int(code.shape[0])
    if n_rows == 0:
        raise ValueError("cannot measure energy from an empty code matrix")
    per_band = band_sums / n_rows
    per_frequency = frequency_sums / n_rows
    total = total_sum / n_rows
    denominator = max(total, 1e-30)
    band_total = float(per_band.sum().item())
    return {
        "unit": "mean squared Frobenius norm per native window",
        "bias_included": False,
        "total": total,
        "dc": float(per_frequency[0].item()),
        "ac": float(per_frequency[1:].sum().item()),
        "dc_share": float(per_frequency[0].item()) / denominator,
        "ac_share": float(per_frequency[1:].sum().item()) / denominator,
        "per_frequency": per_frequency.tolist(),
        "per_band": per_band.tolist(),
        "per_band_share": (per_band / denominator).tolist(),
        "band_additivity_relative_error": abs(band_total - total) / denominator,
    }


@torch.no_grad()
def bias_spectrum(model: Any) -> dict[str, Any]:
    """Return DCT energy of the decoder bias, kept separate from reconstruction."""
    bias = model.b_dec.detach().float().cpu()
    basis = _dct_basis(int(model._T)).to(dtype=bias.dtype)
    coefficients = torch.einsum("ft,td->fd", basis, bias)
    energy = coefficients.square().sum(dim=-1)
    total = float(energy.sum().item())
    denominator = max(total, 1e-30)
    return {
        "total": total,
        "dc": float(energy[0].item()),
        "ac": float(energy[1:].sum().item()),
        "dc_share": float(energy[0].item()) / denominator,
        "ac_share": float(energy[1:].sum().item()) / denominator,
        "per_frequency": energy.tolist(),
    }


@torch.no_grad()
def activation_weighted_coefficient_energy(
    model: Any,
    codes: torch.Tensor,
) -> dict[str, Any]:
    """Additive decoder-energy proxy ``E[z_h^2] ||C_{h,f,:}||_2^2``.

    This deliberately excludes cross-feature interference. It answers whether
    activated atoms allocate their coefficient norm to DC or AC, while
    :func:`decoded_reconstruction_energy` measures the actual decoded result.
    """
    mean_square = codes.detach().float().cpu().square().mean(dim=0)
    per_feature = torch.zeros(int(model._d_sae), dtype=torch.float64)
    per_band = torch.zeros(model.n_bands, dtype=torch.float64)
    per_frequency = torch.zeros(int(model._T), dtype=torch.float64)
    for band_index, (frequencies, (start, end), coefficients) in enumerate(
        zip(model.bands, model.band_slices, model.dec_coef, strict=True)
    ):
        coefficient_norm = coefficients.detach().float().cpu().square().sum(dim=-1)
        weighted = mean_square[start:end, None] * coefficient_norm
        feature_energy = weighted.sum(dim=-1).double()
        per_feature[start:end] = feature_energy
        per_band[band_index] = feature_energy.sum()
        for local_frequency, frequency in enumerate(frequencies):
            per_frequency[int(frequency)] += weighted[:, local_frequency].sum().double()
    total = float(per_feature.sum().item())
    denominator = max(total, 1e-30)
    return {
        "definition": "E[z_h^2] * squared decoder-coefficient norm",
        "total": total,
        "dc": float(per_frequency[0].item()),
        "ac": float(per_frequency[1:].sum().item()),
        "dc_share": float(per_frequency[0].item()) / denominator,
        "ac_share": float(per_frequency[1:].sum().item()) / denominator,
        "per_feature": per_feature.tolist(),
        "per_band": per_band.tolist(),
        "per_band_share": (per_band / denominator).tolist(),
        "per_frequency": per_frequency.tolist(),
        "per_frequency_share": (per_frequency / denominator).tolist(),
    }


def per_band_usage(
    model: Any,
    codes: torch.Tensor,
    decoded_energy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Combine static allocation and realized native-window use for each band."""
    total_l0 = float((codes != 0).sum(dim=1).float().mean().item())
    activation_energy = codes.float().square().mean(dim=0)
    total_activation_energy = float(activation_energy.sum().item())
    rows: list[dict[str, Any]] = []
    for band_index, (frequencies, (start, end)) in enumerate(
        zip(model.bands, model.band_slices, strict=True)
    ):
        band_codes = codes[:, start:end].float()
        l0 = float((band_codes != 0).sum(dim=1).float().mean().item())
        l1 = float(band_codes.abs().sum(dim=1).mean().item())
        code_energy = float(activation_energy[start:end].sum().item())
        rows.append(
            {
                "band": int(band_index),
                "frequencies": [int(value) for value in frequencies],
                "feature_start": int(start),
                "feature_end": int(end),
                "allocated_atoms": int(end - start),
                "allocated_k": int(model.k_per_band[band_index]),
                "realized_l0": l0,
                "realized_l0_share": l0 / max(total_l0, 1e-30),
                "activation_l1_per_window": l1,
                "activation_energy": code_energy,
                "activation_energy_share": code_energy
                / max(total_activation_energy, 1e-30),
                "decoded_energy": float(decoded_energy["per_band"][band_index]),
                "decoded_energy_share": float(
                    decoded_energy["per_band_share"][band_index]
                ),
            }
        )
    return rows


def _as_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().float().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def ridge_r2_by_mask(
    codes: torch.Tensor | np.ndarray,
    hidden: torch.Tensor | np.ndarray,
    masks: Mapping[str, torch.Tensor | np.ndarray],
    *,
    train_fraction: float = 0.8,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Fit per-hidden-feature Ridge probes through named code masks.

    If inputs are three-dimensional, their first dimension is interpreted as
    sequence and the split is made by whole sequence before flattening.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    code_array = _as_numpy(codes)
    hidden_array = _as_numpy(hidden)
    if code_array.ndim not in {2, 3} or hidden_array.ndim != code_array.ndim:
        raise ValueError("codes and hidden must both be 2D or both be 3D")
    if code_array.shape[:-1] != hidden_array.shape[:-1]:
        raise ValueError("codes and hidden must have matching example axes")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")

    split = max(1, min(code_array.shape[0] - 1, int(code_array.shape[0] * train_fraction)))
    if code_array.ndim == 3:
        code_train = code_array[:split].reshape(-1, code_array.shape[-1])
        code_test = code_array[split:].reshape(-1, code_array.shape[-1])
        hidden_train = hidden_array[:split].reshape(-1, hidden_array.shape[-1])
        hidden_test = hidden_array[split:].reshape(-1, hidden_array.shape[-1])
    else:
        code_train, code_test = code_array[:split], code_array[split:]
        hidden_train, hidden_test = hidden_array[:split], hidden_array[split:]

    output: dict[str, Any] = {}
    for name in ("full", "dc", "ac"):
        mask = np.asarray(masks[name], dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != code_array.shape[-1]:
            raise ValueError(f"{name} mask has the wrong shape")
        if not mask.any():
            raise ValueError(f"{name} mask selects no features")
        feature_values: list[float] = []
        for feature in range(hidden_array.shape[-1]):
            probe = Ridge(alpha=float(alpha)).fit(
                code_train[:, mask],
                hidden_train[:, feature],
            )
            prediction = probe.predict(code_test[:, mask])
            feature_values.append(
                float(r2_score(hidden_test[:, feature], prediction))
            )
        output[name] = {
            "n_code_features": int(mask.sum()),
            "mean_r2": statistics.fmean(feature_values),
            "per_hidden_feature_r2": feature_values,
        }
    return output


def hidden_state_ridge_metrics(
    model: Any,
    eval_data: paper_runner.PaperData,
    masks: Mapping[str, torch.Tensor],
    *,
    T: int = 2,
    d_sae: int = 40,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Run full/DC/AC Ridge probes on paper overlap-averaged token codes."""
    if eval_data.hidden_support is None:
        raise ValueError("evaluation data has no hidden support")
    flat_codes = paper_runner.extract_overlapping_latents(
        model,
        eval_data.x,
        T=T,
        d_sae=d_sae,
    )
    n_sequences, sequence_length = eval_data.x.shape[:2]
    codes = flat_codes.reshape(n_sequences, sequence_length, d_sae)
    hidden = eval_data.hidden_support.permute(0, 2, 1).float().cpu().numpy()
    return ridge_r2_by_mask(codes, hidden, masks, alpha=alpha)


def dct_power(values: torch.Tensor, *, T: int = 2) -> dict[str, Any]:
    """True sliding-window DCT power of a time-major tensor."""
    windows = native_windows(values.detach().float().cpu(), T=T)
    basis = _dct_basis(T).to(dtype=windows.dtype)
    coefficients = torch.einsum("ft,btd->bfd", basis, windows)
    power = coefficients.square().sum(dim=-1).mean(dim=0)
    total = float(power.sum().item())
    denominator = max(total, 1e-30)
    input_energy = float(windows.square().sum(dim=(1, 2)).mean().item())
    return {
        "total": total,
        "dc": float(power[0].item()),
        "ac": float(power[1:].sum().item()),
        "dc_share": float(power[0].item()) / denominator,
        "ac_share": float(power[1:].sum().item()) / denominator,
        "per_frequency": power.tolist(),
        "parseval_relative_error": abs(total - input_energy)
        / max(input_energy, 1e-30),
    }


def _support_time_major(
    support: torch.Tensor,
    *,
    sequence_length: int,
) -> torch.Tensor:
    if support.ndim != 3:
        raise ValueError(f"expected three-dimensional support, got {support.shape}")
    if support.shape[1] == sequence_length:
        return support
    if support.shape[2] == sequence_length:
        return support.permute(0, 2, 1)
    raise ValueError("could not identify the support time axis")


def true_data_dct_power(
    eval_data: paper_runner.PaperData,
    *,
    T: int = 2,
) -> dict[str, Any]:
    """DCT power of true observed activations/support and hidden support."""
    if eval_data.support is None or eval_data.hidden_support is None:
        raise ValueError("denoising data is missing observed or hidden support")
    sequence_length = int(eval_data.x.shape[1])
    observed_support = _support_time_major(
        eval_data.support,
        sequence_length=sequence_length,
    )
    hidden_support = _support_time_major(
        eval_data.hidden_support,
        sequence_length=sequence_length,
    )
    return {
        "observed_activation": dct_power(eval_data.x, T=T),
        "observed_support": dct_power(observed_support, T=T),
        "hidden_support": dct_power(hidden_support, T=T),
    }


def validate_paper_model(model: Any) -> None:
    """Fail loudly if analysis is accidentally run on a different cell."""
    if int(model._T) != 2:
        raise ValueError(f"expected T=2, got T={model._T}")
    if int(model.k_pos) != 20:
        raise ValueError(f"expected k_pos=20, got k_pos={model.k_pos}")
    if int(model._d_sae) != 40:
        raise ValueError(f"expected d_sae=40, got d_sae={model._d_sae}")
    if [list(map(int, band)) for band in model.bands] != [[0], [1]]:
        raise ValueError(f"expected separate T=2 DC/AC bands, got {model.bands}")


def analyze_denoising_frequency_usage(
    model: Any,
    eval_data: paper_runner.PaperData,
    *,
    native_batch_size: int = 1024,
    ridge_alpha: float = 1.0,
    require_paper_cell: bool = True,
) -> dict[str, Any]:
    """Run the complete DC-versus-AC diagnostic on independent eval data."""
    if require_paper_cell:
        validate_paper_model(model)
    codes = encode_native_windows(
        model,
        eval_data.x,
        T=int(model._T),
        batch_size=native_batch_size,
    )
    masks = frequency_feature_masks(model)
    decoded = decoded_reconstruction_energy(
        model,
        codes,
        batch_size=native_batch_size,
    )
    return {
        "native_window_count": int(codes.shape[0]),
        "per_feature": per_feature_usage(codes),
        "bands": per_band_usage(model, codes, decoded),
        "decoded_reconstruction_energy": decoded,
        "bias_spectrum": bias_spectrum(model),
        "activation_weighted_decoder_coefficient_energy": (
            activation_weighted_coefficient_energy(model, codes)
        ),
        "hidden_state_ridge_r2": hidden_state_ridge_metrics(
            model,
            eval_data,
            masks,
            T=int(model._T),
            d_sae=int(model._d_sae),
            alpha=ridge_alpha,
        ),
        "true_dct_power": true_data_dct_power(eval_data, T=int(model._T)),
        "feature_masks": {
            name: torch.nonzero(mask, as_tuple=False).flatten().tolist()
            for name, mask in masks.items()
        },
    }


def _important_scalars(row: Mapping[str, Any]) -> dict[str, float]:
    analysis = row["analysis"]
    decoded = analysis["decoded_reconstruction_energy"]
    coefficient = analysis["activation_weighted_decoder_coefficient_energy"]
    ridge = analysis["hidden_state_ridge_r2"]
    return {
        "decoded_dc_share": float(decoded["dc_share"]),
        "decoded_ac_share": float(decoded["ac_share"]),
        "coefficient_dc_share": float(coefficient["dc_share"]),
        "coefficient_ac_share": float(coefficient["ac_share"]),
        "ridge_full_mean_r2": float(ridge["full"]["mean_r2"]),
        "ridge_dc_mean_r2": float(ridge["dc"]["mean_r2"]),
        "ridge_ac_mean_r2": float(ridge["ac"]["mean_r2"]),
    }


def aggregate_seed_results(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean/std summary of the small set of headline scalar diagnostics."""
    if not rows:
        return {}
    values = [_important_scalars(row) for row in rows]
    output: dict[str, Any] = {}
    for key in values[0]:
        samples = [row[key] for row in values]
        output[key] = {
            "mean": statistics.fmean(samples),
            "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "values": samples,
        }
    return output


def run_seeds(
    *,
    seeds: Iterable[int] = (1, 2, 42),
    n_steps: int = 30_000,
    train_n_sequences: int = 4096,
    eval_n_sequences: int = 2000,
    train_data_seed: int = 0,
    eval_data_seed: int = 42,
    batch_size: int = 1024,
    learning_rate: float = 3e-4,
    precision: str = "bf16",
    native_batch_size: int = 1024,
) -> dict[str, Any]:
    """Train and analyze a paired three-seed replay without remote orchestration."""
    if train_data_seed == eval_data_seed:
        raise ValueError("training and evaluation data seeds must be independent")
    seed_list = [int(seed) for seed in seeds]
    config = paper_training_config(
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        precision=precision,
    )
    train_data = paper_runner.paper_markov_data(
        n_seqs=int(train_n_sequences),
        seed=int(train_data_seed),
    )
    eval_data = paper_runner.paper_markov_data(
        n_seqs=int(eval_n_sequences),
        seed=int(eval_data_seed),
    )
    rows: list[dict[str, Any]] = []
    budget = _NoOpBudget()
    for seed in seed_list:
        cell = paper_cell(seed=seed, n_steps=n_steps)
        model, training = paper_runner.train_cell(
            config,
            cell,
            train_data,
            budget,
        )
        analysis = analyze_denoising_frequency_usage(
            model,
            eval_data,
            native_batch_size=native_batch_size,
        )
        rows.append(
            {
                "seed": seed,
                "cell": cell,
                "training": training,
                "analysis": analysis,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "schema_version": 1,
        "experiment": "denoising_frequency_usage",
        "train_data_seed": int(train_data_seed),
        "eval_data_seed": int(eval_data_seed),
        "train_n_sequences": int(train_n_sequences),
        "eval_n_sequences": int(eval_n_sequences),
        "seeds": seed_list,
        "rows": rows,
        "aggregate": aggregate_seed_results(rows),
    }


def _finite_json(value: Any) -> None:
    """Raise before writing if a nested result contains NaN or infinity."""
    if isinstance(value, Mapping):
        for nested in value.values():
            _finite_json(nested)
    elif isinstance(value, list):
        for nested in value:
            _finite_json(nested)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite result: {value}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 42])
    parser.add_argument("--n-steps", type=int, default=30_000)
    parser.add_argument("--train-n-sequences", type=int, default=4096)
    parser.add_argument("--eval-n-sequences", type=int, default=2000)
    parser.add_argument("--train-data-seed", type=int, default=0)
    parser.add_argument("--eval-data-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--native-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use two steps, 64 train sequences, and 32 eval sequences.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.smoke:
        args.n_steps = 2
        args.train_n_sequences = 64
        args.eval_n_sequences = 32
    result = run_seeds(
        seeds=args.seeds,
        n_steps=args.n_steps,
        train_n_sequences=args.train_n_sequences,
        eval_n_sequences=args.eval_n_sequences,
        train_data_seed=args.train_data_seed,
        eval_data_seed=args.eval_data_seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        precision=args.precision,
        native_batch_size=args.native_batch_size,
    )
    _finite_json(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
