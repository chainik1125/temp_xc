"""Unified evaluation for all architecture types.

A single evaluate_model() dispatches based on ArchSpec.data_format and
returns a consistent EvalResult. Feature recovery AUC is computed by
matching decoder columns to true feature directions via cosine similarity.

Adapted from Han's eval_unified.py and temporal_crosscoders/metrics.py.
"""

from dataclasses import dataclass, asdict

import numpy as np
import torch

from src.bench.architectures.base import ArchSpec, EvalOutput


@dataclass
class EvalResult:
    """Canonical evaluation result — every model returns all these fields."""

    nmse: float
    l0: float
    auc: float | None = None
    r90: float | None = None
    mean_max_cos: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def feature_recovery_auc(
    decoder_directions: torch.Tensor,
    true_features: torch.Tensor,
    n_thresholds: int = 50,
) -> dict[str, float]:
    """Compute feature recovery AUC and related metrics.

    For each true feature, finds the best-matching decoder column by
    absolute cosine similarity, then sweeps a threshold to compute
    the recovery curve.

    Args:
        decoder_directions: (d_in, d_sae) decoder weight columns.
        true_features: (d_in, n_features) ground-truth feature directions.
        n_thresholds: Number of threshold points for the AUC curve.

    Returns:
        Dict with auc, mean_max_cos, frac_recovered_90.
    """
    # Normalize columns
    dd = decoder_directions / decoder_directions.norm(dim=0, keepdim=True).clamp(min=1e-8)
    tf = true_features / true_features.norm(dim=0, keepdim=True).clamp(min=1e-8)

    sims = (dd.T @ tf).abs()  # (d_sae, n_features)
    max_per_feature = sims.max(dim=0).values  # (n_features,)

    thresholds = np.linspace(0, 1, n_thresholds)
    max_np = max_per_feature.cpu().numpy()
    curve = np.array([(max_np >= t).mean() for t in thresholds])
    _integrate = getattr(np, "trapezoid", None) or np.trapz
    auc = float(_integrate(curve, thresholds))

    return {
        "auc": auc,
        "mean_max_cos": max_per_feature.mean().item(),
        "frac_recovered_90": (max_per_feature >= 0.9).float().mean().item(),
    }


@torch.no_grad()
def evaluate_model(
    spec: ArchSpec,
    model: torch.nn.Module,
    eval_data: torch.Tensor,
    device: torch.device,
    true_features: torch.Tensor | None = None,
    seq_len: int = 64,
) -> EvalResult:
    """Evaluate a model on eval_data and return unified metrics.

    Args:
        spec: ArchSpec for the model.
        model: Trained model.
        eval_data: (n_seq, seq_len, d) full-sequence eval data.
        device: Torch device.
        true_features: (d_in, n_features) for AUC. None to skip.
        seq_len: Sequence length (for windowing).

    Returns:
        EvalResult with all metrics.
    """
    model.eval()

    acc = EvalOutput(sum_se=0.0, sum_signal=0.0, sum_l0=0.0, n_tokens=0)

    if spec.data_format == "flat":
        _eval_flat(spec, model, eval_data, device, acc)
    elif spec.data_format == "seq":
        _eval_seq(spec, model, eval_data, device, acc)
    elif spec.data_format == "window":
        _eval_window(spec, model, eval_data, device, acc, seq_len)
    else:
        raise ValueError(f"Unknown data_format: {spec.data_format}")

    nmse = acc.sum_se / max(acc.sum_signal, 1e-12)
    l0 = acc.sum_l0 / max(acc.n_tokens, 1)

    auc = None
    r90 = None
    mean_max_cos = None
    if true_features is not None:
        dd = _get_decoder_averaged(spec, model, device)
        tf = true_features.to(device)
        recovery = feature_recovery_auc(dd, tf)
        auc = recovery["auc"]
        r90 = recovery["frac_recovered_90"]
        mean_max_cos = recovery["mean_max_cos"]

    return EvalResult(nmse=nmse, l0=l0, auc=auc, r90=r90, mean_max_cos=mean_max_cos)


def _get_decoder_averaged(
    spec: ArchSpec, model: torch.nn.Module, device: torch.device
) -> torch.Tensor:
    """Get decoder directions, averaged across positions if applicable."""
    if spec.n_decoder_positions is None:
        return spec.decoder_directions(model).to(device)
    dds = [
        spec.decoder_directions(model, pos=pos).to(device)
        for pos in range(spec.n_decoder_positions)
    ]
    return torch.stack(dds).mean(dim=0)


def _eval_flat(spec, model, eval_data, device, acc):
    d = eval_data.shape[-1]
    flat = eval_data.reshape(-1, d)
    for s in range(0, flat.shape[0], 4096):
        x = flat[s : s + 4096].to(device)
        out = spec.eval_forward(model, x)
        acc.sum_se += out.sum_se
        acc.sum_signal += out.sum_signal
        acc.sum_l0 += out.sum_l0
        acc.n_tokens += out.n_tokens


def _eval_seq(spec, model, eval_data, device, acc):
    for s in range(0, eval_data.shape[0], 256):
        x = eval_data[s : s + 256].to(device)
        out = spec.eval_forward(model, x)
        acc.sum_se += out.sum_se
        acc.sum_signal += out.sum_signal
        acc.sum_l0 += out.sum_l0
        acc.n_tokens += out.n_tokens


def _eval_window(spec, model, eval_data, device, acc, seq_len):
    T = spec.n_decoder_positions
    for s in range(0, eval_data.shape[0], 256):
        seqs = eval_data[s : s + 256].to(device)
        for t in range(seq_len - T + 1):
            w = seqs[:, t : t + T, :]
            out = spec.eval_forward(model, w)
            acc.sum_se += out.sum_se
            acc.sum_signal += out.sum_signal
            acc.sum_l0 += out.sum_l0
            acc.n_tokens += out.n_tokens
