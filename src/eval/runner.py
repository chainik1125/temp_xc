"""Unified evaluation for all architecture types.

A single evaluate_model() dispatches based on ArchSpec.data_format and
returns a consistent EvalResult. Feature recovery AUC is computed by
matching decoder columns to true feature directions via cosine similarity.

For coupled-feature data, computes dual AUC:
  - local AUC: decoder vs emission features (M directions)
  - global AUC: decoder vs hidden features (K directions)

Adapted from Han's eval_unified.py and Aniket's original metrics module.
"""

from dataclasses import dataclass, asdict

import numpy as np
import torch

from src.architectures.base import ArchSpec, EvalOutput


@dataclass
class EvalResult:
    """Canonical evaluation result — every model returns all these fields.

    For coupled-feature data, auc/r90/mean_max_cos measure local (emission)
    recovery, while global_auc/global_r90/global_mean_max_cos measure
    hidden-state-level recovery.

    `temporal_mi`, `span_stats`, and `cluster` are populated on real-LM
    cached-activation runs (see src.eval.temporal). They stay None
    on toy Markov data where they're not informative.
    """

    nmse: float
    l0: float
    auc: float | None = None
    r90: float | None = None
    mean_max_cos: float | None = None
    global_auc: float | None = None
    global_r90: float | None = None
    global_mean_max_cos: float | None = None
    # Architecture-specific L0 breakdown. For TFA this holds
    # {"l0_novel": ..., "l0_pred": ...}. Empty for other architectures.
    l0_components: dict[str, float] | None = None
    # Real-LM temporal metrics (None on toy data)
    temporal_mi: dict | None = None
    span_stats: dict | None = None
    cluster: dict | None = None

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
    global_features: torch.Tensor | None = None,
    seq_len: int = 64,
) -> EvalResult:
    """Evaluate a model on eval_data and return unified metrics.

    Args:
        spec: ArchSpec for the model.
        model: Trained model.
        eval_data: (n_seq, seq_len, d) full-sequence eval data.
        device: Torch device.
        true_features: (d_in, n_features) for local AUC. None to skip.
        global_features: (d_in, K) for global AUC (coupled mode). None to skip.
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
    l0_components = (
        {k: v / max(acc.n_tokens, 1) for k, v in acc.extra.items()}
        if acc.extra
        else None
    )

    # Compute decoder directions once (reused for local and global AUC)
    auc = r90 = mean_max_cos = None
    global_auc = global_r90 = global_mean_max_cos = None

    if true_features is not None or global_features is not None:
        dd = _get_decoder_averaged(spec, model, device)

        if true_features is not None:
            recovery = feature_recovery_auc(dd, true_features.to(device))
            auc = recovery["auc"]
            r90 = recovery["frac_recovered_90"]
            mean_max_cos = recovery["mean_max_cos"]

        if global_features is not None:
            global_recovery = feature_recovery_auc(dd, global_features.to(device))
            global_auc = global_recovery["auc"]
            global_r90 = global_recovery["frac_recovered_90"]
            global_mean_max_cos = global_recovery["mean_max_cos"]

    # Optional: real-LM temporal metrics. Only run if we have a proper
    # (n_seq, seq_len, d) tensor and the architecture exposes an encode.
    temporal_mi_d: dict | None = None
    span_d: dict | None = None
    cluster_d: dict | None = None
    if _want_temporal_metrics(spec, eval_data):
        feats = _encode_for_metrics(spec, model, eval_data, device, seq_len)
        if feats is not None:
            from src.eval.temporal import (
                temporal_mi as _temporal_mi,
                activation_span_stats,
                cluster_features,
            )
            try:
                temporal_mi_d = _temporal_mi(feats).to_dict()
            except Exception as e:
                temporal_mi_d = {"error": str(e)}
            try:
                span_d = activation_span_stats(feats).to_dict()
            except Exception as e:
                span_d = {"error": str(e)}
            try:
                dd = _get_decoder_averaged(spec, model, device)
                cluster_d = cluster_features(dd).to_dict()
            except Exception as e:
                cluster_d = {"error": str(e)}

    return EvalResult(
        nmse=nmse, l0=l0,
        auc=auc, r90=r90, mean_max_cos=mean_max_cos,
        global_auc=global_auc, global_r90=global_r90,
        global_mean_max_cos=global_mean_max_cos,
        l0_components=l0_components,
        temporal_mi=temporal_mi_d,
        span_stats=span_d,
        cluster=cluster_d,
    )


def _want_temporal_metrics(spec: ArchSpec, eval_data: torch.Tensor) -> bool:
    """Only compute temporal metrics on real-LM data (3D and no ground truth)."""
    return eval_data.dim() == 3 and eval_data.shape[1] > 1


def _encode_for_metrics(
    spec: ArchSpec,
    model: torch.nn.Module,
    eval_data: torch.Tensor,
    device: torch.device,
    seq_len: int,
    max_seqs: int = 512,
) -> torch.Tensor | None:
    """Best-effort feature extraction for temporal metrics.

    Runs the model on a slice of eval_data and tries to return activations of
    shape (B, T, F). Falls back to None if the architecture doesn't expose a
    hookable encode path.
    """
    n = min(eval_data.shape[0], max_seqs)
    x = eval_data[:n].to(device)
    encode = getattr(spec, "encode", None) or getattr(model, "encode", None)
    if encode is None:
        return None
    try:
        with torch.no_grad():
            feats = encode(x) if callable(encode) else None
    except TypeError:
        try:
            with torch.no_grad():
                feats = encode(model, x)
        except Exception:
            return None
    except Exception:
        return None
    if feats is None:
        return None
    if feats.dim() == 2:
        # flat path — reshape to (B, T, F)
        feats = feats.reshape(n, -1, feats.shape[-1])
    if feats.dim() != 3:
        return None
    return feats.detach().float().cpu()


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


def _accumulate(acc: EvalOutput, out: EvalOutput) -> None:
    acc.sum_se += out.sum_se
    acc.sum_signal += out.sum_signal
    acc.sum_l0 += out.sum_l0
    acc.n_tokens += out.n_tokens
    for k, v in out.extra.items():
        acc.extra[k] = acc.extra.get(k, 0.0) + v


def _eval_flat(spec, model, eval_data, device, acc):
    d = eval_data.shape[-1]
    flat = eval_data.reshape(-1, d)
    for s in range(0, flat.shape[0], 4096):
        x = flat[s : s + 4096].to(device)
        _accumulate(acc, spec.eval_forward(model, x))


def _eval_seq(spec, model, eval_data, device, acc):
    for s in range(0, eval_data.shape[0], 256):
        x = eval_data[s : s + 256].to(device)
        _accumulate(acc, spec.eval_forward(model, x))


def _eval_window(spec, model, eval_data, device, acc, seq_len):
    T = spec.n_decoder_positions
    for s in range(0, eval_data.shape[0], 256):
        seqs = eval_data[s : s + 256].to(device)
        for t in range(seq_len - T + 1):
            w = seqs[:, t : t + T, :]
            _accumulate(acc, spec.eval_forward(model, w))
