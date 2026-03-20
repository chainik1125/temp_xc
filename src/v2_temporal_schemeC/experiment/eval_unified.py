"""Unified evaluation function for all model types.

A single evaluate_model() that dispatches based on ModelSpec.data_format
and returns a consistent EvalResult for every model.
"""

from dataclasses import dataclass, asdict

import numpy as np
import torch

from src.v2_temporal_schemeC.experiment.model_specs import EvalOutput
from src.v2_temporal_schemeC.feature_recovery import feature_recovery_score


@dataclass
class EvalResult:
    """Canonical evaluation result — every model returns all these fields."""
    nmse: float
    novel_l0: float
    pred_l0: float
    total_l0: float
    pred_energy_frac: float
    auc: float | None = None
    r90: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_model(
    spec,
    model: torch.nn.Module,
    eval_data: torch.Tensor,
    device: torch.device,
    true_features: torch.Tensor | None = None,
    seq_len: int = 64,
) -> EvalResult:
    """Evaluate a model on eval_data and return unified metrics.

    Args:
        spec: ModelSpec instance (SAEModelSpec, TFAModelSpec, TXCDRModelSpec).
        model: Trained model.
        eval_data: Shape (n_seq, seq_len, d) — always full sequences.
        device: Torch device.
        true_features: (n_features, d) for AUC computation. None to skip.
        seq_len: Sequence length (for TXCDR windowing).

    Returns:
        EvalResult with all metrics.
    """
    model.eval()

    acc = EvalOutput(
        sum_se=0.0, sum_signal=0.0,
        sum_novel_l0=0.0, sum_pred_l0=0.0, sum_total_l0=0.0,
        n_tokens=0,
    )

    with torch.no_grad():
        if spec.data_format == "flat":
            _eval_flat(spec, model, eval_data, device, acc)
        elif spec.data_format == "seq":
            _eval_seq(spec, model, eval_data, device, acc)
        elif spec.data_format == "window":
            _eval_window(spec, model, eval_data, device, acc, seq_len)
        else:
            raise ValueError(f"Unknown data_format: {spec.data_format}")

    nmse = acc.sum_se / acc.sum_signal
    novel_l0 = acc.sum_novel_l0 / acc.n_tokens
    pred_l0 = acc.sum_pred_l0 / acc.n_tokens
    total_l0 = acc.sum_total_l0 / acc.n_tokens
    # pred_energy_frac would require tracking pred/novel energy separately,
    # which not all old scripts did. Set to 0 for now.
    pred_energy_frac = 0.0

    # AUC
    auc = None
    r90 = None
    if true_features is not None:
        auc, r90 = _compute_auc(spec, model, true_features, device)

    return EvalResult(
        nmse=nmse,
        novel_l0=novel_l0,
        pred_l0=pred_l0,
        total_l0=total_l0,
        pred_energy_frac=pred_energy_frac,
        auc=auc,
        r90=r90,
    )


def _eval_flat(spec, model, eval_data, device, acc: EvalOutput):
    """Evaluate flat (per-token) models like SAE."""
    d = eval_data.shape[-1]
    flat = eval_data.reshape(-1, d)
    n = flat.shape[0]
    for s in range(0, n, 4096):
        x = flat[s:min(s + 4096, n)].to(device)
        out = spec.eval_forward(model, x)
        acc.sum_se += out.sum_se
        acc.sum_signal += out.sum_signal
        acc.sum_novel_l0 += out.sum_novel_l0
        acc.sum_pred_l0 += out.sum_pred_l0
        acc.sum_total_l0 += out.sum_total_l0
        acc.n_tokens += out.n_tokens


def _eval_seq(spec, model, eval_data, device, acc: EvalOutput):
    """Evaluate sequence models like TFA."""
    n_seq = eval_data.shape[0]
    for s in range(0, n_seq, 256):
        x = eval_data[s:min(s + 256, n_seq)].to(device)
        out = spec.eval_forward(model, x)
        acc.sum_se += out.sum_se
        acc.sum_signal += out.sum_signal
        acc.sum_novel_l0 += out.sum_novel_l0
        acc.sum_pred_l0 += out.sum_pred_l0
        acc.sum_total_l0 += out.sum_total_l0
        acc.n_tokens += out.n_tokens


def _eval_window(spec, model, eval_data, device, acc: EvalOutput, seq_len: int):
    """Evaluate windowed models like TXCDR."""
    T = spec.n_decoder_positions
    for s in range(0, eval_data.shape[0], 256):
        seqs = eval_data[s:min(s + 256, eval_data.shape[0])].to(device)
        for t in range(seq_len - T + 1):
            w = seqs[:, t:t + T, :]
            out = spec.eval_forward(model, w)
            acc.sum_se += out.sum_se
            acc.sum_signal += out.sum_signal
            acc.sum_novel_l0 += out.sum_novel_l0
            acc.sum_pred_l0 += out.sum_pred_l0
            acc.sum_total_l0 += out.sum_total_l0
            acc.n_tokens += out.n_tokens


def _compute_auc(spec, model, true_features, device) -> tuple[float, float]:
    """Compute feature recovery AUC, handling per-position decoders for TXCDR."""
    tf = true_features.T.to(device)

    if spec.n_decoder_positions is None:
        dd = spec.decoder_directions(model).to(device)
        result = feature_recovery_score(dd, tf)
        return result["auc"], result["frac_recovered_90"]
    else:
        aucs = []
        r90s = []
        for pos in range(spec.n_decoder_positions):
            dd = spec.decoder_directions(model, pos=pos).to(device)
            result = feature_recovery_score(dd, tf)
            aucs.append(result["auc"])
            r90s.append(result["frac_recovered_90"])
        return float(np.mean(aucs)), float(np.mean(r90s))
