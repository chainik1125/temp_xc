"""Synthetic feature recovery — § 4 evaluator.

For each (arch, synthetic data) cell, measures three numbers:

- ``eauc``  — best-matching dictionary atom cosine vs each EMISSION
              feature (local). Averaged over emissions then thresholded
              to AUC.
- ``gauc``  — same metric vs HIDDEN-chain features (global). Only
              meaningful for the coupled bench.
- ``nmse``  — reconstruction NMSE on a held-out batch from the
              synthetic generator.

These map to the paper's "local vs global feature recovery" narrative
(§ 4). The synthetic generator is the source of truth: its
``emission_features`` and ``hidden_features`` define the targets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from temp_bench.core.config import load_datasource
from temp_bench.data.synthetic import materialise
from temp_bench.interfaces.architecture import TempBenchArch
from temp_bench.interfaces.evaluator import EvalSpec, Evaluator


def _decoder_directions_normed(model: TempBenchArch) -> torch.Tensor:
    """``(d_sae, d_in)`` unit-norm decoder columns."""
    D = model.decoder_directions().detach().cpu().float()      # (d_sae, d_in)
    norms = D.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return D / norms


def _feature_recovery_auc(
    decoder: torch.Tensor,
    targets: torch.Tensor,
    n_thresholds: int = 50,
) -> dict[str, float]:
    """Best-matching cosine AUC + summary stats.

    For each target row, find ``max_j |cos(decoder[j], target)|``.
    AUC = area under ROC of (max-cos > threshold) vs threshold ∈ [0, 1].
    """
    D = decoder / decoder.norm(dim=1, keepdim=True).clamp(min=1e-8)
    F = targets / targets.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cos = (D @ F.T).abs()                              # (d_sae, n_targets)
    max_cos_per_target = cos.max(dim=0).values         # (n_targets,)
    thresholds = torch.linspace(0, 1, n_thresholds + 1)
    fracs = torch.stack([
        (max_cos_per_target >= t).float().mean() for t in thresholds
    ])
    # Trapezoidal AUC.
    auc = torch.trapz(fracs, thresholds).item()
    return {
        "auc": float(auc),
        "mean_max_cos": float(max_cos_per_target.mean()),
        "frac_recovered_90": float((max_cos_per_target >= 0.9).float().mean()),
        "frac_recovered_80": float((max_cos_per_target >= 0.8).float().mean()),
    }


def _reconstruction_nmse(
    model: TempBenchArch,
    x_eval: torch.Tensor,
    n_batches: int = 4,
    batch_size: int = 256,
) -> float:
    """E[||x - x_hat||^2] / E[||x||^2].

    Shape handling: the trained arch's ``consumes`` mode tells us what
    shape to feed it. For window/sequence archs, sample a window of
    the appropriate length. For token archs, flatten.
    """
    device = next(model.parameters()).device
    n_total = x_eval.shape[0]
    rng = np.random.default_rng(0)

    # Inference shape contract:
    # - token archs (consumes="token"): (B, d_in) per sample (flat tokens)
    # - window/sequence archs: (B, T, d_in) windows. T is the arch's
    #   inference window size, exposed via model.config.T.
    consumes = getattr(model, "consumes", "token")
    T = int(getattr(model.config, "T", 1) or 1)
    if consumes == "token":
        T = 1   # tokens are window-of-1

    num = 0.0
    den = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            idx = rng.integers(0, n_total, size=batch_size)
            x = x_eval[idx].to(device, dtype=torch.float32)   # (B, seq_len, d_in)
            if T > 1 and x.shape[1] >= T:
                # Sample random T-window per row.
                offsets = torch.randint(0, x.shape[1] - T + 1, (x.shape[0],), device=device)
                rng_t = torch.arange(T, device=device)
                pos_grid = offsets.unsqueeze(1) + rng_t.unsqueeze(0)
                bidx = torch.arange(x.shape[0], device=device).unsqueeze(1).expand(-1, T)
                x = x[bidx, pos_grid]                        # (B, T, d_in)
            else:
                # Token mode: flatten (B, seq_len, d_in) → (B*seq_len, d_in)
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
            try:
                x_hat = model(x)
            except Exception:
                x_hat = model.decode(model.encode(x))
            num += float((x - x_hat).pow(2).sum().item())
            den += float(x.pow(2).sum().item())
    return float(num / max(den, 1e-12))


class SyntheticRecovery(Evaluator):
    """§ 4 evaluator: feature recovery AUC + NMSE on synthetic data."""

    name = "synthetic_recovery"
    # v1.1.0 (2026-05-28): eval now re-materialises with the TRAINING
    # seed, not seed=0. Fixes a bug where the synthetic generator's
    # feature directions (deterministic in the seed) differed between
    # training and eval, making it impossible for dictionary atoms to
    # match ground-truth features.
    protocol_version = "1.1.0"

    def eval(self, model: TempBenchArch, spec: EvalSpec) -> dict[str, float]:
        # Re-materialise the dataset from the registry using the SAME
        # seed the model was trained on, so the synthetic generator's
        # feature directions (which depend on the seed) match what the
        # trained dictionary atoms learned.
        ds = load_datasource(spec.datasource)
        seed = int(spec.extra.get("training_seed", spec.extra.get("eval_seed", 0)))
        data = materialise(ds, seed=seed)
        decoder = _decoder_directions_normed(model)

        out: dict[str, float] = {}

        emission_stats = _feature_recovery_auc(decoder, data.emission_features)
        out["eauc"] = emission_stats["auc"]
        out["e_mean_max_cos"] = emission_stats["mean_max_cos"]
        out["e_frac_recovered_90"] = emission_stats["frac_recovered_90"]
        out["e_frac_recovered_80"] = emission_stats["frac_recovered_80"]

        if data.hidden_features is not None:
            hidden_stats = _feature_recovery_auc(decoder, data.hidden_features)
            out["gauc"] = hidden_stats["auc"]
            out["g_mean_max_cos"] = hidden_stats["mean_max_cos"]
            out["g_frac_recovered_90"] = hidden_stats["frac_recovered_90"]
            out["g_frac_recovered_80"] = hidden_stats["frac_recovered_80"]

        n_batches = 1 if spec.smoke else 4
        out["nmse"] = _reconstruction_nmse(model, data.x, n_batches=n_batches, batch_size=64 if spec.smoke else 256)

        # AC-only signed-motion add-on (FrequencyBench § 5). Only fires for
        # the signed_motion datasource, which exposes a hidden ±1 sign in
        # `extra`. For every other bench `extra` is None → this block is a
        # no-op and the metrics above are unchanged (so protocol_version,
        # and all committed coupling/denoising eval_keys, stay put).
        if getattr(data, "extra", None) and "sign_labels" in data.extra:
            from temp_bench.evals.signed_motion_recovery import signed_motion_metrics
            out.update(signed_motion_metrics(model, data))

        return out

    def primary_metric(self) -> str:
        return "gauc"   # § 4 headline = global feature recovery
