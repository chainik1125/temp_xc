"""Multi-architecture dispatch for Stage B baselines.

Architectures supported:
  - "txc"        — TemporalCrosscoder (already used in Stage B sprint)
                   shared latent across T positions, per-position W_enc/W_dec
  - "topk_sae"   — flat TopK SAE applied independently to each token in the
                   T-window (reshape to (B*T, d) for training)
  - "stacked_sae"— StackedSAE: T independent TopK SAEs, one per offset slot
  - "tsae"       — Han's TemporalSAE (attention-based predicted+novel codes,
                   from references/TemporalFeatureAnalysis vendored under
                   temporal_crosscoders/han_tsae/)

Common interface (all dispatched through this file):
  build_arch(arch, d_in, d_sae, T, k, **arch_kwargs) -> nn.Module
  arch_forward(arch, model, x_btd) -> (loss, latents_dict)
      latents_dict["window"]: (B, d_sae) window-level latent for mining
      latents_dict["per_pos"]: (B, T, d_sae) per-position latent (None if N/A)
  arch_decoder_directions(arch, model) -> dict
      "pos0": (d_sae, d) — decoder at first window slot (or only slot)
      "union": (d_sae, d) — averaged across T slots (== pos0 for arch w/o T)
      "per_pos": (d_sae, T, d) — full tensor (None if not per-position)
  arch_encode_window(arch, model, x_btd) -> (B, d_sae)
      The "window-level" activation used for D+/D- selectivity ranking.

Conventions
-----------
- All input tensors are float32 on the GPU at call time.
- For TopKSAE we use total per-token k = round(window_k / T) so that the
  total active count summed over the T-window matches the TXC's window L0.
- For TSAE (Han's) we wire kval_topk = window_k (paper convention: total
  k across the window's z_novel, since z_pred is dense by construction).
- All dictionaries on disk are saved with config dict so the loader knows
  which arch and which hparams to instantiate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_crosscoders.models import (
    TopKSAE,
    StackedSAE,
    TemporalCrosscoder,
)
from temporal_crosscoders.han_tsae import TemporalSAE


def build_arch(arch: str, d_in: int, d_sae: int, T: int, k: int, **kwargs):
    """Instantiate the requested architecture with matched-sparsity sizing.

    `k` is the per-position topk count (matching TXC's `k_per_position`).
    Each architecture interprets it slightly differently — see comments.
    """
    if arch == "txc":
        # TemporalCrosscoder has shared latent across T positions and uses
        # k*T total active slots (matching StackedSAE's window L0).
        return TemporalCrosscoder(d_in=d_in, d_sae=d_sae, T=T, k=k)
    elif arch == "stacked_sae":
        # StackedSAE: independent SAE per position with k active each.
        return StackedSAE(d_in=d_in, d_sae=d_sae, T=T, k=k)
    elif arch == "topk_sae":
        # Flat SAE applied to each token. We use k_per_position so that the
        # total active count summed over the T-window matches TXC's window L0.
        return TopKSAE(d_in=d_in, d_sae=d_sae, k=k)
    elif arch == "tsae":
        # Han's TemporalSAE. `width` is the dictionary size = d_sae.
        # n_heads must divide width / bottleneck_factor (see ManualAttention).
        # bottleneck_factor=64 with d_sae=16384 → 256 valid head counts;
        # 8 is a safe default. n_attn_layers=1 to match Dmitry's default.
        n_heads = kwargs.get("n_heads", 8)
        bottleneck_factor = kwargs.get("bottleneck_factor", 64)
        sae_diff_type = kwargs.get("sae_diff_type", "topk")
        n_attn_layers = kwargs.get("n_attn_layers", 1)
        return TemporalSAE(
            dimin=d_in,
            width=d_sae,
            n_heads=n_heads,
            sae_diff_type=sae_diff_type,
            kval_topk=k * T,                      # total k across T window
            tied_weights=kwargs.get("tied_weights", True),
            n_attn_layers=n_attn_layers,
            bottleneck_factor=bottleneck_factor,
        )
    raise ValueError(f"unknown arch: {arch}")


def arch_forward(arch: str, model: nn.Module, x_btd: torch.Tensor):
    """Train-time forward: returns (recon_loss, latents_dict).

    latents_dict has at least one of:
      - "window": (B, d_sae) — TXC's natural latent
      - "per_pos": (B, T, d_sae) — per-position latents (StackedSAE, TSAE,
        and TopKSAE-applied-per-position).
    """
    if arch == "txc":
        loss, x_hat, z = model(x_btd)             # z: (B, d_sae)
        return loss, {"window": z, "per_pos": None}
    elif arch == "stacked_sae":
        loss, x_hat, u = model(x_btd)             # u: (B, T, d_sae)
        return loss, {"window": u.mean(dim=1), "per_pos": u}
    elif arch == "topk_sae":
        # Flatten window into batch axis so the flat SAE sees (B*T, d).
        B, T, d = x_btd.shape
        x_flat = x_btd.reshape(B * T, d)
        loss, x_hat, u = model(x_flat)            # u: (B*T, d_sae)
        per_pos = u.reshape(B, T, -1)
        return loss, {"window": per_pos.mean(dim=1), "per_pos": per_pos}
    elif arch == "tsae":
        x_hat, results = model(x_btd)             # x_hat: (B, T, d)
        loss = (x_hat - x_btd).pow(2).sum(dim=-1).mean()
        codes = results["pred_codes"] + results["novel_codes"]   # (B, T, d_sae)
        return loss, {"window": codes.mean(dim=1), "per_pos": codes}
    raise ValueError(f"unknown arch: {arch}")


@torch.no_grad()
def arch_encode_window(arch: str, model: nn.Module, x_btd: torch.Tensor) -> torch.Tensor:
    """Inference-time encode → (B, d_sae) window-level activation. Used for
    feature mining (D+/D- selectivity ranking).
    """
    _, lat = arch_forward(arch, model, x_btd)
    return lat["window"]


@torch.no_grad()
def arch_decoder_directions(arch: str, model: nn.Module) -> dict:
    """Return decoder directions per architecture.

    Output keys:
      "pos0":     (d_sae, d) — single-vector-per-feature steering source
      "union":    (d_sae, d) — averaged across T positions where applicable
      "per_pos":  (d_sae, T, d) or None — full per-position tensor for TXC/Stacked
    """
    if arch == "txc":
        # W_dec is (d_sae, T, d); pos0 = W_dec[:, 0, :], union = mean over T.
        W = model.W_dec                          # (d_sae, T, d)
        return {
            "pos0": W[:, 0, :],
            "union": W.mean(dim=1),
            "per_pos": W,
        }
    elif arch == "stacked_sae":
        # Per-position decoders: gather as (d_sae, T, d).
        per_pos = torch.stack(
            [sae.W_dec.T for sae in model.saes], dim=1
        )                                        # (d_sae, T, d)
        return {
            "pos0": per_pos[:, 0, :],
            "union": per_pos.mean(dim=1),
            "per_pos": per_pos,
        }
    elif arch == "topk_sae":
        # Single shared decoder (no T axis). pos0 == union.
        W = model.W_dec.T                        # (d_sae, d)
        return {"pos0": W, "union": W, "per_pos": None}
    elif arch == "tsae":
        # Han's TSAE has a single shared dictionary D of shape (width, dimin).
        # All temporal structure lives in the attention layer; the steering
        # direction per feature is the dictionary row.
        W = model.D                              # (d_sae, d)
        return {"pos0": W, "union": W, "per_pos": None}
    raise ValueError(f"unknown arch: {arch}")


def arch_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
