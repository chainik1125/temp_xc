"""V3 latent-space steering driver — TXC encode → perturb z → decode.

V3 is mathematically the cleanest "make the TXC think feature f
fired more strongly" intervention: encode the most recent T-window
through ``arch.encode``, get ``z_base``, set
``z' = z_base + α · e_f``, decode to a ``(T, d_in)`` perturbation,
overwrite the trailing-window of the residual stream.

This isn't a forward hook — it requires the **full T-window of
residuals at the steering layer**, plus a TXC encode + decode per
intervention step. Implemented here as a standalone driver that wraps
the model's forward pass with a residual-overwrite.

V3 is intended as an **upper bound** diagnostic: if V0/V1/V2/V4 all
underperform V3 substantially, the heuristic protocols are leaving
signal on the table; the field-test budget should swing toward V3.
If V3 ties V0 + V1 + V2 + V4, the trajectory question is moot for
this feature.

Usage::

    from temp_bench.eval.steering_protocols import latent_space_steer

    deltas = latent_space_steer(
        arch=txc_arch,
        residual_window=residual,        # (B, T, d_in) — last T
                                         #   residuals at the steering layer
        feature_id=top_feature,
        magnitude=alpha,                 # scalar or (B,) tensor
        ref_norm=ref_norm,               # scale Δz to match per-token energy
    )
    # deltas: (B, T, d_in) — add to the trailing-T positions of the
    # residual stream BEFORE the next forward step.

Caller responsibility: residual capture + overwrite are at the same
layer as ``arch`` was trained on.

This driver is **separate from the forward-hook abstraction** by
design. Wrapping V3 in a forward hook would mean encoding through the
TXC inside the subject model's forward graph — slow, fragile, and
mixing two computation graphs. Instead, callers run a
"capture → encode → perturb → decode → overwrite" loop around each
generation step. The C7 / C5 driver scripts implement this loop;
``latent_space_steer`` is the per-step primitive.
"""

from __future__ import annotations

import torch

from temp_bench.architectures.base import TempBenchArch


def latent_space_steer(
    arch: TempBenchArch,
    residual_window: torch.Tensor,
    *,
    feature_id: int,
    magnitude: float | torch.Tensor,
    ref_norm: float | None = None,
) -> torch.Tensor:
    """One V3 step: TXC encode → perturb feature_id → decode → return Δ.

    Args:
        arch: a TXC-family arch (W_enc / W_dec are 3-D).
        residual_window: ``(B, T, d_in)`` — the trailing-T residuals
            at the steering layer for B continuations.
        feature_id: the feature to amplify in latent space.
        magnitude: scalar or ``(B,)`` tensor — added to ``z[:, feature_id]``.
        ref_norm: if not None, ``Δ`` is rescaled per-row to L2 norm
            ``ref_norm × |magnitude|`` so cross-arch magnitudes remain
            comparable. (Without this, V3 magnitudes are in a
            different unit than V0/V1/V2/V4 — they're latent-space
            additions, not residual-space additions.)

    Returns:
        ``(B, T, d_in)`` Δ tensor — add to the residual stream's
        trailing T positions BEFORE the next forward step.
    """
    if residual_window.dim() != 3:
        raise ValueError(
            f"residual_window must be (B, T, d_in); got {tuple(residual_window.shape)}"
        )
    B, T_in, d_in = residual_window.shape
    arch_T = arch.T if hasattr(arch, "T") else getattr(arch.config, "T", T_in)
    if T_in != arch_T:
        raise ValueError(
            f"residual_window T={T_in} != arch T={arch_T}"
        )

    with torch.no_grad():
        z_base = arch.encode(residual_window)                # (B, 1 or T, d_sae)
        # Build z'  = z + α·e_f  in latent space.
        z_perturb = z_base.clone()
        if isinstance(magnitude, torch.Tensor):
            mag = magnitude.to(device=z_perturb.device, dtype=z_perturb.dtype)
        else:
            mag = torch.full(
                (B,), float(magnitude),
                device=z_perturb.device, dtype=z_perturb.dtype,
            )
        # z shape varies: (B, 1, d_sae) for shared-z TXC, (B, T, d_sae)
        # for per-position-z. Apply α to feature_id of every t-slot.
        if z_perturb.dim() == 3:
            z_perturb[..., int(feature_id)] = (
                z_perturb[..., int(feature_id)]
                + mag.view(B, *([1] * (z_perturb.dim() - 2)))
            )
        else:
            raise RuntimeError(f"unexpected z shape {tuple(z_perturb.shape)}")

        x_base = arch.decode(z_base)                          # (B, T, d_in)
        x_perturb = arch.decode(z_perturb)                    # (B, T, d_in)
        delta = x_perturb - x_base                            # (B, T, d_in)

        if ref_norm is not None:
            # Rescale per-row so ‖delta_b‖_F = ref_norm · |α_b|. Keeps
            # V3 magnitudes interpretable on the same axis as V0/V4.
            rms = delta.flatten(1).norm(dim=-1).clamp_min(1e-12)  # (B,)
            target = ref_norm * mag.abs()                          # (B,)
            scale = (target / rms).view(B, 1, 1)
            delta = delta * scale

    return delta
