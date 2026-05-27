"""Position-aware TXC steering hooks (V0/V1/V2/V4) + helpers.

The C7 ``SteeringHook`` adds a constant ``d_in`` vector at every
continuation token. For per-token archs (TopK-SAE, T-SAE, MLC) this
is correct. For TXC, it averages away the per-position decoder
trajectory — the entire point of the architecture. The five protocols
here decompose the steering question:

* **V0** mean-decoder constant — current C7 baseline. TopK-SAE-equivalent.
* **V1** position-cycled — each continuation token gets the next slice
  of ``W_dec[f, t mod T, :]``. Cheap drop-in.
* **V2** trailing-window — most-recent T positions get the full
  trajectory ``W_dec[f, T-1, :], W_dec[f, T-2, :], ..., W_dec[f, 0, :]``.
  Most TXC-faithful (matches the training reconstruction objective).
* **V4** encoder pre-image — ``sum_t W_enc[t, :, f]`` normalised. The
  constant vector that maximally activates feature f per unit ‖Δx‖.
  Differs from V0 by exactly the encoder–decoder divergence after
  training.

(V3 latent-space steering — encode current window, perturb in z, decode
— is sufficiently different that it needs its own driver, not a
forward hook. See :mod:`temp_bench.eval.steering_protocols`.)

All four hook modes share:

* ``magnitudes: (B,) tensor`` of per-row scalars — set before each
  forward pass; same convention as the legacy
  :class:`temp_bench.case_studies.backtracking.SteeringHook`.
* energy correction: V1/V2 inject vectors at multiple positions, so
  per-position L2 is divided by ``√T`` to keep total energy comparable
  to the per-token baseline at the same ``magnitude``. Without it,
  V1/V2 silently inject ``√T``-fold more residual energy than V0/V4
  at the same nominal magnitude, breaking cross-arch comparability.

Two diagnostics live alongside:

* :func:`position_variance` — per-feature ``Var_t(W_dec[f, t, :])`` /
  ``mean energy``. Predicts whether the V0-vs-V1/V2 gap is worth
  pursuing per feature.
* :func:`encoder_preimage` — ``sum_t W_enc[t, :, f]``; the V4 vector.

Usage::

    from temp_bench.eval.steering_hooks import (
        TXCSteeringHook, encoder_preimage, position_variance,
    )

    W_dec_f = arch.W_dec.data[feature_id]               # (T, d_in)
    W_enc_pre = encoder_preimage(arch, feature_id)       # (d_in,)
    hook = TXCSteeringHook(
        decoder_trajectory=W_dec_f,
        encoder_preimage=W_enc_pre,
        mode="v2",
        ref_norm=ref_norm,
        T=arch.T,
    )
    handle = layer_module.register_forward_hook(hook)
    hook.magnitudes = mags        # (B,)
    out = model(...)              # forward pass — hook fires
    handle.remove()
"""

from __future__ import annotations

from typing import Literal

import torch

from temp_bench.architectures.base import TempBenchArch

SteeringMode = Literal["v0", "v1", "v2", "v4"]
ALL_MODES: tuple[SteeringMode, ...] = ("v0", "v1", "v2", "v4")


# ── Diagnostics ────────────────────────────────────────────────────────


def position_variance(W_dec: torch.Tensor) -> torch.Tensor:
    """Per-feature normalised position variance of a TXC decoder.

    Args:
        W_dec: ``(d_sae, T, d_in)`` decoder atoms.

    Returns:
        ``(d_sae,)`` floats. Defined as
        ``∑_{t,d} (W_dec[f,t,d] - mean_t W_dec[f,:,d])^2``
        divided by ``∑_{t,d} W_dec[f,t,d]^2``.

    Interpretation:
        * ≈ 0 → trajectory is roughly constant in t. V0 mean-decoder is
          approximately faithful; V1/V2/V3 give little additional signal.
        * ≈ 1 → trajectory averages to ~zero (mean cancels). The mean
          vector is a poor representative; V1/V2/V4 will likely matter.

    The ratio is in ``[0, 1]`` and equals ``1 - ‖mean‖^2 / mean(‖slice‖^2)``
    after weighing with per-position frequencies (always uniform here).
    """
    if W_dec.dim() != 3:
        raise ValueError(f"position_variance expects (d_sae, T, d_in); got {tuple(W_dec.shape)}")
    mean_t = W_dec.mean(dim=1, keepdim=True)              # (d_sae, 1, d_in)
    diff = (W_dec - mean_t).pow(2).sum(dim=(1, 2))        # (d_sae,)
    norm = W_dec.pow(2).sum(dim=(1, 2)).clamp_min(1e-12)  # (d_sae,)
    return diff / norm


def encoder_preimage(
    arch: TempBenchArch,
    feature_id: int | None = None,
) -> torch.Tensor:
    """Encoder pre-image — the constant vector that maximally activates
    feature ``f`` per unit ``‖Δx‖``.

    For TXC archs (W_enc shape ``(T, d_in, d_sae)``):
        ``sum_t W_enc[t, :, f]`` — what feature f's pre-activation
        responds to under a constant Δx.
    For per-token archs (W_enc shape ``(d_in, d_sae)`` or similar):
        Falls back to ``arch.decoder_directions()[f]`` (= W_dec column),
        since ``∂z[f]/∂x = W_enc[:, f]`` and for TopK SAE that equals
        the decoder column at init under tied weights.

    Args:
        arch: any :class:`TempBenchArch`.
        feature_id: a single int → returns ``(d_in,)``. ``None`` →
            returns the full ``(d_sae, d_in)`` matrix (one row per
            feature). The latter is convenient for batched mining.
    """
    W_enc = getattr(arch, "W_enc", None)
    if W_enc is not None and W_enc.dim() == 3:
        # TXC convention: (T, d_in, d_sae)
        if feature_id is None:
            return W_enc.data.sum(dim=0).T.contiguous().clone()  # (d_sae, d_in)
        return W_enc.data[:, :, int(feature_id)].sum(dim=0).clone()  # (d_in,)

    # Per-token / non-TXC fallback. Use decoder direction (TopK SAE's
    # encoder is approximately the decoder column at init under tied
    # weights; gives a reasonable behavioural pre-image).
    dirs = arch.decoder_directions()  # (d_sae, d_in)
    if feature_id is None:
        return dirs.detach().clone()
    return dirs[int(feature_id)].detach().clone()


def encoder_decoder_divergence(
    arch: TempBenchArch,
    feature_id: int,
) -> dict[str, float]:
    """Quantify how far ``encoder_preimage(f)`` has drifted from the
    tied-init prediction ``T · mean_t W_dec[f, :, :]``.

    At init both vectors are equal (W_enc[t] = W_dec[:, t, :].T per
    txc_base init). After training they diverge; the gap is what V4
    captures over V0.

    Returns::

        {
            "cos_sim": float,           # cos(W_enc_pre, T * mean_dec_traj)
            "norm_ratio": float,        # ‖W_enc_pre‖ / ‖T * mean_dec‖
            "rel_residual": float,      # ‖W_enc_pre - T * mean_dec‖ / ‖W_enc_pre‖
        }

    Big rel_residual / low cos_sim → V4 will materially differ from V0.
    """
    pre = encoder_preimage(arch, feature_id).float()
    W_dec = arch.W_dec.data
    if W_dec.dim() != 3:
        raise ValueError(
            f"encoder_decoder_divergence assumes TXC-style W_dec (d_sae, T, d_in); "
            f"got shape {tuple(W_dec.shape)}"
        )
    T = W_dec.shape[1]
    mean_dec = W_dec[int(feature_id)].mean(dim=0).float()  # (d_in,)
    pred = T * mean_dec
    cos_sim = torch.nn.functional.cosine_similarity(
        pre.unsqueeze(0), pred.unsqueeze(0), dim=-1, eps=1e-12,
    ).item()
    norm_pre = pre.norm().item()
    norm_pred = pred.norm().item()
    rel_residual = (pre - pred).norm().item() / max(norm_pre, 1e-12)
    return {
        "cos_sim": float(cos_sim),
        "norm_ratio": float(norm_pre / max(norm_pred, 1e-12)),
        "rel_residual": float(rel_residual),
    }


# ── Hook ───────────────────────────────────────────────────────────────


class TXCSteeringHook:
    """Position-aware steering hook with V0/V1/V2/V4 modes.

    Drop-in successor to :class:`temp_bench.case_studies.backtracking.SteeringHook`
    that respects the TXC decoder trajectory.

    Args:
        decoder_trajectory: ``(T, d_in)`` — ``W_dec[f, :, :]`` for the
            chosen feature ``f``.
        encoder_preimage: ``(d_in,)`` — required only for ``mode="v4"``;
            see :func:`encoder_preimage`.
        mode: ``"v0"`` mean-decoder constant (TopK-SAE-equivalent) /
            ``"v1"`` position-cycled / ``"v2"`` trailing-window /
            ``"v4"`` encoder pre-image.
        ref_norm: target L2 norm — energy match across archs. The same
            ``dom_base_union.norm()`` already used by C7.
        T: window length (must equal ``decoder_trajectory.shape[0]``).
        cycle_phase: V1 only. Starting position phase ∈ [0, T). Useful
            for sweeping all T phases and selecting the best.
        sqrt_t_correction: True (default) divides V1/V2 per-position
            vectors by ``√T`` so total energy injected over T positions
            equals the per-token baseline. Set False to disable
            (matches an under-corrected analysis where V1/V2 magnitudes
            are not directly comparable to V0/V4).

    Attributes:
        magnitudes: ``(B,)`` tensor — set before every forward pass.
            Same convention as C7's ``SteeringHook``.
        token_count: incremented per call. Used by V1 to track phase
            across multi-step generation. Reset via :meth:`reset` between
            cohort qids.
    """

    def __init__(
        self,
        decoder_trajectory: torch.Tensor,
        *,
        mode: SteeringMode,
        ref_norm: float,
        T: int,
        encoder_preimage: torch.Tensor | None = None,
        cycle_phase: int = 0,
        sqrt_t_correction: bool = True,
    ):
        if mode not in ALL_MODES:
            raise ValueError(f"unknown mode={mode!r}; expected one of {ALL_MODES}")
        if decoder_trajectory.dim() != 2:
            raise ValueError(
                f"decoder_trajectory must be (T, d_in); got {tuple(decoder_trajectory.shape)}"
            )
        if decoder_trajectory.shape[0] != T:
            raise ValueError(
                f"decoder_trajectory.shape[0]={decoder_trajectory.shape[0]} != T={T}"
            )
        if mode == "v4":
            if encoder_preimage is None:
                raise ValueError("mode='v4' requires encoder_preimage")
            if encoder_preimage.dim() != 1 or encoder_preimage.shape[0] != decoder_trajectory.shape[1]:
                raise ValueError(
                    f"encoder_preimage must be (d_in={decoder_trajectory.shape[1]},); "
                    f"got {tuple(encoder_preimage.shape)}"
                )
        if not 0 <= cycle_phase < T:
            raise ValueError(f"cycle_phase must be in [0, T={T}); got {cycle_phase}")

        self.W = decoder_trajectory.detach().clone()
        self.enc = encoder_preimage.detach().clone() if encoder_preimage is not None else None
        self.mode: SteeringMode = mode
        self.T = int(T)
        self.ref_norm = float(ref_norm)
        self.cycle_phase = int(cycle_phase)
        self.sqrt_t_correction = bool(sqrt_t_correction)
        self.magnitudes: torch.Tensor | None = None
        self.token_count: int = 0
        # Per-device cache for the materialised vectors
        self._cache_device: torch.device | None = None
        self._cache_dtype: torch.dtype | None = None
        self._W_cached: torch.Tensor | None = None
        self._enc_cached: torch.Tensor | None = None

    # ── Lifecycle ──

    def reset(self) -> None:
        """Reset position-cycle counter. Call between cohort qids so
        V1 starts from ``cycle_phase`` again instead of continuing the
        previous qid's phase."""
        self.token_count = 0

    def _materialize(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Move (and cache) trajectory + pre-image onto the ref tensor's
        device + dtype. Avoids reallocating on every forward pass."""
        if (
            self._W_cached is None
            or self._cache_device != ref.device
            or self._cache_dtype != ref.dtype
        ):
            self._W_cached = self.W.to(device=ref.device, dtype=ref.dtype)
            self._enc_cached = (
                self.enc.to(device=ref.device, dtype=ref.dtype) if self.enc is not None else None
            )
            self._cache_device = ref.device
            self._cache_dtype = ref.dtype
        return self._W_cached, self._enc_cached

    def _scale(self) -> float:
        """Per-position L2 budget. V1/V2 inject at multiple positions →
        divide by √T to match per-token energy."""
        if self.sqrt_t_correction and self.mode in ("v1", "v2"):
            return self.ref_norm / (self.T ** 0.5)
        return self.ref_norm

    # ── Forward hook ──

    def __call__(self, _module, _inp, output):
        if self.magnitudes is None or torch.count_nonzero(self.magnitudes) == 0:
            return output
        x = output[0] if isinstance(output, tuple) else output
        # x: (B, S, d_in)
        if x.dim() != 3:
            raise RuntimeError(
                f"TXCSteeringHook expects (B, S, d_in); got {tuple(x.shape)}"
            )
        W_cached, enc_cached = self._materialize(x)
        B, S, D = x.shape
        scale = self._scale()
        mags = self.magnitudes.to(device=x.device, dtype=x.dtype)

        if self.mode == "v0":
            v = W_cached.mean(dim=0)                                   # (d_in,)
            v = v / v.norm().clamp_min(1e-8) * scale
            delta = mags.view(B, 1, 1) * v.view(1, 1, D)
        elif self.mode == "v1":
            # Per-step decoder slice; deterministic across the batch.
            slot = (
                torch.arange(S, device=x.device, dtype=torch.long)
                + self.token_count
                + self.cycle_phase
            ) % self.T                                                  # (S,)
            v_step = W_cached[slot]                                     # (S, d_in)
            v_step = v_step / v_step.norm(dim=-1, keepdim=True).clamp_min(1e-8) * scale
            delta = mags.view(B, 1, 1) * v_step.view(1, S, D)
        elif self.mode == "v2":
            # Trailing window: last T positions of the current forward
            # batch get the trajectory in reverse-position order. The
            # most recent slot gets W[T-1, :], second-most gets W[T-2, :],
            # ..., (T-1)-back gets W[0, :]. Earlier positions are
            # untouched. This matches the TXC training objective
            # (reconstruct the past T positions from one window-level z).
            #
            # Cumulative steering across past forward calls is OUT OF
            # SCOPE for a stateless forward hook (would need a per-row
            # rolling buffer). For the C7 cut-and-continue protocol
            # (one forward per generated token after KV-caching kicks in)
            # this is equivalent to "trajectory once, at the start of
            # the continuation". For longer batched continuations the
            # interpretation is "trajectory inside this batch's window".
            n_apply = min(self.T, S)
            delta = torch.zeros_like(x)
            for j in range(n_apply):
                v = W_cached[self.T - 1 - j]
                v = v / v.norm().clamp_min(1e-8) * scale
                delta[:, S - 1 - j, :] = mags.view(B, 1) * v.view(1, D)
        elif self.mode == "v4":
            assert enc_cached is not None
            v = enc_cached / enc_cached.norm().clamp_min(1e-8) * scale  # (d_in,)
            delta = mags.view(B, 1, 1) * v.view(1, 1, D)
        else:                                                            # pragma: no cover
            raise RuntimeError(f"unhandled mode {self.mode!r}")

        self.token_count += S
        if isinstance(output, tuple):
            return (x + delta,) + output[1:]
        return x + delta


# ── A/B harness ────────────────────────────────────────────────────────


def build_hook(
    arch: TempBenchArch,
    *,
    feature_id: int,
    mode: SteeringMode,
    ref_norm: float,
    cycle_phase: int = 0,
    sqrt_t_correction: bool = True,
) -> TXCSteeringHook:
    """Convenience constructor: pull the trajectory + pre-image off the
    arch and wire them up correctly for ``mode``."""
    W_dec = arch.W_dec.data
    if W_dec.dim() != 3:
        raise ValueError(
            "build_hook assumes TXC-style W_dec (d_sae, T, d_in); "
            f"got shape {tuple(W_dec.shape)}. Use the legacy "
            "SteeringHook for per-token archs."
        )
    traj = W_dec[int(feature_id)].clone()                  # (T, d_in)
    pre = encoder_preimage(arch, feature_id) if mode == "v4" else None
    return TXCSteeringHook(
        decoder_trajectory=traj,
        encoder_preimage=pre,
        mode=mode,
        ref_norm=float(ref_norm),
        T=W_dec.shape[1],
        cycle_phase=cycle_phase,
        sqrt_t_correction=sqrt_t_correction,
    )
