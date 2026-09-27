"""TFA (Temporal Feature Analysis) — TempBenchArch wrapper.

Source: ``origin/wasteland-canonical @ [scrubbed-sha]:src/architectures/tfa.py``
adapted to the unified :class:`TempBenchArch` framework.

Wraps the :class:`TemporalSAE` module (ported from
``_tfa_module.py``). TFA processes full sequences ``(B, T, d)`` and uses
causal attention to decompose each token into:
- ``pred_codes``: codes predictable from prior context via attention
- ``novel_codes``: residual codes identified by per-token TopK SAE

Locked-config knobs (from ``configs/locked_archs.yaml::tfa``):
- ``d_sae=18432`` (default), ``32768`` (c7 override)
- ``k_pos=20``
- ``n_heads=4``
- ``use_pos_embedding=False`` (TFA-pos = the same class with this True)

For training stability on real-LM activations, TFA's internal
``lam = 1/(4 * d_in)`` assumes inputs have norm ~ sqrt(d). Real LLM
activations are 100-1000× larger, so we compute a scaling factor on
the first training batch and apply it uniformly. Matches the reference
TFA paper's scaling_factor convention.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.architectures.base import ArchConfig, TempBenchArch
from temp_bench.architectures._tfa_module import TemporalSAE


class TFA(TempBenchArch):
    """TFA wrapper — temporal SAE with causal-attention pred + per-token novel."""

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        k_pos: int,
        T: int = 5,
        n_heads: int = 4,
        n_attn_layers: int = 1,
        bottleneck_factor: int = 64,
        use_pos_embedding: bool = False,
        max_seq_len: int = 512,
    ):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name="tfa", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T,
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        self.k_pos = k_pos
        self.k_win = k_pos * T  # window-level L0; passed to TemporalSAE.kval_topk

        # Adjust bottleneck_factor so dimin (=d_sae) is divisible by it × n_heads.
        # ManualAttention asserts `dimin % (bottleneck_factor * n_heads) == 0`.
        bf = bottleneck_factor
        while bf > 1 and (d_sae % (bf * n_heads) != 0):
            bf //= 2
        if d_sae % (bf * n_heads) != 0:
            bf = 1
        self._bottleneck_factor = bf

        self.tfa = TemporalSAE(
            dimin=d_in, width=d_sae, n_heads=n_heads,
            sae_diff_type="topk", kval_topk=self.k_win,
            tied_weights=True, n_attn_layers=n_attn_layers,
            bottleneck_factor=bf,
            use_pos_encoding=use_pos_embedding,
            max_seq_len=max_seq_len,
        )

        # Cached scaling factor (computed on first train batch).
        self.register_buffer(
            "_scaling_factor",
            torch.tensor(0.0, dtype=torch.float32),
        )

    # ── scaling helper ───────────────────────────────────────────────

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Apply (or compute on first call) the sqrt(d)/<||x||> scaling."""
        if self._scaling_factor.item() == 0.0:
            with torch.no_grad():
                d = float(x.shape[-1])
                mean_norm = float(x.norm(dim=-1).mean())
                if mean_norm < 1e-8:
                    factor = 1.0
                else:
                    factor = math.sqrt(d) / mean_norm
                self._scaling_factor = self._scaling_factor.new_tensor(factor)
        return x * self._scaling_factor

    # ── encode / decode (TempBenchArch contract) ─────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, T, d_in) → (B, T, d_sae)`` — total codes (pred + novel).

        For C7 mining, this is the canonical "feature activation" view.
        Causal attention requires sequence input; if a single token
        ``(B, d_in)`` is passed we treat it as a singleton sequence.
        """
        squeeze_t = x.dim() == 2
        if squeeze_t:
            x = x.unsqueeze(1)
        x_scaled = self._scale(x)
        _, info = self.tfa(x_scaled)
        z = info["pred_codes"] + info["novel_codes"]
        if squeeze_t:
            z = z.squeeze(1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode ``z @ D + b``. Note: this is the SAE-style decode (no
        attention) — ``z`` here is treated as totals, not split."""
        if z.dim() == 2:
            z = z.unsqueeze(1)
        x_hat_scaled = z @ self.tfa.D + self.tfa.b
        # Undo scaling for the output to live in input units.
        if self._scaling_factor.item() > 0.0:
            x_hat_scaled = x_hat_scaled / self._scaling_factor
        return x_hat_scaled

    # ── train_step: full TFA forward + recon + sparsity ──

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Args:
            x: (B, seq_len, d_in) from canonical batch_iter.
               TFA needs causal context; we treat the whole seq as one
               window (T_seq = seq_len at train-time; encode at C7 eval
               time uses T=6 windows separately).

        Returns:
            (loss, info) — info has 'mse', 'l0', 'z'.
        """
        if x.dim() != 3:
            raise ValueError(
                f"TFA.train_step expects (B, seq_len, d_in); got {tuple(x.shape)}."
            )
        x_scaled = self._scale(x)
        recons, info = self.tfa(x_scaled)
        n_tokens = x_scaled.shape[0] * x_scaled.shape[1]
        mse = F.mse_loss(recons, x_scaled, reduction="sum") / n_tokens
        z_full = info["pred_codes"] + info["novel_codes"]
        l0 = (info["novel_codes"] > 0).float().sum(dim=-1).mean()
        return mse, {"mse": mse.detach(), "l0": l0.detach(), "z": z_full.detach()}

    def post_step(self) -> None:
        """Decoder unit-norm renormalisation per atom (D rows)."""
        with torch.no_grad():
            norms = self.tfa.D.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.tfa.D.data.div_(norms)

    def decoder_directions(self) -> torch.Tensor:
        """``(d_sae, d_in)`` decoder directions = D (already that shape)."""
        return self.tfa.D.data.clone()
