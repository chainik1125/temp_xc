"""SAEBench-compatible wrapper over our ArchSpec-based architectures.

SAEBench's sparse-probing eval duck-types the SAE interface: it needs
`encode()`, `decode()`, `forward()` methods plus attributes
`{W_enc, W_dec, b_enc, b_dec, dtype, device, cfg}`. This class
exposes all that over our spec-based TopKSAE / TemporalCrosscoder
models (MLC pending — see plan § 11).

**Shape contract:** `encode((B, L, d_in)) → (B, L, d_sae_effective)`
where `d_sae_effective = T × d_sae` under `full_window` aggregation
and `d_sae` otherwise.

**Aggregation logic lives here** (not in the underlying model) because
SAE has no T axis and needs identity aggregation, while TempXC needs
to window, encode each T-slice, and collapse T via one of four
strategies.

Downstream, SAEBench calls `get_sae_meaned_activations`, which
mean-pools the encoded `(B, L, d_sae_effective)` tensor across
non-BOS/pad positions to `(B, d_sae_effective)` and hands that to
the k-sparse logistic-regression probe.

See docs/aniket/experiments/sparse_probing/plan.md § 5.
See docs/aniket/experiments/sparse_probing/saebench_notes.md § 2 for
the SAEBench `BaseSAE` contract we duck-type against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from src.bench.architectures.base import ArchSpec
from src.bench.saebench.aggregation import (
    AggregationName,
    aggregate,
    effective_d_sae,
)
from src.bench.saebench.configs import (
    ArchName,
    D_MODEL,
    D_SAE,
    HOOK_NAME,
    LAYER,
    SUBJECT_MODEL,
)


@dataclass
class AdapterConfig:
    """Shim for SAEBench's `CustomSAEConfig` — just the fields sparse_probing
    reads. Other SAEBench evals read more; extend when we run them.
    """

    d_in: int
    d_sae: int                    # effective d_sae after aggregation
    hook_layer: int = LAYER
    hook_name: str = HOOK_NAME
    model_name: str = SUBJECT_MODEL
    architecture: str = "custom"  # free-form label SAEBench logs
    context_size: int = 128
    apply_b_dec_to_input: bool = False
    finetuning_scaling_factor: bool = False
    activation_fn_str: str = "topk"
    prepend_bos: bool = True
    normalize_activations: str = "none"
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class SAEBenchAdapter(nn.Module):
    """Wraps an ArchSpec + trained model into SAEBench's SAE contract.

    Use:
        spec = CrosscoderSpec(T=5)
        model = load_checkpoint(...)
        adapter = SAEBenchAdapter(
            arch="tempxc", spec=spec, model=model,
            t=5, aggregation="mean",
        )
        # hand `adapter` to SAEBench's run_eval via selected_saes list.

    The adapter's `dtype` and `device` attributes are read directly by
    SAEBench, not inferred from parameters — we expose them as plain
    attributes on `__init__`.
    """

    def __init__(
        self,
        arch: ArchName,
        spec: ArchSpec,
        model: nn.Module,
        t: int,
        aggregation: AggregationName,
        d_in: int = D_MODEL,
        d_sae: int = D_SAE,
        hook_layer: int = LAYER,
        hook_name: str = HOOK_NAME,
        shuffle_seed: int | None = None,
    ):
        """
        shuffle_seed: if set, `encode(x)` permutes tokens within each
            sequence using `torch.Generator().manual_seed(shuffle_seed + seq_idx)`
            before calling the underlying arch. Enables the
            ordered + shuffled paired probing the team agreed on in the
            4/18 meeting. Two adapters called with the same shuffle_seed
            see identically permuted inputs — paired delta metrics are
            fair. None (default) disables shuffling.
        """
        super().__init__()
        self._arch_name = arch
        self._spec = spec
        self._model = model
        self._t = t
        self._aggregation = aggregation
        self._base_d_sae = d_sae
        self._effective_d_sae = effective_d_sae(d_sae, t, aggregation)
        self._shuffle_seed = shuffle_seed

        self.cfg = AdapterConfig(
            d_in=d_in,
            d_sae=self._effective_d_sae,
            hook_layer=hook_layer,
            hook_name=hook_name,
            architecture=f"{arch}_t{t}_{aggregation}",
        )

        # SAEBench reads these directly (not via next(self.parameters()))
        first_param = next(self._model.parameters(), None)
        if first_param is not None:
            self.dtype = first_param.dtype
            self.device = first_param.device
        else:
            self.dtype = torch.float32
            self.device = torch.device("cpu")

    # ─── required SAEBench duck-typed attributes ─────────────────────
    @property
    def W_enc(self) -> torch.Tensor:
        """SAEBench shape `(d_in, d_sae_effective)`.

        sparse_probing doesn't use W_enc (it only calls `encode()`),
        but other SAEBench evals (SCR, TPP, absorption) do. For SAE
        this is the native weight; for TempXC we return a
        position-averaged view of `(T, d_in, d_sae) → (d_in, d_sae)`.
        Full-window aggregation isn't supported here (effective d_sae
        is T×d_sae, which has no natural W_enc).
        """
        if self._arch_name == "sae":
            return self._model.W_enc.T.contiguous()  # TopKSAE: (d_sae, d_in) → (d_in, d_sae)
        if self._arch_name in ("tempxc", "mlc"):
            # Both use (T-or-n_layers, d_in, d_sae) native shape; mean over
            # the leading axis for SAEBench's (d_in, d_sae) view.
            with torch.no_grad():
                return self._model.W_enc.mean(dim=0).contiguous()
        raise ValueError(f"unknown arch: {self._arch_name}")

    @property
    def W_dec(self) -> torch.Tensor:
        """SAEBench shape `(d_sae_effective, d_in)` with unit-norm rows."""
        if self._arch_name == "sae":
            return self._model.W_dec.T.contiguous()  # TopKSAE: (d_in, d_sae) → (d_sae, d_in)
        if self._arch_name in ("tempxc", "mlc"):
            # Both: (d_sae, T-or-n_layers, d_in); mean across axis 1.
            with torch.no_grad():
                return self._model.W_dec.mean(dim=1).contiguous()
        raise ValueError(f"unknown arch: {self._arch_name}")

    @property
    def b_enc(self) -> torch.Tensor:
        return self._model.b_enc

    @property
    def b_dec(self) -> torch.Tensor:
        if self._arch_name == "sae":
            return self._model.b_dec
        # TempXC / MLC: b_dec is (T-or-n_layers, d_in); mean across axis 0.
        with torch.no_grad():
            return self._model.b_dec.mean(dim=0)

    # ─── required SAEBench methods ───────────────────────────────────
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, L, d_in)` → `(B, L, d_sae_effective)`.

        SAEBench calls this inside `get_sae_meaned_activations` after
        zeroing BOS/pad positions in `x`. We forward to the underlying
        architecture, applying T-windowing + aggregation for TempXC.

        If `self._shuffle_seed` is set, tokens within each sequence are
        permuted BEFORE the encode (but after BOS/pad zeroing, since
        SAEBench's caller handles that). This is the harness's ordered +
        shuffled control knob; two adapters sharing a shuffle_seed
        receive identically permuted inputs.
        """
        assert x.ndim == 3, f"expected (B, L, d_in), got {tuple(x.shape)}"
        B, L, d_in = x.shape

        if self._shuffle_seed is not None:
            # Canonical shuffle helper. Keeps the data-pipeline concern
            # out of architecture code AND out of this wrapper's
            # implementation (pushback on duplicate inline loops).
            from src.bench.architectures.base import shuffle_within_sequence
            x = shuffle_within_sequence(x, self._shuffle_seed)

        if self._arch_name == "sae":
            # TopKSAE is per-token, no T axis. Flatten → encode → reshape.
            flat = x.reshape(B * L, d_in)
            z = self._model.encode(flat)           # (B*L, d_sae)
            return z.reshape(B, L, -1).contiguous()

        if self._arch_name == "tempxc":
            return self._encode_tempxc(x)

        if self._arch_name == "mlc":
            # MLC crosscodes across *layers* not time. SAEBench's
            # sparse_probing only hooks one layer, so the (B, L, d_in)
            # input here is single-layer activations at layer 12 —
            # insufficient for MLC which needs (B, L, 5, d_model)
            # simultaneous layer acts {10..14}. Resolving this requires
            # a multi-hook variant of SAEBench's run_eval (plan § 11).
            # Until that lands, MLC probing is a deliberate TODO.
            raise NotImplementedError(
                "MLC probing through SAEBench's standard pipeline is not "
                "supported — SAEBench hooks only one layer, MLC needs 5. "
                "See docs/aniket/experiments/sparse_probing/plan.md § 11 "
                "for the multi-hook integration roadmap. The MLC "
                "architecture itself is trainable via the sweep runner "
                "once the multi-layer data pipeline lands."
            )
        raise ValueError(f"unknown arch: {self._arch_name}")

    def _encode_tempxc(self, x: torch.Tensor) -> torch.Tensor:
        """TempXC long-sequence encode with T-windowing and aggregation.

        Dispatches to `_encode_tempxc_batch` in chunks of B to bound peak
        VRAM. With SAEBench's probing passing B ~200+, a single-shot
        (B, L, T, d_sae) tensor for Gemma-2-2B at T=5 is 10+ GB, and
        two of them (intermediate z + padded windows) OOMs an 80 GB H100
        after Gemma's 5 GB and SAEBench's cached activations are resident.
        """
        B = x.shape[0]
        # Target peak alloc per chunk ~1.5 GB for the padded windows tensor.
        # chunk_B * L * T * d_sae * 4B ≈ 1.5e9
        #   at L=128, d_sae=18432, T=5  → chunk_B ≈ 32
        #   at L=128, d_sae=18432, T=20 → chunk_B ≈ 8
        L = x.shape[1]
        target_bytes = 1_500_000_000
        chunk_B = max(
            1,
            target_bytes // (L * self._t * self._base_d_sae * 4),
        )
        if B <= chunk_B:
            return self._encode_tempxc_batch(x)

        outs: list[torch.Tensor] = []
        for i in range(0, B, chunk_B):
            outs.append(self._encode_tempxc_batch(x[i : i + chunk_B]))
        return torch.cat(outs, dim=0)

    def _encode_tempxc_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Inner single-chunk TempXC encode.

        Strategy:
          1. Build stride-1 T-wide windows from the `(B, L, d_in)` input
             → `(B, n_windows, T, d_in)` where n_windows = L - T + 1.
          2. Encode each window via the spec's `_encode_window` (returns
             `(B * n_windows, T, d_sae)` with Option-B per-position
             contribution values).
          3. Pad back to per-token shape `(B, L, T, d_sae)` so every
             token `t` has an associated T-window. Tokens near edges
             (without enough context) are zero-filled.
          4. Apply the aggregation strategy → `(B, L, d_sae_effective)`.

        SAE/MLC use different paths; only TempXC goes through here.
        """
        from src.bench.architectures.crosscoder import CrosscoderSpec
        assert isinstance(self._spec, CrosscoderSpec), (
            f"_encode_tempxc_batch called with spec type {type(self._spec).__name__}"
        )

        B, L, d_in = x.shape
        T = self._t
        if L < T:
            # Short-sequence fallback: pad and encode as a single window
            pad = torch.zeros(B, T - L, d_in, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=1)
            z = self._spec._encode_window(self._model, x_padded)  # (B, T, d_sae)
            windows = z.unsqueeze(1).expand(-1, L, -1, -1).contiguous()
            return aggregate(windows, self._aggregation)

        # L >= T: full windowing
        n_windows = L - T + 1
        win_in = x.unfold(1, T, 1).permute(0, 1, 3, 2).contiguous()  # (B, n_windows, T, d_in)
        flat = win_in.reshape(B * n_windows, T, d_in)

        # Chunk through _encode_window to keep peak VRAM bounded for the
        # inner einsum; at chunk=1024 that's ~380 MB for T=5 d_sae=18432.
        chunk = 1024
        outs: list[torch.Tensor] = []
        for i in range(0, flat.shape[0], chunk):
            outs.append(self._spec._encode_window(self._model, flat[i : i + chunk]))
        z_flat = torch.cat(outs, dim=0)  # (B * n_windows, T, d_sae)
        z = z_flat.view(B, n_windows, T, self._base_d_sae)
        del z_flat

        # Pad per-token: token at position t uses the window ending at t.
        # window index w covers tokens [w .. w+T-1]; the window ending at
        # token t (for t >= T-1) is at w = t - T + 1.
        windows = torch.zeros(
            B, L, T, self._base_d_sae,
            device=z.device, dtype=z.dtype,
        )
        for t in range(T - 1, L):
            windows[:, t, :, :] = z[:, t - (T - 1), :, :]
        del z

        return aggregate(windows, self._aggregation)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """`(B, L, d_sae_effective)` → `(B, L, d_in)`.

        sparse_probing doesn't call decode. Provided for interface
        compatibility with other SAEBench evals (SCR / TPP). For
        aggregated/full-window outputs this is a pseudoinverse path
        that isn't exact; we only use it as a contract filler.
        """
        raise NotImplementedError(
            "decode() not implemented for the probing wrapper. "
            "Sparse-probing doesn't call this; other evals may need it."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Contract stub: return encoded features.

        SAEBench's sparse_probing only calls `encode()`. `forward()` is
        required by `BaseSAE` ABC but not used on this path. We alias
        to encode so `isinstance` checks and contract probes pass.
        """
        return self.encode(x)
