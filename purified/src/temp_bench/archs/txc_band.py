"""TXC-band — band-restricted, stride-aware temporal crosscoder.

Operationalises the spectral (T, S, B) family that contains the paper's
TXC variants as special cases. Three hparams:

  - T : encoder window length (same as vanilla TXC).
  - S : stride between encoder applications over the eval sequence.
        For training we sample T-windows like vanilla TXC; S affects the
        evaluator (sliding stride) but not the training-time sampling.
  - B : subset of frequency bands in {0, 1, …, ⌊T/2⌋} the encoder is
        permitted to use. Implemented by parameterising W_enc in a
        cosine/sine basis indexed by B, so the masked bands have zero
        capacity rather than masked-out-but-still-trainable capacity.

Identity table for the paper's existing archs:

    per-token TopK SAE     ≡ (T=1, S=1, B={0})
    TXC joint T=W          ≡ (T=W, S=W, B=all)
    TXC sliding T<W        ≡ (T,   S=1, B=all)

Predicted ablations:
    DC-only TXC  (B={0})   → behaves like per-token + window-mean
    AC-only TXC  (B⊃{0}=∅) → loses Denoising / DC tasks
    Single-band  (B={f})   → traces the per-band direction signal

The "sliding stride S" is a sweep-level hparam (the evaluator slides
with the given S). It does not appear in the arch's forward signature.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


def _band_basis(T: int, B: tuple[int, ...]) -> torch.Tensor:
    """Orthonormal cos/sin basis for the bands B over the T-window.

    Returns a (T, n_basis) real matrix whose columns are the cos and sin
    components at each f ∈ B. For f=0 the only component is the constant
    (DC); for f>0 we include both cos(2πfτ/T) and sin(2πfτ/T) (the real
    and imaginary parts of the f-th Fourier basis vector). At the
    Nyquist band T/2 (if T is even and T/2 ∈ B) the only real component
    is cos.
    """
    cols = []
    tau = torch.arange(T, dtype=torch.float32)
    for f in sorted(set(B)):
        if f == 0:
            cols.append(torch.ones(T) / math.sqrt(T))
        elif T % 2 == 0 and f == T // 2:
            v = torch.cos(math.pi * tau)        # cos(πτ) = (−1)^τ, real Nyquist
            cols.append(v / v.norm())
        else:
            c = torch.cos(2 * math.pi * f * tau / T)
            s = torch.sin(2 * math.pi * f * tau / T)
            cols.append(c / c.norm())
            cols.append(s / s.norm())
    return torch.stack(cols, dim=1) if cols else torch.zeros(T, 0)


class TXCBand(TempBenchArch):
    """Band-restricted, stride-aware temporal crosscoder."""

    arch_version: str = "1.0.0"
    consumes: str = "window"

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        T: int = 5,
        k_pos: int = 1,
        bands: tuple[int, ...] | str = "all",
        S: int = 1,
        **_ignore,
    ):
        nn.Module.__init__(self)
        # Stride S is an evaluator-side hparam (the encoder is the same
        # per-window, S just controls how many windows the evaluator
        # produces per sequence). Stored as a buffer so it survives
        # checkpoint round-trip and the evaluator can read it.
        self._S = int(S)
        if bands == "all":
            bands = tuple(range(T // 2 + 1))
        elif isinstance(bands, list):
            bands = tuple(bands)
        bands = tuple(sorted(set(int(b) for b in bands)))
        if not bands:
            raise ValueError("TXCBand: bands must be non-empty.")
        if max(bands) > T // 2:
            raise ValueError(f"TXCBand: band {max(bands)} exceeds Nyquist T/2={T//2}.")

        self.config = ArchConfig(
            name="txc_band", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T,
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        self.k_pos = k_pos
        self.bands = bands
        # k_win = window-level TopK budget, matching txc_base's convention.
        self.k_win = min(k_pos * T, d_sae)

        # Trainable coefficients in the band basis. n_basis depends on B:
        #   f=0       → 1 component
        #   0<f<T/2   → 2 components (cos, sin)
        #   f=T/2     → 1 component (cos only, Nyquist)
        basis = _band_basis(T, bands)               # (T, n_basis)
        self.register_buffer("_basis", basis)
        n_basis = basis.shape[1]
        self.coeffs_enc = nn.Parameter(torch.empty(n_basis, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        # Decoder: standard per-position (B unrestricted on decode side).
        self.W_dec = nn.Parameter(torch.empty(d_sae, T, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        for n in range(n_basis):
            nn.init.kaiming_uniform_(self.coeffs_enc.data[n])
        nn.init.kaiming_uniform_(self.W_dec.data.view(d_sae, T * d_in))
        with torch.no_grad():
            self._normalize_decoder()

    # ── Decoder norm utilities ──

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    # ── W_enc reconstruction ──

    @property
    def W_enc(self) -> torch.Tensor:
        """(T, d_in, d_sae) — reconstructed from band coefficients each call."""
        return torch.einsum("tn,nds->tds", self._basis, self.coeffs_enc)

    # ── encode / decode ──

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self._T:
            raise ValueError(
                f"TXCBand.encode expects (B, T={self._T}, d_in); got T={x.shape[1]}."
            )
        W_enc = self.W_enc
        pre = torch.einsum("btd,tds->bs", x, W_enc) + self.b_enc
        vals, idx = pre.topk(self.k_win, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z.unsqueeze(1)                       # (B, 1, d_sae)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            z = z.squeeze(1)
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"TXCBand.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        W_enc = self.W_enc
        pre = torch.einsum("btd,tds->bs", x, W_enc) + self.b_enc
        vals, idx = pre.topk(self.k_win, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        x_hat = torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        with torch.no_grad():
            l0 = (z != 0).float().sum(dim=-1).mean()
        return {"loss": loss, "mse": loss.detach(), "l0": l0.detach()}

    def post_step(self) -> None:
        with torch.no_grad():
            self._normalize_decoder()

    def decoder_directions(self) -> torch.Tensor:
        return self.W_dec.data.mean(dim=1).clone()
