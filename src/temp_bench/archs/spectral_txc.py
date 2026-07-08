"""Spectral (DCT-band) temporal crosscoder on the fair BatchTopK backbone.

Port of Dmitry's ``SpectralTXC`` (``origin/dmitry-spectral-sprint2``,
``docs/dmitry/sprints/2026-06-10_freqbench_sprint/code/fb_core.py``) onto our
strong backbone — BatchTopK during training → fixed JumpReLU threshold at
inference (Bussmann et al.) + AuxK dead-feature revival + decoder unit-norm +
grad-parallel removal — matching the rest of the BatchTopK family
(``experiments/explorations/synthetic/STATUS.md`` § 4). The sprint's arch used
plain per-branch TopK and no AuxK; this re-implements the **band
parameterisation** on our backbone.

The window crosscoder squashes each ``T``-window into ONE shared code and
reconstructs all ``T`` positions from it (like ``txc_batchtopk_post``). The
spectral twist: encoder/decoder kernels are constrained to **DCT bands** of the
time axis. Each atom lives in one band; the code is the concatenation of the
per-band codes. Vanilla TXC is the single-full-band special case (the DCT is an
orthonormal rotation of the raw time axis).

Band parameterisation (per band ``b`` with DCT-index set ``band``):

    e_{h,τ,:} = Σ_{w∈band} ψ_w(τ) · C^enc_{h,w,:}          (time-domain kernel)

where ``ψ`` is the orthonormal DCT-II basis. Constraining the coefficient
support to ``band`` **is** the band constraint (the kernel is band-limited). By
Parseval (ψ orthonormal) the coefficient L2 equals the time-domain L2, so
unit-norming the decoder coefficient vector unit-norms the decoder atom.

**Per-band budgets — the fair-backbone port note.** BatchTopK is applied
**per band** (a batch-level top-``k_b·B`` mask restricted to each band's atoms),
NOT a single global BatchTopK over all bands — otherwise the equal-L0-per-band
guarantee (which drives the band decomposition) disappears. The total window
budget is ``k_win = k_pos·T`` split across bands, so the spectral arch sits at
the same total-per-window budget as ``txc_batchtopk_pre`` (whose shared code
also supports up to ``k_pos·T`` atoms) — making spectral-vs-pre the clean
band-structure test.

Band modes (``bands`` hparam):
- ``"multiband"`` (default, 4 bands): DC ``{0}`` + three contiguous AC groups
  (for ``T=16``: ``{1-5},{6-10},{11-15}`` — the sprint split). Degenerates to
  the available bands at small ``T`` (``T=2`` → DC/AC).
- ``"dcac"`` (2 bands): DC ``{0}`` and AC ``{1..T-1}`` (the frequency_lens split).
- ``"full"`` (1 band): the vanilla DCT crosscoder (== ``txc_batchtopk_post`` up
  to the orthonormal time-axis rotation).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


def _dct_basis(T: int) -> torch.Tensor:
    """(T, T) orthonormal DCT-II basis; row ``w`` = temporal-frequency index w."""
    tau = np.arange(T)
    psi = np.zeros((T, T), dtype=np.float32)
    for w in range(T):
        if w == 0:
            psi[w] = np.sqrt(1.0 / T)
        else:
            psi[w] = np.sqrt(2.0 / T) * np.cos(np.pi * (tau + 0.5) * w / T)
    return torch.from_numpy(psi)


def _build_bands(T: int, mode: str) -> list[list[int]]:
    """DCT-index sets per band, adapted to the window length ``T``."""
    if mode == "full":
        return [list(range(T))]
    if mode == "dcac":
        ac = list(range(1, T))
        return [[0]] + ([ac] if ac else [])
    if mode == "multiband":
        edges = [1, 1 + (T - 1) // 3, 1 + 2 * (T - 1) // 3, T]
        bands = [[0]]
        for i in range(3):
            seg = list(range(edges[i], edges[i + 1]))
            if seg:
                bands.append(seg)
        return bands
    raise ValueError(f"unknown bands mode {mode!r} (use multiband|dcac|full)")


def _split_evenly(total: int, n: int, minimum: int = 0) -> list[int]:
    """Split ``total`` into ``n`` parts as evenly as possible (each ≥ minimum)."""
    base = max(minimum, total // n)
    out = [base] * n
    rem = total - sum(out)
    i = 0
    while rem > 0:
        out[i % n] += 1
        rem -= 1
        i += 1
    return out


class SpectralTXCBatchTopK(TempBenchArch):
    """DCT-band window crosscoder on the BatchTopK → JumpReLU backbone."""

    arch_version: str = "1.0.0"
    consumes: str = "window"
    _registry_name: str = "spectral_txc"

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        T: int = 8,
        k_pos: int = 1,
        bands: str = "multiband",
        auxk_alpha: float = 1.0 / 32.0,
        dead_threshold_tokens: int = 10_000_000,
        aux_k: int | None = None,
        threshold_start_step: int = 1000,
        threshold_beta: float = 0.999,
    ):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name=self._registry_name, d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T,
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        self.k_pos = int(k_pos)
        self.bands_mode = bands
        self.auxk_alpha = auxk_alpha
        self.dead_threshold_tokens = dead_threshold_tokens
        self.aux_k = aux_k if aux_k is not None else min(512, d_in // 2)
        self.threshold_start_step = threshold_start_step
        self.threshold_beta = threshold_beta

        self.bands = _build_bands(T, bands)
        self.n_bands = len(self.bands)
        if d_sae < self.n_bands:
            raise ValueError(f"d_sae ({d_sae}) < n_bands ({self.n_bands})")
        # Atom counts + per-band per-window budget (k_win = k_pos·T split across bands).
        self.h_per_band = _split_evenly(d_sae, self.n_bands, minimum=1)
        k_win = self.k_pos * T
        self.k_per_band = _split_evenly(k_win, self.n_bands, minimum=1)
        # Guard: each band needs ≥ its budget atoms.
        for b, (h_b, k_b) in enumerate(zip(self.h_per_band, self.k_per_band)):
            if h_b < k_b:
                raise ValueError(
                    f"band {b}: h_b ({h_b}) < k_b ({k_b}); raise d_sae or lower k_pos."
                )
        # Contiguous slices of the concatenated (B, d_sae) code, one per band.
        self.band_slices = []
        s = 0
        for h_b in self.h_per_band:
            self.band_slices.append((s, s + h_b))
            s += h_b

        # Per-band coefficient params + DCT-row buffers.
        self.enc_coef = nn.ParameterList()
        self.dec_coef = nn.ParameterList()
        self.b_enc = nn.ParameterList()
        psi = _dct_basis(T)
        for band, h_b in zip(self.bands, self.h_per_band):
            nb = len(band)
            scale = 1.0 / float(np.sqrt(nb * d_in))
            self.enc_coef.append(nn.Parameter(torch.randn(h_b, nb, d_in) * scale))
            self.dec_coef.append(nn.Parameter(torch.randn(h_b, nb, d_in) * scale))
            self.b_enc.append(nn.Parameter(torch.zeros(h_b)))
            self.register_buffer(f"psi_{len(self.b_enc) - 1}",
                                 psi[band].clone(), persistent=False)
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        # Parseval decoder unit-norm at init; tie encoder = decoder.
        with torch.no_grad():
            self._normalize_decoder()
            for b in range(self.n_bands):
                self.enc_coef[b].data.copy_(self.dec_coef[b].data)

        # BatchTopK → per-band JumpReLU threshold + dead tracker.
        self.register_buffer("threshold", torch.full((self.n_bands,), -1.0))
        self.register_buffer("num_tokens_since_fired",
                             torch.zeros(d_sae, dtype=torch.long))
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        for b in range(self.n_bands):
            self.dec_coef[b].register_post_accumulate_grad_hook(self._project_dec_grad)

    # ── kernel synthesis ──

    def _psi(self, b: int) -> torch.Tensor:
        return getattr(self, f"psi_{b}")                       # (nb, T)

    def _enc_kernel(self, b: int) -> torch.Tensor:
        return torch.einsum("wt,hwd->htd", self._psi(b), self.enc_coef[b])  # (h_b,T,d)

    def _dec_kernel(self, b: int) -> torch.Tensor:
        return torch.einsum("wt,hwd->htd", self._psi(b), self.dec_coef[b])  # (h_b,T,d)

    def _dec_full(self) -> torch.Tensor:
        """(d_sae, T, d_in) — concatenated decoder atoms."""
        return torch.cat([self._dec_kernel(b) for b in range(self.n_bands)], dim=0)

    # ── decoder-norm utilities (Parseval on coefficients) ──

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        for D in self.dec_coef:
            n = D.data.flatten(1).norm(dim=1).clamp(min=1e-8)     # coeff L2 == kernel L2
            D.data /= n[:, None, None]

    @staticmethod
    def _project_dec_grad(param: torch.Tensor) -> None:
        if param.grad is None:
            return
        h = param.shape[0]
        W = param.data.view(h, -1)
        g = param.grad.data.view(h, -1)
        normed = W / (W.norm(dim=1, keepdim=True) + 1e-6)
        parallel = (g * normed).sum(dim=1, keepdim=True)
        g.sub_(parallel * normed)

    # ── pre-activation + per-band BatchTopK ──

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        """Squashed (summed over T) ReLU pre-activation ``(B, d_sae)``."""
        outs = []
        for b in range(self.n_bands):
            E = self._enc_kernel(b)                            # (h_b, T, d_in)
            pre_b = torch.einsum("btd,htd->bh", x, E) + self.b_enc[b]
            outs.append(pre_b)
        return F.relu(torch.cat(outs, dim=-1))                 # (B, d_sae)

    def _batchtopk_band(self, pre_b: torch.Tensor, k_b: int) -> torch.Tensor:
        """Flat BatchTopK within one band; budget ``k_b`` per window (pool B)."""
        B = pre_b.shape[0]
        k_total = k_b * B
        flat = pre_b.reshape(-1)
        if k_total >= flat.numel():
            return pre_b
        tk = flat.topk(k_total, sorted=False)
        return (torch.zeros_like(flat)
                .scatter_(-1, tk.indices, tk.values)
                .reshape(pre_b.shape))

    def _select(self, pre: torch.Tensor) -> torch.Tensor:
        """Per-band BatchTopK (train) or per-band JumpReLU threshold (eval)."""
        z = torch.empty_like(pre)
        use_threshold = (not self.training) and bool((self.threshold >= 0).all())
        for b, (s, e) in enumerate(self.band_slices):
            pre_b = pre[:, s:e]
            if use_threshold:
                z[:, s:e] = pre_b * (pre_b > self.threshold[b])
            else:
                z[:, s:e] = self._batchtopk_band(pre_b, self.k_per_band[b])
        return z

    # ── encode / decode ──

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self._T:
            raise ValueError(
                f"{type(self).__name__}.encode expects (B, T={self._T}, d_in); "
                f"got T={x.shape[1]}."
            )
        return self._select(self._pre(x)).unsqueeze(1)         # (B, 1, d_sae)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            if z.shape[1] != 1:
                raise ValueError(
                    f"{type(self).__name__}.decode expects (B, 1, d_sae); "
                    f"got T={z.shape[1]}."
                )
            z = z.squeeze(1)
        return torch.einsum("bs,std->btd", z, self._dec_full()) + self.b_dec

    # ── train_step ──

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"{type(self).__name__}.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, T, _d = x.shape
        pre = self._pre(x)                                     # (B, d_sae)
        z = self._select(pre)                                  # per-band BatchTopK

        # Per-band JumpReLU threshold EMA (min surviving activation in the band).
        step = int(self.global_step.item())
        if step > self.threshold_start_step:
            with torch.no_grad():
                for b, (s, e) in enumerate(self.band_slices):
                    zb = z[:, s:e]
                    active = zb[zb > 0]
                    cur = (active.min().float() if active.numel() > 0
                           else torch.tensor(0.0, device=z.device))
                    if self.threshold[b].item() < 0:
                        self.threshold[b] = cur
                    else:
                        self.threshold[b] = (self.threshold_beta * self.threshold[b]
                                             + (1 - self.threshold_beta) * cur)

        W_dec = self._dec_full()                               # (d_sae, T, d_in)
        x_hat = torch.einsum("bs,std->btd", z, W_dec) + self.b_dec
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        # Dead-feature tracking on the shared code.
        with torch.no_grad():
            active_feat = (z > 0).any(dim=0)
            self.num_tokens_since_fired += B * T
            self.num_tokens_since_fired[active_feat] = 0
            dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
            n_dead = int(dead_mask.sum().item())

        if n_dead > 0:
            k_aux = min(self.aux_k, n_dead)
            auxk_pre = pre.masked_fill(~dead_mask.unsqueeze(0), 0.0)
            vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
            aux_buf = torch.zeros_like(pre).scatter_(-1, idx_a, vals_a)
            aux_decode = torch.einsum("bs,std->btd", aux_buf, W_dec)
            residual = (x - x_hat).detach()
            l2_a = (residual - aux_decode).pow(2).sum(dim=-1).mean()
            mu = residual.mean(dim=(0, 1), keepdim=True)
            denom = (residual - mu).pow(2).sum(dim=-1).mean()
            l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        else:
            l_auxk = torch.zeros((), device=x.device, dtype=x.dtype)

        loss = l_recon + self.auxk_alpha * l_auxk

        with torch.no_grad():
            self.global_step += 1
            l0 = (z != 0).float().sum(dim=-1).mean()

        return {
            "loss": loss,
            "mse": l_recon.detach(),
            "l0": l0.detach(),
            "auxk": l_auxk.detach(),
            "dead": torch.tensor(float(n_dead)),
            "threshold": self.threshold.detach().mean().clone(),
        }

    # ── hooks / introspection ──

    def post_step(self) -> None:
        with torch.no_grad():
            self._normalize_decoder()

    def decoder_directions(self) -> torch.Tensor:
        """(d_sae, d_in) — decoder atoms averaged over the T positions."""
        return self._dec_full().detach().mean(dim=1).clone()

    def band_of_features(self) -> list[tuple[int, int]]:
        """The (start, end) code slice owned by each band (for per-branch probes)."""
        return list(self.band_slices)
