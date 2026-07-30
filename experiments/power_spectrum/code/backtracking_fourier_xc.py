"""Plain Fourier TopK crosscoder for the matched backtracking replication.

This module intentionally changes only the temporal parameterisation of
Aniket's ``TXCBase``:

- one window-level code with per-example TopK followed by ReLU;
- ``k_win = k_pos * T``;
- the same reconstruction, AuxK, decoder-normalisation, and dead-feature
  bookkeeping;
- no Matryoshka, learned frequency weighting, or BatchTopK threshold.

Each atom is restricted to one real-Fourier frequency band.  Inputs and
reconstructions are projected without materialising a dense
``(d_sae, T, d_in)`` kernel, which is important for the parameter-matched
4096-wide experiment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def real_fourier_basis(length: int) -> torch.Tensor:
    """Return a real orthonormal DFT basis in DC/cos/sin/Nyquist order."""

    if length < 1:
        raise ValueError("length must be positive")
    time = torch.arange(length, dtype=torch.float32)
    rows = [torch.ones(length) / math.sqrt(length)]
    for frequency in range(1, (length - 1) // 2 + 1):
        angle = 2.0 * torch.pi * frequency * time / length
        scale = math.sqrt(2.0 / length)
        rows.extend((scale * torch.cos(angle), scale * torch.sin(angle)))
    if length % 2 == 0:
        rows.append(torch.cos(torch.pi * time) / math.sqrt(length))
    return torch.stack(rows)


def fourier_bands(length: int, mode: str = "multiband") -> tuple[list[list[int]], list[list[int]]]:
    """Return basis-row bands and the corresponding physical frequencies.

    ``multiband`` matches the existing Fourier XC: DC plus at most three
    contiguous AC groups, with each sine/cosine quadrature pair kept intact.
    """

    if length < 1:
        raise ValueError("length must be positive")
    if mode == "full":
        return [list(range(length))], [list(range(length // 2 + 1))]
    if mode == "dcac":
        ac_rows = list(range(1, length))
        return [[0]] + ([ac_rows] if ac_rows else []), [[0]] + (
            [list(range(1, length // 2 + 1))] if ac_rows else []
        )
    if mode != "multiband":
        raise ValueError(f"unknown band mode {mode!r}")

    units: list[tuple[list[int], int]] = []
    row = 1
    for frequency in range(1, (length - 1) // 2 + 1):
        units.append(([row, row + 1], frequency))
        row += 2
    if length % 2 == 0:
        units.append(([row], length // 2))
    if not units:
        return [[0]], [[0]]

    group_count = min(3, len(units))
    base, remainder = divmod(len(units), group_count)
    sizes = [base] * group_count
    for offset in range(remainder):
        sizes[group_count - 1 - offset] += 1

    row_bands = [[0]]
    frequency_bands = [[0]]
    start = 0
    for size in sizes:
        group = units[start : start + size]
        row_bands.append([index for rows, _ in group for index in rows])
        frequency_bands.append([frequency for _, frequency in group])
        start += size
    return row_bands, frequency_bands


def split_evenly(total: int, parts: int) -> list[int]:
    """Split ``total`` over ``parts`` with remainders assigned from the left."""

    if parts < 1 or total < parts:
        raise ValueError("total must be at least the positive number of parts")
    base, remainder = divmod(total, parts)
    return [base + int(index < remainder) for index in range(parts)]


def txc_parameter_count(*, d_in: int, d_sae: int, window: int) -> int:
    """Trainable parameter count of Aniket's dense ``TXCBase``."""

    return 2 * window * d_in * d_sae + d_sae + window * d_in


def fourier_parameter_count(
    *,
    d_in: int,
    d_sae: int,
    window: int,
    bands_mode: str = "multiband",
) -> int:
    """Trainable parameter count of :class:`FourierTopKXC`."""

    bands, _ = fourier_bands(window, bands_mode)
    atoms = split_evenly(d_sae, len(bands))
    coefficient_rows = sum(count * len(band) for count, band in zip(atoms, bands))
    return 2 * d_in * coefficient_rows + d_sae + window * d_in


def matched_fourier_width(
    *,
    d_in: int,
    txc_d_sae: int,
    window: int,
    bands_mode: str = "multiband",
) -> int:
    """Closest-total-parameter Fourier width for a reference dense TXC.

    The count is monotone in dictionary width, so a binary search avoids
    constructing candidate models.  Ties prefer the smaller model.
    """

    target = txc_parameter_count(d_in=d_in, d_sae=txc_d_sae, window=window)
    band_count = len(fourier_bands(window, bands_mode)[0])
    low = band_count
    high = max(txc_d_sae, band_count)
    while fourier_parameter_count(
        d_in=d_in,
        d_sae=high,
        window=window,
        bands_mode=bands_mode,
    ) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        count = fourier_parameter_count(
            d_in=d_in,
            d_sae=middle,
            window=window,
            bands_mode=bands_mode,
        )
        if count < target:
            low = middle + 1
        else:
            high = middle
    candidates = [low]
    if low > band_count:
        candidates.append(low - 1)
    return min(
        candidates,
        key=lambda width: (
            abs(
                fourier_parameter_count(
                    d_in=d_in,
                    d_sae=width,
                    window=window,
                    bands_mode=bands_mode,
                )
                - target
            ),
            width,
        ),
    )


class FourierTopKXC(nn.Module):
    """Band-limited real-Fourier XC on Aniket's per-example TopK backbone."""

    arch_version = "backtracking-fourier-topk-v1"
    consumes = "window"

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        T: int,
        k_pos: int,
        bands: str = "multiband",
        auxk_alpha: float = 1.0 / 32.0,
        dead_threshold_tokens: int = 10_000_000,
        aux_k: int | None = None,
    ) -> None:
        super().__init__()
        if min(d_in, d_sae, T, k_pos) < 1:
            raise ValueError("d_in, d_sae, T, and k_pos must be positive")

        self.d_in = int(d_in)
        self._d_sae = int(d_sae)
        self._T = int(T)
        self.k_pos = int(k_pos)
        self.k_win = min(self.k_pos * self._T, self._d_sae)
        self.bands_mode = bands
        self.bands, self.frequency_bin_bands = fourier_bands(self._T, bands)
        self.n_bands = len(self.bands)
        self.h_per_band = split_evenly(self._d_sae, self.n_bands)
        self.band_slices: list[tuple[int, int]] = []
        start = 0
        for width in self.h_per_band:
            self.band_slices.append((start, start + width))
            start += width

        self.auxk_alpha = float(auxk_alpha)
        self.dead_threshold_tokens = int(dead_threshold_tokens)
        self.aux_k = int(aux_k if aux_k is not None else min(512, d_in // 2))

        self.enc_coef = nn.ParameterList()
        self.dec_coef = nn.ParameterList()
        self.b_enc = nn.ParameterList()
        basis = real_fourier_basis(self._T)
        self.register_buffer("temporal_basis", basis, persistent=False)
        for band, atoms in zip(self.bands, self.h_per_band):
            shape = (atoms, len(band), self.d_in)
            encoder = nn.Parameter(torch.empty(shape))
            decoder = nn.Parameter(torch.empty(shape))
            nn.init.kaiming_uniform_(decoder.data.view(atoms, -1))
            self.enc_coef.append(encoder)
            self.dec_coef.append(decoder)
            self.b_enc.append(nn.Parameter(torch.zeros(atoms)))
        self.b_dec = nn.Parameter(torch.zeros(self._T, self.d_in))

        with torch.no_grad():
            self._normalize_decoder()
            for encoder, decoder in zip(self.enc_coef, self.dec_coef):
                encoder.copy_(decoder)

        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(self._d_sae, dtype=torch.long),
        )
        for decoder in self.dec_coef:
            decoder.register_post_accumulate_grad_hook(self._project_dec_grad)

    @property
    def d_sae(self) -> int:
        return self._d_sae

    @property
    def T(self) -> int:
        return self._T

    def pre_step(self) -> None:
        """Compatibility no-op matching the shared architecture contract."""

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        for decoder in self.dec_coef:
            norms = decoder.flatten(1).norm(dim=1).clamp(min=1e-8)
            decoder.div_(norms[:, None, None])

    @staticmethod
    def _project_dec_grad(parameter: torch.Tensor) -> None:
        if parameter.grad is None:
            return
        weights = parameter.data.flatten(1)
        gradient = parameter.grad.data.flatten(1)
        unit = weights / (weights.norm(dim=1, keepdim=True) + 1e-6)
        gradient.sub_((gradient * unit).sum(dim=1, keepdim=True) * unit)

    def _project_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("wt,btd->bwd", self.temporal_basis, x)

    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.shape[1:] != (self._T, self.d_in):
            raise ValueError(
                f"expected (B, {self._T}, {self.d_in}), got {tuple(x.shape)}"
            )
        projected = self._project_input(x)
        pieces = []
        for band, encoder, bias in zip(self.bands, self.enc_coef, self.b_enc):
            pieces.append(
                torch.einsum("bwd,hwd->bh", projected[:, band], encoder) + bias
            )
        return torch.cat(pieces, dim=-1)

    def select_topk(self, pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ReLUed values and indices under the exact TXC TopK rule."""

        values, indices = pre.topk(self.k_win, dim=-1)
        return F.relu(values), indices

    def dense_code(
        self, values: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        code = values.new_zeros((values.shape[0], self._d_sae))
        return code.scatter_(1, indices, values)

    def encode_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.select_topk(self.preactivations(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        values, indices = self.encode_topk(x)
        return self.dense_code(values, indices).unsqueeze(1)

    def _decode_code_without_bias(self, code: torch.Tensor) -> torch.Tensor:
        reconstruction = code.new_zeros((code.shape[0], self._T, self.d_in))
        for band, decoder, (start, stop) in zip(
            self.bands,
            self.dec_coef,
            self.band_slices,
        ):
            coefficients = torch.einsum(
                "bh,hwd->bwd",
                code[:, start:stop],
                decoder,
            )
            reconstruction = reconstruction + torch.einsum(
                "bwd,wt->btd",
                coefficients,
                self.temporal_basis[band],
            )
        return reconstruction

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        if code.dim() == 3:
            if code.shape[1] != 1:
                raise ValueError("window code must have a singleton time axis")
            code = code.squeeze(1)
        return self._decode_code_without_bias(code) + self.b_dec

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1:] != (self._T, self.d_in):
            raise ValueError(
                f"expected (B, {self._T}, {self.d_in}), got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        pre = self.preactivations(x)
        values, indices = self.select_topk(pre)
        code = self.dense_code(values, indices)
        reconstruction = self.decode(code)
        reconstruction_loss = (x - reconstruction).square().sum(dim=-1).mean()

        with torch.no_grad():
            active = (code > 0).any(dim=0)
            self.num_tokens_since_fired += batch * self._T
            self.num_tokens_since_fired[active] = 0
            dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
            dead_count = int(dead_mask.sum().item())

        if dead_count:
            aux_count = min(self.aux_k, dead_count)
            aux_pre = F.relu(pre).masked_fill(~dead_mask.unsqueeze(0), 0.0)
            aux_values, aux_indices = aux_pre.topk(
                aux_count,
                dim=-1,
                sorted=False,
            )
            aux_code = self.dense_code(aux_values, aux_indices)
            aux_reconstruction = self._decode_code_without_bias(aux_code)
            residual = (x - reconstruction).detach()
            aux_error = (
                residual - aux_reconstruction
            ).square().sum(dim=-1).mean()
            residual_mean = residual.mean(dim=(0, 1), keepdim=True)
            denominator = (
                residual - residual_mean
            ).square().sum(dim=-1).mean()
            aux_loss = (aux_error / denominator.clamp(min=1e-8)).nan_to_num(0.0)
        else:
            aux_loss = x.new_zeros(())

        loss = reconstruction_loss + self.auxk_alpha * aux_loss
        with torch.no_grad():
            l0 = (code != 0).float().sum(dim=-1).mean()
        return {
            "loss": loss,
            "mse": reconstruction_loss.detach(),
            "l0": l0.detach(),
            "auxk": aux_loss.detach(),
            "dead": x.new_tensor(float(dead_count)),
        }

    def post_step(self) -> None:
        self._normalize_decoder()

    def decoder_directions(self) -> torch.Tensor:
        """Return time-averaged decoder atoms for interface compatibility."""

        directions = []
        for band, decoder in zip(self.bands, self.dec_coef):
            kernel = torch.einsum(
                "wt,hwd->htd",
                self.temporal_basis[band],
                decoder,
            )
            directions.append(kernel.mean(dim=1))
        return torch.cat(directions, dim=0).detach().clone()

    def band_of_features(self) -> Sequence[tuple[int, int]]:
        return tuple(self.band_slices)
