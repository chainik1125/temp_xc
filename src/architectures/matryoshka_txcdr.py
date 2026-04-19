"""Matryoshka-TXCDR (position-nested) — Phase 5.3 primary novel arch.

Motivation (from the Phase 5 brief's architecture menu §1):

    Vanilla TXCDR applies a single TopK to a shared pre-activation
    summed over T positions. There is no mechanism to force ANY latent
    to carry per-position (local) signal — its features can only live
    at the window level. This is why Aniket observed TempXC losing on
    language-ID and code-detection tasks, where local-token info
    dominates.

    A Matryoshka variant forces prefixes of the latent vector to
    reconstruct SUB-WINDOWS of increasing size:

        latents[:m_1]        must reconstruct each position alone
        latents[:m_1+m_2]    must reconstruct windows of size 2
        latents[:m_1+...+mT] reconstructs the full T-position window

    Concretely: for each nested window size t ∈ {1, 2, ..., T}, we
    train a per-scale decoder W_dec_scale[t] (a separate decoder matrix
    per scale) that uses only the first m_1+...+m_t latents to
    reconstruct the central t-token sub-window. Each scale contributes
    an MSE term to the loss.

Shape of the dictionary:

    Latent prefix sizes: m = [m_1, ..., m_T]. Defaults to a uniform
    split m_t = d_sae / T, so the "scale-1" prefix has d_sae/T latents,
    the "scale-2" prefix has 2*d_sae/T, ..., the full prefix has d_sae.

    Encoder: same shape as vanilla TXCDR — (T, d_in, d_sae) with bias (d_sae,).
    Decoders: T of them, W_dec_scale[t] with shape
        (prefix_sum[t], t, d_in)      # latent-prefix -> t-token window
    and biases (t, d_in).

Loss:

    total_recon = mean over scales t of MSE(decode_scale_t(z[:prefix_sum[t]]),
                                           x[central t tokens])

    The center sub-window for scale t is x[:, floor((T-t)/2) : floor((T-t)/2) + t, :].

Sparsity:

    One TopK over the full pre-activation (shape (B, d_sae)) with k = k_win.
    Features with indices beyond prefix_sum[t] are ignored by scale-t
    decoder but still compete in the TopK selection — so the architecture
    naturally places "global" features in later indices and "local"
    features in earlier indices through gradient flow.

This position-nested formulation is the novel contribution; a
feature-index-nested (classic Matryoshka-SAE) variant is a drop-in via
the ``FeatureNestedMatryoshkaTXCDR`` class (not registered by default).

Independent implementation — not ported from any prior Matryoshka-SAE code.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.base import ArchSpec, EvalOutput


class PositionMatryoshkaTXCDR(nn.Module):
    """Matryoshka-TXCDR with prefix-length decoders for nested sub-windows.

    Args:
        d_in: residual-stream width.
        d_sae: total dictionary size.
        T: full window length.
        k: window-level TopK (same convention as vanilla TXCDR).
        latent_splits: tuple of length T summing to d_sae; m_t latents
            per scale. Defaults to uniform split.
    """

    def __init__(
        self, d_in: int, d_sae: int, T: int, k: int | None,
        latent_splits: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k

        if latent_splits is None:
            base = d_sae // T
            # distribute any remainder into the earliest prefixes
            splits = [base + (1 if i < (d_sae - base * T) else 0) for i in range(T)]
        else:
            splits = list(latent_splits)
            assert sum(splits) == d_sae, (
                f"latent_splits must sum to d_sae={d_sae}, got {sum(splits)}"
            )
        self.latent_splits = tuple(splits)
        # prefix_sum[t] = sum of first (t+1) latent-splits, i.e. number of
        # latents usable by scale-t decoder.
        self.prefix_sum = tuple(
            sum(splits[:i + 1]) for i in range(T)
        )

        # Shared encoder (same as vanilla TXCDR).
        self.W_enc = nn.Parameter(
            torch.randn(T, d_in, d_sae) * (1.0 / d_in**0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # One decoder per scale; shape (prefix, t_size, d_in).
        self.W_decs = nn.ParameterList()
        self.b_decs = nn.ParameterList()
        for t_idx in range(T):
            prefix = self.prefix_sum[t_idx]
            t_size = t_idx + 1
            W = torch.randn(prefix, t_size, d_in) * (1.0 / prefix**0.5)
            self.W_decs.append(nn.Parameter(W))
            self.b_decs.append(nn.Parameter(torch.zeros(t_size, d_in)))

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        # Each scale's decoder atom is (t_size, d_in) — normalize jointly.
        for W in self.W_decs:
            norms = W.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            W.data = W.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) -> z: (B, d_sae) with k non-zeros."""
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
        else:
            z = F.relu(pre)
        return z

    def _window_center(self, x: torch.Tensor, t_size: int) -> torch.Tensor:
        """Extract the central t_size-token sub-window from a full x."""
        T = x.shape[1]
        start = (T - t_size) // 2
        return x[:, start:start + t_size, :]

    def decode_scale(self, z: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Scale-t_idx decoder uses the first prefix_sum[scale_idx] latents."""
        prefix = self.prefix_sum[scale_idx]
        W = self.W_decs[scale_idx]       # (prefix, t_size, d_in)
        b = self.b_decs[scale_idx]       # (t_size, d_in)
        z_prefix = z[:, :prefix]
        return torch.einsum("bs,std->btd", z_prefix, W) + b

    def forward(self, x: torch.Tensor):
        """x: (B, T, d_in) -> (total_loss, x_hat_full, z).

        Total loss = mean over scales of MSE(decode_scale, center_window_scale).
        """
        z = self.encode(x)
        losses = []
        for t_idx in range(self.T):
            t_size = t_idx + 1
            x_center = self._window_center(x, t_size)   # (B, t_size, d_in)
            x_hat = self.decode_scale(z, t_idx)
            # Use per-element MSE averaged over batch; sum over (t_size, d_in)
            loss_scale = (x_hat - x_center).pow(2).sum(dim=-1).mean()
            losses.append(loss_scale)
        total_loss = torch.stack(losses).mean()
        # For the "full" reconstruction we return the final-scale decode.
        x_hat_full = self.decode_scale(z, self.T - 1)
        return total_loss, x_hat_full, z

    @property
    def decoder_dirs_averaged(self) -> torch.Tensor:
        """Average across all scales' decoders for a coarse feature direction.

        For probing / UMAP / feature-recovery analyses we want one
        (d_in, d_sae) matrix. Each latent is "shared" across the scales
        that include it in their prefix; we average across those scales'
        per-position columns.
        """
        # For each latent j, collect its decoder column from scales
        # where j < prefix_sum[t], averaged over those scales' per-position
        # vectors.
        dirs = torch.zeros(self.d_in, self.d_sae, device=self.W_decs[0].device)
        counts = torch.zeros(self.d_sae, device=self.W_decs[0].device)
        for t_idx in range(self.T):
            prefix = self.prefix_sum[t_idx]
            W = self.W_decs[t_idx]  # (prefix, t_size, d_in)
            # per-latent direction is the mean over t_size positions.
            avg = W.mean(dim=1).T  # (d_in, prefix)
            dirs[:, :prefix] += avg
            counts[:prefix] += 1
        counts = counts.clamp(min=1e-8)
        return dirs / counts


class MatryoshkaTXCDRSpec(ArchSpec):
    """ArchSpec for position-nested Matryoshka-TXCDR."""

    data_format = "window"

    def __init__(self, T: int):
        self.T = T
        self.name = f"MatryoshkaTXCDR T={T}"

    @property
    def n_decoder_positions(self) -> int:
        return self.T

    def create(self, d_in, d_sae, k, device):
        # Same budget convention as vanilla TXCDR: caller passes per-token k,
        # internally multiplied by T for window-level TopK.
        k_eff = k * self.T if k is not None else None
        return PositionMatryoshkaTXCDR(d_in, d_sae, self.T, k_eff).to(device)

    def train(
        self, model, gen_fn, total_steps, batch_size, lr, device,
        log_every: int = 500, grad_clip: float = 1.0,
    ):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        log: dict[str, list[float]] = {"loss": [], "l0": []}
        model.train()
        for step in range(total_steps):
            x = gen_fn(batch_size).to(device)
            loss, _, z = model(x)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            model._normalize_decoder()
            if step % log_every == 0 or step == total_steps - 1:
                with torch.no_grad():
                    l0 = (z > 0).float().sum(dim=-1).mean().item()
                log["loss"].append(loss.item())
                log["l0"].append(l0)
        model.eval()
        return log

    def eval_forward(self, model, x):
        loss, x_hat, z = model(x)
        # x_hat is for the innermost t=T scale which is the central full window.
        # For a fair recon metric against the raw x, align on the central t=T slice.
        T = model.T
        x_center = model._window_center(x, T)
        se = (x_hat - x_center).pow(2).sum().item()
        signal = x_center.pow(2).sum().item()
        l0 = (z > 0).float().sum(dim=-1).sum().item()
        return EvalOutput(
            sum_se=se, sum_signal=signal, sum_l0=l0,
            n_tokens=x.shape[0],
        )

    def decoder_directions(self, model, pos=None):
        if pos is None:
            return model.decoder_dirs_averaged
        # Per-position direction from the final-scale decoder.
        return model.W_decs[-1][:, pos, :].T
