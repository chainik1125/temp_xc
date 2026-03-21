"""Minimal TopK Sparse Autoencoder for baseline experiments."""

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """TopK sparse autoencoder.

    Architecture:
        encode: h = TopK(ReLU(W_enc @ (x - b_pre)))
        decode: x_hat = h @ W_dec + b_pre

    W_dec rows are kept at unit norm after each optimizer step.
    """

    def __init__(self, d_input: int, n_latents: int, k: int) -> None:
        """Initialize the TopK SAE.

        Args:
            d_input: Input dimension.
            n_latents: Number of latent features (dictionary size).
            k: Number of top activations to keep per input.
        """
        super().__init__()
        self.d_input = d_input
        self.n_latents = n_latents
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_input, n_latents))
        self.W_dec = nn.Parameter(torch.empty(n_latents, d_input))
        self.b_pre = nn.Parameter(torch.zeros(d_input))

        # Kaiming uniform init for encoder
        nn.init.kaiming_uniform_(self.W_enc)
        # Initialize decoder as transpose of encoder
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            # Normalize decoder rows to unit norm
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse latent activations.

        Args:
            x: Input tensor of shape (batch, d_input).

        Returns:
            Sparse activations of shape (batch, n_latents) with at most k
            nonzero entries per row.
        """
        h = torch.relu(torch.mm(x - self.b_pre, self.W_enc))
        # TopK: keep only the k largest, zero the rest
        topk_vals, topk_idx = torch.topk(h, self.k, dim=-1)
        h_sparse = torch.zeros_like(h)
        h_sparse.scatter_(1, topk_idx, topk_vals)
        return h_sparse

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode sparse latents to reconstruction.

        Args:
            h: Sparse activations of shape (batch, n_latents).

        Returns:
            Reconstruction of shape (batch, d_input).
        """
        return torch.mm(h, self.W_dec) + self.b_pre

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass: encode, decode, compute losses.

        Args:
            x: Input tensor of shape (batch, d_input).

        Returns:
            Tuple of (x_hat, h, loss_dict) where:
                x_hat: Reconstruction (batch, d_input)
                h: Sparse activations (batch, n_latents)
                loss_dict: {"recon_loss": scalar, "l0": scalar}
        """
        h = self.encode(x)
        x_hat = self.decode(h)
        recon_loss = (x - x_hat).pow(2).mean()
        l0 = (h > 0).float().sum(dim=-1).mean()
        return x_hat, h, {"recon_loss": recon_loss, "l0": l0}

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Normalize W_dec rows to unit norm."""
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))

    @torch.no_grad()
    def remove_parallel_grads(self) -> None:
        """Remove gradient component parallel to decoder weight vectors.

        Call before optimizer.step() to prevent the optimizer from changing
        the norm of decoder rows.
        """
        if self.W_dec.grad is not None:
            # Project out component parallel to each row
            parallel = (self.W_dec.grad * self.W_dec).sum(dim=1, keepdim=True)
            self.W_dec.grad.sub_(parallel * self.W_dec)
