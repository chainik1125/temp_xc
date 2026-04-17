"""TopK SAE — single-token sparse autoencoder with TopK activation.

This is the standard SAE baseline. It processes each token independently
(no temporal structure). When used in a windowed comparison, it sees
flattened (B*T, d) input.

Ported from temporal_crosscoders/models.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bench.architectures.base import ArchSpec, EvalOutput


class TopKSAE(nn.Module):
    """Standard SAE with TopK activation — exactly k latents active per token.

    Input:  x in R^d
    Latent: z in R^h  (k non-zero)
    Output: x_hat in R^d
    """

    def __init__(self, d_in: int, d_sae: int, k: int | None):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - self.b_dec
        pre = x_c @ self.W_enc.T + self.b_enc
        if self.k is not None:
            topk_vals, topk_idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(-1, topk_idx, F.relu(topk_vals))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor):
        """x: (B, d) -> (recon_loss, x_hat, z)"""
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, z


class TopKSAESpec(ArchSpec):
    """ArchSpec for the single-token TopK SAE baseline."""

    name = "TopKSAE"
    data_format = "flat"

    def create(self, d_in, d_sae, k, device):
        return TopKSAE(d_in, d_sae, k).to(device)

    def train(self, model, gen_fn, total_steps, batch_size, lr, device,
              log_every=500, grad_clip=1.0):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        log = {"loss": [], "l0": []}
        model.train()

        for step in range(total_steps):
            x = gen_fn(batch_size).to(device)
            loss, _, z = model(x)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model._normalize_decoder()

            if step % log_every == 0 or step == total_steps - 1:
                with torch.no_grad():
                    l0 = (z > 0).float().sum(dim=-1).mean().item()
                log["loss"].append(loss.item())
                log["l0"].append(l0)
                print(
                    f"      [{self.name}] step {step:5d}/{total_steps} "
                    f"| loss={loss.item():.4f} | l0={l0:.2f}",
                    flush=True,
                )
                # Stream to W&B if active (silent fallback if not).
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"train/loss": loss.item(), "train/l0": l0},
                                  step=step)
                except Exception:
                    pass

        model.eval()
        return log

    def eval_forward(self, model, x):
        loss, x_hat, z = model(x)
        se = (x_hat - x).pow(2).sum().item()
        signal = x.pow(2).sum().item()
        l0 = (z > 0).float().sum(dim=-1).sum().item()
        return EvalOutput(sum_se=se, sum_signal=signal, sum_l0=l0, n_tokens=x.shape[0])

    def decoder_directions(self, model, pos=None):
        return model.W_dec.data  # (d_in, d_sae)

    def encode(self, model, x):
        """(B, T, d_in) → (B, T, d_sae) via per-token application.

        TopKSAE's native encode takes (B, d_in); we flatten positions,
        run encode, then reshape back. This is the intentional null
        baseline for the auto-MI metric — TopKSAE has no architectural
        mechanism for temporal binding, so any auto-MI it shows is
        just token-level feature auto-correlation.
        """
        B, T, d = x.shape
        flat = x.reshape(B * T, d)
        z = model.encode(flat)  # (B*T, d_sae)
        return z.reshape(B, T, -1)
