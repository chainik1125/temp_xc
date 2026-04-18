"""TopK SAE — single-token sparse autoencoder.

This is the standard SAE baseline. It processes each token independently
(no temporal structure). When used in a windowed comparison, it sees
flattened (B*T, d) input.

Supports both TopK sparsity (k set) and ReLU+L1 sparsity (k=None plus
l1_coeff>0 in training). Training also supports LR warmup and Adam
weight decay.

Ported from Aniket's original crosscoder models and Han's relu_sae.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bench.architectures.base import ArchSpec, EvalOutput


class TopKSAE(nn.Module):
    """Standard SAE — TopK when k is set, ReLU when k is None.

    Input:  x in R^d
    Latent: z in R^h  (k non-zero if k is set, else non-negative)
    Output: x_hat in R^d
    """

    def __init__(self, d_in: int, d_sae: int, k: int | None = None):
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
              log_every=500, grad_clip=1.0, l1_coeff=0.0, warmup_steps=0,
              weight_decay=0.0):
        """Train the SAE.

        Extra args:
            l1_coeff: L1 penalty coefficient on latent codes. Required
                when model.k is None (ReLU mode) to enforce sparsity.
            warmup_steps: linear LR warmup from 0 to lr over the first N
                steps. 0 disables warmup.
            weight_decay: Adam weight decay (L2 on parameters, separate
                from the L1 sparsity penalty).
        """
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        log = {"loss": [], "mse": [], "l1": [], "l0": []}
        model.train()

        for step in range(total_steps):
            if warmup_steps > 0 and step < warmup_steps:
                cur_lr = lr * (step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr

            x = gen_fn(batch_size).to(device)
            mse, _, z = model(x)

            if l1_coeff > 0:
                l1 = z.abs().sum(dim=-1).mean()
                loss = mse + l1_coeff * l1
            else:
                l1 = torch.zeros((), device=mse.device)
                loss = mse

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model._normalize_decoder()

            if step % log_every == 0 or step == total_steps - 1:
                with torch.no_grad():
                    l0 = (z > 0).float().sum(dim=-1).mean().item()
                log["loss"].append(loss.item())
                log["mse"].append(mse.item())
                log["l1"].append(l1.item())
                log["l0"].append(l0)

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
