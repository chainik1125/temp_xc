"""BatchTopK SAE — per-token baseline on the strong backbone.

Fair-backbone redo (``synthetic/STATUS.md`` § 4). This is the
per-token (T=1) member of the BatchTopK family: the same backbone as
T-SAE (BatchTopK during training → fixed JumpReLU threshold at inference,
Bussmann et al.) + AuxK dead-feature revival + decoder unit-norm +
grad-parallel removal — but WITHOUT T-SAE's matryoshka split or temporal
contrastive loss. It is the "plain BatchTopK SAE" that replaces the plain
TopK SAE so the only thing separating it from the crosscoders is the
decode structure, not the backbone.

Per-token contract (``consumes = "token"``): the trainer feeds ``(B,
d_in)`` tokens i.i.d. from :class:`ActivationBuffer`. BatchTopK pools over
the ``B`` tokens with budget ``k_pos`` per token (``k_pos · B`` actives).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


@torch.no_grad()
def _set_decoder_norm_to_unit(W_dec_DF: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(W_dec_DF.dtype).eps
    norm = torch.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


def _remove_grad_parallel_to_decoder(
    W_dec_DF: torch.Tensor, grad_DF: torch.Tensor
) -> torch.Tensor:
    normed_W = W_dec_DF / (torch.norm(W_dec_DF, dim=0, keepdim=True) + 1e-6)
    parallel = torch.einsum("df,df->f", grad_DF, normed_W)
    return grad_DF - parallel.unsqueeze(0) * normed_W


class BatchTopKSAE(TempBenchArch):
    """Per-token BatchTopK SAE (strong backbone, no matryoshka/contrastive)."""

    arch_version: str = "1.0.0"
    consumes: str = "token"

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        k_pos: int,
        T: int = 1,
        auxk_alpha: float = 1.0 / 32.0,
        dead_feature_threshold_tokens: int = 10_000_000,
        threshold_start_step: int = 1000,
        threshold_beta: float = 0.999,
    ):
        nn.Module.__init__(self)
        if int(T) != 1:
            raise ValueError(f"BatchTopKSAE is per-token; T must be 1, got {T}.")
        self.config = ArchConfig(
            name="batchtopk_sae", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=1
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self.k_pos = int(k_pos)
        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold_tokens = dead_feature_threshold_tokens
        self.threshold_start_step = threshold_start_step
        self.threshold_beta = threshold_beta
        self.top_k_aux = d_in // 2

        # Params (tsae convention: W_enc (d_in,d_sae), W_dec (d_sae,d_in)).
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_sae, d_in)))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self.W_dec.data = _set_decoder_norm_to_unit(self.W_dec.data.T).T.contiguous()
        self.W_enc.data = self.W_dec.data.clone().T.contiguous()

        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.register_buffer(
            "num_tokens_since_fired", torch.zeros(d_sae, dtype=torch.long)
        )
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    @staticmethod
    def _project_dec_grad(param: torch.Tensor) -> None:
        if param.grad is None:
            return
        new_grad = _remove_grad_parallel_to_decoder(param.data.T, param.grad.data.T).T
        param.grad.data.copy_(new_grad)

    def _batchtopk(self, post_relu: torch.Tensor) -> torch.Tensor:
        """Flat BatchTopK over (B·d_sae); budget k_pos per token."""
        flat = post_relu.flatten()
        k_total = self.k_pos * post_relu.shape[0]
        if k_total >= flat.numel():
            return post_relu
        tk = flat.topk(k_total, sorted=False)
        return (
            torch.zeros_like(flat)
            .scatter_(-1, tk.indices, tk.values)
            .reshape(post_relu.shape)
        )

    # ── encode / decode ──

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d_in) or (B, T, d_in) → matching z. BatchTopK / threshold."""
        squeeze_t = x.dim() == 2
        if squeeze_t:
            x = x.unsqueeze(1)
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)
        post_relu = F.relu((x_flat - self.b_dec) @ self.W_enc + self.b_enc)
        if (not self.training) and self.threshold.item() >= 0:
            z_flat = post_relu * (post_relu > self.threshold)
        else:
            z_flat = self._batchtopk(post_relu)
        z = z_flat.reshape(B, T, self._d_sae)
        return z.squeeze(1) if squeeze_t else z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_t = z.dim() == 2
        if squeeze_t:
            z = z.unsqueeze(1)
        x_hat = z @ self.W_dec + self.b_dec
        return x_hat.squeeze(1) if squeeze_t else x_hat

    # ── train_step ──

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(
                f"BatchTopKSAE.train_step expects (B, d_in); got {tuple(x.shape)}."
            )
        B = x.shape[0]
        post_relu = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        z = self._batchtopk(post_relu)

        step = int(self.global_step.item())
        if step > self.threshold_start_step:
            with torch.no_grad():
                active = z[z > 0]
                cur = (
                    active.min().float()
                    if active.numel() > 0
                    else torch.tensor(0.0, device=z.device)
                )
                if self.threshold.item() < 0:
                    self.threshold.copy_(cur)
                else:
                    self.threshold.copy_(
                        self.threshold_beta * self.threshold
                        + (1 - self.threshold_beta) * cur
                    )

        x_hat = z @ self.W_dec + self.b_dec
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        with torch.no_grad():
            did_fire = (z.sum(dim=0) > 0)
            self.num_tokens_since_fired += B
            self.num_tokens_since_fired[did_fire] = 0
            dead = self.num_tokens_since_fired >= self.dead_feature_threshold_tokens
            n_dead = int(dead.sum().item())

        if n_dead > 0:
            k_aux = min(self.top_k_aux, n_dead)
            neg_inf = torch.tensor(float("-inf"), device=x.device)
            masked = torch.where(dead.unsqueeze(0), post_relu, neg_inf)
            vals, idx = masked.topk(k_aux, sorted=False)
            buf = torch.zeros_like(post_relu).scatter_(-1, idx, vals)
            x_aux = buf @ self.W_dec
            residual = (x - x_hat).detach()
            l2 = (residual - x_aux).pow(2).sum(dim=-1).mean()
            mu = residual.mean(dim=0, keepdim=True)
            denom = (residual - mu).pow(2).sum(dim=-1).mean()
            l_auxk = (l2 / denom.clamp(min=1e-8)).nan_to_num(0.0)
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
            "threshold": self.threshold.detach().clone(),
        }

    def post_step(self) -> None:
        with torch.no_grad():
            self.W_dec.data = _set_decoder_norm_to_unit(
                self.W_dec.data.T
            ).T.contiguous()

    def decoder_directions(self) -> torch.Tensor:
        return self.W_dec.data.clone()
