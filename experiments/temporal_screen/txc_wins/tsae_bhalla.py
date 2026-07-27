"""The published temporal SAE (arXiv:2511.05541), vendored so this sprint runs the real one.

WHY THIS FILE EXISTS. Every `tsae_*` number produced by this sprint before it was written
came from `temporal_crosscoders.han_tsae.TemporalSAE` -- an **attention** architecture with
`n_heads=8`, `n_attn_layers=1` and a predicted/novel code split. That is not the published
method. The repo aliases it `tsae_paper` (see `experiments/ward_backtracking_txc/
architectures.py`), and this sprint inherited the alias and the mislabel with it, reporting
the arm as "attention temporal SAE" throughout.

**The published T-SAE has no attention.** It is a per-token BatchTopK SAE whose temporal
structure lives entirely in the *training objective*:

  * Matryoshka groups, 20% high-level / 80% low-level, cumulative reconstruction.
  * A temporal **InfoNCE contrastive** term on the high-level latents between consecutive
    positions -- raw dot-product logits, symmetric cross-entropy against the diagonal.
  * OpenAI-style **AuxK** on dead features.
  * **Threshold inference**: an EMA-tracked threshold replaces BatchTopK at eval, so
    per-token sparsity is variable at inference.

Source of this port: `origin/temp-bench-anon:src/temp_bench/architectures/tsae.py`, itself a
port of https://github.com/AI4LIFE-GROUP/temporal-saes
(`dictionary_learning/trainers/temporal_sequence_top_k.py`). Copied rather than imported
because that branch carries a whole framework (`TempBenchArch`, `ArchConfig`) this harness
does not have; the maths below is unchanged from the branch, with only the base class and
config object removed.

HYPERPARAMETERS come from `origin/temp-bench-anon:configs/locked_archs.yaml::tsae_paper`:
`h_frac = 0.20`, `contrastive_alpha = 1.0`, `auxk_alpha = 1/32`. Two deviations, both forced
and both stated:

  * `d_sae` and `k_pos` are set by the CALLER to whatever the SAE and crosscoder use (4096
    and 8 here), not the locked 16384/20. Matching the dictionaries is the whole point of
    the comparison, and it is the axis the previous tSAE arm could not be matched on --
    its L1 form only reached the 1-32 coefficient band by collapsing.
  * The locked notes say `temp_alpha = 1/10` while both the code default and the locked
    `contrastive_alpha` say 1.0. This file follows the locked config (1.0) because that is
    what the repo actually runs; the discrepancy is recorded here rather than silently
    resolved. `b_dec` is zero-initialised, where the notes mention a geometric-median init
    -- inherited from the branch port, not introduced here.

WHAT "CONSECUTIVE POSITION" MEANS HERE. The paper pairs consecutive *tokens*. This harness
works in windows of `T` contiguous **segments**, so `train_step` pairs consecutive segments,
which is the same relation one level up. Stated because it is an adaptation, not a copy.

THE STRUCTURAL POINT IS UNCHANGED BY THE FIX. `W_dec` is `(d_sae, d_in)`: one direction per
latent, no position axis. So a steered T-SAE latent still reaches only a rank-1 write, exactly
like the TopK SAE and unlike the crosscoder's `(T, d)` slab. Correcting the architecture
corrects the arm's numbers; it does not change which side of the rank argument it sits on.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _set_decoder_norm_to_unit(W_dec_DF: torch.Tensor) -> torch.Tensor:
    """Normalise decoder columns (shape (d_in, d_sae)) to unit L2 norm."""
    eps = torch.finfo(W_dec_DF.dtype).eps
    norm = torch.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


def _remove_grad_parallel_to_decoder(W_dec_DF, grad_DF):
    """Project out the component of `grad` parallel to each decoder column."""
    normed_W = W_dec_DF / (torch.norm(W_dec_DF, dim=0, keepdim=True) + 1e-6)
    parallel = torch.einsum("df,df->f", grad_DF, normed_W)
    return grad_DF - parallel.unsqueeze(0) * normed_W


class TSAEPaper(nn.Module):
    """Faithful port of the published T-SAE. Matryoshka BatchTopK + AuxK + temporal InfoNCE."""

    def __init__(self, *, d_in, d_sae, k_pos, h_frac=0.20, contrastive_alpha=1.0,
                 auxk_alpha=1.0 / 32.0, threshold_start_step=1000, threshold_beta=0.999,
                 dead_feature_threshold_tokens=10_000_000):
        super().__init__()
        self.d_in, self._d_sae, self.h_frac = d_in, d_sae, h_frac
        n_high = max(1, int(round(h_frac * d_sae)))
        self.group_sizes = (n_high, d_sae - n_high)
        self.group_weights = (1.0, 1.0)
        self.active_groups = 2

        self.register_buffer("k", torch.tensor(k_pos, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.register_buffer("num_tokens_since_fired", torch.zeros(d_sae, dtype=torch.long))
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        self.contrastive_alpha = contrastive_alpha
        self.auxk_alpha = auxk_alpha
        self.threshold_start_step = threshold_start_step
        self.threshold_beta = threshold_beta
        self.dead_feature_threshold_tokens = dead_feature_threshold_tokens
        self.top_k_aux = d_in // 2

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_sae, d_in)))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self.W_dec.data = _set_decoder_norm_to_unit(self.W_dec.data.T).T.contiguous()
        self.W_enc.data = self.W_dec.data.clone().T.contiguous()
        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    @staticmethod
    def _project_dec_grad(param):
        if param.grad is None:
            return
        new_grad = _remove_grad_parallel_to_decoder(param.data.T, param.grad.data.T).T
        param.grad.data.copy_(new_grad)

    def _batch_topk(self, post_relu, n_rows):
        flat = post_relu.flatten()
        k_total = int(self.k.item()) * n_rows
        if k_total >= flat.numel():
            return post_relu
        tk = flat.topk(k_total, sorted=False)
        return torch.zeros_like(flat).scatter_(-1, tk.indices, tk.values).reshape(
            post_relu.shape)

    def encode(self, x):
        """(B, T, d_in) -> (B, T, d_sae). BatchTopK while training, threshold at eval."""
        squeeze_t = x.dim() == 2
        if squeeze_t:
            x = x.unsqueeze(1)
        B, T, d = x.shape
        post_relu = F.relu((x.reshape(B * T, d) - self.b_dec) @ self.W_enc + self.b_enc)
        if (not self.training) and self.threshold.item() >= 0:
            z_flat = post_relu * (post_relu > self.threshold)
        else:
            z_flat = self._batch_topk(post_relu, B * T)
        z = z_flat.reshape(B, T, self._d_sae)
        return z.squeeze(1) if squeeze_t else z

    def decode(self, z):
        squeeze_t = z.dim() == 2
        if squeeze_t:
            z = z.unsqueeze(1)
        x_hat = z @ self.W_dec + self.b_dec
        return x_hat.squeeze(1) if squeeze_t else x_hat

    def _encode_per_token(self, x):
        post_relu = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        return self._batch_topk(post_relu, x.shape[0]), post_relu

    def train_step(self, x):
        """x: (B, T, d_in) window of SEGMENT activations. Pairs consecutive segments."""
        if x.dim() != 3 or x.shape[1] < 2:
            raise ValueError(f"T-SAE needs (B, T>=2, d_in); got {tuple(x.shape)}")
        B, T_seq, _ = x.shape
        t = torch.randint(0, T_seq - 1, (1,)).item()
        x_anchor, x_temp = x[:, t, :], x[:, t + 1, :]

        f, post_relu = self._encode_per_token(x_anchor)
        f_temp, _ = self._encode_per_token(x_temp)

        if int(self.global_step.item()) > self.threshold_start_step:
            with torch.no_grad():
                active = f[f > 0]
                cur = (active.min().float() if active.numel() > 0
                       else torch.tensor(0.0, device=f.device))
                if self.threshold.item() < 0:
                    self.threshold.copy_(cur)
                else:
                    self.threshold.copy_(self.threshold_beta * self.threshold
                                         + (1 - self.threshold_beta) * cur)

        W_chunks = torch.split(self.W_dec, list(self.group_sizes), dim=0)
        f_chunks = torch.split(f, list(self.group_sizes), dim=1)
        f_temp_chunks = torch.split(f_temp, list(self.group_sizes), dim=1)

        x_recon = self.b_dec.unsqueeze(0).expand_as(x_anchor).clone()
        x_recon = x_recon + f_chunks[0] @ W_chunks[0]
        total_l2 = ((x_anchor - x_recon).pow(2).sum(-1) * self.group_weights[0]).mean()

        # Temporal contrastive: raw-dot InfoNCE between high-level latents one step apart.
        logits = f_chunks[0] @ f_temp_chunks[0].T
        labels = torch.arange(logits.shape[0], device=logits.device)
        temp_loss = 0.5 * (F.cross_entropy(logits, labels)
                           + F.cross_entropy(logits.T, labels))

        for gi in range(1, self.active_groups):
            x_recon = x_recon + f_chunks[gi] @ W_chunks[gi]
            total_l2 = total_l2 + ((x_anchor - x_recon).pow(2).sum(-1).mean()
                                   * self.group_weights[gi])

        with torch.no_grad():
            self.num_tokens_since_fired += B
            self.num_tokens_since_fired[f.sum(0) > 0] = 0
        auxk = self._auxiliary_loss((x_anchor - x_recon).detach(), post_relu)

        total = total_l2 + self.auxk_alpha * auxk + self.contrastive_alpha * temp_loss
        with torch.no_grad():
            self.global_step += 1
            l0 = (f != 0).float().sum(-1).mean()
        return total, {"mse": total_l2.detach(), "l0": l0.detach(),
                       "auxk": auxk.detach(), "temp": temp_loss.detach()}

    def _auxiliary_loss(self, residual, post_relu):
        dead = self.num_tokens_since_fired >= self.dead_feature_threshold_tokens
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return torch.tensor(0.0, dtype=residual.dtype, device=residual.device)
        k_aux = min(self.top_k_aux, n_dead)
        masked = torch.where(dead.unsqueeze(0), post_relu,
                             torch.tensor(float("-inf"), device=residual.device))
        vals, idx = masked.topk(k_aux, sorted=False)
        x_aux = torch.zeros_like(post_relu).scatter_(-1, idx, vals) @ self.W_dec
        l2 = (residual.float() - x_aux.float()).pow(2).sum(-1).mean()
        mu = residual.mean(0, keepdim=True)
        denom = (residual.float() - mu.float()).pow(2).sum(-1).mean()
        return (l2 / denom).nan_to_num(0.0)

    def post_step(self):
        with torch.no_grad():
            self.W_dec.data = _set_decoder_norm_to_unit(self.W_dec.data.T).T.contiguous()
