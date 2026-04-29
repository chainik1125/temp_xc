"""Faithful port of Bhalla et al. 2025 Temporal SAE (arxiv 2511.05541).
Originally mis-cited as 'Ye et al.' in this docstring and several research-log
docs across the repo; fixed here. Citation is:
    Bhalla, Oesterling, Verdun, Lakkaraju, Calmon (2025).
    "Temporal Sparse Autoencoders: Leveraging the Sequential Nature of
    Language for Interpretability." arXiv:2511.05541.

Sources this file directly mirrors (from
https://github.com/AI4LIFE-GROUP/temporal-saes):

    dictionary_learning/dictionary_learning/trainers/temporal_sequence_top_k.py
        TemporalMatryoshkaBatchTopKSAE, TemporalMatryoshkaBatchTopKTrainer
    dictionary_learning/dictionary_learning/trainers/trainer.py
        set_decoder_norm_to_unit_norm,
        remove_gradient_parallel_to_decoder_directions,
        get_lr_schedule

Differences vs the prior `tsae_ours.py` port (which was a crude sketch
based on the paper's equations alone):

    - **Matryoshka Batch-TopK**: the paper uses BatchTopK (flat top-k*B
      across the batch) with grouped decoder slices, not plain TopK.
    - **Auxiliary K loss** (paper App B.1): recruits dead features into
      a residual reconstruction; the primary mechanism that keeps
      alive fraction ≳ 0.78 (Table 1).
    - **Geometric median init** for `b_dec` at step 0.
    - **Encoder = decoder.T** init (kaiming-uniform on decoder).
    - **Decoder unit-norm constraint** enforced by projecting out the
      parallel-component of the gradient and renormalising after each
      step — this is what makes decoder cosine similarity stay bounded.
    - **Temporal contrastive loss** uses raw dot product (not
      L2-normalised cosine) with `temp_alpha = 1/10` (paper's code).
    - **Threshold-based inference**: an EMA-tracked threshold replaces
      BatchTopK at inference time, so per-token sparsity is variable.
    - **LR schedule**: linear warmup (1000 steps) + optional decay.

This module exposes:

    TemporalMatryoshkaBatchTopKSAE  — the nn.Module (architecture only)
    TemporalMatryoshkaBatchTopKTrainerLite — minimal trainer wrapping
        the paper's `loss()` + `update()` logic into our pair-generator
        training convention. We drop the Wandb/nnsight integration.
"""

from __future__ import annotations

from collections import namedtuple
from math import isclose
from typing import Optional

import einops
import torch as t
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────── decoder-norm utilities ──


@t.no_grad()
def set_decoder_norm_to_unit_norm(
    W_dec_DF: t.Tensor, activation_dim: int, d_sae: int
) -> t.Tensor:
    """Normalise decoder columns (shape (d_in, d_sae)) to unit L2 norm."""
    D, F = W_dec_DF.shape
    assert D == activation_dim and F == d_sae
    eps = t.finfo(W_dec_DF.dtype).eps
    norm = t.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


@t.no_grad()
def remove_gradient_parallel_to_decoder_directions(
    W_dec_DF: t.Tensor, grad: t.Tensor, activation_dim: int, d_sae: int
) -> t.Tensor:
    """Project out the component of `grad` parallel to each decoder column.

    Without this the unit-norm constraint is violated between steps and
    the renormalisation shrinks the update.
    """
    D, F = W_dec_DF.shape
    assert D == activation_dim and F == d_sae
    normed_W = W_dec_DF / (t.norm(W_dec_DF, dim=0, keepdim=True) + 1e-6)
    parallel = einops.einsum(grad, normed_W, "d_in d_sae, d_in d_sae -> d_sae")
    grad -= einops.einsum(parallel, normed_W, "d_sae, d_in d_sae -> d_in d_sae")
    return grad


def get_lr_schedule(total_steps: int, warmup_steps: int,
                    decay_start: Optional[int] = None):
    """Linear warmup + optional linear decay."""
    if decay_start is not None:
        assert warmup_steps < decay_start < total_steps

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if decay_start is not None and step >= decay_start:
            return max(0.0, (total_steps - step) / (total_steps - decay_start))
        return 1.0
    return schedule


# ──────────────────────────────────────────────────── architecture ──


class TemporalMatryoshkaBatchTopKSAE(nn.Module):
    """Paper's T-SAE with Matryoshka groups + BatchTopK + threshold inference.

    Args:
        activation_dim: `d_in` residual width.
        dict_size: `d_sae`.
        k: batch-topk budget; average active features per token = k.
        group_sizes: list of ints summing to `dict_size`. Defines the
            matryoshka partition. Paper uses `[0.2 * d_sae, 0.8 * d_sae]`.
    """

    def __init__(self, activation_dim: int, dict_size: int,
                 k: int, group_sizes: list[int]):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        assert sum(group_sizes) == dict_size
        assert all(s > 0 for s in group_sizes)
        assert isinstance(k, int) and k > 0

        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))
        self.register_buffer("group_sizes", t.tensor(group_sizes))

        self.active_groups = len(group_sizes)
        self.group_indices = [0] + list(t.cumsum(
            t.tensor(group_sizes), dim=0
        ).tolist())

        self.W_enc = nn.Parameter(t.empty(activation_dim, dict_size))
        self.b_enc = nn.Parameter(t.zeros(dict_size))
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(t.empty(dict_size, activation_dim))
        )
        self.b_dec = nn.Parameter(t.zeros(activation_dim))

        # Normalise decoder columns (operate on transposed shape (d_in, d_sae))
        self.W_dec.data = set_decoder_norm_to_unit_norm(
            self.W_dec.data.T, activation_dim, dict_size
        ).T
        # Encoder init = decoder transposed
        self.W_enc.data = self.W_dec.data.clone().T

    # -- encoding --

    def encode(self, x: t.Tensor, return_active: bool = False,
               use_threshold: bool = True) -> t.Tensor | tuple:
        """BatchTopK during training (use_threshold=False); threshold at inference."""
        post_relu = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

        if use_threshold:
            z = post_relu * (post_relu > self.threshold)
        else:
            # flat BatchTopK over (B * d_sae)
            flat = post_relu.flatten()
            topk = flat.topk(int(self.k.item()) * x.size(0), sorted=False)
            z = (t.zeros_like(flat)
                 .scatter_(-1, topk.indices, topk.values)
                 .reshape(post_relu.shape))

        if return_active:
            return z, z.sum(dim=0) > 0, post_relu
        return z

    # -- reconstruction --

    def decode(self, z: t.Tensor) -> t.Tensor:
        return z @ self.W_dec + self.b_dec


# ─────────────────────────────────────────────────────── trainer ──


class TemporalMatryoshkaBatchTopKTrainerLite:
    """Thin wrapper around the paper's `loss()` + `update()` that drives
    an existing nn.Module + pair generator. Drops the Wandb/nnsight hooks.
    """

    def __init__(
        self,
        model: TemporalMatryoshkaBatchTopKSAE,
        group_weights: list[float],
        total_steps: int,
        lr: float,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,
        auxk_alpha: float = 1.0 / 32.0,
        temp_alpha: float = 1.0 / 10.0,
        threshold_start_step: int = 1000,
        threshold_beta: float = 0.999,
        dead_feature_threshold: int = 10_000_000,  # tokens since fired
        contrastive: bool = True,
        device: t.device = t.device("cuda"),
    ):
        self.ae = model
        self.group_weights = group_weights
        self.total_steps = total_steps
        self.auxk_alpha = auxk_alpha
        self.temp_alpha = temp_alpha
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step
        self.dead_feature_threshold = dead_feature_threshold
        self.contrastive = contrastive
        self.device = device

        self.top_k_aux = model.activation_dim // 2   # paper App B.1 heuristic
        self.num_tokens_since_fired = t.zeros(
            model.dict_size, dtype=t.long, device=device
        )
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1.0

        self.optimizer = t.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.999)
        )
        schedule = get_lr_schedule(total_steps, warmup_steps, decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=schedule
        )

    # ---- auxiliary loss for dead features ----

    def get_auxiliary_loss(self, residual: t.Tensor, post_relu: t.Tensor) -> t.Tensor:
        dead = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead.sum().item())
        if self.dead_features == 0:
            self.pre_norm_auxk_loss = -1.0
            return t.tensor(0.0, dtype=residual.dtype, device=residual.device)

        k_aux = min(self.top_k_aux, self.dead_features)
        auxk_pre = t.where(dead[None], post_relu, t.tensor(float("-inf"), device=residual.device))
        auxk_vals, auxk_idx = auxk_pre.topk(k_aux, sorted=False)
        auxk_buf = t.zeros_like(post_relu).scatter_(-1, auxk_idx, auxk_vals)

        # Reconstruct residual from dead features only (no bias)
        x_aux = auxk_buf @ self.ae.W_dec
        l2 = (residual.float() - x_aux.float()).pow(2).sum(dim=-1).mean()
        self.pre_norm_auxk_loss = float(l2.detach())

        # Normalise by variance of residual (OpenAI convention)
        mu = residual.mean(dim=0, keepdim=True).broadcast_to(residual.shape)
        denom = (residual.float() - mu.float()).pow(2).sum(dim=-1).mean()
        return (l2 / denom).nan_to_num(0.0)

    # ---- threshold EMA ----

    def update_threshold(self, f: t.Tensor):
        with t.no_grad():
            active = f[f > 0]
            cur = active.min().float() if active.numel() else t.tensor(0.0, device=f.device)
            if self.ae.threshold < 0:
                self.ae.threshold = cur
            else:
                self.ae.threshold = (
                    self.threshold_beta * self.ae.threshold
                    + (1 - self.threshold_beta) * cur
                )

    # ---- main loss ----

    def loss(self, x_pair: t.Tensor, step: int):
        """x_pair: (B, 2, d_in). Returns scalar loss + dict of metrics."""
        f, active_idx, post_relu = self.ae.encode(
            x_pair[:, 0], return_active=True, use_threshold=False
        )
        f_temp, _, _ = self.ae.encode(
            x_pair[:, 1], return_active=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            self.update_threshold(f)

        total_l2 = t.tensor(0.0, device=self.device)
        x_reconstruct = t.zeros_like(x_pair[:, 0]) + self.ae.b_dec
        W_chunks = t.split(self.ae.W_dec, self.ae.group_sizes.tolist(), dim=0)
        f_chunks = t.split(f, self.ae.group_sizes.tolist(), dim=1)
        f_temp_chunks = t.split(f_temp, self.ae.group_sizes.tolist(), dim=1)

        # group 0: high-level, gets reconstruction + contrastive
        W0 = W_chunks[0]; f0 = f_chunks[0]; f0_temp = f_temp_chunks[0]
        x_reconstruct = x_reconstruct + f0 @ W0
        l2_0 = ((x_pair[:, 0] - x_reconstruct).pow(2).sum(dim=-1)
                * self.group_weights[0]).mean()
        total_l2 = total_l2 + l2_0

        if self.contrastive:
            # Raw dot-product InfoNCE between high-level latents of current & temporal pair
            logits = f0 @ f0_temp.T  # (B, B)
            labels = t.arange(logits.shape[0], device=self.device)
            temp_loss = 0.5 * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)
            )
        else:
            # L1-on-diff, scaled by x cosine (paper fallback)
            x_temp_sim = F.cosine_similarity(x_pair[:, 0], x_pair[:, 1], dim=-1)
            temp_loss = ((f0 - f0_temp).abs().sum(dim=-1) * x_temp_sim
                         * self.group_weights[0]).mean()

        # subsequent groups: cumulative matryoshka reconstruction
        for gi in range(1, self.ae.active_groups):
            x_reconstruct = x_reconstruct + f_chunks[gi] @ W_chunks[gi]
            total_l2 = total_l2 + (
                (x_pair[:, 0] - x_reconstruct).pow(2).sum(dim=-1).mean()
                * self.group_weights[gi]
            )

        self.effective_l0 = int(self.ae.k.item())

        # Update "tokens since fired" for aux loss
        B = x_pair.shape[0]
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_idx] = True
        self.num_tokens_since_fired += B
        self.num_tokens_since_fired[did_fire] = 0

        auxk = self.get_auxiliary_loss(
            (x_pair[:, 0] - x_reconstruct).detach(), post_relu
        )

        total = total_l2 + self.auxk_alpha * auxk + self.temp_alpha * temp_loss
        stats = {
            "l2": float(total_l2.detach()),
            "auxk": float(auxk.detach()),
            "temp": float(temp_loss.detach()),
            "total": float(total.detach()),
            "dead": self.dead_features,
            "threshold": float(self.ae.threshold.item()) if self.ae.threshold.item() >= 0 else -1.0,
        }
        return total, stats, x_reconstruct, f

    # ---- one step ----

    def update(self, step: int, x_pair: t.Tensor, b_dec_init: Optional[t.Tensor] = None):
        if step == 0 and b_dec_init is not None:
            self.ae.b_dec.data = b_dec_init.to(self.ae.b_dec.dtype)

        x_pair = x_pair.to(self.device)
        total, stats, _, _ = self.loss(x_pair, step=step)
        total.backward()

        # Remove gradient parallel to decoder directions (paper's safeguard
        # to prevent the unit-norm renormalise from shrinking each update)
        self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.W_dec.T, self.ae.W_dec.grad.T,
            self.ae.activation_dim, self.ae.dict_size,
        ).T
        nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Re-enforce decoder unit norm after the step
        self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
            self.ae.W_dec.data.T, self.ae.activation_dim, self.ae.dict_size,
        ).T
        return stats


# ─────────────────────────────────────────── geometric median (b_dec init) ──


@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100,
                     tol: float = 1e-5) -> t.Tensor:
    """Weiszfeld iteration to compute the geometric median of rows of `points`."""
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess
        weights = 1.0 / (t.norm(points - guess, dim=1) + 1e-8)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if t.norm(guess - prev) < tol:
            break
    return guess
