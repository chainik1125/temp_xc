"""Temporal Feature Analysis (TFA) — TemporalSAE with causal attention.

Ported from Han's implementation on the `han` branch of
github.com/Astera-org/temp_xc (files: src/v2_temporal_schemeC/tfa/{saeTemporal.py,
utils.py} + src/v2_temporal_schemeC/train_tfa.py). The original TFA is from the
paper "Temporal Feature Analysis" (Han et al.); this port just removes cross-
branch imports so it stands alone in sae_day.

Architecture:
    D: (width, dimin) — shared dictionary
    Attention layers: causal self-attention on encoded representations
    Sparsity: TopK / ReLU / BatchTopK on the novel part

Each token's representation is decomposed into:
    - pred_codes: predicted from prior context via causal attention
    - novel_codes: residual codes from a per-token SAE

Use `use_pos_encoding=True` for TFA-pos (sinusoidal PEs added to Q/K).

TFA assumes inputs with norm ≈ √d (internal `lam = 1/(4·dimin)`). When
applying to real residual-stream activations, precompute `compute_scaling_factor`
once on a training batch and apply to all inputs (train + eval).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding + attention
# ---------------------------------------------------------------------------


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model % 2 == 0:
        pe[:, 1::2] = torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
    return pe


def _get_attention_map(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
    attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight + attn_bias, dim=-1)
    return attn_weight


class ManualAttention(nn.Module):
    """Causal multi-head attention with a bottleneck factor.

    Projects keys/queries through a `dimin // bottleneck_factor` bottleneck while
    keeping values full-dim. Optionally adds sinusoidal positional encoding to
    Q/K inputs (not V).
    """

    def __init__(
        self,
        dimin: int,
        n_heads: int = 4,
        bottleneck_factor: int = 64,
        bias_k: bool = True,
        bias_q: bool = True,
        bias_v: bool = True,
        bias_o: bool = True,
        use_pos_encoding: bool = False,
        max_seq_len: int = 512,
    ):
        super().__init__()
        assert dimin % (bottleneck_factor * n_heads) == 0, (
            f"dimin ({dimin}) must be divisible by bottleneck_factor * n_heads "
            f"({bottleneck_factor * n_heads})"
        )
        self.n_heads = n_heads
        self.n_embds = dimin // bottleneck_factor
        self.dimin = dimin

        self.k_ctx = nn.Linear(dimin, self.n_embds, bias=bias_k)
        self.q_target = nn.Linear(dimin, self.n_embds, bias=bias_q)
        self.v_ctx = nn.Linear(dimin, dimin, bias=bias_v)
        self.c_proj = nn.Linear(dimin, dimin, bias=bias_o)

        with torch.no_grad():
            s = 1 / math.sqrt(self.n_embds // self.n_heads)
            self.k_ctx.weight.copy_(s * self.k_ctx.weight / (1e-6 + torch.linalg.norm(self.k_ctx.weight, dim=1, keepdim=True)))
            self.q_target.weight.copy_(s * self.q_target.weight / (1e-6 + torch.linalg.norm(self.q_target.weight, dim=1, keepdim=True)))

            s = 1 / math.sqrt(self.dimin // self.n_heads)
            self.v_ctx.weight.copy_(s * self.v_ctx.weight / (1e-6 + torch.linalg.norm(self.v_ctx.weight, dim=1, keepdim=True)))

            s = 1 / math.sqrt(self.dimin)
            self.c_proj.weight.copy_(s * self.c_proj.weight / (1e-6 + torch.linalg.norm(self.c_proj.weight, dim=1, keepdim=True)))

        if use_pos_encoding:
            self.register_buffer("pos_enc", sinusoidal_positional_encoding(max_seq_len, dimin))
        else:
            self.pos_enc = None

    def forward(
        self, x_ctx: torch.Tensor, x_target: torch.Tensor, get_attn_map: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.pos_enc is not None:
            T = x_ctx.size(1)
            pe = self.pos_enc[:T, :]
            x_ctx_qk = x_ctx + pe.unsqueeze(0)
            x_target_qk = x_target + pe.unsqueeze(0)
        else:
            x_ctx_qk = x_ctx
            x_target_qk = x_target

        k = self.k_ctx(x_ctx_qk)
        v = self.v_ctx(x_ctx)
        q = self.q_target(x_target_qk)

        B, T, _ = x_ctx.size()
        k = k.view(B, T, self.n_heads, self.n_embds // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.n_embds // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.dimin // self.n_heads).transpose(1, 2)

        attn_map = _get_attention_map(q, k) if get_attn_map else None

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0, is_causal=True,
        )
        d_target = self.c_proj(attn_output.transpose(1, 2).contiguous().view(B, T, self.dimin))
        return d_target, attn_map


# ---------------------------------------------------------------------------
# TemporalSAE (the main TFA module)
# ---------------------------------------------------------------------------


class TemporalSAE(nn.Module):
    """Temporal Feature Analysis SAE.

    Input: `(B, T, dimin)` sequence of activations.
    Output: `(recons, results_dict)` where results_dict has keys
      - `novel_codes`: `(B, T, width)` sparsified per-token codes
      - `pred_codes`:  `(B, T, width)` codes predicted from context
      - `novel_recons`, `pred_recons`: `(B, T, dimin)` reconstructions sans bias
      - `attn_graphs`: attention maps if `return_graph=True`
    """

    def __init__(
        self,
        dimin: int = 2,
        width: int = 5,
        n_heads: int = 8,
        sae_diff_type: str = "relu",
        kval_topk: int | None = None,
        tied_weights: bool = True,
        n_attn_layers: int = 1,
        bottleneck_factor: int = 64,
        inference_mode_batchtopk: bool = False,
        min_act_regularizer_batchtopk: float = 0.999,
        use_pos_encoding: bool = False,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.sae_type = "temporal"
        self.width = width
        self.dimin = dimin
        self.eps = 1e-6
        self.lam = 1 / (4 * dimin)
        self.tied_weights = tied_weights
        self.use_pos_encoding = use_pos_encoding

        self.n_attn_layers = n_attn_layers
        self.attn_layers = nn.ModuleList([
            ManualAttention(
                dimin=width, n_heads=n_heads, bottleneck_factor=bottleneck_factor,
                bias_k=True, bias_q=True, bias_v=True, bias_o=True,
                use_pos_encoding=use_pos_encoding, max_seq_len=max_seq_len,
            )
            for _ in range(n_attn_layers)
        ])

        self.D = nn.Parameter(torch.randn((width, dimin)))
        self.b = nn.Parameter(torch.zeros((1, dimin)))
        if not tied_weights:
            self.E = nn.Parameter(torch.randn((dimin, width)))

        self.sae_diff_type = sae_diff_type
        self.kval_topk = kval_topk if sae_diff_type in ("topk", "batchtopk") else None

        if sae_diff_type == "batchtopk":
            self.inference_mode_batchtopk = inference_mode_batchtopk
            self.expected_min_act = nn.Parameter(torch.zeros(1))
            self.expected_min_act.requires_grad = False
            self.min_act_regularizer_batchtopk = min_act_regularizer_batchtopk

    def forward(
        self, x_input: torch.Tensor, return_graph: bool = False, inf_k: int | None = None
    ) -> tuple[torch.Tensor, dict]:
        B, L, _ = x_input.size()
        E = self.D.T if self.tied_weights else self.E

        x_input = x_input - self.b
        attn_graphs = []

        # Predictable part: iteratively peel off projections via causal attention
        z_pred = torch.zeros((B, L, self.width), device=x_input.device, dtype=x_input.dtype)
        for attn_layer in self.attn_layers:
            z_input = F.relu(torch.matmul(x_input * self.lam, E))
            z_ctx = torch.cat((torch.zeros_like(z_input[:, :1, :]), z_input[:, :-1, :].clone()), dim=1)

            z_pred_, attn_g = attn_layer(z_ctx, z_input, get_attn_map=return_graph)
            z_pred_ = F.relu(z_pred_)

            Dz_pred_ = torch.matmul(z_pred_, self.D)
            Dz_norm_sq = Dz_pred_.norm(dim=-1, keepdim=True).pow(2) + self.eps
            proj_scale = (Dz_pred_ * x_input).sum(dim=-1, keepdim=True) / Dz_norm_sq

            z_pred = z_pred + (z_pred_ * proj_scale)
            x_input = x_input - proj_scale * Dz_pred_

            if return_graph:
                attn_graphs.append(attn_g)

        # Novel part
        if self.sae_diff_type == "relu":
            z_novel = F.relu(torch.matmul(x_input * self.lam, E))
        elif self.sae_diff_type == "topk":
            kval = self.kval_topk if inf_k is None else inf_k
            z_novel = F.relu(torch.matmul(x_input * self.lam, E))
            _, topk_idx = torch.topk(z_novel, kval, dim=-1)
            mask = torch.zeros_like(z_novel)
            mask.scatter_(-1, topk_idx, 1)
            z_novel = z_novel * mask
        elif self.sae_diff_type == "batchtopk":
            kval = self.kval_topk if inf_k is None else inf_k
            kval_full_batch = kval * B * L
            z_novel = F.relu(torch.matmul(x_input * self.lam, E))
            if not self.inference_mode_batchtopk:
                z_flat = z_novel.flatten()
                topk_vals, topk_idx = torch.topk(z_flat, kval_full_batch, dim=-1)
                z_novel_sparse = torch.zeros_like(z_flat)
                z_novel_sparse.scatter_(-1, topk_idx, topk_vals)
                z_novel = z_novel_sparse.reshape(z_novel.shape)
                active = z_flat[z_flat > 0]
                min_activation = active.min().detach().to(dtype=z_novel.dtype) if active.numel() > 0 else torch.tensor(0.0, dtype=z_novel.dtype, device=z_novel.device)
                self.expected_min_act[0] = (
                    self.min_act_regularizer_batchtopk * self.expected_min_act[0]
                    + (1 - self.min_act_regularizer_batchtopk) * min_activation
                )
            else:
                z_novel = z_novel * (z_novel > self.expected_min_act[0])
        elif self.sae_diff_type == "nullify":
            z_novel = torch.zeros_like(z_pred)
        else:
            raise ValueError(f"unknown sae_diff_type: {self.sae_diff_type}")

        x_recons = torch.matmul(z_novel + z_pred, self.D) + self.b

        with torch.no_grad():
            x_pred_recons = torch.matmul(z_pred, self.D)
            x_novel_recons = torch.matmul(z_novel, self.D)

        results_dict = {
            "novel_codes": z_novel,
            "novel_recons": x_novel_recons,
            "pred_codes": z_pred,
            "pred_recons": x_pred_recons,
            "attn_graphs": torch.stack(attn_graphs, dim=1) if return_graph else None,
        }
        return x_recons, results_dict


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@dataclass
class TFATrainingConfig:
    total_steps: int = 5000
    batch_size: int = 64
    lr: float = 1e-3
    min_lr: float = 9e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    log_every: int = 500
    l1_coeff: float = 0.0


def compute_scaling_factor(acts: torch.Tensor) -> float:
    """Compute scaling factor s.t. scaled inputs have mean L2 norm ≈ √d.

    TFA's internal `lam = 1/(4·dimin)` is calibrated for unit-ish activations;
    real residual-stream activations have much larger norms. Apply the returned
    factor multiplicatively to *both* train and eval inputs once at start.
    """
    d = acts.shape[-1]
    mean_norm = acts.float().norm(dim=-1).mean().item()
    if mean_norm < 1e-8:
        return 1.0
    return math.sqrt(d) / mean_norm


def _configure_optimizer(model: nn.Module, cfg: TFATrainingConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
    )


def _get_lr(step: int, cfg: TFATrainingConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def train_tfa(
    model: TemporalSAE,
    sample_batch_fn: Callable[[int], torch.Tensor],
    config: TFATrainingConfig,
    device: torch.device,
    verbose: bool = True,
) -> dict:
    """Train TFA. `sample_batch_fn(batch_size)` must return `(B, T, dimin)` tensor."""
    optimizer = _configure_optimizer(model, config)
    model.train()
    log = {"loss": [], "mse": [], "rel_energy_pred": []}

    for step in range(config.total_steps):
        lr = _get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = sample_batch_fn(config.batch_size).to(device)
        recons, inter = model(batch)

        batch_flat = batch.reshape(-1, batch.shape[-1])
        recons_flat = recons.reshape(-1, recons.shape[-1])
        n_tokens = batch_flat.shape[0]
        mse_loss = F.mse_loss(recons_flat, batch_flat, reduction="sum") / n_tokens
        loss = mse_loss
        if config.l1_coeff > 0:
            loss = loss + config.l1_coeff * inter["novel_codes"].abs().sum() / n_tokens

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if verbose and (step % config.log_every == 0 or step == config.total_steps - 1):
            with torch.no_grad():
                pred_norm = inter["pred_recons"].norm(dim=-1).pow(2).mean()
                novel_norm = inter["novel_recons"].norm(dim=-1).pow(2).mean()
                rel_energy = (pred_norm / (pred_norm + novel_norm + 1e-12)).item()
            log["loss"].append(loss.item())
            log["mse"].append(mse_loss.item())
            log["rel_energy_pred"].append(rel_energy)
            print(
                f"    TFA step {step:5d}/{config.total_steps} "
                f"lr={lr:.2e} MSE={mse_loss.item():.6f} pred_energy={rel_energy:.3f}",
                flush=True,
            )

    model.eval()
    return log
