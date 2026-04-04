"""Temporal Feature Autoencoder ported into the TemporalAE interface."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutput, TemporalAE


def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model % 2 == 0:
        pe[:, 1::2] = torch.cos(position * div_term)
    else:
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
    return pe


class ManualAttention(nn.Module):
    """Small causal attention block used by TFA."""

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int = 4,
        bottleneck_factor: int = 1,
        use_pos_encoding: bool = False,
        max_seq_len: int = 512,
    ):
        super().__init__()
        if d_model % (bottleneck_factor * n_heads) != 0:
            raise ValueError("d_model must be divisible by bottleneck_factor * n_heads")

        self.n_heads = n_heads
        self.d_model = d_model
        self.qk_dim = d_model // bottleneck_factor

        self.k_ctx = nn.Linear(d_model, self.qk_dim, bias=True)
        self.q_target = nn.Linear(d_model, self.qk_dim, bias=True)
        self.v_ctx = nn.Linear(d_model, d_model, bias=True)
        self.c_proj = nn.Linear(d_model, d_model, bias=True)

        self._renorm_weights()

        if use_pos_encoding:
            self.register_buffer(
                "pos_enc",
                _sinusoidal_positional_encoding(max_seq_len, d_model),
            )
        else:
            self.pos_enc = None

    def _renorm_weights(self) -> None:
        with torch.no_grad():
            qk_scale = 1.0 / math.sqrt(self.qk_dim // self.n_heads)
            for proj in (self.k_ctx, self.q_target):
                proj.weight.copy_(qk_scale * F.normalize(proj.weight, dim=1))
            v_scale = 1.0 / math.sqrt(self.d_model // self.n_heads)
            self.v_ctx.weight.copy_(v_scale * F.normalize(self.v_ctx.weight, dim=1))
            out_scale = 1.0 / math.sqrt(self.d_model)
            self.c_proj.weight.copy_(out_scale * F.normalize(self.c_proj.weight, dim=1))

    def forward(
        self,
        x_ctx: torch.Tensor,
        x_target: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.pos_enc is not None:
            T = x_ctx.shape[1]
            pe = self.pos_enc[:T].unsqueeze(0)
            x_ctx_qk = x_ctx + pe
            x_target_qk = x_target + pe
        else:
            x_ctx_qk = x_ctx
            x_target_qk = x_target

        k = self.k_ctx(x_ctx_qk)
        q = self.q_target(x_target_qk)
        v = self.v_ctx(x_ctx)

        B, T, _ = x_ctx.shape
        q = q.view(B, T, self.n_heads, self.qk_dim // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.qk_dim // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)

        attn_map = None
        if return_attention:
            scale = 1.0 / math.sqrt(k.shape[-1])
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.triu(
                torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
            attn_map = scores.softmax(dim=-1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        merged = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.c_proj(merged), attn_map


class TemporalFeatureAutoencoder(TemporalAE):
    """TFA with causal attention over sparse feature codes."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int | None,
        *,
        n_heads: int = 4,
        n_attn_layers: int = 1,
        bottleneck_factor: int = 1,
        tied_weights: bool = True,
        use_pos_encoding: bool = False,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.tied_weights = tied_weights
        self.lam = 1.0 / (4.0 * d_in)

        self.D = nn.Parameter(torch.randn(d_sae, d_in))
        self.b = nn.Parameter(torch.zeros(1, d_in))
        if not tied_weights:
            self.E = nn.Parameter(torch.randn(d_in, d_sae))

        self.attn_layers = nn.ModuleList(
            [
                ManualAttention(
                    d_sae,
                    n_heads=n_heads,
                    bottleneck_factor=bottleneck_factor,
                    use_pos_encoding=use_pos_encoding,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_attn_layers)
            ]
        )

        self._normalize_decoder()

    @property
    def encoder(self) -> torch.Tensor:
        return self.D.T if self.tied_weights else self.E

    def _normalize_decoder(self) -> None:
        self.D.data.copy_(F.normalize(self.D.data, dim=-1))

    def _apply_topk(self, z: torch.Tensor) -> torch.Tensor:
        if self.k is None:
            return z
        topk_vals, topk_idx = z.topk(self.k, dim=-1)
        sparse = torch.zeros_like(z)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse

    def forward(self, x: torch.Tensor, *, return_attention: bool = False) -> ModelOutput:
        B, T, _ = x.shape
        x_resid = x - self.b

        pred_codes = torch.zeros(B, T, self.d_sae, device=x.device, dtype=x.dtype)
        pred_recons = torch.zeros_like(x)
        attn_graphs = [] if return_attention else None

        for attn_layer in self.attn_layers:
            z_input = F.relu(torch.matmul(x_resid * self.lam, self.encoder))
            z_ctx = torch.cat((torch.zeros_like(z_input[:, :1]), z_input[:, :-1]), dim=1)

            z_pred_step, attn_map = attn_layer(
                z_ctx, z_input, return_attention=return_attention
            )
            z_pred_step = F.relu(z_pred_step)
            Dz_pred_step = torch.matmul(z_pred_step, self.D)
            denom = Dz_pred_step.norm(dim=-1, keepdim=True).pow(2).clamp(min=1e-8)
            proj_scale = (Dz_pred_step * x_resid).sum(dim=-1, keepdim=True) / denom

            pred_codes = pred_codes + z_pred_step * proj_scale
            pred_recons = pred_recons + Dz_pred_step * proj_scale
            x_resid = x_resid - Dz_pred_step * proj_scale

            if attn_graphs is not None and attn_map is not None:
                attn_graphs.append(attn_map)

        novel_codes = F.relu(torch.matmul(x_resid * self.lam, self.encoder))
        novel_codes = self._apply_topk(novel_codes)
        novel_recons = torch.matmul(novel_codes, self.D)
        x_hat = pred_recons + novel_recons + self.b

        pred_mask = (pred_codes.abs() > 1e-8).float()
        novel_mask = (novel_codes > 0).float()
        total_mask = torch.maximum(pred_mask, novel_mask)

        metrics: dict[str, float] = {}
        if self.collect_metrics:
            metrics = {
                "recon_loss": (x - x_hat).pow(2).sum(dim=-1).mean().item(),
                "novel_l0": novel_mask.sum(dim=-1).mean().item(),
                "pred_l0": pred_mask.sum(dim=-1).mean().item(),
                "l0": total_mask.sum(dim=-1).mean().item(),
            }

        aux: dict[str, torch.Tensor] = {}
        if self.collect_metrics:
            aux = {
                "pred_codes": pred_codes,
                "novel_codes": novel_codes,
                "pred_recons": pred_recons,
                "novel_recons": novel_recons,
            }
        if attn_graphs:
            aux["attn_graphs"] = torch.stack(attn_graphs, dim=1)

        loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        return ModelOutput(
            x_hat=x_hat,
            latents=pred_codes + novel_codes,
            metric_latents=total_mask,
            loss=loss,
            metrics=metrics,
            aux=aux,
        )

    def latents_for_metrics(self, out: ModelOutput) -> torch.Tensor:
        if out.metric_latents is not None:
            return out.metric_latents
        return super().latents_for_metrics(out)

    def decoder_directions(self, pos: int | None = None) -> torch.Tensor:
        return self.D.T

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()

    def reconstruction_bias(self, pos: int | None = None) -> torch.Tensor:
        return self.b.squeeze(0)
