"""Temporal Feature Analysis (TFA) — TemporalSAE nn.Module.

Copied from Han's src/v2_temporal_schemeC/tfa/ to make it available
on main without cross-branch imports. Original code by the TFA authors,
adapted by Han for toy model experiments.

The TFA decomposes each token's representation into:
    - pred_codes: codes predicted from prior context via causal attention
    - novel_codes: residual codes identified by a per-token SAE

Architecture:
    D: (width, dimin) — shared dictionary
    Attention layers: causal self-attention on encoded representations
    Sparsity: TopK, ReLU, or BatchTopK on the novel part
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Attention utils -----------------------------------------------------------

def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
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


class ManualAttention(nn.Module):
    def __init__(
        self,
        dimin,
        n_heads=4,
        bottleneck_factor=64,
        bias_k=True,
        bias_q=True,
        bias_v=True,
        bias_o=True,
        use_pos_encoding=False,
        max_seq_len=512,
    ):
        super().__init__()
        assert dimin % (bottleneck_factor * n_heads) == 0
        self.n_heads = n_heads
        self.n_embds = dimin // bottleneck_factor
        self.dimin = dimin

        self.k_ctx = nn.Linear(dimin, self.n_embds, bias=bias_k)
        self.q_target = nn.Linear(dimin, self.n_embds, bias=bias_q)
        self.v_ctx = nn.Linear(dimin, dimin, bias=bias_v)
        self.c_proj = nn.Linear(dimin, dimin, bias=bias_o)

        with torch.no_grad():
            scaling = 1 / math.sqrt(self.n_embds // self.n_heads)
            self.k_ctx.weight.copy_(
                scaling
                * self.k_ctx.weight
                / (1e-6 + self.k_ctx.weight.norm(dim=1, keepdim=True))
            )
            self.q_target.weight.copy_(
                scaling
                * self.q_target.weight
                / (1e-6 + self.q_target.weight.norm(dim=1, keepdim=True))
            )
            scaling = 1 / math.sqrt(self.dimin // self.n_heads)
            self.v_ctx.weight.copy_(
                scaling
                * self.v_ctx.weight
                / (1e-6 + self.v_ctx.weight.norm(dim=1, keepdim=True))
            )
            scaling = 1 / math.sqrt(self.dimin)
            self.c_proj.weight.copy_(
                scaling
                * self.c_proj.weight
                / (1e-6 + self.c_proj.weight.norm(dim=1, keepdim=True))
            )

        if use_pos_encoding:
            self.register_buffer(
                "pos_enc", sinusoidal_positional_encoding(max_seq_len, dimin)
            )
        else:
            self.pos_enc = None

    def forward(self, x_ctx, x_target, get_attn_map=False):
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

        attn_map = None
        if get_attn_map:
            scale_factor = 1 / math.sqrt(q.size(-1))
            attn_weight = q @ k.transpose(-2, -1) * scale_factor
            L, S = q.size(-2), k.size(-2)
            attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
            temp_mask = torch.ones(
                L, S, dtype=torch.bool, device=q.device
            ).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_weight = torch.softmax(attn_weight + attn_bias, dim=-1)
            attn_map = attn_weight

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0, is_causal=True
        )

        d_target = self.c_proj(
            attn_output.transpose(1, 2).contiguous().view(B, T, self.dimin)
        )
        return d_target, attn_map


# -- TemporalSAE nn.Module ----------------------------------------------------


class TemporalSAE(nn.Module):
    def __init__(
        self,
        dimin=2,
        width=5,
        n_heads=8,
        sae_diff_type="relu",
        kval_topk=None,
        tied_weights=True,
        n_attn_layers=1,
        bottleneck_factor=64,
        use_pos_encoding=False,
        max_seq_len=512,
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
        self.attn_layers = nn.ModuleList(
            [
                ManualAttention(
                    dimin=width,
                    n_heads=n_heads,
                    bottleneck_factor=bottleneck_factor,
                    bias_k=True,
                    bias_q=True,
                    bias_v=True,
                    bias_o=True,
                    use_pos_encoding=use_pos_encoding,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_attn_layers)
            ]
        )

        self.D = nn.Parameter(torch.randn((width, dimin)))
        self.b = nn.Parameter(torch.zeros((1, dimin)))
        if not tied_weights:
            self.E = nn.Parameter(torch.randn((dimin, width)))

        self.sae_diff_type = sae_diff_type
        self.kval_topk = kval_topk if sae_diff_type in ["topk", "batchtopk"] else None

    def forward(self, x_input, return_graph=False, inf_k=None):
        B, L, _ = x_input.size()
        E = self.D.T if self.tied_weights else self.E

        x_input = x_input - self.b
        attn_graphs = []

        z_pred = torch.zeros(
            (B, L, self.width), device=x_input.device, dtype=x_input.dtype
        )
        for attn_layer in self.attn_layers:
            z_input = F.relu(torch.matmul(x_input * self.lam, E))
            z_ctx = torch.cat(
                (torch.zeros_like(z_input[:, :1, :]), z_input[:, :-1, :].clone()),
                dim=1,
            )
            z_pred_, attn_graphs_ = attn_layer(
                z_ctx, z_input, get_attn_map=return_graph
            )
            z_pred_ = F.relu(z_pred_)
            Dz_pred_ = torch.matmul(z_pred_, self.D)
            Dz_norm_ = Dz_pred_.norm(dim=-1, keepdim=True) + self.eps
            proj_scale = (
                (Dz_pred_ * x_input).sum(dim=-1, keepdim=True) / Dz_norm_.pow(2)
            )
            # Numerical stability for real-LM scale: with dimin=2304+ and
            # lam=1/(4*dimin) the attention output Dz_pred_ is tiny in early
            # steps, which makes Dz_norm_.pow(2) ~ eps^2 and proj_scale
            # explode to ~1e6+. The actual projected update s*Dz_pred_ is
            # bounded (Cauchy-Schwarz), but the raw proj_scale multiplies
            # z_pred_ via the next line — unclamped, this blows gradients
            # and NaNs AdamW before warmup finishes. Toy-model dims were
            # small enough for this to never trigger.
            proj_scale = proj_scale.clamp(min=-10.0, max=10.0)
            z_pred = z_pred + (z_pred_ * proj_scale)
            x_input = x_input - proj_scale * Dz_pred_
            if return_graph:
                attn_graphs.append(attn_graphs_)

        if self.sae_diff_type == "relu":
            z_novel = F.relu(torch.matmul(x_input * self.lam, E))
        elif self.sae_diff_type == "topk":
            kval = self.kval_topk if inf_k is None else inf_k
            z_novel = F.relu(torch.matmul(x_input * self.lam, E))
            _, topk_indices = torch.topk(z_novel, kval, dim=-1)
            mask = torch.zeros_like(z_novel)
            mask.scatter_(-1, topk_indices, 1)
            z_novel = z_novel * mask
        else:
            z_novel = F.relu(torch.matmul(x_input * self.lam, E))

        x_recons = torch.matmul(z_novel + z_pred, self.D) + self.b

        with torch.no_grad():
            x_pred_recons = torch.matmul(z_pred, self.D)
            x_novel_recons = torch.matmul(z_novel, self.D)

        results_dict = {
            "novel_codes": z_novel,
            "novel_recons": x_novel_recons,
            "pred_codes": z_pred,
            "pred_recons": x_pred_recons,
            "attn_graphs": (
                torch.stack(attn_graphs, dim=1) if return_graph else None
            ),
        }
        return x_recons, results_dict
