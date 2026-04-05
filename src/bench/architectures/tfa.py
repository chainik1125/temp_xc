"""TFA (Temporal Feature Analysis) — ArchSpec wrapper.

Wraps the TemporalSAE module with the ArchSpec interface. TFA processes
full sequences (B, T, d) and uses causal attention to decompose each
token into predictable (from context) and novel components.

Training adapted from Han's train_tfa.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bench.architectures.base import ArchSpec, EvalOutput
from src.bench.architectures._tfa_module import TemporalSAE


class TFASpec(ArchSpec):
    """ArchSpec for Temporal Feature Analysis (TFA).

    Args:
        use_pos_encoding: Whether to add sinusoidal positional encodings.
        n_heads: Number of attention heads.
        n_attn_layers: Number of attention layers.
        bottleneck_factor: Attention bottleneck factor.
    """

    data_format = "seq"

    def __init__(
        self,
        use_pos_encoding: bool = False,
        n_heads: int = 4,
        n_attn_layers: int = 1,
        bottleneck_factor: int = 1,
    ):
        self.use_pos_encoding = use_pos_encoding
        self.n_heads = n_heads
        self.n_attn_layers = n_attn_layers
        self.bottleneck_factor = bottleneck_factor
        self.name = "TFA-pos" if use_pos_encoding else "TFA"

    def create(self, d_in, d_sae, k, device):
        sae_diff_type = "topk" if k is not None else "relu"
        tfa = TemporalSAE(
            dimin=d_in,
            width=d_sae,
            n_heads=self.n_heads,
            sae_diff_type=sae_diff_type,
            kval_topk=k,
            tied_weights=True,
            n_attn_layers=self.n_attn_layers,
            bottleneck_factor=self.bottleneck_factor,
            use_pos_encoding=self.use_pos_encoding,
        )
        return tfa.to(device)

    def train(self, model, gen_fn, total_steps, batch_size, lr, device,
              log_every=500, grad_clip=1.0):
        # TFA uses AdamW with cosine LR schedule (from Han's train_tfa.py)
        min_lr = lr * 0.9
        warmup_steps = min(200, total_steps // 10)

        decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 1e-4},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
        )

        log = {"loss": [], "l0": []}
        model.train()

        for step in range(total_steps):
            # Cosine warmup schedule
            if step < warmup_steps:
                current_lr = lr * step / max(1, warmup_steps)
            else:
                decay_ratio = (step - warmup_steps) / max(
                    1, total_steps - warmup_steps
                )
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                current_lr = min_lr + coeff * (lr - min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # TFA expects full sequences: gen_fn(n_seq) -> (B, T, d)
            batch = gen_fn(batch_size).to(device)
            recons, intermediates = model(batch)

            batch_flat = batch.reshape(-1, batch.shape[-1])
            recons_flat = recons.reshape(-1, recons.shape[-1])
            n_tokens = batch_flat.shape[0]
            loss = F.mse_loss(recons_flat, batch_flat, reduction="sum") / n_tokens

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if step % log_every == 0 or step == total_steps - 1:
                with torch.no_grad():
                    novel_l0 = (
                        (intermediates["novel_codes"] > 0)
                        .float()
                        .sum(dim=-1)
                        .mean()
                        .item()
                    )
                log["loss"].append(loss.item())
                log["l0"].append(novel_l0)

        model.eval()
        return log

    def eval_forward(self, model, x):
        recons, inter = model(x)
        B, T, D = x.shape
        se = (recons - x).pow(2).sum().item()
        signal = x.pow(2).sum().item()
        novel_l0 = (inter["novel_codes"] > 0).float().sum(dim=-1).sum().item()
        pred_l0 = (inter["pred_codes"].abs() > 1e-8).float().sum(dim=-1).sum().item()
        return EvalOutput(
            sum_se=se,
            sum_signal=signal,
            sum_l0=novel_l0 + pred_l0,
            n_tokens=B * T,
        )

    def decoder_directions(self, model, pos=None):
        return model.D.data.T  # (dimin, width)
