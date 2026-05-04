"""Train the Bhalla et al. 2025 T-SAE on Andre's mid_res Gemma-2-2b-it
L13 cached residuals — same data + d_sae as the existing three arms,
but with the proper Bhalla architecture (BatchTopK + matryoshka +
temporal contrastive + AuxK + threshold inference).

Architecturally T=1 (per-token at inference). At training time the
contrastive loss is computed between consecutive tokens (anchor /
positive pair) drawn from the same window.

Run:  uv run python safety_research/scripts/train_tsae_paper.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

NLP_DIR = "/home/cs29824/andre/temp_xc/temporal_crosscoders/NLP"
SAFETY_DIR = "/home/cs29824/andre/temp_xc/safety_research"
sys.path.insert(0, NLP_DIR)
os.chdir(NLP_DIR)

from config import (  # type: ignore  # noqa: E402
    D_SAE, LEARNING_RATE, ADAM_BETAS, GRAD_CLIP, DEVICE,
    LAYER_SPECS, SEED,
)
from data import CachedActivationSource, WindowIterator  # type: ignore  # noqa: E402

import wandb

LAYER = "mid_res"
K_POS = 100              # average active features per token (BatchTopK budget)
STEPS = 3000             # match the existing arms
BATCH = 1024             # window batch (each window length T_window=5)
T_WINDOW = 5             # we train pairs (t, t+1) drawn from this window
H_FRAC = 0.20            # paper's matryoshka split
CONTRASTIVE_ALPHA = 1.0
AUXK_ALPHA = 1.0 / 32.0
THRESHOLD_START_STEP = 1000
THRESHOLD_BETA = 0.999
DEAD_TOKENS = 10_000_000

WANDB_PROJECT = "temporal-crosscoders-safety"

CKPT_DIR = Path(SAFETY_DIR) / "results" / "checkpoints"
LOG_DIR = Path(SAFETY_DIR) / "results" / "training_logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Decoder unit-norm helpers (cribbed from the wasteland port) ────────────


@torch.no_grad()
def _set_decoder_norm_to_unit(W_dec: torch.Tensor) -> torch.Tensor:
    """Normalise decoder columns (d_in, d_sae) to unit L2 norm."""
    eps = torch.finfo(W_dec.dtype).eps
    norm = torch.norm(W_dec.data, dim=0, keepdim=True)
    W_dec.data /= norm + eps
    return W_dec.data


def _remove_grad_parallel_to_decoder(W: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    n = W / (torch.norm(W, dim=0, keepdim=True) + 1e-6)
    par = torch.einsum("df,df->f", g, n)
    return g - par.unsqueeze(0) * n


# ─── Bhalla T-SAE ─────────────────────────────────────────────────────────────


class TSAEPaper(nn.Module):
    """Faithful Bhalla et al. 2025 T-SAE.

    Forward (per-token):
        post_relu = ReLU((x - b_dec) @ W_enc + b_enc)               (B, h)
        z         = BatchTopK_{k * B}(post_relu)                    (B, h)
        x_hat     = z @ W_dec + b_dec                               (B, d_in)

    `W_enc` shape (d_in, d_sae); `W_dec` shape (d_sae, d_in)
    (matches the wasteland convention).
    """

    def __init__(self, *, d_in: int, d_sae: int, k_pos: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.register_buffer("k", torch.tensor(k_pos, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.register_buffer("num_tokens_since_fired",
                             torch.zeros(d_sae, dtype=torch.long))
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        n_high = max(1, int(round(H_FRAC * d_sae)))
        n_low = d_sae - n_high
        self.group_sizes = (n_high, n_low)
        self.group_weights = (1.0, 1.0)
        self.top_k_aux = d_in // 2

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(d_sae, d_in)))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Init decoder unit norm + tie encoder to decoder transpose.
        self.W_dec.data = _set_decoder_norm_to_unit(self.W_dec.data.T).T.contiguous()
        self.W_enc.data = self.W_dec.data.clone().T.contiguous()

        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    @staticmethod
    def _project_dec_grad(p: torch.Tensor) -> None:
        if p.grad is None:
            return
        new_g = _remove_grad_parallel_to_decoder(p.data.T, p.grad.data.T).T
        p.grad.data.copy_(new_g)

    @torch.no_grad()
    def renormalise_decoder(self) -> None:
        _set_decoder_norm_to_unit(self.W_dec.data.T)

    def _encode_per_token(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """BatchTopK encode. x: (B, d_in) → (z, post_relu) both (B, d_sae)."""
        post_relu = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        flat = post_relu.flatten()
        k_total = int(self.k.item()) * x.shape[0]
        if k_total >= flat.numel():
            return post_relu, post_relu
        tk = flat.topk(k_total, sorted=False)
        z = (
            torch.zeros_like(flat)
            .scatter_(-1, tk.indices, tk.values)
            .reshape(post_relu.shape)
        )
        return z, post_relu

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Inference encode. x: (B, d_in) or (B, T, d_in) → (..., d_sae).

        Uses threshold gating once the threshold has been initialised.
        """
        squeeze_t = x.dim() == 2
        if squeeze_t:
            x = x.unsqueeze(1)
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)
        post_relu = F.relu((x_flat - self.b_dec) @ self.W_enc + self.b_enc)
        if (not self.training) and self.threshold.item() >= 0:
            z = post_relu * (post_relu > self.threshold)
        else:
            flat = post_relu.flatten()
            k_total = int(self.k.item()) * (B * T)
            if k_total >= flat.numel():
                z = post_relu
            else:
                tk = flat.topk(k_total, sorted=False)
                z = (torch.zeros_like(flat)
                     .scatter_(-1, tk.indices, tk.values)
                     .reshape(post_relu.shape))
        z = z.reshape(B, T, self.d_sae)
        return z.squeeze(1) if squeeze_t else z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_t = z.dim() == 2
        if squeeze_t:
            z = z.unsqueeze(1)
        x_hat = z @ self.W_dec + self.b_dec
        return x_hat.squeeze(1) if squeeze_t else x_hat

    def _auxiliary_loss(self, residual: torch.Tensor, post_relu: torch.Tensor) -> torch.Tensor:
        """Reanimate dead features against the residual after main reconstruction."""
        dead_mask = self.num_tokens_since_fired >= DEAD_TOKENS
        if dead_mask.sum() == 0:
            return torch.zeros((), device=residual.device)
        f_dead = post_relu * dead_mask.unsqueeze(0)
        kk = min(self.top_k_aux, int(dead_mask.sum().item()))
        if kk == 0:
            return torch.zeros((), device=residual.device)
        vals, idx = f_dead.topk(kk, dim=-1)
        f_aux = torch.zeros_like(f_dead).scatter_(-1, idx, vals)
        x_aux = f_aux @ self.W_dec
        return F.mse_loss(x_aux, residual)

    def train_step(self, x_window: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """x_window: (B, T_window, d_in). Sample (t, t+1) and compute the loss."""
        B, T_seq, _ = x_window.shape
        t_off = int(torch.randint(0, T_seq - 1, (1,)).item())
        x_anchor = x_window[:, t_off, :]
        x_temp = x_window[:, t_off + 1, :]

        f, post_relu = self._encode_per_token(x_anchor)
        f_temp, _ = self._encode_per_token(x_temp)

        # threshold EMA (post-warmup)
        step = int(self.global_step.item())
        if step > THRESHOLD_START_STEP:
            with torch.no_grad():
                active = f[f > 0]
                cur = (active.min().float() if active.numel() > 0
                       else torch.tensor(0.0, device=f.device))
                if self.threshold.item() < 0:
                    self.threshold.copy_(cur)
                else:
                    self.threshold.copy_(THRESHOLD_BETA * self.threshold
                                          + (1 - THRESHOLD_BETA) * cur)

        # Matryoshka cumulative reconstruction
        W_chunks = torch.split(self.W_dec, list(self.group_sizes), dim=0)
        f_chunks = torch.split(f, list(self.group_sizes), dim=1)
        f_temp_chunks = torch.split(f_temp, list(self.group_sizes), dim=1)

        x_recon = self.b_dec.unsqueeze(0).expand_as(x_anchor).clone()
        W0, f0, f0_temp = W_chunks[0], f_chunks[0], f_temp_chunks[0]
        x_recon = x_recon + f0 @ W0
        l2 = ((x_anchor - x_recon).pow(2).sum(dim=-1) * self.group_weights[0]).mean()

        # Temporal contrastive on the high-level group
        logits = f0 @ f0_temp.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        temp_loss = 0.5 * (F.cross_entropy(logits, labels)
                            + F.cross_entropy(logits.T, labels))

        for gi in range(1, len(self.group_sizes)):
            x_recon = x_recon + f_chunks[gi] @ W_chunks[gi]
            l2 = l2 + ((x_anchor - x_recon).pow(2).sum(dim=-1).mean()
                       * self.group_weights[gi])

        with torch.no_grad():
            did_fire = (f.sum(dim=0) > 0)
            self.num_tokens_since_fired += B
            self.num_tokens_since_fired[did_fire] = 0

        residual_for_aux = (x_anchor - x_recon).detach()
        aux = self._auxiliary_loss(residual_for_aux, post_relu)

        total = l2 + AUXK_ALPHA * aux + CONTRASTIVE_ALPHA * temp_loss

        with torch.no_grad():
            self.global_step += 1
            l0 = (f != 0).float().sum(dim=-1).mean()

        return total, {
            "mse": l2.detach(),
            "auxk": aux.detach(),
            "temp": temp_loss.detach(),
            "l0": l0.detach(),
            "threshold": float(self.threshold.item()),
        }


def main() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    spec = LAYER_SPECS[LAYER]
    d_in = spec["d_act"]
    print(f"Bhalla T-SAE: layer {LAYER}  d_in={d_in}  d_sae={D_SAE}  k_pos={K_POS}  steps={STEPS}")

    name = f"tsae_paper__{LAYER}__k{K_POS}__T1"
    run = wandb.init(
        project=WANDB_PROJECT,
        name=name,
        tags=["safety", "tsae_paper", "bhalla", LAYER],
        config=dict(arm="tsae_paper", arch="tsae_bhalla", T=1, k_pos=K_POS,
                    layer=LAYER, d_in=d_in, d_sae=D_SAE, steps=STEPS,
                    batch=BATCH, lr=LEARNING_RATE,
                    h_frac=H_FRAC, contrastive_alpha=CONTRASTIVE_ALPHA,
                    auxk_alpha=AUXK_ALPHA),
        reinit=True,
    )
    print(f"  wandb: {run.url}")

    torch.manual_seed(SEED)
    src = CachedActivationSource(LAYER)
    iterator = WindowIterator(src, BATCH, T=T_WINDOW)

    model = TSAEPaper(d_in=d_in, d_sae=D_SAE, k_pos=K_POS).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

    history: list[dict] = []
    pbar = tqdm(range(STEPS), desc=name, ncols=100)
    t0 = time.time()
    for step in pbar:
        x = next(iterator)                           # (B, T_window, d_in)
        loss, info = model.train_step(x)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()
        model.renormalise_decoder()

        if step % 100 == 0 or step == STEPS - 1:
            with torch.no_grad():
                # eval FVU on a fresh window's anchor token
                x_eval = next(iterator)
                anchor = x_eval[:, T_WINDOW // 2, :]
                z = model.encode(anchor)
                x_hat = model.decode(z)
                fvu = ((x_hat - anchor).pow(2).sum() / (anchor - anchor.mean(0)).pow(2).sum()).item()
            row = dict(step=step,
                        loss=float(loss.item()),
                        mse=float(info["mse"]),
                        auxk=float(info["auxk"]),
                        temp=float(info["temp"]),
                        l0=float(info["l0"]),
                        threshold=info["threshold"],
                        fvu=fvu)
            history.append(row)
            wandb.log(row, step=step)
            pbar.set_postfix(loss=f"{row['loss']:.3f}", fvu=f"{fvu:.3f}",
                              L0=int(row["l0"]),
                              temp=f"{row['temp']:.3f}")

    elapsed = time.time() - t0

    ckpt = CKPT_DIR / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(),
                "arch": "tsae_paper", "T": 1, "k_pos": K_POS,
                "d_in": d_in, "d_sae": D_SAE, "layer": LAYER,
                "h_frac": H_FRAC, "group_sizes": list(model.group_sizes),
                "threshold": float(model.threshold.item()),
                "config": {"contrastive_alpha": CONTRASTIVE_ALPHA,
                           "auxk_alpha": AUXK_ALPHA,
                           "threshold_start_step": THRESHOLD_START_STEP,
                           "threshold_beta": THRESHOLD_BETA,
                           "dead_tokens": DEAD_TOKENS}}, ckpt)
    log_path = LOG_DIR / f"{name}.json"
    with open(log_path, "w") as f:
        json.dump(dict(history=history, elapsed_s=elapsed,
                       wandb_url=run.url, ckpt=str(ckpt)), f, indent=1)
    print(f"  saved: {ckpt}\n  log:   {log_path}\n  time:  {elapsed:.0f}s")
    final = history[-1]
    wandb.summary.update({"final_fvu": final["fvu"],
                          "final_loss": final["loss"],
                          "final_l0": final["l0"],
                          "final_temp": final["temp"],
                          "elapsed_s": elapsed})
    run.finish()


if __name__ == "__main__":
    main()
