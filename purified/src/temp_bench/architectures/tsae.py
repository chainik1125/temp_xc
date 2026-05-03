"""T-SAE — faithful port of Ye et al. 2025 (arxiv 2511.05541).

Source: ``origin/han-phase7-unification @ 94119bc0:src/architectures/tsae_paper.py``
which itself ports https://github.com/AI4LIFE-GROUP/temporal-saes
(``dictionary_learning/dictionary_learning/trainers/temporal_sequence_top_k.py``).

Adapted for the unified ``TempBenchArch`` framework (commit 3b70563f):
- Module class subclasses ``TempBenchArch``
- Wasteland's ``TemporalMatryoshkaBatchTopKTrainerLite.loss(x_pair, step)``
  is folded into ``train_step(x)`` — the canonical trainer does not branch
  on arch type, so per-arch loss (matryoshka + AuxK + contrastive) lives
  here.
- Decoder unit-norm projection + grad-parallel removal:
  - Renormalisation runs in ``post_step()`` (after optimizer.step())
  - Grad-parallel removal runs via a ``register_post_accumulate_grad_hook``
    on ``W_dec`` (after backward, before step) — keeps the module
    self-contained without needing a pre-step trainer hook.

Pair generation: wasteland T-SAE expects ``x_pair: (B, 2, d_in)``.
Framework batch_iter returns ``(B, seq_len, d_in)`` activations from
the residual cache. ``train_step`` samples a single random offset per
batch and uses tokens ``(t, t+1)`` as the contrastive pair (matches
the wasteland loader's "consecutive token pair" convention).

Threshold-based inference (paper App C): an EMA-tracked threshold
replaces BatchTopK at inference time, so per-token sparsity is
variable. We update the threshold inside ``train_step`` once
``step > threshold_start_step`` and switch to threshold mode
in ``encode`` when ``self.training`` is False.

DO NOT use ``tsae_ours.py`` from the wasteland — that's a deprecated
crude approximation (see ``configs/locked_archs.yaml`` notes).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.architectures.base import ArchConfig, TempBenchArch


# ── Decoder-norm utilities (cribbed from wasteland) ─────────────────────────


@torch.no_grad()
def _set_decoder_norm_to_unit(W_dec_DF: torch.Tensor) -> torch.Tensor:
    """Normalise decoder columns (shape (d_in, d_sae)) to unit L2 norm."""
    eps = torch.finfo(W_dec_DF.dtype).eps
    norm = torch.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


def _remove_grad_parallel_to_decoder(
    W_dec_DF: torch.Tensor, grad_DF: torch.Tensor
) -> torch.Tensor:
    """Project out the component of ``grad`` parallel to each decoder column.

    Operates on the (d_in, d_sae) view of the decoder. Without this the
    unit-norm constraint is violated between steps and the renormalisation
    shrinks the update.
    """
    normed_W = W_dec_DF / (torch.norm(W_dec_DF, dim=0, keepdim=True) + 1e-6)
    # parallel[f] = sum_d grad[d,f] * normed_W[d,f]
    parallel = torch.einsum("df,df->f", grad_DF, normed_W)
    # subtract per-column projection: grad[d,f] -= parallel[f] * normed_W[d,f]
    grad_DF = grad_DF - parallel.unsqueeze(0) * normed_W
    return grad_DF


# ── Architecture ─────────────────────────────────────────────────────────────


class TSAEPaper(TempBenchArch):
    """Faithful Ye et al. 2025 T-SAE.

    Args (from ``configs/locked_archs.yaml::tsae_paper``):
        d_in:          residual width.
        d_sae:         dictionary width.
        k_pos:         BatchTopK budget (avg active features per token).
        h_frac:        high-level matryoshka split (paper: 0.20).
        contrastive_alpha:  weight on temporal contrastive loss (paper: 1/10).

        auxk_alpha:    AuxK loss weight (paper: 1/32).
        threshold_start_step: when to start tracking the inference threshold.
        threshold_beta:       EMA beta for threshold update.
        dead_feature_threshold_tokens: tokens-since-fired before AuxK targets.
    """

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        k_pos: int,
        h_frac: float = 0.20,
        contrastive_alpha: float = 1.0,
        auxk_alpha: float = 1.0 / 32.0,
        threshold_start_step: int = 1000,
        threshold_beta: float = 0.999,
        dead_feature_threshold_tokens: int = 10_000_000,
    ):
        nn.Module.__init__(self)
        # T=1: T-SAE's encode is per-token at inference. The contrastive pair
        # (B, 2, d_in) is a TRAINING-time construct, handled inside
        # ``train_step`` via a single random offset over the seq_len axis.
        self.config = ArchConfig(
            name="tsae_paper", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=1
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self.h_frac = h_frac

        # Matryoshka groups: high (h_frac) then low (1 - h_frac).
        n_high = max(1, int(round(h_frac * d_sae)))
        n_low = d_sae - n_high
        self.group_sizes = (n_high, n_low)
        self.group_weights = (1.0, 1.0)  # paper sums groups equally
        self.active_groups = 2

        # k_pos here is "average active features per token" (BatchTopK budget).
        self.register_buffer("k", torch.tensor(k_pos, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(d_sae, dtype=torch.long),
        )
        self.register_buffer(
            "global_step", torch.tensor(0, dtype=torch.long)
        )

        # Hyperparams that drive train_step (not hparams of the model
        # itself, but per-arch training behaviour).
        self.contrastive_alpha = contrastive_alpha
        self.auxk_alpha = auxk_alpha
        self.threshold_start_step = threshold_start_step
        self.threshold_beta = threshold_beta
        self.dead_feature_threshold_tokens = dead_feature_threshold_tokens
        self.top_k_aux = d_in // 2

        # Parameters (note: wasteland W_dec is (d_sae, d_in); we keep that
        # convention for direct port-ability of the matryoshka split logic).
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(d_sae, d_in))
        )
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Init decoder to unit norm + tie encoder = decoder transpose.
        self.W_dec.data = _set_decoder_norm_to_unit(self.W_dec.data.T).T
        self.W_enc.data = self.W_dec.data.clone().T

        # Hook: project out grad-parallel-to-decoder before optimizer step.
        # Avoids needing a pre-step trainer hook.
        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    # ── grad hook ──

    @staticmethod
    def _project_dec_grad(param: torch.Tensor) -> None:
        if param.grad is None:
            return
        new_grad = _remove_grad_parallel_to_decoder(param.data.T, param.grad.data.T).T
        param.grad.data.copy_(new_grad)

    # ── encode / decode (TempBenchArch contract) ──

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, T, d_sae) (or (B, d_sae) if input is (B, d_in)).

        Internally flattens to per-token, applies BatchTopK during training
        and threshold during eval.
        """
        squeeze_t = x.dim() == 2
        if squeeze_t:
            x = x.unsqueeze(1)
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)
        post_relu = F.relu((x_flat - self.b_dec) @ self.W_enc + self.b_enc)

        if (not self.training) and self.threshold.item() >= 0:
            z_flat = post_relu * (post_relu > self.threshold)
        else:
            # Flat BatchTopK over (B * T * d_sae) — paper uses k * (B*T) as the
            # global budget so per-token average stays at k.
            flat = post_relu.flatten()
            k_total = int(self.k.item()) * (B * T)
            if k_total >= flat.numel():
                z_flat = post_relu
            else:
                tk = flat.topk(k_total, sorted=False)
                z_flat = (
                    torch.zeros_like(flat)
                    .scatter_(-1, tk.indices, tk.values)
                    .reshape(post_relu.shape)
                )
        z = z_flat.reshape(B, T, self._d_sae)
        return z.squeeze(1) if squeeze_t else z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_t = z.dim() == 2
        if squeeze_t:
            z = z.unsqueeze(1)
        x_hat = z @ self.W_dec + self.b_dec
        return x_hat.squeeze(1) if squeeze_t else x_hat

    # ── per-arch training step (matryoshka + AuxK + temporal contrastive) ──

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """T-SAE's full loss.

        Args:
            x: (B, seq_len, d_in) — full sequences from the batch_iter.
               We sample a random offset ``t`` in [0, seq_len-2] and use
               (x[:, t], x[:, t+1]) as the (anchor, temporal-pair).

        Returns:
            (loss, info) — info has 'mse', 'l0', 'auxk', 'temp', 'dead',
            'threshold', 'z' (detached latents on the anchor token).
        """
        if x.dim() != 3 or x.shape[1] < 2:
            raise ValueError(
                f"T-SAE train_step expects (B, seq_len>=2, d_in); got {tuple(x.shape)}."
            )
        B, T_seq, _ = x.shape
        # Random consecutive-token pair offset; same offset across the batch
        # (matches the wasteland pair generator's conv).
        t_offset = torch.randint(0, T_seq - 1, (1,)).item()
        x_anchor = x[:, t_offset, :]                     # (B, d_in)
        x_temp = x[:, t_offset + 1, :]                   # (B, d_in)

        # Encode both — use_threshold=False during training (BatchTopK).
        f, post_relu = self._encode_per_token(x_anchor)
        f_temp, _ = self._encode_per_token(x_temp)

        # Threshold EMA update (after warmup).
        step = int(self.global_step.item())
        if step > self.threshold_start_step:
            with torch.no_grad():
                active = f[f > 0]
                cur = (
                    active.min().float()
                    if active.numel() > 0
                    else torch.tensor(0.0, device=f.device)
                )
                if self.threshold.item() < 0:
                    self.threshold.copy_(cur)
                else:
                    self.threshold.copy_(
                        self.threshold_beta * self.threshold
                        + (1 - self.threshold_beta) * cur
                    )

        # ── Matryoshka cumulative reconstruction ──
        # W_dec is (d_sae, d_in); split rows into (high, low).
        W_chunks = torch.split(self.W_dec, list(self.group_sizes), dim=0)
        f_chunks = torch.split(f, list(self.group_sizes), dim=1)
        f_temp_chunks = torch.split(f_temp, list(self.group_sizes), dim=1)

        x_recon = self.b_dec.unsqueeze(0).expand_as(x_anchor).clone()

        # Group 0: high-level — gets reconstruction + contrastive.
        W0, f0, f0_temp = W_chunks[0], f_chunks[0], f_temp_chunks[0]
        x_recon = x_recon + f0 @ W0
        l2_0 = ((x_anchor - x_recon).pow(2).sum(dim=-1) * self.group_weights[0]).mean()
        total_l2 = l2_0

        # Temporal contrastive (raw dot InfoNCE between high-level latents).
        logits = f0 @ f0_temp.T                                # (B, B)
        labels = torch.arange(logits.shape[0], device=logits.device)
        temp_loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )

        # Subsequent groups: cumulative matryoshka reconstruction.
        for gi in range(1, self.active_groups):
            x_recon = x_recon + f_chunks[gi] @ W_chunks[gi]
            total_l2 = total_l2 + (
                (x_anchor - x_recon).pow(2).sum(dim=-1).mean()
                * self.group_weights[gi]
            )

        # ── AuxK on dead features ──
        with torch.no_grad():
            did_fire = (f.sum(dim=0) > 0)
            self.num_tokens_since_fired += B
            self.num_tokens_since_fired[did_fire] = 0
        residual_for_auxk = (x_anchor - x_recon).detach()
        auxk_loss = self._auxiliary_loss(residual_for_auxk, post_relu)

        total = (
            total_l2
            + self.auxk_alpha * auxk_loss
            + self.contrastive_alpha * temp_loss
        )

        with torch.no_grad():
            self.global_step += 1
            l0 = (f != 0).float().sum(dim=-1).mean()
            dead = int((self.num_tokens_since_fired >= self.dead_feature_threshold_tokens).sum().item())

        return total, {
            "mse": total_l2.detach(),
            "l0": l0.detach(),
            "auxk": auxk_loss.detach(),
            "temp": temp_loss.detach(),
            "dead": dead,
            "threshold": float(self.threshold.item()),
            "z": f.detach(),
        }

    def _encode_per_token(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token BatchTopK encode used inside train_step.

        Returns (z, post_relu) — post_relu is needed for AuxK on dead features.
        """
        post_relu = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)  # (B, d_sae)
        flat = post_relu.flatten()
        k_total = int(self.k.item()) * x.shape[0]
        if k_total >= flat.numel():
            z = post_relu
        else:
            tk = flat.topk(k_total, sorted=False)
            z = (
                torch.zeros_like(flat)
                .scatter_(-1, tk.indices, tk.values)
                .reshape(post_relu.shape)
            )
        return z, post_relu

    def _auxiliary_loss(
        self, residual: torch.Tensor, post_relu: torch.Tensor
    ) -> torch.Tensor:
        """OpenAI-style AuxK: reconstruct residual using top-k_aux dead features."""
        dead = self.num_tokens_since_fired >= self.dead_feature_threshold_tokens
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return torch.tensor(0.0, dtype=residual.dtype, device=residual.device)
        k_aux = min(self.top_k_aux, n_dead)
        neg_inf = torch.tensor(float("-inf"), device=residual.device)
        masked = torch.where(dead.unsqueeze(0), post_relu, neg_inf)
        vals, idx = masked.topk(k_aux, sorted=False)
        buf = torch.zeros_like(post_relu).scatter_(-1, idx, vals)
        x_aux = buf @ self.W_dec
        l2 = (residual.float() - x_aux.float()).pow(2).sum(dim=-1).mean()
        # Normalise by residual variance (OpenAI convention).
        mu = residual.mean(dim=0, keepdim=True)
        denom = (residual.float() - mu.float()).pow(2).sum(dim=-1).mean()
        return (l2 / denom).nan_to_num(0.0)

    # ── post_step (decoder unit-norm renormalisation) ──

    def post_step(self) -> None:
        with torch.no_grad():
            self.W_dec.data = _set_decoder_norm_to_unit(self.W_dec.data.T).T

    # ── decoder_directions for C4 ──

    def decoder_directions(self) -> torch.Tensor:
        # W_dec is (d_sae, d_in) — that IS the convention TempBenchArch wants.
        return self.W_dec.data.clone()
