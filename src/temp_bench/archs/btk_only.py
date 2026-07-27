"""btk-only variants — the ACTMIX composition fix (ReLU removed from the
sparsity path of the v2 BatchTopK fair-backbone family).

ACTMIX finding (briefings/actmix-shared.md): the v2 hunt backbone composes
ReLU -> BatchTopK. When the positive pre-act pool is thin, BatchTopK picks
exact zeros ("zero-picks") and realized l0 < nominal (sae@T1 realizes
~4.4/8; pre/stacked 5.9->7.9 and post 5.6->8.0 per window as T grows).
These variants remove the ReLU so selection runs over RAW pre-activations.

THE CANONICAL btk-only CONVENTION (single-source rule, actmix-shared.md:
runpod-1/2 and every other agent follow THIS file + the task_hunt LOG
convention note; never fork an independent convention):

1. Selection over raw pre-acts by SIGNED VALUE (largest values win — not
   magnitude): negative slots are selected only when the positive pool
   runs out; selected values pass through signed — no ReLU anywhere in
   the sparsity path. Realized l0 == nominal (ties at exactly 0.0 are
   measure-zero).
2. Threshold path (JumpReLU at eval): the gating expression is UNCHANGED
   (``post * (post > threshold)``) and the EMA rule is UNCHANGED (min
   surviving activation, same beta / warmup), with the EMA source set
   generalized from {survivors > 0} to {survivors != 0} — identical
   whenever no negative is selected. The relu-mix ``-1.0`` sentinel +
   ``>= 0`` validity check cannot represent a legitimately-negative
   threshold (it would silently fall back to batch-dependent TopK at
   eval), so these variants carry an explicit ``threshold_set`` buffer.
3. Fired/dead accounting: fired <=> z != 0 (a negative-firing feature is
   alive; relu-mix used ``> 0`` / ``sum > 0``).
4. AuxK revival path UNCHANGED: operates on ReLU'd pre-acts exactly as
   relu-mix. AuxK is outside the sparsity path (it never touches z or
   realized l0); holding it constant isolates the selection composition
   as the only moved variable.
5. Diagnostic: every train_step logs ``neg_frac`` = (# negative
   survivors) / (# nonzero survivors) of the selection.

Everything else — params, init, decoder unit-norm, grad-parallel removal,
matryoshka / contrastive losses (tsae), batch conventions — is inherited
from the relu-mix parents. Overridden methods are line-for-line copies of
the parents with every deviation tagged ``# btk-only:``; equivalence on
positive-rich inputs is contract-tested (tests/test_btk_only.py).

Registry (configs/archs.yaml): ``batchtopk_sae_btkonly``,
``tsae_btkonly``, ``stacked_batchtopk_btkonly``,
``txc_batchtopk_pre_btkonly``, ``txc_batchtopk_post_btkonly``. Arm label
``btk-only`` everywhere these run; the unsuffixed names ARE the
``relu-mix`` arm. ``relu_mode`` is threaded as an hparam (asserted below)
so every train_key / leaderboard row hashes the arm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.archs import telemetry
from temp_bench.archs.batchtopk_sae import BatchTopKSAE
from temp_bench.archs.stacked_batchtopk import StackedBatchTopK
from temp_bench.archs.tsae import TSAEPaper
from temp_bench.archs.txc_batchtopk import _TXCBatchTopKBase


def _check_relu_mode(relu_mode: str) -> None:
    if relu_mode != "btk-only":
        raise ValueError(
            f"btk_only variants are the btk-only arm; got relu_mode={relu_mode!r}. "
            "The relu-mix arm is the unsuffixed registry name."
        )


def _neg_frac(gated: torch.Tensor) -> torch.Tensor:
    """(# negative survivors) / (# nonzero survivors) — the logged diagnostic."""
    with torch.no_grad():
        nz = (gated != 0).sum().float().clamp(min=1.0)
        return ((gated < 0).sum().float() / nz).detach()


# ── temporal crosscoders (pre / post squash) ─────────────────────────────


class _TXCBatchTopKBTKBase(_TXCBatchTopKBase):
    """btk-only twin of ``_TXCBatchTopKBase``; leaf classes drop the ReLU
    in ``_compute_post``. ``encode`` / ``train_step`` are copies of the
    parent with the ``# btk-only:`` deviations only."""

    def __init__(self, *, relu_mode: str = "btk-only", **kw):
        _check_relu_mode(relu_mode)
        super().__init__(**kw)
        self.relu_mode = relu_mode
        # btk-only: explicit threshold validity flag (threshold may be < 0).
        self.register_buffer("threshold_set", torch.tensor(0, dtype=torch.uint8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) → (B, 1, d_sae) shared window code."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self._T:
            raise ValueError(
                f"{type(self).__name__}.encode expects (B, T={self._T}, d_in); "
                f"got T={x.shape[1]}."
            )
        post = self._compute_post(x)
        # btk-only: validity via the flag, not the >=0 sentinel check.
        if (not self.training) and bool(self.threshold_set.item()):
            gated = post * (post > self.threshold)
        else:
            gated = self._batchtopk(post)
        return self._to_shared(gated).unsqueeze(1)        # (B, 1, d_sae)

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"{type(self).__name__}.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, T, _d_in = x.shape

        post = self._compute_post(x)              # btk-only: RAW pre-acts
        gated = self._batchtopk(post)             # BatchTopK during training
        z_shared = self._to_shared(gated)         # (B, d_sae)

        # JumpReLU threshold EMA (min surviving activation).
        step = int(self.global_step.item())
        if step > self.threshold_start_step:
            with torch.no_grad():
                active = gated[gated != 0]        # btk-only: signed survivors
                cur = (
                    active.min().float()
                    if active.numel() > 0
                    else torch.tensor(0.0, device=gated.device)
                )
                if not bool(self.threshold_set.item()):   # btk-only: flag
                    self.threshold.copy_(cur)
                    self.threshold_set.fill_(1)
                else:
                    self.threshold.copy_(
                        self.threshold_beta * self.threshold
                        + (1 - self.threshold_beta) * cur
                    )

        # Reconstruction (shared code decodes all T positions).
        x_hat = torch.einsum("bs,std->btd", z_shared, self.W_dec) + self.b_dec
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        # Dead-feature tracking on the shared code.
        with torch.no_grad():
            active_feat = (z_shared != 0).any(dim=0)      # btk-only: != 0
            self.num_tokens_since_fired += B * T
            self.num_tokens_since_fired[active_feat] = 0
            dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
            n_dead = int(dead_mask.sum().item())
            if telemetry.due(int(self.global_step.item())):
                nz = gated[gated != 0]
                telemetry.maybe_log(
                    self, step=int(self.global_step.item()), n_dead=n_dead,
                    batch_l0=float(nz.numel()) / B,
                    boundary_min_pre=(float(nz.min().item())
                                      if nz.numel() else 0.0))

        # AuxK on dead features, in shared-code space.
        # btk-only: UNCHANGED — revival stays on ReLU'd pre-acts (conv § 4).
        if n_dead > 0:
            k_aux = min(self.aux_k, n_dead)
            pre_sq = F.relu(self._squashed_preact(x))
            auxk_pre = pre_sq.masked_fill(~dead_mask.unsqueeze(0), 0.0)
            vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
            aux_buf = torch.zeros_like(pre_sq).scatter_(-1, idx_a, vals_a)
            aux_decode = torch.einsum("bs,std->btd", aux_buf, self.W_dec)
            residual = (x - x_hat).detach()
            l2_a = (residual - aux_decode).pow(2).sum(dim=-1).mean()
            mu = residual.mean(dim=(0, 1), keepdim=True)
            denom = (residual - mu).pow(2).sum(dim=-1).mean()
            l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        else:
            l_auxk = torch.zeros((), device=x.device, dtype=x.dtype)

        loss = l_recon + self.auxk_alpha * l_auxk

        with torch.no_grad():
            self.global_step += 1
            l0 = (z_shared != 0).float().sum(dim=-1).mean()

        return {
            "loss": loss,
            "mse": l_recon.detach(),
            "l0": l0.detach(),
            "auxk": l_auxk.detach(),
            "dead": torch.tensor(float(n_dead)),
            "threshold": self.threshold.detach().clone(),
            "neg_frac": _neg_frac(gated),         # btk-only: diagnostic
        }


class TXCBatchTopKPreBTKOnly(_TXCBatchTopKBTKBase):
    """Pre-squash, btk-only: BatchTopK on RAW per-position pre-acts."""

    arch_version = "1.1.0"
    _registry_name = "txc_batchtopk_pre_btkonly"

    def _compute_post(self, x: torch.Tensor) -> torch.Tensor:
        # btk-only: no ReLU — raw per-position pre-activations.
        return torch.einsum("btd,tds->bts", x, self.W_enc) + self.b_enc

    def _to_shared(self, gated: torch.Tensor) -> torch.Tensor:
        return gated.sum(dim=1)                          # (B, d_sae)


class TXCBatchTopKPostBTKOnly(_TXCBatchTopKBTKBase):
    """Post-squash, btk-only: BatchTopK on the RAW squashed code."""

    arch_version = "1.1.0"
    _registry_name = "txc_batchtopk_post_btkonly"

    def _compute_post(self, x: torch.Tensor) -> torch.Tensor:
        # btk-only: no ReLU — raw squashed pre-activations.
        return torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc

    def _to_shared(self, gated: torch.Tensor) -> torch.Tensor:
        return gated                                     # already (B, d_sae)


# ── per-token BatchTopK SAE ──────────────────────────────────────────────


class BatchTopKSAEBTKOnly(BatchTopKSAE):
    """btk-only twin of ``batchtopk_sae``."""

    arch_version = "1.1.0"

    def __init__(self, *, relu_mode: str = "btk-only", **kw):
        _check_relu_mode(relu_mode)
        super().__init__(**kw)
        self.relu_mode = relu_mode
        self.config.name = "batchtopk_sae_btkonly"
        # btk-only: explicit threshold validity flag (threshold may be < 0).
        self.register_buffer("threshold_set", torch.tensor(0, dtype=torch.uint8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_t = x.dim() == 2
        if squeeze_t:
            x = x.unsqueeze(1)
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)
        pre = (x_flat - self.b_dec) @ self.W_enc + self.b_enc  # btk-only: no ReLU
        if (not self.training) and bool(self.threshold_set.item()):  # btk-only
            z_flat = pre * (pre > self.threshold)
        else:
            z_flat = self._batchtopk(pre)
        z = z_flat.reshape(B, T, self._d_sae)
        return z.squeeze(1) if squeeze_t else z

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(
                f"{type(self).__name__}.train_step expects (B, d_in); "
                f"got {tuple(x.shape)}."
            )
        B = x.shape[0]
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc   # btk-only: no ReLU
        z = self._batchtopk(pre)

        step = int(self.global_step.item())
        if step > self.threshold_start_step:
            with torch.no_grad():
                active = z[z != 0]                # btk-only: signed survivors
                cur = (
                    active.min().float()
                    if active.numel() > 0
                    else torch.tensor(0.0, device=z.device)
                )
                if not bool(self.threshold_set.item()):   # btk-only: flag
                    self.threshold.copy_(cur)
                    self.threshold_set.fill_(1)
                else:
                    self.threshold.copy_(
                        self.threshold_beta * self.threshold
                        + (1 - self.threshold_beta) * cur
                    )

        x_hat = z @ self.W_dec + self.b_dec
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        with torch.no_grad():
            did_fire = (z != 0).any(dim=0)        # btk-only: != 0
            self.num_tokens_since_fired += B
            self.num_tokens_since_fired[did_fire] = 0
            dead = self.num_tokens_since_fired >= self.dead_feature_threshold_tokens
            n_dead = int(dead.sum().item())

        # btk-only: AuxK UNCHANGED — revival stays on ReLU'd pre-acts (conv § 4).
        if n_dead > 0:
            post_relu = F.relu(pre)
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
            "neg_frac": _neg_frac(z),             # btk-only: diagnostic
        }


# ── stacked per-position BatchTopK SAEs ──────────────────────────────────


class StackedBatchTopKBTKOnly(StackedBatchTopK):
    """btk-only twin of ``stacked_batchtopk``."""

    arch_version = "1.1.0"

    def __init__(self, *, relu_mode: str = "btk-only", **kw):
        _check_relu_mode(relu_mode)
        super().__init__(**kw)
        self.relu_mode = relu_mode
        self.config.name = "stacked_batchtopk_btkonly"
        # btk-only: explicit threshold validity flag (threshold may be < 0).
        self.register_buffer("threshold_set", torch.tensor(0, dtype=torch.uint8))

    def _post(self, x: torch.Tensor) -> torch.Tensor:
        # btk-only: no ReLU — raw per-position pre-activations.
        return torch.einsum("btd,tds->bts", x, self.W_enc) + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) → (B, T, d_sae) per-position codes."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self._T:
            raise ValueError(
                f"{type(self).__name__}.encode expects (B, T={self._T}, d_in); "
                f"got T={x.shape[1]}."
            )
        post = self._post(x)
        # btk-only: validity via the flag, not the >=0 sentinel check.
        if (not self.training) and bool(self.threshold_set.item()):
            return post * (post > self.threshold)
        return self._batchtopk(post)

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"{type(self).__name__}.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, T, _d_in = x.shape

        post = self._post(x)                      # btk-only: RAW pre-acts
        z = self._batchtopk(post)

        step = int(self.global_step.item())
        if step > self.threshold_start_step:
            with torch.no_grad():
                active = z[z != 0]                # btk-only: signed survivors
                cur = (
                    active.min().float()
                    if active.numel() > 0
                    else torch.tensor(0.0, device=z.device)
                )
                if not bool(self.threshold_set.item()):   # btk-only: flag
                    self.threshold.copy_(cur)
                    self.threshold_set.fill_(1)
                else:
                    self.threshold.copy_(
                        self.threshold_beta * self.threshold
                        + (1 - self.threshold_beta) * cur
                    )

        x_hat = torch.einsum("bts,tsd->btd", z, self.W_dec) + self.b_dec
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        with torch.no_grad():
            did_fire = (z != 0).any(dim=0)        # btk-only: != 0 (T, d_sae)
            self.num_tokens_since_fired += B
            self.num_tokens_since_fired[did_fire] = 0
            dead = self.num_tokens_since_fired >= self.dead_threshold_tokens
            n_dead = int(dead.sum().item())
            if telemetry.due(int(self.global_step.item())):
                nz = z[z != 0]
                telemetry.maybe_log(
                    self, step=int(self.global_step.item()), n_dead=n_dead,
                    batch_l0=float(nz.numel()) / B,
                    boundary_min_pre=(float(nz.min().item())
                                      if nz.numel() else 0.0))

        # btk-only: AuxK UNCHANGED — revival stays on ReLU'd pre-acts (conv § 4).
        if n_dead > 0:
            k_aux = min(self.aux_k, n_dead, self._d_sae)
            auxk_pre = F.relu(post).masked_fill(~dead.unsqueeze(0), 0.0)
            vals, idx = auxk_pre.topk(k_aux, dim=-1, sorted=False)
            aux_buf = torch.zeros_like(post).scatter_(-1, idx, vals)
            aux_decode = torch.einsum("bts,tsd->btd", aux_buf, self.W_dec)
            residual = (x - x_hat).detach()
            l2 = (residual - aux_decode).pow(2).sum(dim=-1).mean()
            mu = residual.mean(dim=(0, 1), keepdim=True)
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
            "neg_frac": _neg_frac(z),             # btk-only: diagnostic
        }


# ── T-SAE (Ye et al. port) ───────────────────────────────────────────────


class TSAEBTKOnly(TSAEPaper):
    """btk-only twin of ``tsae`` (matryoshka + contrastive untouched)."""

    arch_version = "2.1.0-port"

    def __init__(self, *, relu_mode: str = "btk-only", **kw):
        _check_relu_mode(relu_mode)
        super().__init__(**kw)
        self.relu_mode = relu_mode
        self.config.name = "tsae_btkonly"
        # btk-only: explicit threshold validity flag (threshold may be < 0).
        self.register_buffer("threshold_set", torch.tensor(0, dtype=torch.uint8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_t = x.dim() == 2
        if squeeze_t:
            x = x.unsqueeze(1)
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)
        pre = (x_flat - self.b_dec) @ self.W_enc + self.b_enc  # btk-only: no ReLU

        if (not self.training) and bool(self.threshold_set.item()):  # btk-only
            z_flat = pre * (pre > self.threshold)
        else:
            flat = pre.flatten()
            k_total = int(self.k.item()) * (B * T)
            if k_total >= flat.numel():
                z_flat = pre
            else:
                tk = flat.topk(k_total, sorted=False)
                z_flat = (
                    torch.zeros_like(flat)
                    .scatter_(-1, tk.indices, tk.values)
                    .reshape(pre.shape)
                )
        z = z_flat.reshape(B, T, self._d_sae)
        return z.squeeze(1) if squeeze_t else z

    def _encode_per_token(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (z, relu'd pre-acts) — the 2nd feeds AuxK ONLY, which
        stays on ReLU'd pre-acts by convention § 4."""
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc   # btk-only: no ReLU
        flat = pre.flatten()
        k_total = int(self.k.item()) * x.shape[0]
        if k_total >= flat.numel():
            z = pre
        else:
            tk = flat.topk(k_total, sorted=False)
            z = (
                torch.zeros_like(flat)
                .scatter_(-1, tk.indices, tk.values)
                .reshape(pre.shape)
            )
        return z, F.relu(pre)                     # btk-only: AuxK input ReLU'd

    def train_step(self, x: torch.Tensor):
        if x.dim() != 3 or x.shape[1] < 2:
            raise ValueError(
                f"T-SAE train_step expects (B, seq_len>=2, d_in); got {tuple(x.shape)}."
            )
        B, T_seq, _ = x.shape
        t_offset = torch.randint(0, T_seq - 1, (1,)).item()
        x_anchor = x[:, t_offset, :]                     # (B, d_in)
        x_temp = x[:, t_offset + 1, :]                   # (B, d_in)

        f, post_relu = self._encode_per_token(x_anchor)
        f_temp, _ = self._encode_per_token(x_temp)

        step = int(self.global_step.item())
        if step > self.threshold_start_step:
            with torch.no_grad():
                active = f[f != 0]                # btk-only: signed survivors
                cur = (
                    active.min().float()
                    if active.numel() > 0
                    else torch.tensor(0.0, device=f.device)
                )
                if not bool(self.threshold_set.item()):   # btk-only: flag
                    self.threshold.copy_(cur)
                    self.threshold_set.fill_(1)
                else:
                    self.threshold.copy_(
                        self.threshold_beta * self.threshold
                        + (1 - self.threshold_beta) * cur
                    )

        # ── Matryoshka cumulative reconstruction (unchanged) ──
        W_chunks = torch.split(self.W_dec, list(self.group_sizes), dim=0)
        f_chunks = torch.split(f, list(self.group_sizes), dim=1)
        f_temp_chunks = torch.split(f_temp, list(self.group_sizes), dim=1)

        x_recon = self.b_dec.unsqueeze(0).expand_as(x_anchor).clone()

        W0, f0, f0_temp = W_chunks[0], f_chunks[0], f_temp_chunks[0]
        x_recon = x_recon + f0 @ W0
        l2_0 = ((x_anchor - x_recon).pow(2).sum(dim=-1) * self.group_weights[0]).mean()
        total_l2 = l2_0

        logits = f0 @ f0_temp.T                                # (B, B)
        labels = torch.arange(logits.shape[0], device=logits.device)
        temp_loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )

        for gi in range(1, self.active_groups):
            x_recon = x_recon + f_chunks[gi] @ W_chunks[gi]
            total_l2 = total_l2 + (
                (x_anchor - x_recon).pow(2).sum(dim=-1).mean()
                * self.group_weights[gi]
            )

        # ── AuxK on dead features (unchanged; input already ReLU'd) ──
        with torch.no_grad():
            did_fire = (f != 0).any(dim=0)        # btk-only: != 0
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
            dead = int(
                (self.num_tokens_since_fired
                 >= self.dead_feature_threshold_tokens).sum().item()
            )

        return total, {
            "mse": total_l2.detach(),
            "l0": l0.detach(),
            "auxk": auxk_loss.detach(),
            "temp": temp_loss.detach(),
            "dead": dead,
            "threshold": float(self.threshold.item()),
            "z": f.detach(),
            "neg_frac": _neg_frac(f),             # btk-only: diagnostic
        }
