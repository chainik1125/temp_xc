"""TXC-base-BTK — btk-only twin of the paper TXC (``txc_base``).

ACTMIX arm label: **btk-only** (BatchTopK with NO ReLU anywhere in the
sparsity path). This arch exists to answer the ACTMIX re-run gate
(briefings/actmix-shared.md): the paper TXC composes TopK selection with
a ReLU applied AFTER selection (txc_base.py: ``topk`` then
``F.relu(vals)``), so selected-negative slots are zeroed post-hoc — a
harm that grows with T and biases the paper's d(perf)/dT down. This twin
changes the sparsity path and NOTHING else.

Deltas vs ``txc_base`` (everything not listed is verbatim-identical —
same parameterisation, init, AuxK dead-revival stack, decoder unit-norm,
grad-parallel removal, ``consumes = "window"``):

1. Selection is **BatchTopK** (Bussmann et al.): flat top-``(B * k_win)``
   over the ``(B, d_sae)`` squashed pre-activations of the whole batch,
   instead of per-window ``topk(k_win)``.
2. Selection operates on **raw pre-activations** — no ReLU before OR
   after. Selected values pass through signed; when the positive pool is
   thin, negative values can be selected (logged as ``neg_frac``, the
   mixing fingerprint diagnostic required by the shared briefing).
3. Inference uses the family's **JumpReLU threshold**: gating
   expression unchanged (``z = pre * (pre > threshold)``); EMA rule
   unchanged (min surviving activation, beta=0.999, warmup=1000) with
   the source set generalized {survivors > 0} -> {survivors != 0}, and
   an explicit ``threshold_set`` flag replacing the ``-1.0`` sentinel
   (a legitimately-negative threshold is representable) — the CANONICAL
   btk-only convention of mac-a's Stage 1
   (src/temp_bench/archs/btk_only.py + task_hunt LOG note), applied to
   the paper arch.

Budget convention: **k_win = min(k_pos * T, d_sae) per window** — the
paper arch's own budget (including the toy-bench clip), NOT the
``txc_batchtopk_post`` correction to k_pos/window. Rationale: this twin
isolates the activation composition; changing the budget too would
confound the comparison. This follows mac-a's canonical Stage-1 convention (items 1-5) with
the *_btkonly registry-name pattern; the budget rule is the paper
arch's own (the five v2 twins mirror their parents' budgets the same
way).

AuxK note: the dead-feature revival loss keeps the family's
``F.relu(pre)`` — the aux path is a training-only auxiliary objective,
shared verbatim by both ``txc_base`` and the ``txc_batchtopk`` family,
and is not part of the sparsity path the btk-only arm is defined over.

Dead/L0 accounting uses ``z != 0`` (codes are signed here), where
``txc_base`` uses ``z > 0``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


class TXCBaseBTK(TempBenchArch):
    """btk-only twin of the paper TXC: BatchTopK on raw pre-acts, no ReLU."""

    arch_version: str = "1.1.0"
    consumes: str = "window"

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        T: int = 5,
        k_pos: int = 20,
        auxk_alpha: float = 1.0 / 32.0,
        dead_threshold_tokens: int = 10_000_000,
        aux_k: int | None = None,
        threshold_start_step: int = 1000,
        threshold_beta: float = 0.999,
        relu_mode: str = "btk-only",
        k_win: int | None = None,
    ):
        nn.Module.__init__(self)
        if relu_mode not in ("btk-only", "relu-mix", "perwin-raw"):
            raise ValueError(
                f"relu_mode must be 'btk-only', 'relu-mix' or 'perwin-raw'; "
                f"got {relu_mode!r}. The paper-match composition (TopK then "
                "ReLU, per-window) is the frozen arch txc_base."
            )
        self.relu_mode = relu_mode
        _name = {"btk-only": "txc_base_btkonly",
                 "relu-mix": "txc_base_relumix",
                 "perwin-raw": "txc_base_perwinraw"}[relu_mode]
        self.config = ArchConfig(
            name=_name, d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T,
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        self.k_pos = int(k_pos)
        # Paper-arch budget, incl. the toy-bench clip (txc_base.py:65).
        # Budget-scan knob: an explicit ``k_win`` hparam overrides the
        # k_pos*T rule (fixed window budget independent of T); it hashes
        # into train_key like any hparam.
        nominal = k_pos * T if k_win is None else int(k_win)
        self.k_win = min(nominal, d_sae)
        if self.k_win < nominal:
            import warnings
            warnings.warn(
                f"TXCBaseBTK: clipped k_win from {nominal} to {d_sae} "
                f"(d_sae={d_sae}, k_pos={k_pos}, T={T})",
                stacklevel=2,
            )
        self.auxk_alpha = auxk_alpha
        self.dead_threshold_tokens = dead_threshold_tokens
        self.aux_k = aux_k if aux_k is not None else min(512, d_in // 2)
        self.threshold_start_step = threshold_start_step
        self.threshold_beta = threshold_beta

        # Params (txc_base convention).
        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, T, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc.data[t])
        nn.init.kaiming_uniform_(self.W_dec.data.view(d_sae, T * d_in))
        with torch.no_grad():
            self._normalize_decoder()
            for t in range(T):
                self.W_enc.data[t] = self.W_dec.data[:, t, :].T

        self.register_buffer("threshold", torch.tensor(0.0, dtype=torch.float32))
        # btk-only: explicit flag — a -1.0 sentinel cannot represent a
        # legitimately-negative threshold (canonical convention item 2).
        self.register_buffer("threshold_set", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer(
            "num_tokens_since_fired", torch.zeros(d_sae, dtype=torch.long)
        )
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    # ── decoder-norm utilities (verbatim txc_base) ──

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    @staticmethod
    def _project_dec_grad(param: torch.Tensor) -> None:
        if param.grad is None:
            return
        d_sae = param.shape[0]
        W_flat = param.data.view(d_sae, -1)
        g_flat = param.grad.data.view(d_sae, -1)
        normed = W_flat / (W_flat.norm(dim=1, keepdim=True) + 1e-6)
        parallel = (g_flat * normed).sum(dim=1, keepdim=True)
        g_flat.sub_(parallel * normed)

    # ── sparsity path ──

    def _squashed_preact(self, x: torch.Tensor) -> torch.Tensor:
        """Raw squashed pre-activation (B, d_sae). No ReLU."""
        return torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc

    def _selection_pool(self, x: torch.Tensor) -> torch.Tensor:
        """What BatchTopK selects over. btk-only: raw pre-acts (canonical
        item 1). relu-mix control: ReLU'd pre-acts (the v2-family
        composition applied to the paper arch's k_win budget; zero-picks
        possible when the positive pool is thin — that IS the arm)."""
        pre = self._squashed_preact(x)
        return F.relu(pre) if self.relu_mode == "relu-mix" else pre

    def _batchtopk(self, pre: torch.Tensor) -> torch.Tensor:
        """Selection at budget k_win per window.

        btk-only / relu-mix: flat BatchTopK over the batch pool.
        perwin-raw: per-window topk (the paper arch's own selection
        scope) over RAW pre-acts — txc_base with the F.relu deleted.
        Selected values pass through UNCHANGED — signed codes allowed.
        """
        if self.relu_mode == "perwin-raw":
            vals, idx = pre.topk(self.k_win, dim=-1)
            return torch.zeros_like(pre).scatter_(1, idx, vals)
        B = pre.shape[0]
        k_total = self.k_win * B
        flat = pre.reshape(-1)
        if k_total >= flat.numel():
            return pre
        tk = flat.topk(k_total, sorted=False)
        return (
            torch.zeros_like(flat)
            .scatter_(-1, tk.indices, tk.values)
            .reshape(pre.shape)
        )

    # ── encode / decode ──

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) → (B, 1, d_sae) shared window code.

        BatchTopK while training (or before the threshold is tracked);
        JumpReLU threshold at inference.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self._T:
            raise ValueError(
                f"TXCBaseBTK.encode expects (B, T={self._T}, d_in); "
                f"got T={x.shape[1]}."
            )
        pre = self._selection_pool(x)
        if (not self.training) and bool(self.threshold_set.item()):
            z = pre * (pre > self.threshold)
        else:
            z = self._batchtopk(pre)
        return z.unsqueeze(1)               # (B, 1, d_sae)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 1, d_sae) or (B, d_sae) → (B, T, d_in)."""
        if z.dim() == 3:
            if z.shape[1] != 1:
                raise ValueError(
                    f"TXCBaseBTK.decode expects (B, 1, d_sae); got T={z.shape[1]}."
                )
            z = z.squeeze(1)
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    # ── train_step ──

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"TXCBaseBTK.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, T, _d_in = x.shape

        pre = self._selection_pool(x)
        z = self._batchtopk(pre)

        # JumpReLU threshold EMA over the min POSITIVE surviving
        # activation (txc_batchtopk convention, verbatim).
        step = int(self.global_step.item())
        if step > self.threshold_start_step:
            with torch.no_grad():
                active = z[z != 0]     # btk-only: source set {!=0}, not {>0}
                cur = (
                    active.min().float()
                    if active.numel() > 0
                    else torch.tensor(0.0, device=z.device)
                )
                if not bool(self.threshold_set.item()):
                    self.threshold.copy_(cur)
                    self.threshold_set.fill_(1)
                else:
                    self.threshold.copy_(
                        self.threshold_beta * self.threshold
                        + (1 - self.threshold_beta) * cur
                    )

        x_hat = torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        # Dead tracker (z != 0: codes are signed here).
        with torch.no_grad():
            active_feat = (z != 0).any(dim=0)
            self.num_tokens_since_fired += B * T
            self.num_tokens_since_fired[active_feat] = 0
            dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
            n_dead = int(dead_mask.sum().item())

        # AuxK on dead features (family-verbatim: ReLU'd aux path).
        if n_dead > 0:
            k_aux = min(self.aux_k, n_dead)
            auxk_pre = F.relu(pre).masked_fill(~dead_mask.unsqueeze(0), 0.0)
            vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
            aux_buf = torch.zeros_like(pre).scatter_(-1, idx_a, vals_a)
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
            nnz = (z != 0)
            l0 = nnz.float().sum(dim=-1).mean()
            n_sel = nnz.sum()
            neg_frac = (
                (z < 0).sum().float() / n_sel.clamp(min=1).float()
            )

        return {
            "loss": loss,
            "mse": l_recon.detach(),
            "l0": l0.detach(),
            "auxk": l_auxk.detach(),
            "dead": torch.tensor(float(n_dead)),
            "threshold": self.threshold.detach().clone(),
            "neg_frac": neg_frac.detach(),
        }

    # ── hooks / introspection ──

    def post_step(self) -> None:
        with torch.no_grad():
            self._normalize_decoder()

    def decoder_directions(self) -> torch.Tensor:
        """(d_sae, d_in) — average decoder over T positions."""
        return self.W_dec.data.mean(dim=1).clone()
