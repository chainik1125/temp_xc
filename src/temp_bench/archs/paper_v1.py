"""Paper-v1 arch adapters for the ACTMIX `paper-match` arm (EVAL-ONLY).

Wraps the EXACT classes that trained the paper's shipped § 5.1 cells —
vendored verbatim from `origin/han-phase7-unification @ 94119bc08`
(`src/architectures/{topk_sae,txc_bare_antidead,tsae_paper}.py`), the
pin quoted by `experiments/explorations/task_hunt/COMPOSITION_AUDIT.md`
§ 3 (files stable through the training window; no arch classes existed
in purified at camera-ready, so these are the only implementations the
shipped checkpoints can load into). Only the dev-repo import line and
the unused Spec/trainer classes were dropped; compute code is
line-for-line (deviations would be tagged `# v2-adapter:`; there are
none in the math).

PER-ARM composition is preserved verbatim (mac-local's Phase-B ruling —
never collapse to one composition):

- `paper_topk_sae_v1`  — TopK→ReLU per token (selection on raw
  pre-acts; ReLU zeroes selected negatives — the paper-era mixing
  fingerprint on the per-token baseline).
- `paper_txc_base_v1`  — TopK→ReLU per window, k_win = k_pos·T
  (TXCBareAntidead, the paper's TXC-base).
- `paper_tsae_v1`      — ReLU→threshold at eval (BatchTopK only at
  train): `encode(..., use_threshold=True)` default, exactly the path
  the paper's probing pipeline called (`_encode_per_token(model.encode)`).

The adapters are EVAL-ONLY: ``train_step`` raises. Checkpoints are
staged by ``experiments/probing/actmix/phase_b.py`` (state-dict keys
re-prefixed ``inner.*``; provenance manifest maps v2 train_keys to the
shipped `temp-bench-models` train_keys).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch

# ── Vendored v1 classes (verbatim compute) ─────────────────────────────────


class _V1TopKSAE(nn.Module):
    """han-phase7-unification@94119bc08 src/architectures/topk_sae.py:TopKSAE."""

    def __init__(self, d_in: int, d_sae: int, k: int | None = None):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - self.b_dec
        pre = x_c @ self.W_enc.T + self.b_enc
        if self.k is not None:
            topk_vals, topk_idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(-1, topk_idx, F.relu(topk_vals))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_dec


class _V1TXCBareAntidead(nn.Module):
    """han-phase7-unification@94119bc08 src/architectures/txc_bare_antidead.py
    :TXCBareAntidead (encoder/decoder + buffers; training-only members kept
    for state-dict compatibility, loss code dropped)."""

    def __init__(self, d_in: int, d_sae: int, T: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, T, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        for t_ in range(T):
            nn.init.kaiming_uniform_(self.W_enc.data[t_])
        nn.init.kaiming_uniform_(self.W_dec.data.view(d_sae, T * d_in))

        # Shipped state carries EXACTLY one buffer (verified on
        # temp-bench-models cells): the class's other tracker buffers
        # were trainer-side in the run that saved these.
        self.register_buffer(
            "num_tokens_since_fired", torch.zeros(d_sae, dtype=torch.long))

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self._pre_activation(x)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec


class _V1TemporalMatryoshkaBatchTopKSAE(nn.Module):
    """han-phase7-unification@94119bc08 src/architectures/tsae_paper.py
    :TemporalMatryoshkaBatchTopKSAE (encode/decode + buffers)."""

    def __init__(self, activation_dim: int, dict_size: int,
                 k: int, group_sizes: list[int]):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        assert sum(group_sizes) == dict_size
        assert all(s > 0 for s in group_sizes)
        assert isinstance(k, int) and k > 0

        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        # Shipped state (verified): global_step + num_tokens_since_fired
        # ARE in the checkpoint; group_sizes is NOT (plain attr here so
        # strict load matches; the eval threshold path never reads groups).
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "num_tokens_since_fired", torch.zeros(dict_size, dtype=torch.long))
        self.group_sizes = torch.tensor(group_sizes)

        self.active_groups = len(group_sizes)
        self.group_indices = [0] + list(torch.cumsum(
            torch.tensor(group_sizes), dim=0
        ).tolist())

        self.W_enc = nn.Parameter(torch.empty(activation_dim, dict_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(dict_size, activation_dim))
        )
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))
        with torch.no_grad():
            eps = torch.finfo(self.W_dec.dtype).eps
            norm = torch.norm(self.W_dec.data.T, dim=0, keepdim=True)
            self.W_dec.data = (self.W_dec.data.T / (norm + eps)).T
            self.W_enc.data = self.W_dec.data.clone().T

    def encode(self, x: torch.Tensor, return_active: bool = False,
               use_threshold: bool = True):
        post_relu = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        if use_threshold:
            z = post_relu * (post_relu > self.threshold)
        else:
            flat = post_relu.flatten()
            topk = flat.topk(int(self.k.item()) * x.size(0), sorted=False)
            z = (torch.zeros_like(flat)
                 .scatter_(-1, topk.indices, topk.values)
                 .reshape(post_relu.shape))
        if return_active:
            return z, z.sum(dim=0) > 0, post_relu
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec


# ── v2 adapters (EVAL-ONLY) ────────────────────────────────────────────────


class _EvalOnlyMixin:
    """``src_tag`` on every adapter is PROVENANCE-ONLY: it has zero
    compute effect and exists so distinct shipped checkpoints that share
    (arch, seed, training_cfg) — e.g. the six bug-artifact "T10/T20"
    cells whose weights are T5-shaped — hash to distinct train_keys.
    """

    def train_step(self, x):
        raise NotImplementedError(
            f"{type(self).__name__} is an EVAL-ONLY paper-match adapter; "
            "its checkpoints are the paper's shipped cells staged by "
            "experiments/probing/actmix/phase_b.py — never retrain it."
        )


class PaperTopKSAEV1(_EvalOnlyMixin, TempBenchArch):
    """Paper § 5.1 TopK SAE (TopK→ReLU per token), eval-only adapter."""

    arch_version = "upstream-94119bc08"
    consumes = "token"

    def __init__(self, d_in: int, d_sae: int = 18432, k_pos: int = 20,
                 src_tag: str = ""):
        nn.Module.__init__(self)
        del src_tag  # provenance-only (train_key disambiguation)
        self.config = ArchConfig(
            name="paper_topk_sae_v1", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=1)
        self.inner = _V1TopKSAE(d_in=d_in, d_sae=d_sae, k=k_pos)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.inner.encode(x)
        B, S, d = x.shape
        return self.inner.encode(x.reshape(B * S, d)).reshape(B, S, -1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.inner.decode(z)


class PaperTSAEV1(_EvalOnlyMixin, TempBenchArch):
    """Paper § 5.1 T-SAE (ReLU→threshold at eval), eval-only adapter.

    ``encode`` uses the v1 default ``use_threshold=True`` — the exact
    path the paper's probing pipeline exercised.
    """

    arch_version = "upstream-94119bc08"
    consumes = "token"

    def __init__(self, d_in: int, d_sae: int = 16384, k_pos: int = 20,
                 h_frac: float = 0.2, src_tag: str = ""):
        nn.Module.__init__(self)
        del src_tag  # provenance-only (train_key disambiguation)
        self.config = ArchConfig(
            name="paper_tsae_v1", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=1)
        n_high = max(1, int(round(h_frac * d_sae)))
        self.inner = _V1TemporalMatryoshkaBatchTopKSAE(
            activation_dim=d_in, dict_size=d_sae, k=k_pos,
            group_sizes=[n_high, d_sae - n_high])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.inner.encode(x)
        B, S, d = x.shape
        return self.inner.encode(x.reshape(B * S, d)).reshape(B, S, -1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.inner.decode(z)


class PaperTXCBaseV1(_EvalOnlyMixin, TempBenchArch):
    """Paper § 5.1 TXC-base (TopK→ReLU per window, k_win = k_pos·T),
    eval-only adapter. ``encode`` returns ``(B, d_sae)`` per window —
    the probing evaluator's window path handles that shape natively.
    """

    arch_version = "upstream-94119bc08"
    consumes = "window"

    def __init__(self, d_in: int, d_sae: int = 18432, k_pos: int = 20,
                 T: int = 5, src_tag: str = ""):
        nn.Module.__init__(self)
        del src_tag  # provenance-only (train_key disambiguation)
        self.config = ArchConfig(
            name="paper_txc_base_v1", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T)
        self.inner = _V1TXCBareAntidead(
            d_in=d_in, d_sae=d_sae, T=T, k=k_pos * T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.encode(x)                      # (B, d_sae)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.inner.decode(z)
