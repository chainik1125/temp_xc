"""SAE architectures: TopK, Matryoshka, Hierarchical, Temporal Crosscoder, and
Multi-Layer Crosscoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """Standard TopK sparse autoencoder.

    z = TopK(ReLU(W_enc @ (x - b_dec) + b_enc), k)            (use_relu=True, default)
    z = TopK(W_enc @ (x - b_dec) + b_enc, k)                  (use_relu=False)
    x_hat = W_dec @ z + b_dec

    ``use_relu=False`` matches the Bussmann et al. 2024 BatchTopK formulation
    (no ReLU before TopK; latents may be negative). ``use_relu=True`` matches
    Marks/Karvonen/Mueller `dictionary_learning` and our default.
    """

    def __init__(self, d_in: int, d_sae: int, k: int, use_relu: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.use_relu = use_relu

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        if self.use_relu:
            pre = torch.relu(pre)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_hat, z)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()


class MatryoshkaSAE(nn.Module):
    """Matryoshka SAE: batch_topk activation + nested reconstruction losses.

    Same encoder/decoder as TopK, but:
    - Uses batch_topk: keeps top-(k * batch_size) activations across the batch
      rather than top-k per sample. Targets L0 ~ k on average.
    - Training loss adds reconstruction at each prefix width w in
      matryoshka_widths, using only the first w latents. This pressures early
      features to capture the most important structure.

    The hypothesis: with hierarchy pressure, the first few features should
    learn component-level (omega) information, while later features capture
    within-component (eta) detail.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        matryoshka_widths: list[int] | None = None,
        inner_weight: float = 1.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.inner_weight = inner_weight

        if matryoshka_widths is None:
            # Default: powers of 2 up to d_sae
            widths = []
            w = 4
            while w < d_sae:
                widths.append(w)
                w *= 2
            widths.append(d_sae)
            self.matryoshka_widths = widths
        else:
            self.matryoshka_widths = sorted(matryoshka_widths)
            if self.matryoshka_widths[-1] != d_sae:
                self.matryoshka_widths.append(d_sae)

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def _batch_topk(self, pre: torch.Tensor) -> torch.Tensor:
        """Keep top-(k * batch_size) activations across the batch."""
        batch_size = pre.shape[0]
        total_k = self.k * batch_size
        flat = pre.reshape(-1)
        actual_k = min(total_k, flat.numel())
        topk_vals, topk_idx = flat.topk(actual_k)
        result = torch.zeros_like(flat)
        result.scatter_(0, topk_idx, topk_vals)
        return result.reshape_as(pre)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._batch_topk(self._pre_activation(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_hat, z)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute matryoshka loss: weighted combination of nested recon losses.

        Returns (total_loss, info_dict).
        """
        pre = self._pre_activation(x)
        z = self._batch_topk(pre)
        x_hat = z @ self.W_dec + self.b_dec
        full_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        # Nested reconstruction losses at inner widths
        inner_losses = []
        per_width = {self.matryoshka_widths[-1]: full_recon.item()}
        for w in self.matryoshka_widths[:-1]:
            z_inner = z[:, :w]
            x_hat_inner = z_inner @ self.W_dec[:w] + self.b_dec
            inner_recon = (x - x_hat_inner).pow(2).sum(dim=-1).mean()
            inner_losses.append(inner_recon)
            per_width[w] = inner_recon.item()

        # Weighted combination
        if inner_losses:
            inner_mean = sum(inner_losses) / len(inner_losses)
            total = (full_recon + self.inner_weight * inner_mean) / (
                1.0 + self.inner_weight
            )
        else:
            total = full_recon

        info = {
            "total_loss": total.item(),
            "full_recon": full_recon.item(),
            "per_width": per_width,
        }
        return total, info

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()


class HierarchicalSAE(nn.Module):
    """Two-layer hierarchical SAE inspired by Cagnetta's Random Hierarchy Model.

    Architecture:
        Layer 1 (parents): K sparse features, k_parent active per input.
            These should learn component identity (omega).
        Layer 2 (leaves): d_leaf sparse features, k_leaf active per input.
            Each leaf is assigned to exactly one parent.
            These capture within-component detail (eta).

    Only the leaf layer has a decoder. Parents are "virtual" — defined as the
    sum of their children's activations, enforced by a hierarchy constraint.

    Hierarchy constraint (from HSAE / "Atoms to Trees"):
        z_parent_j ~ sum_{i in children(j)} z_leaf_i
    """

    def __init__(
        self,
        d_in: int,
        n_parents: int,
        d_leaf: int,
        k_parent: int = 1,
        k_leaf: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_parents = n_parents
        self.d_leaf = d_leaf
        self.k_parent = k_parent
        self.k_leaf = k_leaf
        self.alpha = alpha

        assert d_leaf % n_parents == 0, f"d_leaf={d_leaf} must divide by n_parents={n_parents}"
        self.leaves_per_parent = d_leaf // n_parents

        # Parent encoder
        self.W_enc_parent = nn.Parameter(torch.empty(d_in, n_parents))
        self.b_enc_parent = nn.Parameter(torch.zeros(n_parents))

        # Leaf encoder + decoder
        self.W_enc_leaf = nn.Parameter(torch.empty(d_in, d_leaf))
        self.b_enc_leaf = nn.Parameter(torch.zeros(d_leaf))
        self.W_dec_leaf = nn.Parameter(torch.empty(d_leaf, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_enc_parent, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_enc_leaf, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec_leaf.copy_(self.W_enc_leaf.T)
            self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        norms = self.W_dec_leaf.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec_leaf.data.div_(norms)

    def _topk(self, pre: torch.Tensor, k: int) -> torch.Tensor:
        topk_vals, topk_idx = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def encode_parents(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu((x - self.b_dec) @ self.W_enc_parent + self.b_enc_parent)
        return self._topk(pre, self.k_parent)

    def encode_leaves(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu((x - self.b_dec) @ self.W_enc_leaf + self.b_enc_leaf)
        return self._topk(pre, self.k_leaf)

    def decode(self, z_leaf: torch.Tensor) -> torch.Tensor:
        return z_leaf @ self.W_dec_leaf + self.b_dec

    def parent_from_children(self, z_leaf: torch.Tensor) -> torch.Tensor:
        """Parent activations = sum of children's leaf activations."""
        B = z_leaf.shape[0]
        grouped = z_leaf.view(B, self.n_parents, self.leaves_per_parent)
        return grouped.sum(dim=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_hat, z_parent, z_leaf)."""
        z_parent = self.encode_parents(x)
        z_leaf = self.encode_leaves(x)
        x_hat = self.decode(z_leaf)
        return x_hat, z_parent, z_leaf

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Recon + alpha * hierarchy constraint."""
        x_hat, z_parent, z_leaf = self.forward(x)
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        parent_from_children = self.parent_from_children(z_leaf)
        hierarchy_loss = (z_parent - parent_from_children).pow(2).sum(dim=-1).mean()
        total = recon_loss + self.alpha * hierarchy_loss
        return total, {
            "total_loss": total.item(),
            "recon_loss": recon_loss.item(),
            "hierarchy_loss": hierarchy_loss.item(),
        }

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()

    @torch.no_grad()
    def reassign_children(self) -> None:
        """Reassign leaves to parents by encoder cosine similarity (HSAE-style)."""
        leaf_dirs = nn.functional.normalize(self.W_enc_leaf, dim=0)
        parent_dirs = nn.functional.normalize(self.W_enc_parent, dim=0)
        sim = parent_dirs.T @ leaf_dirs  # (n_parents, d_leaf)

        # Greedy balanced assignment: each parent gets leaves_per_parent leaves
        assigned = set()
        final_order = []
        for p in range(self.n_parents):
            scores = sim[p].clone()
            scores[list(assigned)] = -float("inf") if assigned else scores[list(assigned)]
            children = []
            for _ in range(self.leaves_per_parent):
                # Mask already-assigned
                for idx in assigned:
                    scores[idx] = -float("inf")
                best = scores.argmax().item()
                children.append(best)
                assigned.add(best)
                scores[best] = -float("inf")
            final_order.extend(children)

        perm = torch.tensor(final_order, device=self.W_enc_leaf.device)
        self.W_enc_leaf.data = self.W_enc_leaf.data[:, perm]
        self.b_enc_leaf.data = self.b_enc_leaf.data[perm]
        self.W_dec_leaf.data = self.W_dec_leaf.data[perm, :]


class TemporalCrosscoder(nn.Module):
    """Shared-latent temporal crosscoder (ckkissane-style).

    Encodes a window of T positions into a single shared sparse latent,
    then decodes back to T positions using per-position decoder weights.

    Architecture:
        z = TopK(sum_t W_enc[t] @ x_t + b_enc, k_total)
        x_hat_t = W_dec[t] @ z + b_dec[t]

    The encoder sums per-position projections into a single latent vector.
    The shared latent captures structure spanning the whole window.

    Sparsity can be specified either as:
    - ``k_per_pos``: total active features = ``k_per_pos * T`` (legacy behavior)
    - ``k_total``: fixed total active features across the whole shared latent
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k_per_pos: int | None = None,
        k_total: int | None = None,
        use_relu: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.use_relu = use_relu
        if k_total is not None and k_total <= 0:
            raise ValueError(f"k_total must be positive, got {k_total}")
        if k_per_pos is not None and k_per_pos <= 0:
            raise ValueError(f"k_per_pos must be positive, got {k_per_pos}")
        if k_total is None and k_per_pos is None:
            raise ValueError("Specify either k_per_pos or k_total")

        self.k_per_pos = k_per_pos
        self.k_total = k_total if k_total is not None else int(k_per_pos * T)
        if self.k_total > d_sae:
            raise ValueError(f"k_total={self.k_total} exceeds d_sae={d_sae}")

        # Per-position encoder: (T, d_in, d_sae)
        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Per-position decoder: (T, d_sae, d_in)
        self.W_dec = nn.Parameter(torch.empty(T, d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc[t], a=math.sqrt(5))
            with torch.no_grad():
                self.W_dec.data[t] = self.W_enc.data[t].T
        self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        # Joint norm across (T, d_in) for each latent
        norms = self.W_dec.data.pow(2).sum(dim=(0, 2), keepdim=True).sqrt().clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) -> z: (B, d_sae)"""
        # Sum per-position projections
        pre = torch.einsum("btd,tdm->bm", x, self.W_enc) + self.b_enc
        if self.use_relu:
            pre = torch.relu(pre)
        topk_vals, topk_idx = pre.topk(self.k_total, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_sae) -> x_hat: (B, T, d)"""
        return torch.einsum("bm,tmd->btd", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_hat, z). x: (B, T, d), x_hat: (B, T, d), z: (B, d_sae)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        x_hat, z = self.forward(x)
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        return recon_loss, {"recon_loss": recon_loss.item()}

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()


class MatryoshkaTemporalCrosscoder(TemporalCrosscoder):
    """Temporal crosscoder with Matryoshka-style nested prefix losses.

    The shared sparse latent is unchanged from TemporalCrosscoder. The only
    change is the training objective: we add reconstruction losses using only
    the first w latent channels for widths in ``matryoshka_widths``.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k_per_pos: int | None = None,
        k_total: int | None = None,
        matryoshka_widths: list[int] | None = None,
        inner_weight: float = 1.0,
        use_relu: bool = True,
    ):
        super().__init__(
            d_in=d_in, d_sae=d_sae, T=T,
            k_per_pos=k_per_pos, k_total=k_total, use_relu=use_relu,
        )
        self.inner_weight = inner_weight
        if matryoshka_widths is None:
            widths = []
            w = 4
            while w < d_sae:
                widths.append(w)
                w *= 2
            widths.append(d_sae)
            self.matryoshka_widths = widths
        else:
            self.matryoshka_widths = sorted(matryoshka_widths)
            if self.matryoshka_widths[-1] != d_sae:
                self.matryoshka_widths.append(d_sae)

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        x_hat, z = self.forward(x)
        full_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        inner_losses = []
        per_width = {self.matryoshka_widths[-1]: full_recon.item()}
        for w in self.matryoshka_widths[:-1]:
            x_hat_inner = torch.einsum("bm,tmd->btd", z[:, :w], self.W_dec[:, :w, :]) + self.b_dec
            inner_recon = (x - x_hat_inner).pow(2).sum(dim=-1).mean()
            inner_losses.append(inner_recon)
            per_width[w] = inner_recon.item()

        if inner_losses:
            inner_mean = sum(inner_losses) / len(inner_losses)
            total = (full_recon + self.inner_weight * inner_mean) / (1.0 + self.inner_weight)
        else:
            total = full_recon

        return total, {
            "total_loss": total.item(),
            "full_recon": full_recon.item(),
            "per_width": per_width,
        }


class MultiLayerCrosscoder(TemporalCrosscoder):
    """Cross-layer sparse crosscoder (Anthropic crosscoder-style).

    Reads ``L`` residual-stream positions — typically
    ``blocks.<layer>.hook_resid_post`` for several layers — at a single
    time step, encodes them to one shared sparse latent, and decodes
    per-layer reconstructions.

    The math is identical to ``TemporalCrosscoder`` with ``T`` replaced
    by ``L``; this class exists to make the semantics explicit (no
    time-windowing is applied — every (time-step, L-layer-stack) is an
    independent sample).

    Expected input shape is ``(B, L, d_in)`` where ``L`` is the number
    of layer hooks being read.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        L: int,
        k_per_layer: int | None = None,
        k_total: int | None = None,
        use_relu: bool = True,
    ):
        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            T=L,
            k_per_pos=k_per_layer,
            k_total=k_total,
            use_relu=use_relu,
        )
        self.L = L
        self.k_per_layer = k_per_layer


class MatryoshkaMultiLayerCrosscoder(MatryoshkaTemporalCrosscoder):
    """Matryoshka-style nested-width multi-layer crosscoder.

    Same relationship to :class:`MultiLayerCrosscoder` as
    :class:`MatryoshkaTemporalCrosscoder` has to :class:`TemporalCrosscoder`.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        L: int,
        k_per_layer: int | None = None,
        k_total: int | None = None,
        matryoshka_widths: list[int] | None = None,
        inner_weight: float = 1.0,
        use_relu: bool = True,
    ):
        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            T=L,
            k_per_pos=k_per_layer,
            k_total=k_total,
            matryoshka_widths=matryoshka_widths,
            inner_weight=inner_weight,
            use_relu=use_relu,
        )
        self.L = L
        self.k_per_layer = k_per_layer


class TemporalBatchTopKSAE(nn.Module):
    """Temporal Matryoshka BatchTopK SAE — port of the architecture from

        Oesterling et al., "Temporal Sparse Autoencoders: Leveraging the
        Sequential Nature of Language for Interpretability"
        https://github.com/AI4LIFE-GROUP/temporal-saes (Apache 2.0).

    Key ingredients:
      - Single-position SAE over activations ``x: (B, d_in)`` (no windowing
        during forward — compare to our :class:`TemporalCrosscoder` which
        consumes windows).
      - **BatchTopK** sparsity: topk is taken across the flattened ``(B *
        d_sae)`` tensor of pre-activations, then scattered back. Samples in the
        batch can carry different numbers of active features.
      - **Matryoshka groups**: ``d_sae`` is partitioned into ``group_sizes``.
        Reconstruction loss is computed cumulatively — first group, then first
        two groups, ..., each weighted by ``group_weights``.
      - **Temporal regulariser**: training receives pairs ``x_pair: (B, 2,
        d_in)`` of temporally-adjacent activations. The first-group features
        are pushed to agree across the pair, weighted by the cosine similarity
        between the activation pairs (L1 variant). Encourages smooth feature
        dynamics in time while not over-regularising when the underlying state
        actually changes.

    Intentional simplifications vs the reference implementation:
      - No aux-k (dead feature revival) loss — our training runs are short
        enough that dead features aren't an issue.
      - No threshold adaptation at inference — we use the topk scatter directly.
      - No contrastive temporal loss variant — the L1-cosine variant is the one
        the paper highlights.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        group_sizes: list[int],
        temporal: bool = True,
    ):
        super().__init__()
        assert sum(group_sizes) == d_sae, (
            f"group_sizes sum {sum(group_sizes)} != d_sae {d_sae}"
        )
        assert all(s > 0 for s in group_sizes), "all group sizes must be positive"
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = int(k)
        self.group_sizes = tuple(int(s) for s in group_sizes)
        self.active_groups = len(group_sizes)
        # Cumulative group boundaries (inclusive at start of group i).
        self.group_indices = [0]
        for s in self.group_sizes:
            self.group_indices.append(self.group_indices[-1] + s)
        self.temporal = bool(temporal)

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self._normalize_decoder()
        self.W_enc.data = self.W_dec.data.clone().T

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        # Per-feature decoder norms (each row is one feature's readout).
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()

    def encode(self, x: torch.Tensor, return_pre: bool = False) -> torch.Tensor:
        """Encode ``x: (B, d_in)`` → ``z: (B, d_sae)`` via BatchTopK."""
        pre = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        B = x.shape[0]
        flat = pre.reshape(-1)
        k_total = self.k * B
        if k_total > flat.numel():
            k_total = flat.numel()
        vals, idx = flat.topk(k_total, sorted=False)
        z = torch.zeros_like(flat).scatter_(-1, idx, vals).reshape_as(pre)
        # Zero any groups beyond `active_groups` (allows staged training).
        cut = self.group_indices[self.active_groups]
        if cut < self.d_sae:
            z[:, cut:] = 0
        if return_pre:
            return z, pre
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z

    def compute_loss(
        self,
        x_pair: torch.Tensor,
        group_weights: list[float],
        temporal_alpha: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Matryoshka + temporal loss.

        Args:
            x_pair: ``(B, 2, d_in)`` — x_pair[:, 0] is the "current" activation
                that we reconstruct; x_pair[:, 1] is the temporal neighbour
                used only in the regulariser.
            group_weights: length = number of groups; sums to roughly 1.
            temporal_alpha: weight of the temporal term vs reconstruction.
        """
        assert x_pair.dim() == 3 and x_pair.shape[1] == 2, (
            f"expected (B, 2, d_in), got {tuple(x_pair.shape)}"
        )
        x = x_pair[:, 0]
        x_next = x_pair[:, 1]

        f, _ = self.encode(x, return_pre=True)
        f_next, _ = self.encode(x_next, return_pre=True)

        W_dec_chunks = torch.split(self.W_dec, list(self.group_sizes), dim=0)
        f_chunks = torch.split(f, list(self.group_sizes), dim=1)
        f_next_chunks = torch.split(f_next, list(self.group_sizes), dim=1)

        # Cumulative reconstruction across groups
        x_hat = self.b_dec.unsqueeze(0).expand_as(x).clone()
        total_l2 = torch.zeros((), device=x.device, dtype=x.dtype)
        per_group_l2 = []
        for i in range(self.active_groups):
            x_hat = x_hat + f_chunks[i] @ W_dec_chunks[i]
            l2 = (x - x_hat).pow(2).sum(dim=-1).mean() * float(group_weights[i])
            total_l2 = total_l2 + l2
            per_group_l2.append(float(l2.item()))

        # Temporal regulariser: L1 between f[0] and f[0]-of-next, weighted by
        # cosine similarity of the raw activations (so the penalty scales down
        # when activations genuinely differ).
        if self.temporal:
            cos = torch.nn.functional.cosine_similarity(x, x_next, dim=-1)
            # Apply to first (high-level) group only — the paper's convention.
            temp_term = (torch.abs(f_chunks[0] - f_next_chunks[0]).sum(dim=-1) * cos * float(group_weights[0])).mean()
        else:
            temp_term = torch.zeros((), device=x.device, dtype=x.dtype)

        # Normalise reconstruction by number of groups to keep scale comparable
        mean_l2 = total_l2 / max(1, self.active_groups)
        loss = mean_l2 + temporal_alpha * temp_term
        return loss, {
            "total_l2": float(total_l2.item()),
            "mean_l2": float(mean_l2.item()),
            "temp_loss": float(temp_term.item()),
            "per_group_l2": per_group_l2,
        }
