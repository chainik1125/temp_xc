"""Adapters to load sweep-trained src.bench.architectures.* checkpoints into
the NLP autointerp pipeline.

The sweep (src/bench/sweep.py) saves state dicts from the training-time
classes: StackedSAE (with saes.N.* sub-modules), TemporalCrosscoder,
TemporalSAE (TFA). The autointerp pipeline was originally written against
fast_models.{FastStackedSAE, FastTemporalCrosscoder}, which pack per-
position weights into stacked tensors and don't have a TFA sibling at all.
Rather than re-saving checkpoints, we load the training classes directly
and expose the (loss, x_hat, feat_acts) signature TopKFinder expects.

For TFA, the adapter exposes **novel_codes only** as feat_acts. This is
the sparse (topk) per-token signal — the natural analog to StackedSAE's
per-token z. pred_codes are dense (all d_sae entries potentially nonzero)
and represent attention-predicted context carryover; if we included them
in feat_acts, every feature would appear to "fire" at every position and
the top-K ranking would degenerate into magnitude ordering of dense
vectors. pred_codes are preserved on the adapter (set keep_pred_novel=True)
for downstream analyses that want to classify features as novel-driven
vs pred-driven.
"""

import math
import torch
import torch.nn as nn


def load_bench_stacked_sae(d_in: int, d_sae: int, T: int, k: int) -> nn.Module:
    from src.bench.architectures.stacked_sae import StackedSAE
    return StackedSAE(d_in, d_sae, T, k)


def load_bench_crosscoder(d_in: int, d_sae: int, T: int, k: int) -> nn.Module:
    from src.bench.architectures.crosscoder import TemporalCrosscoder
    return TemporalCrosscoder(d_in, d_sae, T, k)


class BenchTFAAdapter(nn.Module):
    """Wraps TemporalSAE so autointerp sees (loss, x_hat, feat_acts).

    feat_acts: (B, T, d_sae) = novel_codes only (sparse, topk per token).
    TopKFinder treats this identically to StackedSAE (mean-over-T → per-
    window activation for heap ranking).

    Scaling: TemporalSAE was trained with x rescaled so ||x|| ~ sqrt(d),
    per TFASpec. The scaling factor is not in the state dict; we recompute
    it from the first batch passed through forward() — the activation
    cache is the same distribution used at training time, so the recomputed
    factor matches within sampling noise.

    Pred/novel decomposition is stashed on `self.last_novel` and
    `self.last_pred` for analysis code that wants to split features by
    origin. Not written every forward in production (would inflate memory)
    — set `keep_pred_novel=True` to enable.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int,
        use_pos_encoding: bool = True,
        n_heads: int = 4,
        n_attn_layers: int = 1,
        bottleneck_factor: int = 8,
        keep_pred_novel: bool = False,
        feat_source: str = "novel",
    ):
        assert feat_source in ("novel", "pred", "sum"), feat_source
        super().__init__()
        from src.bench.architectures._tfa_module import TemporalSAE
        self._inner = TemporalSAE(
            dimin=d_in,
            width=d_sae,
            n_heads=n_heads,
            sae_diff_type="topk",
            kval_topk=k,
            tied_weights=True,
            n_attn_layers=n_attn_layers,
            bottleneck_factor=bottleneck_factor,
            use_pos_encoding=use_pos_encoding,
        )
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        self._scaling_factor: float | None = None
        self.keep_pred_novel = keep_pred_novel
        self.feat_source = feat_source
        self.last_novel: torch.Tensor | None = None
        self.last_pred: torch.Tensor | None = None

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        if self._scaling_factor is None:
            d = x.shape[-1]
            mean_norm = x.norm(dim=-1).mean().item()
            self._scaling_factor = (
                math.sqrt(d) / mean_norm if mean_norm > 1e-8 else 1.0
            )
        return x * self._scaling_factor

    def forward(self, x: torch.Tensor):
        x_scaled = self._scale(x)
        x_recons, inter = self._inner(x_scaled)
        novel = inter["novel_codes"]   # (B, T, d_sae) — sparse (topk)
        pred = inter["pred_codes"]     # (B, T, d_sae) — dense
        if self.keep_pred_novel:
            self.last_novel = novel.detach()
            self.last_pred = pred.detach()
        if self.feat_source == "novel":
            feat_acts = novel
        elif self.feat_source == "pred":
            feat_acts = pred
        else:  # "sum"
            feat_acts = novel + pred
        loss = (x_recons - x_scaled).pow(2).sum(dim=-1).mean()
        return loss, x_recons, feat_acts
