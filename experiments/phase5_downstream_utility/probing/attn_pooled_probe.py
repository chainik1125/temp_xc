"""Attention-pooled probing baseline — Eq. 2 of Kantamneni et al. 2025.

The paper (`papers/are_saes_useful.md`, §5) identifies that SAE probes
can *appear* to outperform simple last-token logistic regression in
multi-token regimes, but that the advantage vanishes when the baseline
is replaced with an attention-pooled probe of the form:

    a_t = softmax_t(X_t · q)                    # attention weights, R^T
    v_t = X_t · v                               # per-position scores, R^T
    logit = a · v + b                           # scalar per example

where `q, v ∈ R^{d}` and `X ∈ R^{T × d}` is the per-position activation
tensor. This baseline is required in every Phase 5 headline comparison —
without it, any SAE-probe win on a multi-token task is unsafe.

Training:
- BCE loss on the sigmoid of the logit.
- Adam, 200 epochs with early stopping on validation AUC.
- Scalar output head has a learnable scale `s` to let the probe match
  logistic-regression-style absolute-scale logits; without it the softmax
  normalization can leave logits too close to zero for BCE to converge.

No L2 regularization, no weight decay — the d-dim q and v already have
very few parameters (~5k on gemma-2-2b-it) relative to the dataset sizes
SAEBench uses (4000 train examples). We are deliberately underpowering
this probe; it is a baseline, not a competitor.

Independent Phase 5 implementation from the paper equation — not ported
from Aniket's code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooledProbe(nn.Module):
    """Scalar attention-pooled logistic probe.

    Args:
        d: activation dimension at the probed layer.
    """

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * (1.0 / d**0.5))
        self.v = nn.Parameter(torch.randn(d) * (1.0 / d**0.5))
        # Scalar scale + bias on the pooled output — lets logistic regression
        # calibrate the magnitude. Equivalent to training an LR with one input.
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, X: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute logit for a batch.

        Args:
            X: (B, T, d) activation tensor. Padded positions are allowed;
               pass a boolean mask of shape (B, T) with True for real tokens.
            mask: optional (B, T) bool mask, True for attended positions.

        Returns:
            (B,) logit.
        """
        scores = torch.einsum("btd,d->bt", X, self.q)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        attn = F.softmax(scores, dim=-1)
        values = torch.einsum("btd,d->bt", X, self.v)
        pooled = (attn * values).sum(dim=-1)
        return self.scale * pooled + self.bias


@dataclass
class AttnProbeConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0  # intentionally unregularized baseline
    n_epochs: int = 200
    batch_size: int = 64
    patience: int = 20           # early stopping on validation AUC
    seed: int = 42


def _split_train_val(
    X: torch.Tensor,
    y: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """ROC-AUC; matches `sklearn.metrics.roc_auc_score` on binary data."""
    scores = logits.detach().cpu().numpy()
    labels = targets.detach().cpu().numpy().astype(int)
    from sklearn.metrics import roc_auc_score
    if labels.min() == labels.max():
        return 0.5
    return float(roc_auc_score(labels, scores))


def train_attn_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    mask_train: torch.Tensor | None = None,
    mask_test: torch.Tensor | None = None,
    cfg: AttnProbeConfig | None = None,
    device: torch.device | str = "cuda",
) -> dict[str, float]:
    """Fit the attention-pooled probe and return test AUC + metadata.

    Args:
        X_train, X_test: (N, T, d) activation tensors on any device.
        y_train, y_test: (N,) binary labels.
        mask_train, mask_test: optional (N, T) attention masks.
        cfg: optimizer config.

    Returns:
        Dict with `test_auc`, `best_val_auc`, `epochs_to_best`,
        `stopped_early`.
    """
    cfg = cfg or AttnProbeConfig()
    torch.manual_seed(cfg.seed)

    # Carve out 10% of train for validation. Reuse the same permutation
    # for X, y, and mask so they stay aligned.
    n = X_train.shape[0]
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * 0.1))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    Xtr, Xval = X_train[train_idx], X_train[val_idx]
    ytr, yval = y_train[train_idx], y_train[val_idx]
    if mask_train is not None:
        mtr = mask_train[train_idx].bool()
        mval = mask_train[val_idx].bool()
    else:
        mtr = mval = None

    d = Xtr.shape[-1]
    probe = AttentionPooledProbe(d).to(device)
    opt = torch.optim.Adam(
        probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    bce = nn.BCEWithLogitsLoss()

    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    best_val = 0.0
    best_epoch = 0
    patience = cfg.patience

    for epoch in range(cfg.n_epochs):
        probe.train()
        idx = torch.randperm(Xtr.shape[0])
        for i in range(0, Xtr.shape[0], cfg.batch_size):
            b = idx[i:i + cfg.batch_size]
            Xb = Xtr[b].to(device, non_blocking=True)
            yb = ytr[b].to(device, non_blocking=True).float()
            mb = mtr[b].to(device, non_blocking=True) if mtr is not None else None
            logit = probe(Xb, mb)
            loss = bce(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            val_logit = probe(Xval.to(device), mval.to(device) if mval is not None else None)
        val_auc = _auc(val_logit, yval)

        if val_auc > best_val:
            best_val = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            patience = cfg.patience
        else:
            patience -= 1
            if patience <= 0:
                break

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        test_logit = probe(
            X_test.to(device),
            mask_test.to(device) if mask_test is not None else None,
        )
    test_auc = _auc(test_logit, y_test)
    test_acc = float(
        ((test_logit.detach().cpu() > 0).long() == y_test.long()).float().mean()
    )

    return {
        "test_auc": test_auc,
        "test_acc": test_acc,
        "best_val_auc": best_val,
        "epochs_to_best": best_epoch,
        "stopped_early": patience <= 0,
        "n_params": 2 * d + 2,
    }
