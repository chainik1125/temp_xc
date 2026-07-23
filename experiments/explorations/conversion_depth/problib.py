"""Shared probe stack for the conversion-depth exploration.

Faithful port of the FreqBench sprint probe conventions
(`origin/dmitry-spectral-sprint2` `fb_core.fit_probe` + `bt_freq.probe_with_auc`),
extended with the § 8 execution rule from the synthetic README:
raw-access ceilings are **threshold-optimized** (a plain probe can sit at
chance under class imbalance while real access exists), and binary probes
report rank-AUC (threshold-free) as the primary ceiling statistic.

The stack is FROZEN after phase-1 validation on GPT-2 day-stride
(RECORD.md § 1) — no per-target retuning downstream. Probe budget scales
only with input dim (full-batch Adam, fixed epochs/lr/wd per the sprint).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sprint-frozen hyperparameters (fb_core.fit_probe defaults; bt_freq used
# epochs=300, lr=1e-2, wd=1e-4 for the class-weighted binary variant too).
EPOCHS = 300
LR = 1e-2
WD = 1e-4
MLP_HIDDEN = 512


def _standardize(train: torch.Tensor, test: torch.Tensor):
    """Per-dim z-score on TRAIN stats (fb_core convention)."""
    mu = train.mean(0, keepdim=True)
    sd = train.std(0, keepdim=True).clamp(min=1e-6)
    return (train - mu) / sd, (test - mu) / sd


def rank_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney rank AUC (bt_freq.auc port)."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _balacc(pred: torch.Tensor, y: torch.Tensor) -> float:
    m1, m0 = y == 1, y == 0
    return 0.5 * ((pred[m1] == 1).float().mean().item()
                  + (pred[m0] == 0).float().mean().item())


def _opt_threshold(logit_tr: torch.Tensor, ytr: torch.Tensor) -> float:
    """Threshold over train logits maximizing train balanced accuracy
    (the § 8 'threshold-optimized ceiling' rule; optimized on TRAIN only)."""
    qs = torch.quantile(logit_tr, torch.linspace(0.01, 0.99, 99,
                                                 device=logit_tr.device))
    best_t, best_b = 0.0, -1.0
    for t in qs.tolist():
        b = _balacc((logit_tr > t).long(), ytr)
        if b > best_b:
            best_b, best_t = b, t
    return best_t


def fit_probe(feats_tr, y_tr, feats_te, y_te, n_classes: int,
              hidden: int = 0, epochs: int = EPOCHS, lr: float = LR,
              wd: float = WD, seed: int = 0,
              class_weight: bool = False) -> dict:
    """Linear (hidden=0) or 1-hidden-layer MLP probe, full-batch Adam.

    Port of fb_core.fit_probe; `class_weight=True` adds bt_freq's inverse-
    frequency CE weights (binary). Binary probes additionally report
    rank-AUC and threshold-optimized balanced accuracy.

    Inputs may be float16/bfloat16 tensors on any device; standardization
    happens on DEVICE at fp32.
    """
    torch.manual_seed(seed)
    ftr = feats_tr.to(DEVICE).float()
    fte = feats_te.to(DEVICE).float()
    ytr = y_tr.to(DEVICE).long()
    yte = y_te.to(DEVICE).long()
    ftr, fte = _standardize(ftr, fte)
    D = ftr.shape[1]
    if hidden:
        probe = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(),
                              nn.Linear(hidden, n_classes)).to(DEVICE)
    else:
        probe = nn.Linear(D, n_classes).to(DEVICE)
    if class_weight and n_classes == 2:
        w = torch.tensor([1.0, ((ytr == 0).sum() / (ytr == 1).sum()).item()],
                         device=DEVICE)
    else:
        w = None
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        loss = F.cross_entropy(probe(ftr), ytr, weight=w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits_tr, logits_te = probe(ftr), probe(fte)
        acc_tr = (logits_tr.argmax(-1) == ytr).float().mean().item()
        pred_te = logits_te.argmax(-1)
        acc_te = (pred_te == yte).float().mean().item()
        per_class = []
        for c in range(n_classes):
            m = yte == c
            per_class.append(((pred_te[m] == c).float().mean().item()
                              if m.any() else float("nan")))
        out = {"acc_train": acc_tr, "acc_test": acc_te,
               "per_class": per_class, "n_train": int(ytr.numel()),
               "n_test": int(yte.numel())}
        if n_classes == 2:
            lt = logits_te[:, 1] - logits_te[:, 0]
            l0 = logits_tr[:, 1] - logits_tr[:, 0]
            out["auc"] = rank_auc(lt.cpu().numpy(), yte.cpu().numpy())
            out["balacc"] = _balacc((lt > 0).long(), yte)
            thr = _opt_threshold(l0, ytr)
            out["balacc_opt"] = _balacc((lt > thr).long(), yte)
            out["n_pos_test"] = int((yte == 1).sum())
    del ftr, fte
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out


def ceilings_for_target(Xtr_tok, Xte_tok, Xtr_win, Xte_win, ytr, yte,
                        n_classes: int, *, seed: int = 0,
                        class_weight: bool = False) -> dict:
    """The three § 1 ceilings for one (layer, target) cell.

    - per-token linear ceiling  : linear probe on the single-position acts
    - window linear ceiling     : linear probe on the flattened T-window
    - presence checks (MLP)     : MLP(512) on both, so blindness ≠ absence
    """
    return {
        "per_token_linear": fit_probe(Xtr_tok, ytr, Xte_tok, yte, n_classes,
                                      seed=seed, class_weight=class_weight),
        "window_linear": fit_probe(Xtr_win, ytr, Xte_win, yte, n_classes,
                                   seed=seed, class_weight=class_weight),
        "per_token_mlp": fit_probe(Xtr_tok, ytr, Xte_tok, yte, n_classes,
                                   hidden=MLP_HIDDEN, seed=seed,
                                   class_weight=class_weight),
        "window_mlp": fit_probe(Xtr_win, ytr, Xte_win, yte, n_classes,
                                hidden=MLP_HIDDEN, seed=seed,
                                class_weight=class_weight),
    }
