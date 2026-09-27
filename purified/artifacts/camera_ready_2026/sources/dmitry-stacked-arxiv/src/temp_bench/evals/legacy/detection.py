"""Cross-case-study detection protocol — encode-and-pool → top-S probe → PR-AUC.

Detection complements steering on the three behavioural case studies
(C5 sentiment, C6 emergent misalignment, C7 backtracking). Where
steering asks "can we causally exploit the SAE's basis to flip the
behaviour?", detection asks "is the behaviour linearly readable from
the SAE's features?". Two near-orthogonal axes; neither subsumes the
other. C7 already implements its own version (``compute_pr_auc_at_S``);
this module hoists the protocol so C5 and C6 can adopt it without
copying ~150 lines of probing logic each.

The TXC-specific load-bearing rule is **encode with axis preserved**:
TXC's value over a TopK SAE is the per-position decoder trajectory.
Mean-pooling residuals before ``arch.encode`` collapses that axis
(SAEBench's recurring confound). The encode-and-pool contract this
module enforces is the opposite: stride-1 sliding T-windows over the
sentence/rollout/continuation, encode each window, ``amax`` over
windows. Position information survives encode; only the window-level
detection signal is pooled.

Usage::

    from temp_bench.eval.detection import detect_case_study, DetectionResult

    result = detect_case_study(
        arch=arch,                     # any TempBenchArch
        sentence_acts=X,               # (n_sent, T, d_in) — windowed residuals
        labels=y,                      # (n_sent,) 0/1 behavior labels
        question_ids=qids,             # (n_sent,) groups for GroupKFold
        S_grid=(1, 2, 4, 8, 16, 32),
        shuffle_seed=42,               # paired within-window shuffle ablation
    )
    # result.pr_auc:        {S: float}
    # result.pr_auc_shuffled: {S: float}
    # result.shuffle_gap:   {S: float}  — pr_auc - pr_auc_shuffled

The shuffle ablation is **mandatory** for any TXC detection cell. A
TXC's "temporal" PR-AUC that survives within-window token shuffle is
window-density, not temporal. Per-feature decision rule documented in
``docs/cross_component/det_steer_detection.md`` (gap ≥ 0.02 across S
to claim genuine temporal detection).

Scope: this module is **window-arch friendly** but does NOT assume
TXC-specific shapes. For per-token archs (TopK-SAE, MLC), pass the
sentence_acts in shape (n_sent, T=1, d_in) — encode returns
(n_sent, 1, d_sae) and the pool collapses the singleton axis. For
window archs (TXC, T-SAE, TFA, Stacked), shape is (n_sent, T_arch, d_in).
The encode-side stride-1 sliding-window construction for sentences
**longer** than T_arch lives in the case-study modules
(``case_studies.backtracking.extract_labeled_sentence_acts`` is the
template — Aniket's window-of-T-positions-before-each-sentence rule).

Compute footprint: encode in batches (cap d_sae × n_sent latents at
~4 GB peak); LogReg fit + score is CPU-bound and parallelisable across
S × folds via joblib (defer to caller).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch

from temp_bench.interfaces.architecture import TempBenchArch
from temp_bench.utils.shuffles import shuffle_within_window

DEFAULT_S_GRID: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


@dataclass
class DetectionResult:
    """Output of :func:`detect_case_study`.

    All numeric fields are means across folds.

    Fields:
        pr_auc: ``{S: PR-AUC}`` on unshuffled features. Headline number.
        pr_auc_shuffled: ``{S: PR-AUC}`` after within-window token
            permutation (None if ``shuffle_seed=None`` was passed).
        shuffle_gap: ``pr_auc[S] - pr_auc_shuffled[S]`` (None if not run).
        n_sent: number of input rows used (post-NaN-filter).
        positive_rate: fraction of positives in the cohort.
        n_folds: GroupKFold splits used.
        encode_shape: shape of the encoded feature matrix
            ``(n_sent, d_sae)`` after pooling.
        meta: free-form dict — useful for caller-attached context like
            ``{"arch": "txc_base", "case_study": "c7", "T": 5}``.
    """
    pr_auc: dict[int, float]
    pr_auc_shuffled: dict[int, float] | None
    shuffle_gap: dict[int, float] | None
    n_sent: int
    positive_rate: float
    n_folds: int
    encode_shape: tuple[int, int]
    meta: dict[str, Any] = field(default_factory=dict)


def encode_and_pool(
    arch: TempBenchArch,
    sentence_acts: np.ndarray | torch.Tensor,
    *,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """TXC-aware encode-and-pool contract.

    Input ``sentence_acts``: ``(n_sent, T, d_in)`` — one T-window per
    cohort element. (The case study is responsible for stride-1
    windowing over longer sequences before calling this.)

    Pipeline per batch:
      1. ``arch.encode(x)`` → ``(B, T_z, d_sae)`` with ``T_z ∈ {1, T}``.
         For TXC-base/pro: 1 (window-level latent).
         For per-token archs: T (per-token latent).
      2. ``.abs()`` (matches Aniket's mining convention — sign-agnostic).
      3. ``amax(dim=1)`` if ``T_z > 1`` else squeeze. Max-pool over the
         window axis: a feature that fires sharply on one position is
         the detection signal we want; mean-pool would dilute it.

    Returns ``(n_sent, d_sae)`` numpy float32.
    """
    if isinstance(sentence_acts, np.ndarray):
        x_full = torch.from_numpy(sentence_acts)
    else:
        x_full = sentence_acts
    if x_full.dim() != 3:
        raise ValueError(
            f"encode_and_pool expects (n_sent, T, d_in); got {tuple(x_full.shape)}"
        )
    if device is None:
        device = next(arch.parameters()).device
    n = x_full.shape[0]
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = x_full[i:i + batch_size].to(device)
            z = arch.encode(xb).abs()
            if z.dim() == 3 and z.shape[1] > 1:
                z = z.amax(dim=1)
            elif z.dim() == 3 and z.shape[1] == 1:
                z = z.squeeze(1)
            chunks.append(z.detach().to(torch.float32).cpu().numpy())
            del z, xb
    return np.concatenate(chunks, axis=0)


def _pr_auc_from_features(
    feature_acts: np.ndarray,
    labels: np.ndarray,
    question_ids: np.ndarray | None,
    *,
    S_grid: tuple[int, ...],
    n_folds: int,
    C: float,
    random_state: int,
) -> dict[int, float]:
    """Sparse-probe PR-AUC at top-S features with GroupKFold-by-qid.

    Per-fold:
      * Top-S feature selection on the **train** fold via D+/D- mean
        difference (no test-fold leakage).
      * Fit ``LogisticRegression(penalty="l1", C=C, solver="liblinear")``
        on those S features.
      * Score test fold via ``average_precision_score``.

    Mirrors :func:`temp_bench.case_studies.backtracking.compute_pr_auc_at_S`
    intentionally — they're the same function. We re-implement here so
    C5 / C6 can depend on this module without round-tripping through
    the C7 case study.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold, KFold

    if question_ids is not None:
        cv = GroupKFold(n_splits=n_folds)
        splits = list(cv.split(feature_acts, labels, groups=question_ids))
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(cv.split(feature_acts, labels))

    pr_auc: dict[int, float] = {}
    for S in S_grid:
        fold_aps: list[float] = []
        for train_idx, test_idx in splits:
            X_tr, X_te = feature_acts[train_idx], feature_acts[test_idx]
            y_tr, y_te = labels[train_idx], labels[test_idx]
            mean_pos = X_tr[y_tr == 1].mean(axis=0) if (y_tr == 1).any() else np.zeros(X_tr.shape[1])
            mean_neg = X_tr[y_tr == 0].mean(axis=0) if (y_tr == 0).any() else np.zeros(X_tr.shape[1])
            md = np.abs(mean_pos - mean_neg)
            top = np.argsort(md)[-S:]
            clf = LogisticRegression(
                penalty="l1",
                C=C,
                solver="liblinear",
                max_iter=2000,
                random_state=random_state,
            )
            clf.fit(X_tr[:, top], y_tr)
            proba = clf.predict_proba(X_te[:, top])[:, 1]
            fold_aps.append(float(average_precision_score(y_te, proba)))
        pr_auc[S] = float(np.mean(fold_aps))
    return pr_auc


def detect_case_study(
    arch: TempBenchArch,
    sentence_acts: np.ndarray,
    labels: np.ndarray,
    question_ids: np.ndarray | None = None,
    *,
    S_grid: tuple[int, ...] = DEFAULT_S_GRID,
    n_folds: int = 5,
    C: float = 1.0,
    random_state: int = 42,
    shuffle_seed: int | None = 42,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
    meta: dict[str, Any] | None = None,
) -> DetectionResult:
    """Run detection PR-AUC + within-window shuffle ablation on one (arch, cohort).

    Args:
        arch: any :class:`TempBenchArch`. Per-token archs accept
            ``(n_sent, 1, d_in)``; window archs (TXC, T-SAE, TFA, MLC,
            Stacked) accept ``(n_sent, T_arch, d_in)``.
        sentence_acts: ``(n_sent, T, d_in)`` float numpy array. Caller
            is responsible for the encode-side windowing convention
            (stride-1 windows over longer sequences are case-study
            responsibility — see ``case_studies.backtracking``'s
            ``extract_labeled_sentence_acts`` for the C7 pattern).
        labels: ``(n_sent,)`` 0/1 binary behavior labels.
        question_ids: ``(n_sent,)`` group ids for GroupKFold. Pass ``None``
            to fall back to KFold (NOT recommended for case studies —
            same-prompt sentences will leak across folds).
        S_grid: top-S grid for the sparse probe. Default ``(1,2,4,8,16,32)``.
        n_folds: GroupKFold splits. 5 is the case-study default.
        C: L1 LogReg regularisation strength. Larger = less penalty.
        random_state: seed for KFold + LogReg.
        shuffle_seed: seed for the within-window shuffle ablation. Pass
            ``None`` to skip the ablation (not recommended for TXC
            cells — see findings doc).
        batch_size: encode-batch size to cap GPU memory.
        device: where to encode (default: arch's device).
        meta: free-form annotations attached to the result.
    """
    if sentence_acts.ndim != 3:
        raise ValueError(
            f"sentence_acts must be (n_sent, T, d_in); got {sentence_acts.shape}"
        )
    n_sent, T, d_in = sentence_acts.shape
    if labels.shape != (n_sent,):
        raise ValueError(
            f"labels shape {labels.shape} mismatches n_sent={n_sent}"
        )
    if question_ids is not None and question_ids.shape != (n_sent,):
        raise ValueError(
            f"question_ids shape {question_ids.shape} mismatches n_sent={n_sent}"
        )

    # ── Unshuffled encode + probe ──
    X = encode_and_pool(arch, sentence_acts, batch_size=batch_size, device=device)
    pr_auc = _pr_auc_from_features(
        X, labels, question_ids,
        S_grid=S_grid, n_folds=n_folds, C=C, random_state=random_state,
    )

    pr_auc_shuffled: dict[int, float] | None = None
    shuffle_gap: dict[int, float] | None = None
    if shuffle_seed is not None and T > 1:
        # Shuffle within window then re-encode + re-probe.
        x_t = torch.from_numpy(sentence_acts) if isinstance(sentence_acts, np.ndarray) else sentence_acts
        x_shuffled = shuffle_within_window(x_t, T=T, seed=shuffle_seed)
        X_sh = encode_and_pool(
            arch, x_shuffled, batch_size=batch_size, device=device,
        )
        pr_auc_shuffled = _pr_auc_from_features(
            X_sh, labels, question_ids,
            S_grid=S_grid, n_folds=n_folds, C=C, random_state=random_state,
        )
        shuffle_gap = {S: pr_auc[S] - pr_auc_shuffled[S] for S in S_grid}

    return DetectionResult(
        pr_auc=pr_auc,
        pr_auc_shuffled=pr_auc_shuffled,
        shuffle_gap=shuffle_gap,
        n_sent=n_sent,
        positive_rate=float(labels.mean()),
        n_folds=n_folds,
        encode_shape=tuple(X.shape),
        meta=meta or {},
    )


def detection_table(
    results: dict[str, DetectionResult],
    S_grid: tuple[int, ...] = DEFAULT_S_GRID,
) -> str:
    """Render a markdown table of detection results across architectures.

    ``results`` is keyed by arch name. The output has columns
    ``S, <arch1>, <arch2>, ...`` with sub-rows for ``shuffled`` and
    ``gap`` per arch when shuffle results are available.
    """
    archs = list(results.keys())
    if not archs:
        return "(no detection results)"

    has_shuffle = any(r.shuffle_gap is not None for r in results.values())

    lines = []
    header = ["S", *(f"{a} unshuf" for a in archs)]
    if has_shuffle:
        header.extend(f"{a} shuf" for a in archs)
        header.extend(f"{a} gap" for a in archs)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for S in S_grid:
        row = [str(S)]
        for a in archs:
            row.append(f"{results[a].pr_auc.get(S, float('nan')):.3f}")
        if has_shuffle:
            for a in archs:
                sh = results[a].pr_auc_shuffled
                row.append(f"{sh[S]:.3f}" if sh is not None else "—")
            for a in archs:
                gap = results[a].shuffle_gap
                row.append(f"{gap[S]:+.3f}" if gap is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
