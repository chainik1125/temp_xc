"""Sparse probing evaluation — C3 (Gemma-IT-L13).

Shared so we don't re-implement the SAEBench-style probe for every
component that wants it. PROTOCOL.md § 11 *Code reuse contract*.

Public API:

- :func:`mean_pool_probe(model, X_train, y_train, X_test, y_test, k_feat) -> dict`
  — operates on PRE-ENCODED features; just does top-k selection +
  L1 logistic regression.
- :func:`s_tail_probe(model, X_train, y_train, X_test, y_test, S, k_feat) -> dict`
  — operates on RAW activation sequences; encodes via ``model``,
  mean-pools over the last-S tokens, then delegates to
  :func:`mean_pool_probe`.
- :func:`run_task_suite(model, tasks, k_feats, S) -> list[dict]`
  — loops :func:`s_tail_probe` over every task × k_feat combination.

Convention (Kantamneni et al. § 2.2 + Phase 7 S-tail):

1. Feature aggregation:
   - Per-token archs (``model.T == 1``): encode every position, mean-pool
     latent vectors over the last-S tokens.
   - Window archs (``model.T > 1``): slide a T-window over the last-S
     tokens, encode each window (collapses to ``(B, 1, d_sae)``),
     mean-pool the window-level latents over (S - T + 1) windows.
2. Top-k feature selection on TRAIN: sort by
   ``|mean(z[y=1]) - mean(z[y=0])|``, take top-k_feat.
3. L1 logistic regression on the top-k features; report test AUC + acc.

Validity rule: ``S >= T`` (Phase 7 corrected formula). The wrapper
enforces this and raises on misconfiguration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from temp_bench.architectures.base import TempBenchArch


def mean_pool_probe(
    model: TempBenchArch,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_feat: int,
    n_jobs: int = -1,
) -> dict[str, float]:
    """Top-k sparse logistic-regression probe on PRE-ENCODED features.

    Args:
        model: kept for API parity (unused — features already encoded).
        X_train, X_test: ``(N, d_sae)`` arrays of mean-pooled SAE latents.
        y_train, y_test: ``(N,)`` int 0/1 labels.
        k_feat: top-k features by class-mean absolute difference.
        n_jobs: liblinear is single-threaded; kept for API parity.

    Returns:
        dict with ``"auc"``, ``"acc"``, ``"k_feat"``, ``"n_train"``,
        ``"n_test"``.
    """
    del model, n_jobs  # unused (kept in signature for the upstream contract)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train).astype(int).ravel()
    y_test = np.asarray(y_test).astype(int).ravel()

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            f"mean_pool_probe expects 2-D X arrays; got "
            f"train={X_train.shape}, test={X_test.shape}"
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"d_sae mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}"
        )

    d_sae = X_train.shape[1]
    if k_feat > d_sae:
        raise ValueError(f"k_feat={k_feat} > d_sae={d_sae}")

    # Top-k features by absolute class-mean difference (on TRAIN only).
    pos = y_train == 1
    neg = y_train == 0
    if not pos.any() or not neg.any():
        raise ValueError("Train labels must contain both classes (0 and 1).")
    diff = np.abs(X_train[pos].mean(axis=0) - X_train[neg].mean(axis=0))
    top_idx = np.argsort(diff)[-k_feat:]

    Xtr = X_train[:, top_idx]
    Xte = X_test[:, top_idx]

    # L1 logistic regression — Kantamneni convention. liblinear handles L1.
    # Note: sklearn 1.8 deprecates `penalty=` kwarg in favour of `l1_ratio=` for
    # newer solvers, but liblinear-with-L1 still uses penalty='l1'. Wrap to
    # silence the FutureWarning until sklearn drops the old API entirely.
    import warnings as _warn
    with _warn.catch_warnings():
        _warn.simplefilter("ignore", FutureWarning)
        _warn.simplefilter("ignore", UserWarning)
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0, max_iter=1000, random_state=0,
        )
        clf.fit(Xtr, y_train)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # NOTE: only float metrics returned (LeaderboardRow.metrics validates each
    # value as float). Categorical / int diagnostics (agg, S, k_feat, n_train,
    # n_test) belong in eval_cfg or a separate metadata dict.
    return {
        "auc": float(roc_auc_score(y_test, proba)),
        "acc": float(accuracy_score(y_test, pred)),
    }


def s_tail_probe(
    model: TempBenchArch,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    S: int,
    k_feat: int,
    encode_batch_size: int = 64,
    device: str | torch.device | None = None,
    n_jobs: int = -1,
) -> dict[str, float]:
    """Phase 7 S-tail protocol — encode last-S tokens, mean-pool, probe.

    Args:
        model: subclass of ``TempBenchArch``; in eval mode.
        X_train, X_test: ``(N, seq_len, d_in)`` activation arrays. ``seq_len``
            must be ``>= S``; we slice the last-S tail.
        y_train, y_test: ``(N,)`` int 0/1 labels.
        S: tail length (Phase 7 default = 32). Must be ``>= model.T``.
        k_feat: top-k feature selection.
        encode_batch_size: forward-pass micro-batch size.
        device: ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect from model).

    Returns:
        dict (see :func:`mean_pool_probe`) plus ``"S"`` and ``"agg"``
        ('per_token' or 'window').
    """
    if S < model.T:
        raise ValueError(
            f"S={S} < T={model.T}; Phase 7 S-tail protocol requires S >= T."
        )
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            f"s_tail_probe expects (N, seq_len, d_in) arrays; got "
            f"train={X_train.shape}, test={X_test.shape}"
        )
    if X_train.shape[1] < S or X_test.shape[1] < S:
        raise ValueError(
            f"seq_len < S={S}: train={X_train.shape[1]}, test={X_test.shape[1]}"
        )

    device = torch.device(device) if device is not None else next(model.parameters()).device

    train_feats = _encode_pool(model, X_train, S=S, batch_size=encode_batch_size, device=device)
    test_feats = _encode_pool(model, X_test, S=S, batch_size=encode_batch_size, device=device)

    return mean_pool_probe(
        model, X_train=train_feats, y_train=y_train,
        X_test=test_feats, y_test=y_test, k_feat=k_feat, n_jobs=n_jobs,
    )


def run_task_suite(
    model: TempBenchArch,
    *,
    tasks: list[dict[str, Any]],
    k_feats: tuple[int, ...] = (5, 20),
    S: int = 32,
    encode_batch_size: int = 64,
    n_jobs: int = -1,
) -> list[dict[str, Any]]:
    """Run :func:`s_tail_probe` over every task × k_feat combination.

    Args:
        tasks: list of dicts with keys
            ``"task_name"``, ``"X_train"``, ``"y_train"``,
            ``"X_test"``, ``"y_test"``.
            X-arrays are pre-cached activations ``(N, seq_len, d_in)``
            from a probe-cache builder (analog of the wasteland's
            ``build_probe_cache.py``).
        k_feats: tuple of feature-selection sparsities to evaluate.
        S: tail length (Phase 7 default = 32).
        encode_batch_size: forward-pass micro-batch size.

    Returns:
        list of result dicts, one per (task, k_feat) cell. Each row has
        ``task_name`` (str), ``k_feat`` (int), and the metric keys
        (``auc``, ``acc``) as floats. S and agg are constant per call
        and live on the caller's eval_cfg, not the per-row metrics
        (LeaderboardRow.metrics validates each value as float).
    """
    out: list[dict[str, Any]] = []
    for task in tasks:
        for k in k_feats:
            r = s_tail_probe(
                model,
                X_train=task["X_train"], y_train=task["y_train"],
                X_test=task["X_test"], y_test=task["y_test"],
                S=S, k_feat=k,
                encode_batch_size=encode_batch_size, n_jobs=n_jobs,
            )
            # task_name + k_feat are categorical/int — kept on the row for
            # downstream aggregation but NOT inside the float-only metrics.
            row = {"task_name": task["task_name"], "k_feat": int(k), **r}
            out.append(row)
    return out


# ── Internal helpers ────────────────────────────────────────────────────────


def _encode_pool(
    model: TempBenchArch,
    X: np.ndarray,
    *,
    S: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Encode last-S tokens of each sequence and mean-pool latents.

    Returns ``(N, d_sae)`` numpy array. Per-token vs window aggregation
    is dispatched on ``model.T``.
    """
    N = X.shape[0]
    tail = X[:, -S:, :]  # (N, S, d_in)
    out: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        if model.T == 1:
            # Per-token encode then mean-pool over S
            for start in range(0, N, batch_size):
                batch = torch.from_numpy(tail[start:start + batch_size]).to(device)
                z = model.encode(batch)                    # (B, S, d_sae)
                pooled = z.mean(dim=1)                     # (B, d_sae)
                out.append(pooled.float().cpu().numpy())
        else:
            # Window encode: slide T-window over the S-tail
            T = model.T
            n_windows = S - T + 1
            for start in range(0, N, batch_size):
                batch = torch.from_numpy(tail[start:start + batch_size]).to(device)  # (B, S, d_in)
                # Stack windows along a new dim so we can encode in one batched pass
                # via reshape: (B, S, d_in) -> (B * n_windows, T, d_in).
                # (Memory: B * n_windows * T * d_in floats — for B=64, S=32, T=5,
                # n_windows=28: 64*28*5*d_in. At d_in=2304: ~17 MB. Fine.)
                wins = torch.stack(
                    [batch[:, w:w + T] for w in range(n_windows)], dim=1
                )                                          # (B, n_windows, T, d_in)
                B = wins.shape[0]
                wins_flat = wins.reshape(B * n_windows, T, batch.shape[-1])
                z = model.encode(wins_flat)                # (B*n_windows, 1, d_sae)
                z = z.squeeze(1).reshape(B, n_windows, -1)  # (B, n_windows, d_sae)
                pooled = z.mean(dim=1)                      # (B, d_sae)
                out.append(pooled.float().cpu().numpy())

    return np.concatenate(out, axis=0)
